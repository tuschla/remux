use remux_sdks::remux::{AudioCodec, TranscodeReasons, VideoCodec};
use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::AtomicU32},
    time::{Duration, Instant},
};
use tokio::sync::{Notify, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum TranscodeState {
    Starting,
    Running,
    Complete,
    Error(String),
}

/// Container ffmpeg writes HLS segments in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentContainer {
    Ts,
    Fmp4,
}

impl SegmentContainer {
    /// iOS Safari (and the HLS spec) require fMP4 for HEVC, so copied HEVC
    /// video gets fMP4 segments; everything else uses MPEG-TS.
    pub fn for_codecs(video_codec: &str, source_video_codec: Option<&str>) -> Self {
        let hevc_copy = video_codec == "copy"
            && source_video_codec
                .and_then(|s| {
                    s.parse::<VideoCodec>()
                        .ok()
                })
                .is_some_and(|codec| codec.is_hevc());
        if hevc_copy { Self::Fmp4 } else { Self::Ts }
    }

    /// Whether source audio can be stream-copied into these segments.
    ///
    /// TrueHD, FLAC, and PCM have no MPEG-TS stream type. fMP4 carries FLAC,
    /// but TrueHD in MP4 is experimental in ffmpeg and browsers can't play
    /// PCM in fMP4. Unknown codecs are assumed copyable.
    pub fn can_copy_audio(self, source_audio_codec: Option<&str>) -> bool {
        match source_audio_codec.and_then(|s| {
            s.parse::<AudioCodec>()
                .ok()
        }) {
            Some(AudioCodec::TrueHd | AudioCodec::Pcm) => false,
            Some(AudioCodec::Flac) => self == Self::Fmp4,
            _ => true,
        }
    }
}

/// Client requests and playlist discontinuities of a live session.
pub struct LiveState {
    /// Last playlist or segment request. The engine stops the ffmpeg of a
    /// client that went away without a Stop (`engine::LIVE_IDLE`).
    last_request: Mutex<Instant>,
    /// Media sequence numbers of segments that follow an
    /// `#EXT-X-DISCONTINUITY`, kept after they slide out of the playlist.
    discontinuities: Mutex<BTreeSet<u64>>,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            last_request: Mutex::new(Instant::now()),
            discontinuities: Mutex::default(),
        }
    }
}

impl LiveState {
    /// Record a playlist or segment request.
    pub fn touch(&self) {
        *self
            .last_request
            .lock()
            .expect("last_request lock is not poisoned") = Instant::now();
    }

    /// Time since the last playlist or segment request.
    pub fn idle_for(&self) -> Duration {
        self.last_request
            .lock()
            .expect("last_request lock is not poisoned")
            .elapsed()
    }

    /// `#EXT-X-DISCONTINUITY-SEQUENCE` of a sliding-window live playlist: the
    /// number of discontinuities before its first segment (RFC 8216 §6.2.2).
    /// ffmpeg doesn't write the tag, and a discontinuity that slid out of the
    /// window can't be counted from the playlist alone, so every served
    /// playlist records the discontinuities it lists.
    pub fn discontinuity_sequence(&self, playlist: &str) -> u64 {
        let mut seen = self
            .discontinuities
            .lock()
            .expect("discontinuities lock is not poisoned");
        let mut media_sequence: u64 = 0;
        let mut next = 0;
        let mut after_discontinuity = false;
        for line in playlist
            .lines()
            .map(str::trim)
        {
            if let Some(n) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                media_sequence = n
                    .parse()
                    .unwrap_or(0);
                next = media_sequence;
            } else if line == "#EXT-X-DISCONTINUITY" {
                after_discontinuity = true;
            } else if !line.is_empty() && !line.starts_with('#') {
                if after_discontinuity {
                    seen.insert(next);
                    after_discontinuity = false;
                }
                next += 1;
            }
        }
        seen.range(..media_sequence)
            .count() as u64
    }
}

pub struct TranscodeSession {
    pub id: String,
    pub item_id: Uuid,
    pub media_source_id: Uuid,
    pub output_dir: PathBuf,
    pub input_url: String,
    pub state: TranscodeState,
    /// Broadcasts state transitions so waiters can react immediately.
    pub state_tx: Arc<watch::Sender<TranscodeState>>,
    pub created_at: Instant,
    pub video_codec: String,
    pub audio_codec: String,
    pub audio_stream_index: Option<i32>,
    pub subtitle_stream_index: Option<i32>,
    pub burn_subtitle: bool,
    pub segment_length: u32,
    pub transcode_reasons: TranscodeReasons,
    /// Kill channel and done notifier for the ffmpeg subprocess.
    pub kill_tx: Option<tokio::sync::oneshot::Sender<()>>,
    pub wait_done: Arc<Notify>,
    /// Cancelled when the session is stopped, so a live session doesn't
    /// restart ffmpeg afterwards.
    pub stopped: CancellationToken,
    /// Used by live sessions only.
    pub live_state: Arc<LiveState>,
    /// Index of the last segment the client has requested (0-based).
    pub last_segment_index: Arc<AtomicU32>,
    /// Start offset of this transcode in seconds (from start_time_ticks).
    pub start_time_secs: u32,
    /// Playback offset in seconds relative to start_time_secs, updated from progress reports.
    pub playback_offset_secs: Arc<AtomicU32>,
    /// Total runtime of the media in Jellyfin ticks (100-ns units).
    pub runtime_ticks: i64,
    /// True for live TV — variant playlist is served from the ffmpeg-written
    /// sliding-window file.
    pub is_live: bool,
    /// Codec name of the source video stream (e.g. "hevc", "h264"), used when
    /// rebuilding `TranscodeParams` on seek-restart.
    pub source_video_codec: Option<String>,
    /// Codec name of the source audio stream (e.g. "eac3", "aac").
    pub source_audio_codec: Option<String>,
    /// HEVC sample-entry tag (`hvc1`/`hev1`) resolved from the client's
    /// DeviceProfile, used when stream-copying HEVC into fMP4.
    pub hevc_copy_tag: Option<String>,
    /// Profile of the source video stream (e.g. "Main 10"), used to generate
    /// the correct HLS CODECS attribute string for HEVC.
    pub source_video_profile: Option<String>,
    /// Level of the source video stream (e.g. 150.0 for level 5.0).
    pub source_video_level: Option<f64>,
    /// HDR type of the source video (SDR/HDR10/HLG/…), used for VIDEO-RANGE in master playlist.
    pub source_video_range_type: Option<remux_sdks::remux::VideoRangeType>,
    /// Source video width in pixels, used for RESOLUTION in master playlist.
    pub source_video_width: Option<i64>,
    /// Source video height in pixels, used for RESOLUTION in master playlist.
    pub source_video_height: Option<i64>,
    /// Source video frame rate (fps), used for FRAME-RATE in master playlist.
    pub source_frame_rate: Option<f32>,
    /// Target output video bitrate in bps (None for direct-copy or unknown).
    pub video_bitrate: Option<u32>,
    /// Hardware acceleration type used for this transcode, e.g. "VAAPI", "QSV".
    pub hardware_acceleration_type: Option<String>,
}

impl TranscodeSession {
    pub fn new(
        play_session_id: String,
        item_id: Uuid,
        media_source_id: Uuid,
        input_url: String,
        output_dir: PathBuf,
        video_codec: String,
        audio_codec: String,
        audio_stream_index: Option<i32>,
        subtitle_stream_index: Option<i32>,
        burn_subtitle: bool,
        segment_length: u32,
        transcode_reasons: TranscodeReasons,
        runtime_ticks: i64,
        is_live: bool,
        source_video_codec: Option<String>,
        source_audio_codec: Option<String>,
        hevc_copy_tag: Option<String>,
        source_video_profile: Option<String>,
        source_video_level: Option<f64>,
        source_video_range_type: Option<remux_sdks::remux::VideoRangeType>,
        source_video_width: Option<i64>,
        source_video_height: Option<i64>,
        source_frame_rate: Option<f32>,
        video_bitrate: Option<u32>,
        hardware_acceleration_type: Option<String>,
    ) -> Arc<tokio::sync::RwLock<Self>> {
        let _ = std::fs::create_dir_all(&output_dir);
        let (state_tx, _) = watch::channel(TranscodeState::Starting);
        Arc::new(tokio::sync::RwLock::new(Self {
            id: play_session_id,
            item_id,
            media_source_id,
            output_dir,
            input_url,
            state: TranscodeState::Starting,
            state_tx: Arc::new(state_tx),
            created_at: Instant::now(),
            video_codec,
            audio_codec,
            audio_stream_index,
            subtitle_stream_index,
            burn_subtitle,
            segment_length,
            transcode_reasons,
            kill_tx: None,
            wait_done: Arc::new(Notify::new()),
            stopped: CancellationToken::new(),
            live_state: Arc::default(),
            last_segment_index: Arc::new(AtomicU32::new(0)),
            start_time_secs: 0,
            playback_offset_secs: Arc::new(AtomicU32::new(0)),
            runtime_ticks,
            is_live,
            source_video_codec,
            source_audio_codec,
            hevc_copy_tag,
            source_video_profile,
            source_video_level,
            source_video_range_type,
            source_video_width,
            source_video_height,
            source_frame_rate,
            video_bitrate,
            hardware_acceleration_type,
        }))
    }

    pub fn master_playlist_path(&self) -> PathBuf {
        self.output_dir
            .join("master.m3u8")
    }

    pub fn variant_playlist_path(&self) -> PathBuf {
        self.output_dir
            .join("main.m3u8")
    }

    pub fn segment_container(&self) -> SegmentContainer {
        SegmentContainer::for_codecs(
            &self.video_codec,
            self.source_video_codec
                .as_deref(),
        )
    }

    /// Returns true if this session should use fragmented MP4 (fMP4) segments
    /// rather than MPEG-TS.
    pub fn use_fmp4(&self) -> bool {
        self.segment_container() == SegmentContainer::Fmp4
    }

    pub fn segment_path(&self, segment_id: &str) -> PathBuf {
        let ext = if self.use_fmp4() { "m4s" } else { "ts" };
        self.output_dir
            .join(format!("{}.{}", segment_id, ext))
    }

    /// Path of the fMP4 initialization segment written by ffmpeg.
    pub fn init_segment_path(&self) -> PathBuf {
        self.output_dir
            .join("init.mp4")
    }

    /// Generate a Jellyfin-compatible transcoding URL
    pub fn transcoding_url(&self) -> String {
        format!(
            "/videos/{}/master.m3u8?PlaySessionId={}&VideoCodec={}&AudioCodec=aac&SegmentContainer=ts&SegmentLength={}&MediaSourceId={}",
            self.item_id
                .as_simple(),
            self.id,
            self.video_codec,
            self.segment_length,
            self.media_source_id
                .as_simple(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discontinuity_sequence_counts_discontinuities_that_slid_out() {
        let live_state = LiveState::default();
        // ffmpeg restarted with append_list: segment 17 follows a discontinuity.
        let before = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n\
                      #EXT-X-MEDIA-SEQUENCE:15\n\
                      #EXTINF:6.0,\nsegment_00015.ts\n\
                      #EXTINF:6.0,\nsegment_00016.ts\n\
                      #EXT-X-DISCONTINUITY\n\
                      #EXTINF:6.0,\nsegment_00017.ts\n";
        // Still listed: the tag itself carries it.
        assert_eq!(live_state.discontinuity_sequence(before), 0);
        let after = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n\
                     #EXT-X-MEDIA-SEQUENCE:18\n\
                     #EXTINF:6.0,\nsegment_00018.ts\n\
                     #EXT-X-DISCONTINUITY\n\
                     #EXTINF:6.0,\nsegment_00019.ts\n";
        // Segment 17's discontinuity slid out; 19's is still listed.
        assert_eq!(live_state.discontinuity_sequence(after), 1);
        // Both slid out.
        let later =
            "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:20\n#EXTINF:6.0,\nsegment_00020.ts\n";
        assert_eq!(live_state.discontinuity_sequence(later), 2);
    }
}

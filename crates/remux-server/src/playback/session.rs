use remux_sdks::remux::{AudioCodec, TranscodeReasons, VideoCodec};
use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
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

/// Least time without a segment or playlist request after which a client
/// that left a transcode for a newer one has abandoned it. Once its buffer
/// is full, a playing HLS client (hls.js) requests one segment per segment
/// duration; twice that, and at least this, leaves room for jitter. The
/// longer this wait, the more abandoned retries run at once (see
/// `PlaybackSessionManager::supersede_abandoned_transcodes`).
const ABANDONED_AFTER_MIN: Duration = Duration::from_secs(10);

/// The segment and playlist requests of a transcode's client.
#[derive(Debug, Default)]
pub struct RequestActivity {
    in_flight: AtomicU32,
    times: Mutex<RequestTimes>,
}

#[derive(Debug, Default)]
struct RequestTimes {
    /// When the last request started.
    started: Option<Instant>,
    /// When the last request started or finished.
    active: Option<Instant>,
}

impl RequestActivity {
    /// Marks a request in flight until the returned guard drops: a client
    /// waiting for a slow segment is still on this transcode.
    pub fn begin(self: &Arc<Self>) -> ActiveRequest {
        self.in_flight
            .fetch_add(1, Ordering::Relaxed);
        let now = Some(Instant::now());
        *self.times() = RequestTimes {
            started: now,
            active: now,
        };
        ActiveRequest(self.clone())
    }

    pub fn in_flight(&self) -> bool {
        self.in_flight
            .load(Ordering::Relaxed)
            > 0
    }

    /// When the last request started; `None` if there was none.
    pub fn last_started(&self) -> Option<Instant> {
        self.times()
            .started
    }

    /// When the last request started or finished; `None` if there was none.
    pub fn last_active(&self) -> Option<Instant> {
        self.times()
            .active
    }

    fn times(&self) -> std::sync::MutexGuard<'_, RequestTimes> {
        self.times
            .lock()
            .expect("request activity lock is not poisoned")
    }

    /// Backdates the last request (tests can't advance `Instant`).
    #[cfg(test)]
    pub fn requested_at(&self, at: Instant) {
        *self.times() = RequestTimes {
            started: Some(at),
            active: Some(at),
        };
    }
}

/// A request in flight; see `RequestActivity::begin`.
pub struct ActiveRequest(Arc<RequestActivity>);

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.0
            .times()
            .active = Some(Instant::now());
        self.0
            .in_flight
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// Client requests and playlist discontinuities of a live session.
///
/// Playlist and segment requests are recorded so the engine can stop the live
/// ffmpeg of a client that went away without a Stop and restart it when the
/// client comes back (see `engine::LIVE_IDLE`).
pub struct LiveActivity {
    last_request: Mutex<Instant>,
    /// Media sequence numbers of segments that follow an
    /// `#EXT-X-DISCONTINUITY`, kept after they slide out of the playlist.
    discontinuities: Mutex<BTreeSet<u64>>,
}

impl Default for LiveActivity {
    fn default() -> Self {
        Self {
            last_request: Mutex::new(Instant::now()),
            discontinuities: Mutex::default(),
        }
    }
}

impl LiveActivity {
    /// Record a playlist or segment request.
    pub fn touch(&self) {
        *self
            .last_request
            .lock()
            .expect("last_request lock is not poisoned") = Instant::now();
    }

    /// Time since the last playlist or segment request.
    pub fn idle_for(&self) -> Duration {
        self.idle_at(Instant::now())
    }

    fn idle_at(&self, now: Instant) -> Duration {
        now.saturating_duration_since(
            *self
                .last_request
                .lock()
                .expect("last_request lock is not poisoned"),
        )
    }

    /// `#EXT-X-DISCONTINUITY-SEQUENCE` of a sliding-window live playlist: the
    /// number of discontinuities before its first segment (RFC 8216 §6.2.2).
    /// ffmpeg doesn't write the tag, and a discontinuity that slid out of the
    /// window can't be counted from the playlist alone, so every playlist read
    /// records the discontinuities it lists.
    pub fn discontinuity_sequence(&self, playlist: &str) -> u64 {
        let (media_sequence, found) = playlist_discontinuities(playlist);
        let mut seen = self
            .discontinuities
            .lock()
            .expect("discontinuities lock is not poisoned");
        seen.extend(found);
        seen.range(..media_sequence)
            .count() as u64
    }
}

/// The playlist's `#EXT-X-MEDIA-SEQUENCE` and the media sequence numbers of
/// its segments that follow an `#EXT-X-DISCONTINUITY`.
fn playlist_discontinuities(playlist: &str) -> (u64, Vec<u64>) {
    let mut media_sequence = 0;
    let mut next = 0;
    let mut pending = false;
    let mut found = Vec::new();
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
            pending = true;
        } else if !line.is_empty() && !line.starts_with('#') {
            if pending {
                found.push(next);
                pending = false;
            }
            next += 1;
        }
    }
    (media_sequence, found)
}

pub struct TranscodeSession {
    pub id: String,
    /// The user whose request created this transcode.
    pub user_id: Uuid,
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
    /// Cancelled when the session is stopped: the transcode loop never
    /// (re)spawns ffmpeg once cancelled, so a live session doesn't restart
    /// ffmpeg afterwards.
    pub stopped: CancellationToken,
    /// Segment and playlist requests: one played elsewhere goes quiet.
    pub requests: Arc<RequestActivity>,
    /// Client requests and playlist discontinuities, used for live sessions.
    pub live_activity: Arc<LiveActivity>,
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
        user_id: Uuid,
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
            user_id,
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
            requests: Default::default(),
            live_activity: Arc::default(),
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

    /// Whether the client has made no segment or playlist request for long
    /// enough to have abandoned this transcode, as of `now`, if its segments
    /// are as long as requested. Alone this also holds for a paused player;
    /// see `left_for`.
    pub fn abandoned(&self, now: Instant) -> bool {
        self.abandoned_with_segments_of(now, Duration::ZERO)
    }

    /// `abandoned`, for segments up to `longest_segment` long: with a copied
    /// video, segments end at the source's keyframes, so a playing client
    /// may request less often than the requested length suggests.
    pub fn abandoned_with_segments_of(
        &self,
        now: Instant,
        longest_segment: Duration,
    ) -> bool {
        let segment =
            longest_segment.max(Duration::from_secs(u64::from(self.segment_length)));
        let after = ABANDONED_AFTER_MIN.max(2 * segment);
        let idle_since = self
            .requests
            .last_active()
            .unwrap_or(self.created_at);
        !self
            .requests
            .in_flight()
            && now.saturating_duration_since(idle_since) >= after
    }

    /// Whether the client has requested nothing of this transcode since a
    /// transcode created at `other_created_at` (another of its item, on its
    /// device) began: a retry's predecessor goes quiet before the retry
    /// exists, while a second player of the item (a browser tab) requested
    /// after the other one started, however long it is paused since. A
    /// transcode never requested lost to any other.
    pub fn left_for(&self, other_created_at: Instant) -> bool {
        self.requests
            .last_started()
            .is_none_or(|started| started < other_created_at)
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

/// The longest `#EXTINF` duration of an HLS playlist (zero if none).
pub fn longest_segment(playlist: &str) -> Duration {
    playlist
        .lines()
        .filter_map(|line| {
            let secs = line
                .strip_prefix("#EXTINF:")?
                .split(',')
                .next()?
                .trim()
                .parse::<f64>()
                .ok()?;
            Duration::try_from_secs_f64(secs).ok()
        })
        .max()
        .unwrap_or_default()
}
#[cfg(test)]
mod tests {
    use super::*;

    fn session(base: &std::path::Path, segment_length: u32) -> TranscodeSession {
        let ts = TranscodeSession::new(
            "play-session".to_string(),
            Uuid::nil(),
            Uuid::nil(),
            Uuid::nil(),
            String::new(),
            base.join("play-session"),
            "copy".to_string(),
            "aac".to_string(),
            None,
            None,
            false,
            segment_length,
            Default::default(),
            0,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        Arc::into_inner(ts)
            .expect("only reference")
            .into_inner()
    }

    #[test]
    fn a_transcode_is_abandoned_after_two_segment_durations_without_requests() {
        let temp = tempfile::tempdir().unwrap();
        let start = Instant::now();
        for (segment_length, after) in [(6, 12), (10, 20), (2, 10)] {
            let ts = session(temp.path(), segment_length);
            ts.requests
                .requested_at(start);
            let at = |secs| start + Duration::from_secs(secs);
            assert!(!ts.abandoned(at(after - 1)), "{segment_length}");
            assert!(ts.abandoned(at(after)), "{segment_length}");
        }
    }

    #[test]
    fn a_request_in_flight_keeps_its_transcode_from_being_abandoned() {
        let temp = tempfile::tempdir().unwrap();
        let ts = session(temp.path(), 6);
        let much_later = Instant::now() + Duration::from_secs(60);
        let request = ts
            .requests
            .begin();
        assert!(
            ts.requests
                .in_flight()
        );
        assert!(!ts.abandoned(much_later));

        // Idle again from the answer on.
        drop(request);
        assert!(
            !ts.requests
                .in_flight()
        );
        let answered = ts
            .requests
            .last_active()
            .expect("a request was answered");
        assert!(!ts.abandoned(answered + Duration::from_secs(11)));
        assert!(ts.abandoned(answered + Duration::from_secs(12)));
    }

    #[test]
    fn a_transcode_never_requested_is_idle_since_its_creation() {
        let temp = tempfile::tempdir().unwrap();
        let ts = session(temp.path(), 6);
        assert!(!ts.abandoned(ts.created_at + Duration::from_secs(11)));
        assert!(ts.abandoned(ts.created_at + Duration::from_secs(12)));
        // It lost to any other transcode, older or newer.
        assert!(ts.left_for(ts.created_at - Duration::from_secs(1)));
        assert!(ts.left_for(ts.created_at + Duration::from_secs(1)));
    }

    #[test]
    fn a_transcode_is_left_only_for_one_created_after_its_last_request() {
        let temp = tempfile::tempdir().unwrap();
        let ts = session(temp.path(), 6);
        let start = Instant::now();
        ts.requests
            .requested_at(start);
        assert!(ts.left_for(start + Duration::from_millis(1)));
        assert!(!ts.left_for(start));
        assert!(!ts.left_for(start - Duration::from_secs(30)));
    }

    #[test]
    fn longer_segments_than_requested_take_longer_to_abandon() {
        let temp = tempfile::tempdir().unwrap();
        let ts = session(temp.path(), 6);
        let start = Instant::now();
        ts.requests
            .requested_at(start);
        let at = |secs| start + Duration::from_secs(secs);
        // A copied video with 16 s keyframe intervals.
        let longest = Duration::from_secs(16);
        assert!(ts.abandoned(at(12)));
        assert!(!ts.abandoned_with_segments_of(at(31), longest));
        assert!(ts.abandoned_with_segments_of(at(32), longest));
        // Shorter segments than requested don't shorten the wait.
        assert!(!ts.abandoned_with_segments_of(at(11), Duration::from_secs(2)));
    }

    #[test]
    fn longest_segment_reads_the_playlist_durations() {
        let playlist = "#EXTM3U\n\
            #EXT-X-VERSION:3\n\
            #EXT-X-TARGETDURATION:17\n\
            #EXTINF:16.016000,\n\
            segment_00000.ts\n\
            #EXTINF:4.170833,\n\
            segment_00001.ts\n";
        assert_eq!(longest_segment(playlist), Duration::from_secs_f64(16.016));
        assert_eq!(longest_segment("#EXTM3U\n"), Duration::ZERO);
    }

    #[test]
    fn live_activity_measures_time_since_the_last_request() {
        let activity = LiveActivity::default();
        let now = Instant::now();
        assert!(activity.idle_at(now) < Duration::from_secs(1));
        assert!(
            activity.idle_at(now + Duration::from_secs(121)) > Duration::from_secs(120)
        );
        activity.touch();
        assert!(activity.idle_at(Instant::now()) < Duration::from_secs(1));
    }

    #[test]
    fn discontinuity_sequence_counts_discontinuities_that_slid_out() {
        let activity = LiveActivity::default();
        // ffmpeg restarted with append_list: segment 17 follows a discontinuity.
        let before = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n\
                      #EXT-X-MEDIA-SEQUENCE:15\n\
                      #EXTINF:6.0,\nsegment_00015.ts\n\
                      #EXTINF:6.0,\nsegment_00016.ts\n\
                      #EXT-X-DISCONTINUITY\n\
                      #EXTINF:6.0,\nsegment_00017.ts\n";
        // Still listed: the tag itself carries it.
        assert_eq!(activity.discontinuity_sequence(before), 0);
        let after = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n\
                     #EXT-X-MEDIA-SEQUENCE:18\n\
                     #EXTINF:6.0,\nsegment_00018.ts\n\
                     #EXT-X-DISCONTINUITY\n\
                     #EXTINF:6.0,\nsegment_00019.ts\n";
        // Segment 17's discontinuity slid out; 19's is still listed.
        assert_eq!(activity.discontinuity_sequence(after), 1);
        // Both slid out.
        let later =
            "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:20\n#EXTINF:6.0,\nsegment_00020.ts\n";
        assert_eq!(activity.discontinuity_sequence(later), 2);
    }

    #[test]
    fn playlist_discontinuities_number_segments_from_the_media_sequence() {
        let playlist = "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:7\n\
                        #EXT-X-DISCONTINUITY\n#EXTINF:6.0,\nsegment_00007.ts\n\
                        #EXTINF:6.0,\nsegment_00008.ts\n\
                        #EXT-X-DISCONTINUITY\n#EXTINF:6.0,\nsegment_00009.ts\n";
        assert_eq!(playlist_discontinuities(playlist), (7, vec![7, 9]));
        assert_eq!(playlist_discontinuities("#EXTM3U\n"), (0, vec![]));
    }
}

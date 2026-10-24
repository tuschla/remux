use anyhow::{Result, anyhow};
#[cfg(unix)]
use libc;
use std::{
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::{
    common::{HideConsole, TickUnit, ToRunTimeTicks},
    device_profile::{AudioCodec, VideoCodec},
    playback::hw_accel::{Accelerator, HdrTreatment, NoAccel, TonemapOptions},
};
use remux_sdks::remux::{EncodingPreset, HardwareAccelerationType, VideoRangeType};

use super::session::{
    LiveActivity, SegmentContainer, TranscodeSession, TranscodeState,
};

pub async fn detect_hardware_acceleration() -> HardwareAccelerationType {
    let detected = probe_hw_accel().await;
    info!(hw_accel = ?detected, "Hardware acceleration detected");
    detected
}

/// Detect the VAAPI driver name for the primary DRM render node.
/// Returns "iHD" for Intel (vendor 0x8086), empty string for others.
/// Probe the VAAPI device by running ffmpeg in verbose mode and inspecting
/// the driver string it reports.  Mirrors Jellyfin's `CheckVaapiDeviceByDriverName`.
///
/// Returns "iHD" for Intel iHD driver, "i965" for Intel legacy, or "" if
/// the device is unavailable or the driver is unknown.
pub async fn detect_vaapi_driver(vaapi_device: &str) -> String {
    let device = if vaapi_device.is_empty() {
        "/dev/dri/renderD128"
    } else {
        vaapi_device
    };

    let result = tokio::process::Command::new(ffmpeg_bin())
        .args([
            "-v",
            "verbose",
            "-hide_banner",
            "-init_hw_device",
            &format!("vaapi=va:{device}"),
        ])
        .hide_console()
        .output()
        .await;

    let output = match result {
        Ok(o) => String::from_utf8_lossy(&o.stderr).into_owned(),
        Err(e) => {
            warn!("Could not probe VAAPI device for driver detection: {e}");
            return String::new();
        }
    };

    if output.contains("Intel iHD driver") {
        "iHD".to_string()
    } else if output.contains("Intel i965 driver") {
        "i965".to_string()
    } else {
        String::new()
    }
}

/// Probe whether ffmpeg can derive an OpenCL device from the VAAPI device and
/// has `tonemap_opencl`. Mirrors Jellyfin's `IsOpenclFullSupported`: without
/// an OpenCL runtime (e.g. no intel-compute-runtime) the derive fails.
pub async fn detect_opencl_tonemap(vaapi_device: &str, vaapi_driver: &str) -> bool {
    let device = if vaapi_device.is_empty() {
        "/dev/dri/renderD128"
    } else {
        vaapi_device
    };
    let driver_opt = if vaapi_driver.is_empty() {
        String::new()
    } else {
        format!(",driver={vaapi_driver}")
    };

    let filters = match tokio::process::Command::new(ffmpeg_bin())
        .args(["-hide_banner", "-filters"])
        .hide_console()
        .output()
        .await
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(e) => {
            warn!("Could not list ffmpeg filters for OpenCL detection: {e}");
            return false;
        }
    };
    if !filters
        .split_whitespace()
        .any(|w| w == "tonemap_opencl")
    {
        return false;
    }

    match tokio::process::Command::new(ffmpeg_bin())
        .args([
            "-v",
            "error",
            "-hide_banner",
            "-init_hw_device",
            &format!("vaapi=va:{device}{driver_opt}"),
            "-init_hw_device",
            "opencl=ocl@va",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=64x64",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ])
        .hide_console()
        .output()
        .await
    {
        Ok(o) => o
            .status
            .success(),
        Err(e) => {
            warn!("Could not probe OpenCL device: {e}");
            false
        }
    }
}

async fn probe_hw_accel() -> HardwareAccelerationType {
    let supported = match tokio::process::Command::new(ffmpeg_bin())
        .args(["-hide_banner", "-hwaccels"])
        .hide_console()
        .output()
        .await
    {
        Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(
                "ffmpeg not found — transcoding and hardware acceleration detection will not work. \
                 Set the FFMPEG_PATH environment variable to the ffmpeg binary path."
            );
            String::new()
        }
        Err(e) => {
            warn!("Could not run ffmpeg to detect hwaccels: {e}");
            String::new()
        }
    };

    select_hw_accel(
        &supported,
        |p| std::path::Path::new(p).exists(),
        || {
            std::fs::read_to_string("/sys/class/drm/renderD128/device/vendor")
                .ok()
                .map(|s| {
                    s.trim()
                        .to_string()
                })
        },
    )
}

pub(crate) fn select_hw_accel(
    hwaccels_output: &str,
    device_exists: impl Fn(&str) -> bool,
    drm_vendor: impl Fn() -> Option<String>,
) -> HardwareAccelerationType {
    let has = |name: &str| {
        hwaccels_output
            .lines()
            .any(|l| l.trim() == name)
    };

    let has_render_node = device_exists("/dev/dri/renderD128");
    let is_intel = drm_vendor().as_deref() == Some("0x8086");

    if has("cuda") && device_exists("/dev/nvidia0") {
        HardwareAccelerationType::Nvenc
    } else if has("qsv") && has_render_node && is_intel {
        HardwareAccelerationType::Qsv
    } else if has("vaapi") && has_render_node {
        HardwareAccelerationType::Vaapi
    } else if has("videotoolbox") && cfg!(target_os = "macos") {
        HardwareAccelerationType::VideoToolbox
    } else if has("v4l2m2m") && device_exists("/dev/video0") {
        HardwareAccelerationType::V4l2m2m
    } else if has("rkmpp") && device_exists("/dev/mpp_service") {
        HardwareAccelerationType::Rkmpp
    } else {
        HardwareAccelerationType::None
    }
}

/// Max seconds to buffer ahead of the current playback position.
const MAX_BUFFER_SECS: u32 = 300;
/// Seconds behind the playback position before a segment is eligible for deletion.
const SEGMENT_KEEP_SECS: u32 = 30;
/// Seconds of live output listed in the sliding-window live playlist. A
/// jellyfin-web (hls.js) client that reports real positions can pause this
/// long and resume in place, as with the EVENT playlist live TV had before,
/// where the buffer throttle stopped ffmpeg `MAX_BUFFER_SECS` ahead.
const LIVE_WINDOW_SECS: u32 = MAX_BUFFER_SECS;
/// How far ffmpeg may get ahead of the newest segment the client requested,
/// once the live source turned out to be read faster than real time (a file
/// or VOD playlist; see `LiveRun`). Half the window: copying a file writes up
/// to a minute of output per second and the throttle checks once a second, so
/// the client's next segment stays listed. Also the lead over the wall clock
/// that marks a source as read faster than real time: real live sources send
/// a backlog on connect (the HLS demuxer starts 3 segments behind the live
/// edge, servers burst their buffer), which must not mark them.
const LIVE_THROTTLE_SECS: u32 = LIVE_WINDOW_SECS / 2;
/// How often the live maintenance task throttles ffmpeg, deletes old segments
/// and checks for an abandoned client.
const LIVE_MAINTENANCE_INTERVAL: Duration = Duration::from_secs(1);
/// A live session without a playlist or segment request for this long has
/// lost its client: players reload a live playlist every few seconds, also
/// while paused. Its ffmpeg is stopped until the next request.
const LIVE_IDLE: Duration = Duration::from_secs(120);
/// ffmpeg (`-v error`) messages showing that a live input failed rather than
/// ended. ffmpeg still exits with status 0 after them.
const INPUT_ERROR_MESSAGES: &[&str] = &[
    // fftools: any read error other than EOF; not fatal without `-xerror`.
    "Error during demuxing",
    // libavformat/http.c: the body ended before its Content-Length, or a body
    // without Content-Length or chunked framing ended. Every end of such a
    // live HTTP stream logs this, also a clean close by the server.
    "Stream ends prematurely",
];
/// Log contexts of the network protocols and the HLS demuxer: their errors
/// (connection reset, TLS, timeouts, segment failures) are input failures.
const INPUT_ERROR_CONTEXTS: &[&str] =
    &["[http @", "[https @", "[tcp @", "[tls @", "[hls @"];
/// Upper bound for the upstream playlist read by `upstream_hls_playlist_ends`.
const MAX_UPSTREAM_PLAYLIST_BYTES: usize = 2 * 1024 * 1024;

/// Entries in the live playlist window (`-hls_list_size`).
fn live_window_segments(segment_length: u32) -> u32 {
    LIVE_WINDOW_SECS.div_ceil(segment_length)
}

/// Segments below this index can be deleted from a live session.
///
/// Live clients' reported positions can't be trusted (Jellyfin Android TV
/// reports 0, then epoch-based values after an error), and a paused client
/// stops requesting segments while ffmpeg keeps writing. So cleanup trails the
/// playlist window, plus `SEGMENT_KEEP_SECS` for players still holding an
/// older copy of the playlist. Segments behind that are no longer listed, so
/// players don't request them.
fn live_cleanup_cutoff(highest: u32, segment_length: u32) -> u32 {
    highest.saturating_sub(
        live_window_segments(segment_length) + SEGMENT_KEEP_SECS / segment_length,
    )
}

/// Whether a live run that started writing at segment `first` and reached
/// `highest` in `elapsed` got more than `LIVE_THROTTLE_SECS` ahead of the wall
/// clock, which a live source can't, even with a backlog sent on connect.
fn outran_the_clock(
    first: u32,
    highest: Option<u32>,
    segment_length: u32,
    elapsed: Duration,
) -> bool {
    let produced_secs = highest
        .filter(|&h| h >= first)
        .map_or(0, |h| u64::from(h - first + 1) * u64::from(segment_length));
    produced_secs > elapsed.as_secs() + u64::from(LIVE_THROTTLE_SECS)
}

/// The ffmpeg runs of a live session, shared by the transcode loop and
/// `spawn_live_maintenance`.
///
/// Xtream VOD, M3U entries pointing at files and VOD upstream playlists are TV
/// channels too. Their URL, ffprobe duration or playlist type don't tell them
/// apart from real live channels reliably, but their pace does: ffmpeg reads
/// them faster than real time. Such a source is throttled to the client's
/// requests. The mark only drives the throttle; whether the source ended is
/// decided from how ffmpeg exited (`live_source_ended`).
struct LiveRun {
    /// First segment and start time of the current run.
    current: std::sync::Mutex<(u32, std::time::Instant)>,
    /// Some run outran the wall clock. Never reset: restarts read the same
    /// source.
    faster_than_realtime: AtomicBool,
    /// The maintenance task stopped ffmpeg because the client went away.
    idle_stopped: AtomicBool,
}

impl Default for LiveRun {
    fn default() -> Self {
        Self {
            current: std::sync::Mutex::new((0, std::time::Instant::now())),
            faster_than_realtime: AtomicBool::new(false),
            idle_stopped: AtomicBool::new(false),
        }
    }
}

impl LiveRun {
    /// A new ffmpeg run starts writing at `first_segment`.
    fn start(&self, first_segment: u32) {
        *self
            .current
            .lock()
            .expect("live run lock is not poisoned") =
            (first_segment, std::time::Instant::now());
    }

    /// Whether the source is read faster than real time, given the highest
    /// segment on disk. Latches once the current run outran the clock.
    fn faster_than_realtime(&self, highest: Option<u32>, segment_length: u32) -> bool {
        if self
            .faster_than_realtime
            .load(Ordering::Relaxed)
        {
            return true;
        }
        let (first, started) = *self
            .current
            .lock()
            .expect("live run lock is not poisoned");
        let outran =
            outran_the_clock(first, highest, segment_length, started.elapsed());
        if outran {
            self.faster_than_realtime
                .store(true, Ordering::Relaxed);
        }
        outran
    }
}

/// Throttle a live ffmpeg by the newest segment the client requested
/// (`requested`), never by reported positions (#146).
///
/// Only a source read faster than real time is paused. A real live source
/// can't get ahead of a client that follows the live edge, and a paused
/// client must not stop it either: a stopped live ffmpeg freezes the playlist
/// and ExoPlayer fails with PlaylistStuckException. Its client can pause for
/// the length of the window instead.
fn live_throttle(
    highest: Option<u32>,
    requested: u32,
    segment_length: u32,
    paused: bool,
    faster_than_realtime: bool,
) -> Option<ThrottleSignal> {
    let ahead = highest
        .map_or(0, |h| h.saturating_sub(requested))
        .saturating_mul(segment_length);
    if !paused && faster_than_realtime && ahead >= LIVE_THROTTLE_SECS {
        Some(ThrottleSignal::Pause { ahead })
    } else if paused && ahead < LIVE_THROTTLE_SECS.saturating_sub(segment_length * 2) {
        Some(ThrottleSignal::Resume { ahead })
    } else {
        None
    }
}

/// What follows a live ffmpeg run.
#[derive(Debug, PartialEq, Eq)]
enum LiveRunEnd {
    /// The session was stopped.
    Stopped,
    /// The client went away: wait for its next request, then restart.
    Idle,
    /// The source ended: end the playlist.
    Ended,
    /// Upstream failure: restart ffmpeg.
    Restart,
}

/// Decide what follows a live ffmpeg run. `source_ended` is
/// `live_source_ended`; every other exit restarts. A source that ended wins
/// over an idle stop that raced its exit.
fn live_run_end(
    stopped: bool,
    idle_stopped: bool,
    killed: bool,
    source_ended: bool,
) -> LiveRunEnd {
    if stopped {
        LiveRunEnd::Stopped
    } else if source_ended {
        LiveRunEnd::Ended
    } else if idle_stopped {
        LiveRunEnd::Idle
    } else if killed {
        LiveRunEnd::Stopped
    } else {
        LiveRunEnd::Restart
    }
}

/// Whether ffmpeg's stderr (`-v error`) shows that the input failed; see
/// `INPUT_ERROR_MESSAGES` and `INPUT_ERROR_CONTEXTS`. Decoder errors don't
/// count: a damaged frame doesn't end the input.
fn stderr_reports_input_error(stderr: &str) -> bool {
    stderr
        .lines()
        .map(str::trim_start)
        .any(|line| {
            INPUT_ERROR_MESSAGES
                .iter()
                .any(|message| line.contains(message))
                || INPUT_ERROR_CONTEXTS
                    .iter()
                    .any(|context| line.starts_with(context))
        })
}

/// Whether a live ffmpeg run that exited by itself read its source to the end.
///
/// The exit must be clean: status 0 without input errors in stderr. Over HTTP
/// that already rules out every end of a body without Content-Length or
/// chunked framing (ffmpeg logs `Stream ends prematurely` for those) and a
/// body cut short of its Content-Length. A live server can still end a
/// chunked body cleanly, and the HLS demuxer exits cleanly when a live
/// playlist disappears, so the input must also have an end
/// (`live_input_ends`).
async fn live_source_ended(
    status: Option<&std::process::ExitStatus>,
    stderr: &str,
    input_url: &str,
) -> bool {
    status.is_some_and(std::process::ExitStatus::success)
        && !stderr_reports_input_error(stderr)
        && live_input_ends(input_url).await
}

/// Whether the input has an end: a local file, an HTTP resource with a length
/// (`probe_input_end`), or an HLS playlist with `#EXT-X-ENDLIST` or
/// `PLAYLIST-TYPE:VOD`. Asked after ffmpeg exited, so the request doesn't
/// compete with ffmpeg for a provider's connection limit. Unreachable
/// upstreams and other protocols (RTSP, UDP, …) count as live.
///
/// A live HLS publisher that reconnects writes `#EXT-X-ENDLIST` for a moment,
/// which ends the channel if read then; main does the same.
async fn live_input_ends(input_url: &str) -> bool {
    match url::Url::parse(input_url) {
        Ok(url) if matches!(url.scheme(), "http" | "https") => {}
        Ok(url) if url.scheme() == "file" => return true,
        // A Windows drive letter parses as a one-letter scheme.
        Ok(url)
            if url
                .scheme()
                .len()
                > 1 =>
        {
            return false;
        }
        _ => return true,
    }
    if is_hls_input_url(input_url) {
        return upstream_hls_playlist_ends(input_url).await;
    }
    probe_input_end(input_url).await
}

/// Tries of the length probe: providers with a connection limit refuse a new
/// connection for a moment after ffmpeg's closed.
const LENGTH_PROBE_ATTEMPTS: u32 = 3;
const LENGTH_PROBE_BACKOFF: Duration = Duration::from_secs(2);
/// Bound for all tries; the live restart waits for the answer.
const LENGTH_PROBE_BUDGET: Duration = Duration::from_secs(8);

/// Whether a response status (`None`: transport error) to the length probe is
/// worth another try: a provider's connection limit (403, 429, 509) or a
/// temporary outage (503) rather than an answer about the resource.
///
/// A provider that keeps refusing for longer than the tries (about 4 s, e.g.
/// a 10 s linger after a connection closed) makes a file count as live: it
/// restarts and replays, up to the restart cap.
fn length_probe_retries(status: Option<u16>) -> bool {
    matches!(status, None | Some(403 | 429 | 503 | 509))
}

/// What the length probe learned about an HTTP input.
enum ProbedInput {
    /// An HLS playlist behind a URL without .m3u8 (by its content type).
    Hls,
    /// Whether it declares a length (`declares_length`).
    Length(bool),
}

/// Ask an HTTP input for its length with `Range: bytes=0-0`, retried per
/// `length_probe_retries`. An HLS playlist found that way is judged like an
/// `.m3u8` input.
async fn probe_input_end(input_url: &str) -> bool {
    let tries = async {
        for attempt in 0..LENGTH_PROBE_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(LENGTH_PROBE_BACKOFF).await;
            }
            let response = UPSTREAM_CLIENT
                .get(input_url)
                .header(reqwest::header::RANGE, "bytes=0-0")
                .send()
                .await;
            match response {
                Ok(response)
                    if !length_probe_retries(Some(
                        response
                            .status()
                            .as_u16(),
                    )) =>
                {
                    return probe_response(&response);
                }
                _ => {}
            }
        }
        ProbedInput::Length(false)
    };
    match tokio::time::timeout(LENGTH_PROBE_BUDGET, tries).await {
        Ok(ProbedInput::Hls) => upstream_hls_playlist_ends(input_url).await,
        Ok(ProbedInput::Length(declared)) => declared,
        Err(_) => false,
    }
}

fn probe_response(response: &reqwest::Response) -> ProbedInput {
    if !response
        .status()
        .is_success()
    {
        return ProbedInput::Length(false);
    }
    let header = |name: reqwest::header::HeaderName| {
        response
            .headers()
            .get(name)
            .and_then(|value: &reqwest::header::HeaderValue| {
                value
                    .to_str()
                    .ok()
            })
    };
    if header(reqwest::header::CONTENT_TYPE).is_some_and(is_hls_content_type) {
        return ProbedInput::Hls;
    }
    ProbedInput::Length(declares_length(
        response
            .status()
            .as_u16(),
        header(reqwest::header::CONTENT_RANGE),
        header(reqwest::header::CONTENT_LENGTH),
    ))
}

/// Whether a response to `Range: bytes=0-0` gives the resource's length: a
/// 206 with `Content-Range: bytes 0-0/<length>`, or a 200 (Range ignored) with
/// a `Content-Length`. Live streams have neither, or a bogus length that
/// ffmpeg can't have reached when it stopped short of it.
fn declares_length(
    status: u16,
    content_range: Option<&str>,
    content_length: Option<&str>,
) -> bool {
    match status {
        206 => content_range
            .and_then(|range| range.rsplit_once('/'))
            .is_some_and(|(_, total)| {
                total
                    .trim()
                    .parse::<u64>()
                    .is_ok()
            }),
        200 => content_length.is_some_and(|length| {
            length
                .trim()
                .parse::<u64>()
                .is_ok()
        }),
        _ => false,
    }
}

fn is_hls_content_type(content_type: &str) -> bool {
    content_type
        .to_ascii_lowercase()
        .contains("mpegurl")
}

/// What an HLS playlist says about the end of its stream.
#[derive(Debug, PartialEq, Eq)]
enum HlsPlaylistEnd {
    /// VOD: `#EXT-X-ENDLIST` or `#EXT-X-PLAYLIST-TYPE:VOD`.
    Ends,
    /// Live or EVENT playlist without an end yet.
    Open,
    /// Master playlist: the URI of its first variant.
    Variant(String),
}

fn hls_playlist_end(playlist: &str) -> HlsPlaylistEnd {
    let mut variant_next = false;
    for line in playlist
        .lines()
        .map(str::trim)
    {
        if line == "#EXT-X-ENDLIST" || line == "#EXT-X-PLAYLIST-TYPE:VOD" {
            return HlsPlaylistEnd::Ends;
        }
        if line.starts_with("#EXT-X-STREAM-INF") {
            variant_next = true;
        } else if variant_next && !line.is_empty() && !line.starts_with('#') {
            return HlsPlaylistEnd::Variant(line.to_owned());
        }
    }
    HlsPlaylistEnd::Open
}

static UPSTREAM_CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build upstream client")
    });

/// Whether an upstream HLS playlist is VOD, following a master playlist to
/// its first variant. An unreachable or oversized playlist counts as live.
async fn upstream_hls_playlist_ends(url: &str) -> bool {
    let mut url = url.to_owned();
    // The master playlist and one variant.
    for _ in 0..2 {
        let Some(playlist) = fetch_upstream_playlist(&url).await else {
            return false;
        };
        match hls_playlist_end(&playlist) {
            HlsPlaylistEnd::Ends => return true,
            HlsPlaylistEnd::Open => return false,
            HlsPlaylistEnd::Variant(uri) => {
                match url::Url::parse(&url).and_then(|base| base.join(&uri)) {
                    Ok(variant) => url = variant.into(),
                    Err(_) => return false,
                }
            }
        }
    }
    false
}

async fn fetch_upstream_playlist(url: &str) -> Option<String> {
    let mut response = UPSTREAM_CLIENT
        .get(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .ok()?
    {
        body.extend_from_slice(&chunk);
        if body.len() > MAX_UPSTREAM_PLAYLIST_BYTES {
            return None;
        }
    }
    String::from_utf8(body).ok()
}

/// Append `#EXT-X-ENDLIST` to the live playlist of a source that ended, so
/// players finish playback instead of polling for segments that never come.
/// Appending keeps the file whole for concurrent playlist reads.
fn end_live_playlist(playlist: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let content = std::fs::read_to_string(playlist)?;
    if content.contains("#EXT-X-ENDLIST") {
        return Ok(());
    }
    let separator = if content.ends_with('\n') { "" } else { "\n" };
    std::fs::OpenOptions::new()
        .append(true)
        .open(playlist)?
        .write_all(format!("{separator}#EXT-X-ENDLIST\n").as_bytes())
}

#[derive(Debug, PartialEq, Eq)]
enum ThrottleSignal {
    Pause { ahead: u32 },
    Resume { ahead: u32 },
}

/// What one buffer-monitor tick does to ffmpeg and its output dir.
#[derive(Debug, PartialEq, Eq)]
struct MonitorStep {
    signal: Option<ThrottleSignal>,
    /// Delete segments below this index.
    delete_below: Option<u32>,
}

/// One VOD buffer-monitor tick. `highest` is the highest segment index on
/// disk; `playback_secs` is how far the client has played relative to the
/// start of this run (from progress reports); `cleanup_due` is set every 30
/// ticks.
fn monitor_step(
    highest: Option<u32>,
    start_index: u32,
    segment_length: u32,
    playback_secs: u32,
    paused: bool,
    cleanup_due: bool,
) -> MonitorStep {
    let buffered_secs =
        buffered_secs_from_segments(highest, start_index, segment_length);
    let ahead = buffered_secs.saturating_sub(playback_secs);

    let signal = if !paused && ahead >= MAX_BUFFER_SECS {
        Some(ThrottleSignal::Pause { ahead })
    } else if paused && ahead < MAX_BUFFER_SECS.saturating_sub(segment_length * 2) {
        Some(ThrottleSignal::Resume { ahead })
    } else {
        None
    };
    // Delete segments more than SEGMENT_KEEP_SECS behind the playback position.
    let delete_below = (cleanup_due && playback_secs > SEGMENT_KEEP_SECS)
        .then(|| (playback_secs - SEGMENT_KEEP_SECS) / segment_length);
    MonitorStep {
        signal,
        delete_below,
    }
}

fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".into())
}

/// Send SIGSTOP/SIGCONT to a process by PID on Unix.
#[cfg(unix)]
fn send_signal(pid: u32, sig: libc::c_int) {
    unsafe { libc::kill(pid as libc::pid_t, sig) };
}
#[cfg(not(unix))]
fn send_signal(_pid: u32, _sig: i32) {}

/// Spawn the VOD buffer-throttle task. It pauses/resumes ffmpeg so it never
/// encodes more than MAX_BUFFER_SECS ahead of what the client has requested,
/// and deletes old segments. See `monitor_step`. Live sessions use
/// `spawn_live_maintenance` instead.
///
/// `start_index` is this run's `-start_number` (see `segment_start_index`):
/// segment file names are numbered from the seek position, not from 0, so
/// it must be subtracted from whatever index ffmpeg has reached before
/// converting to a duration — otherwise a seek into the middle of a long
/// file reads as already having buffered the seek offset itself and pauses
/// ffmpeg almost immediately.
fn spawn_buffer_monitor(
    output_dir: PathBuf,
    segment_length: u32,
    start_index: u32,
    playback_offset_secs: Arc<AtomicU32>,
    ffmpeg_pid: Arc<AtomicU32>,
    mut stop_rx: tokio::sync::oneshot::Receiver<()>,
    play_session_id: String,
) {
    tokio::spawn(async move {
        let mut paused = false;
        let mut ticks: u32 = 0;
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
            }
            ticks += 1;

            let pid = ffmpeg_pid.load(Ordering::Relaxed);
            let step = monitor_step(
                highest_segment_index(&output_dir),
                start_index,
                segment_length,
                playback_offset_secs.load(Ordering::Relaxed),
                paused,
                ticks % 30 == 0,
            );
            match step.signal {
                Some(ThrottleSignal::Pause { ahead }) if pid != 0 => {
                    debug!(play_session_id, pid, ahead, "Buffer full — pausing ffmpeg");
                    #[cfg(unix)]
                    send_signal(pid, libc::SIGSTOP);
                    paused = true;
                }
                Some(ThrottleSignal::Resume { ahead }) if pid != 0 => {
                    debug!(
                        play_session_id,
                        pid, ahead, "Buffer drained — resuming ffmpeg"
                    );
                    #[cfg(unix)]
                    send_signal(pid, libc::SIGCONT);
                    paused = false;
                }
                _ => {}
            }
            if let Some(cutoff_idx) = step.delete_below {
                delete_old_segments(&output_dir, cutoff_idx);
            }
        }

        // Ensure ffmpeg isn't left paused when we stop monitoring.
        if paused {
            let pid = ffmpeg_pid.load(Ordering::Relaxed);
            if pid != 0 {
                #[cfg(unix)]
                send_signal(pid, libc::SIGCONT);
            }
        }
    });
}

/// Session-level upkeep of a live session, for as long as the session lasts
/// rather than per ffmpeg run (a run can end within seconds):
/// - pauses ffmpeg reading a source faster than real time once it is
///   `LIVE_THROTTLE_SECS` ahead of the client's requests (`live_throttle`);
/// - deletes segments behind the playlist window (`live_cleanup_cutoff`);
/// - records the playlist's discontinuities before they slide out of the
///   window (`LiveActivity::discontinuity_sequence`);
/// - stops the ffmpeg of a source read in real time once the client has made
///   no playlist or segment request for `LIVE_IDLE`, setting
///   `LiveRun::idle_stopped` for the transcode loop.
///
/// Live clients' reported positions are ignored: Jellyfin Android TV reports
/// 0, then epoch-based values after an error. `start_index` is the session's
/// first segment. Runs until `ended` is cancelled.
fn spawn_live_maintenance(
    session: Arc<RwLock<TranscodeSession>>,
    start_index: u32,
    run: Arc<LiveRun>,
    ffmpeg_pid: Arc<AtomicU32>,
    ended: CancellationToken,
) {
    tokio::spawn(async move {
        let (output_dir, segment_length, requested, activity) = {
            let s = session
                .read()
                .await;
            (
                s.output_dir
                    .clone(),
                s.segment_length,
                s.last_segment_index
                    .clone(),
                s.live_activity
                    .clone(),
            )
        };
        // The ffmpeg this task paused; a restarted one runs.
        let mut paused_pid = 0;
        loop {
            tokio::select! {
                _ = ended.cancelled() => break,
                _ = tokio::time::sleep(LIVE_MAINTENANCE_INTERVAL) => {}
            }
            let highest = highest_segment_index(&output_dir);
            // 0 between runs; kill(0, …) would signal our process group.
            let pid = ffmpeg_pid.load(Ordering::Relaxed);
            let faster_than_realtime =
                run.faster_than_realtime(highest, segment_length);
            let signal = live_throttle(
                highest,
                requested
                    .load(Ordering::Relaxed)
                    .max(start_index),
                segment_length,
                pid != 0 && pid == paused_pid,
                faster_than_realtime,
            );
            match signal {
                Some(ThrottleSignal::Pause { ahead }) if pid != 0 => {
                    debug!(
                        pid,
                        ahead, "Live source ahead of the client — pausing ffmpeg"
                    );
                    #[cfg(unix)]
                    send_signal(pid, libc::SIGSTOP);
                    paused_pid = pid;
                }
                // Only signalled while paused, so `pid` is the paused ffmpeg.
                Some(ThrottleSignal::Resume { ahead }) => {
                    debug!(pid, ahead, "Live client caught up — resuming ffmpeg");
                    #[cfg(unix)]
                    send_signal(pid, libc::SIGCONT);
                    paused_pid = 0;
                }
                _ => {}
            }
            if let Some(highest) = highest {
                delete_old_segments(
                    &output_dir,
                    live_cleanup_cutoff(highest, segment_length),
                );
            }
            if let Ok(playlist) =
                tokio::fs::read_to_string(output_dir.join("main.m3u8")).await
            {
                activity.discontinuity_sequence(&playlist);
            }
            // A source read faster than real time is paused by the throttle
            // once its client is gone, which costs no CPU, and the session
            // reaper ends it. Stopping it instead would replay it from the
            // start when the client returns: live inputs get no `-ss`.
            if !faster_than_realtime && activity.idle_for() >= LIVE_IDLE {
                let mut s = session
                    .write()
                    .await;
                if let Some(kill_tx) = s
                    .kill_tx
                    .take()
                {
                    info!(
                        session_id = %s.id,
                        idle_secs = LIVE_IDLE.as_secs(),
                        "No live playlist or segment request — stopping ffmpeg"
                    );
                    run.idle_stopped
                        .store(true, Ordering::Relaxed);
                    // run_ffmpeg resumes a paused ffmpeg before killing it.
                    let _ = kill_tx.send(());
                }
            }
        }
    });
}

/// Wait for the client of an idle live session to request the playlist or a
/// segment again. False if the session was stopped first.
async fn wait_for_live_client(
    activity: &LiveActivity,
    stopped: &CancellationToken,
) -> bool {
    loop {
        if activity.idle_for() < LIVE_IDLE {
            return true;
        }
        tokio::select! {
            _ = stopped.cancelled() => return false,
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
    }
}

/// Parses the numeric index out of a segment filename — `segment_00042.ts`,
/// `segment_00042.m4s`, etc. Extension-agnostic on purpose: HEVC-copy
/// sessions write fMP4 `.m4s` segments instead of `.ts`, and every caller
/// here only cares about the index, not the container.
fn segment_index(name: &str) -> Option<u32> {
    name.rsplit('_')
        .next()?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Delete segment files whose index is less than `cutoff_idx`.
fn delete_old_segments(dir: &PathBuf, cutoff_idx: u32) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(idx) = segment_index(&name.to_string_lossy()) else {
            continue;
        };
        if idx < cutoff_idx {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Highest segment index currently on disk, or `None` if there are no
/// segments yet. Used instead of a file count so the buffer-ahead
/// calculation in `spawn_buffer_monitor` stays correct after
/// `delete_old_segments` starts pruning old segments (file count would
/// then undercount how far ffmpeg has actually gotten).
fn highest_segment_index(dir: &PathBuf) -> Option<u32> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            segment_index(
                &e.file_name()
                    .to_string_lossy(),
            )
        })
        .max()
}

/// The HLS segment index a seek offset lands on — this session's
/// `-start_number` / `#EXT-X-MEDIA-SEQUENCE` base. Segment files are named
/// from this index, not from 0, so anything comparing segment indexes across
/// a seek (the buffer monitor, the playlist) must account for it.
fn segment_start_index(start_time_ticks: Option<i64>, segment_length: u32) -> u32 {
    start_time_ticks
        .map(|t| (t as f64 / 10_000_000.0 / segment_length as f64).floor() as u32)
        .unwrap_or(0)
}

/// Seconds of ffmpeg output currently buffered, given the highest segment
/// index found on disk. `start_index` (see `segment_start_index`) must be
/// subtracted first: segment numbering is absolute (offset by the seek
/// position), while the buffer-ahead comparison against `playback_secs`
/// needs a duration relative to where this run started.
fn buffered_secs_from_segments(
    highest_index: Option<u32>,
    start_index: u32,
    segment_length: u32,
) -> u32 {
    highest_index.map_or(0, |idx| {
        (idx.saturating_sub(start_index) + 1) * segment_length
    })
}

/// Parameters for starting a new HLS transcode job.
pub struct TranscodeParams {
    pub input_url: String,
    pub output_dir: PathBuf,
    pub video_codec: String, // "copy", "libx264", "libx265"
    pub audio_codec: String, // "aac", "copy"
    pub segment_length: u32, // seconds (default 6)
    pub start_time_ticks: Option<i64>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub video_bitrate: Option<u32>,
    pub audio_bitrate: Option<u32>,
    pub audio_channels: Option<u32>,
    pub audio_stream_index: Option<i32>,
    pub subtitle_stream_index: Option<i32>,
    pub burn_subtitle: bool,
    /// Native dimensions of the subtitle bitmap (PGS canvas size), used to
    /// scale the subtitle to match the output video resolution.
    pub subtitle_width: Option<u32>,
    pub subtitle_height: Option<u32>,
    pub encoding_preset: Option<EncodingPreset>,
    /// Codec of the source video stream (e.g. "hevc", "h264"), used to apply
    /// codec-specific output flags such as `-tag:v hvc1` for HEVC in HLS.
    pub source_video_codec: Option<String>,
    /// Codec of the source audio stream (e.g. "aac", "ac3"), used to apply
    /// codec-specific bitstream filters such as `aac_adtstoasc` when copying.
    pub source_audio_codec: Option<String>,
    /// HEVC sample-entry tag (`hvc1`/`hev1`) resolved from the client's
    /// DeviceProfile; see `DeviceProfileExt::hevc_copy_tag`.
    pub hevc_copy_tag: Option<String>,
    pub accelerator: Box<dyn Accelerator>,
    /// HDR type of the source video, used to decide whether tone-mapping or
    /// SDR colour-space override is needed.
    pub source_video_range_type: Option<VideoRangeType>,
    /// Software HDR→SDR tonemapping via tonemapx filter (CPU).
    pub enable_tonemapping: bool,
    /// Hardware HDR→SDR tonemapping via tonemap_vaapi (Intel VAAPI/QSV only).
    pub enable_vpp_tonemapping: bool,
    /// Algorithm for tonemapx: hable, reinhard, mobius, bt2390, bt2446a, none.
    pub tonemapping_algorithm: String,
    /// Desaturation coefficient for tonemapx.
    pub tonemapping_desat: f32,
    /// Peak luminance for tonemapx (nits). 0 = auto.
    pub tonemapping_peak: f32,
    /// When false, HEVC encoding requests fall back to H.264.
    pub allow_hevc_encoding: bool,
    /// When false, AV1 encoding requests fall back to H.264.
    pub allow_av1_encoding: bool,
    /// CRF quality for software H.264 (libx264).
    pub h264_crf: u32,
    /// CRF quality for software H.265 (libx265).
    pub h265_crf: u32,
    /// True for live TV / RTSP streams — disables seeking and enables auto-restart on exit.
    pub is_live: bool,
    /// Apply `loudnorm=I=-14:TP=-1:LRA=11` when transcoding audio. Has no effect
    /// when audio_codec is "copy". See EncodingOptions::normalize_audio_loudness.
    pub normalize_audio_loudness: bool,
}

impl Default for TranscodeParams {
    fn default() -> Self {
        Self {
            input_url: String::new(),
            output_dir: PathBuf::new(),
            video_codec: "copy".to_string(),
            audio_codec: "aac".to_string(),
            segment_length: 6,
            start_time_ticks: None,
            max_width: None,
            max_height: None,
            video_bitrate: None,
            audio_bitrate: None,
            audio_channels: None,
            audio_stream_index: None,
            subtitle_stream_index: None,
            burn_subtitle: false,
            subtitle_width: None,
            subtitle_height: None,
            encoding_preset: None,
            source_video_codec: None,
            source_audio_codec: None,
            hevc_copy_tag: None,
            accelerator: Box::new(NoAccel),
            source_video_range_type: None,
            enable_tonemapping: false,
            enable_vpp_tonemapping: false,
            tonemapping_algorithm: "hable".to_string(),
            tonemapping_desat: 0.0,
            tonemapping_peak: 0.0,
            allow_hevc_encoding: false,
            allow_av1_encoding: false,
            h264_crf: 23,
            h265_crf: 28,
            is_live: false,
            normalize_audio_loudness: false,
        }
    }
}

/// Sample-entry tag for a stream-copied HEVC track. The real decision belongs
/// to `DeviceProfileExt::hevc_copy_tag`, which weighs the client's declared
/// constraints against the source's own fourcc at PlaybackInfo time and hands
/// the answer down on the transcode URL. This only narrows that answer back to
/// the two legal values, falling back to `hvc1` for requests that carry no tag
/// at all — a direct hit on the stream endpoint, or a session predating the
/// parameter.
fn hevc_copy_tag(tag: Option<&str>) -> &str {
    match tag {
        Some(t) if t.eq_ignore_ascii_case("hev1") => "hev1",
        _ => "hvc1",
    }
}

/// Return the expected output video dimensions based on transcode params.
fn output_dimensions(params: &TranscodeParams) -> (Option<u32>, Option<u32>) {
    (params.max_width, params.max_height)
}

/// Build a scale filter string for FFmpeg, if needed.
fn build_scale_filter(params: &TranscodeParams) -> Option<String> {
    match (params.max_width, params.max_height) {
        (Some(w), Some(h)) => Some(format!(
            "scale='min({},iw)':'min({},ih)':force_original_aspect_ratio=decrease",
            w, h
        )),
        (Some(w), None) => Some(format!("scale='min({},iw)':-2", w)),
        (None, Some(h)) => Some(format!("scale=-2:'min({},ih)'", h)),
        _ => None,
    }
}

/// Build a VAAPI hardware scale filter for QSV/VAAPI transcoding.
///
/// Both QSV and plain VAAPI decode via `-hwaccel vaapi`, so frames live in
/// VAAPI GPU memory either way — we must use `scale_vaapi` instead of the CPU
/// `scale` filter. Always returns `Some` — at minimum a format-conversion
/// pass keeps the pool sized/formatted consistently for whatever the
/// accelerator's `hw_filter_suffix` does next (a further hop into QSV memory,
/// or nothing at all for a native VAAPI encoder).
/// `extra_hw_frames=24` follows Jellyfin's recommendation for VAAPI VPP pools.
///
/// GPU tone mappers need the 10-bit P010 surface: fed NV12, `tonemap_vaapi`
/// reads the truncated 8-bit PQ samples as full-range HDR and outputs a solid
/// cyan frame. Scale to P010 on those paths and let the tone map emit NV12,
/// as Jellyfin does.
fn build_vaapi_scale_filter(
    treatment: HdrTreatment,
    max_width: Option<u32>,
    max_height: Option<u32>,
) -> String {
    let fmt = match treatment {
        HdrTreatment::VppTonemap | HdrTreatment::OclTonemap => "p010",
        _ => "nv12",
    };
    match (max_width, max_height) {
        (Some(w), Some(h)) => format!(
            "scale_vaapi=w='min({w},iw)':h='min({h},ih)':format={fmt}:extra_hw_frames=24:force_original_aspect_ratio=decrease"
        ),
        (Some(w), None) => {
            format!("scale_vaapi=w='min({w},iw)':h=-2:format={fmt}:extra_hw_frames=24")
        }
        (None, Some(h)) => {
            format!("scale_vaapi=w=-2:h='min({h},ih)':format={fmt}:extra_hw_frames=24")
        }
        _ => format!("scale_vaapi=format={fmt}:extra_hw_frames=24"),
    }
}

/// Main-video chain for QSV when frames stay on the GPU from decode to encode:
/// `scale_vaapi` on the VAAPI-decoded surface, then the accelerator's suffix
/// (VPP tone map, BT.709 relabel, or plain `hwmap` into QSV memory). Used for
/// both the `-vf` path and the `overlay_qsv` main branch — QSV-only, since
/// `overlay_qsv` has no VAAPI equivalent (see `supports_gpu_resident_overlay`).
fn build_vaapi_main_video(
    accel: &dyn Accelerator,
    treatment: HdrTreatment,
    tonemap: &TonemapOptions,
    max_width: Option<u32>,
    max_height: Option<u32>,
) -> String {
    let scale = build_vaapi_scale_filter(treatment, max_width, max_height);
    match accel.hw_filter_suffix(treatment, tonemap) {
        Some(suffix) => format!("{scale},{suffix}"),
        None => scale,
    }
}

/// True when the pipeline keeps frames in GPU memory end to end on QSV or
/// plain VAAPI — both decode via `-hwaccel vaapi`. Only an explicit software
/// tone map pulls frames onto the CPU; every other HDR treatment stays
/// resident for scale/tone-map purposes. This is *not* the right check for
/// subtitle burn-in GPU residency — `overlay_qsv` has no VAAPI equivalent, so
/// that decision uses `Accelerator::supports_gpu_resident_overlay` instead,
/// which stays QSV-only.
fn hw_gpu_resident(accel: &dyn Accelerator, treatment: HdrTreatment) -> bool {
    matches!(
        accel.as_type(),
        HardwareAccelerationType::Qsv | HardwareAccelerationType::Vaapi
    ) && treatment != HdrTreatment::SwTonemap
}

/// Build the main video processing chain for a CPU-based subtitle overlay.
///
/// Applies optional scaling and, for HDR sources, the appropriate colour treatment:
/// - `do_sw_tonemap` → tonemapx (outputs yuv420p SDR)
/// - HDR without tonemap → setparams clears BT.2020/PQ metadata + format=yuv420p
/// - SDR → plain scale (or empty string if no scale needed)
fn build_cpu_main_video_filters(
    scale: Option<String>,
    hdr: bool,
    do_sw_tonemap: bool,
    algo: &str,
    desat: f32,
    peak: f32,
) -> String {
    let scale_part = scale.unwrap_or_default();
    if !hdr {
        return scale_part;
    }
    let hdr_part = if do_sw_tonemap {
        format!(
            "tonemapx=tonemap={algo}:desat={desat:.1}:peak={peak:.1}:t=bt709:m=bt709:p=bt709:format=yuv420p"
        )
    } else {
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709,format=yuv420p"
            .to_string()
    };
    if scale_part.is_empty() {
        hdr_part
    } else {
        format!("{scale_part},{hdr_part}")
    }
}

/// Compose a filter_complex string for a CPU-based (non-QSV) subtitle overlay.
///
/// Places `sub_preproc` on the subtitle stream, composites it over `main_video`
/// via `overlay`, then optionally passes through `hw_suffix` (e.g.
/// `format=nv12,hwupload` for VAAPI) before the final `[v]` output.
fn build_cpu_overlay_complex(
    sub_idx: i32,
    sub_preproc: &str,
    main_video: &str,
    hw_suffix: Option<&str>,
) -> String {
    let overlay = "overlay=eof_action=pass:repeatlast=0";
    if main_video.is_empty() {
        match hw_suffix {
            Some(suf) => format!(
                "[0:{sub_idx}]{sub_preproc}[sub];[0:v:0][sub]{overlay}[vraw];[vraw]{suf}[v]"
            ),
            None => {
                format!("[0:{sub_idx}]{sub_preproc}[sub];[0:v:0][sub]{overlay}[v]")
            }
        }
    } else {
        match hw_suffix {
            Some(suf) => format!(
                "[0:{sub_idx}]{sub_preproc}[sub];[0:v:0]{main_video}[main];[main][sub]{overlay}[vraw];[vraw]{suf}[v]"
            ),
            None => format!(
                "[0:{sub_idx}]{sub_preproc}[sub];[0:v:0]{main_video}[main];[main][sub]{overlay}[v]"
            ),
        }
    }
}

fn is_hdr(range_type: Option<&VideoRangeType>) -> bool {
    matches!(
        range_type,
        Some(VideoRangeType::Hdr10)
            | Some(VideoRangeType::Hdr10Plus)
            | Some(VideoRangeType::Hlg)
            | Some(VideoRangeType::Dovi)
            | Some(VideoRangeType::DoviWithHdr10)
            | Some(VideoRangeType::DoviWithHlg)
    )
}

fn is_hls_input_url(input_url: &str) -> bool {
    let Ok(url) = url::Url::parse(input_url) else {
        return false;
    };
    let path = url
        .path()
        .to_ascii_lowercase();
    let is_hls_endpoint = path.ends_with("/hls");
    path.ends_with(".m3u8")
        // Stream relays expose their HLS manifest through a generic endpoint
        // such as `/api/manifest?url=https://origin/live.m3u8`, or through
        // `/hls?url=...` where the target may have no extension at all.
        || url.query_pairs().any(|(name, value)| {
            name.eq_ignore_ascii_case("url")
                && url::Url::parse(&value).ok().is_some_and(|target| {
                    let target_path = target.path().to_ascii_lowercase();
                    target_path.ends_with(".m3u8")
                        || (is_hls_endpoint && !target_path.ends_with(".mpd"))
                })
        })
}

/// FFmpeg HTTP input options, checked against the resolved input string (what
/// FFmpeg actually opens), not the originating `StreamDescriptor`.
///
/// HTTP persistence is disabled for HLS because some stream proxies redirect a
/// manifest to a different host. FFmpeg otherwise tries to reuse the proxy
/// connection for the redirected host and aborts the input.
/// `-http_persistent` is an option of FFmpeg's HLS demuxer only; passing it for
/// any other input makes FFmpeg abort with "Option http_persistent not found".
/// Plain HTTP inputs are already non-persistent by default.
///
/// The reconnect options are intentionally omitted for HLS: FFmpeg's HLS
/// demuxer opens the playlists and segments itself, while `-reconnect_streamed`
/// can leave proxied master playlists waiting indefinitely before the first
/// rendition is opened.
pub(crate) fn ffmpeg_http_input_args(input_url: &str) -> &'static [&'static str] {
    match url::Url::parse(input_url) {
        Ok(url)
            if matches!(url.scheme(), "http" | "https")
                && is_hls_input_url(input_url) =>
        {
            &["-http_persistent", "0"]
        }
        Ok(url) if matches!(url.scheme(), "http" | "https") => &[
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
        ],
        _ => &[],
    }
}

fn add_hls_extension_compat_args(args: &mut Vec<String>, input_url: &str) {
    if is_hls_input_url(input_url) {
        args.extend([
            "-allowed_extensions".into(),
            "ALL".into(),
            "-allowed_segment_extensions".into(),
            "ALL".into(),
            "-extension_picky".into(),
            "0".into(),
        ]);
    }
}

/// Build the ffmpeg CLI args for an HLS transcode.
pub(crate) fn build_hls_args(params: &TranscodeParams) -> Vec<String> {
    let accel = params
        .accelerator
        .as_ref();
    let is_hw = accel.is_hw();
    let hdr = is_hdr(
        params
            .source_video_range_type
            .as_ref(),
    );

    // The API layer must explicitly select a video encoder after checking
    // server and user permissions. Never turn a constrained copy request back
    // into a transcode here merely because SubtitleMethod=Encode was supplied.
    let burn_subtitle_filter = params.burn_subtitle
        && params
            .subtitle_stream_index
            .is_some_and(|index| index >= 0)
        && params.video_codec != "copy";

    // Tone-map decision (only applies to HDR + transcode, never to copy). On
    // QSV the subtitle overlay runs on the GPU after tone mapping, so
    // burn-in doesn't affect the choice there; other VPP-tonemap-capable
    // accelerators (VAAPI) have no GPU-resident overlay and still need to
    // fall back to software tone mapping when burning a subtitle in.
    let treatment = HdrTreatment::for_source(
        hdr,
        accel,
        params.enable_tonemapping,
        params.enable_vpp_tonemapping,
        burn_subtitle_filter,
    );
    let do_sw_tonemap = treatment == HdrTreatment::SwTonemap;

    let ffmpeg_video_codec = {
        let base = match params
            .video_codec
            .as_str()
        {
            "copy" => "copy",
            _ => "libx264",
        };
        if base != "copy" && is_hw {
            accel.encoder_name(base)
        } else {
            base.to_string()
        }
    };
    // fMP4 (fragmented MP4) is required for HEVC on iOS Safari per Apple's HLS
    // authoring specification.  MPEG-TS cannot carry HEVC correctly in HLS.
    // Shared with TranscodeSession so segment paths and playlists agree.
    let segment_container = SegmentContainer::for_codecs(
        &ffmpeg_video_codec,
        params
            .source_video_codec
            .as_deref(),
    );
    let is_hevc_copy = segment_container == SegmentContainer::Fmp4;

    let ffmpeg_audio_codec = match params
        .audio_codec
        .as_str()
    {
        "copy" => {
            // IMPORTANT: do not remove this override.
            //
            // Some codecs can't be carried in the segment container (TrueHD,
            // FLAC, and PCM have no MPEG-TS stream type). FFmpeg either errors
            // out or silently drops the audio track when you try to mux them.
            // Clients (iOS Safari, ExoPlayer) then see a broken/silent stream.
            //
            // The client asked for "copy" because it trusts the server to only
            // honour that when the codec can actually be carried in the
            // container. We must downgrade to AAC here; do not "fix" this by
            // removing the override thinking the client knows best. The HLS
            // handler applies the same rule first so it can enforce the audio
            // transcoding permission.
            if segment_container.can_copy_audio(
                params
                    .source_audio_codec
                    .as_deref(),
            ) {
                "copy"
            } else {
                "aac"
            }
        }
        _ => "aac",
    };

    let mut args: Vec<String> = vec![
        "-v".into(),
        "error".into(),
        "-analyzeduration".into(),
        "1000000".into(),
        "-probesize".into(),
        "1000000".into(),
    ];

    args.extend(
        accel.decode_input_args(
            params
                .source_video_codec
                .as_deref(),
            treatment,
            burn_subtitle_filter,
        ),
    );

    // Input seek (fast, before -i) — not applicable to live streams
    if !params.is_live {
        if let Some(ticks) = params.start_time_ticks {
            let secs = ticks as f64 / 10_000_000.0;
            args.extend(["-ss".into(), format!("{:.6}", secs)]);
        }
    }

    // RTSP reliability: force TCP transport and set a 5 s connection timeout
    if params.is_live
        && params
            .input_url
            .starts_with("rtsp://")
    {
        args.extend([
            "-rtsp_transport".into(),
            "tcp".into(),
            "-timeout".into(),
            "5000000".into(),
        ]);
    }

    add_hls_extension_compat_args(&mut args, &params.input_url);

    args.extend(
        ffmpeg_http_input_args(&params.input_url)
            .iter()
            .map(|s| (*s).into()),
    );

    args.extend([
        "-copyts".into(),
        "-i".into(),
        params
            .input_url
            .clone(),
        "-avoid_negative_ts".into(),
        "disabled".into(),
        "-max_muxing_queue_size".into(),
        "2048".into(),
    ]);

    let tonemap = TonemapOptions {
        algorithm: &params.tonemapping_algorithm,
        desat: params.tonemapping_desat,
        peak: params.tonemapping_peak,
    };
    let hw_suffix = accel.hw_filter_suffix(treatment, &tonemap);

    // Stream mapping
    if burn_subtitle_filter {
        if let Some(sub_idx) = params.subtitle_stream_index {
            // Image subtitle (PGS/DVD): bitmap overlay via filter_complex.
            // Scale subtitle bitmap to output dimensions (matching Jellyfin's approach).
            // When output size is known, scale to that; otherwise pass through as-is.
            let (out_w, out_h) = output_dimensions(params);
            let sub_scale = match (out_w, out_h) {
                (Some(w), Some(h)) => format!("scale={w}:{h}:fast_bilinear"),
                (Some(w), None) => format!("scale={w}:-1:fast_bilinear"),
                (None, Some(h)) => format!("scale=-1:{h}:fast_bilinear"),
                _ => String::new(),
            };

            // PGS bitmaps are BGRA; bare `scale` converts to a pixel format
            // compatible with overlay before any resize step (mirrors Jellyfin).
            let sub_preproc = if sub_scale.is_empty() {
                "scale".to_string()
            } else {
                format!("scale,{sub_scale}")
            };

            let filter = if is_hw
                && accel.supports_gpu_resident_overlay()
                && treatment != HdrTreatment::SwTonemap
            {
                // QSV path: keep VAAPI hw decode throughout. Sub is prepared on CPU
                // (format=bgra) then uploaded to the QSV device. overlay_qsv composites
                // both surfaces on-GPU — no CPU round-trip for video frames. Any tone
                // map or BT.709 relabel happens on the main branch before the overlay,
                // so tonemap_vaapi never sees CPU overlay output.
                // Mirrors Jellyfin's GetIntelQsvVaapiVidFiltersPrefered graphical-sub path.
                // QSV-only: `overlay_qsv` has no VAAPI equivalent (see
                // `supports_gpu_resident_overlay`), so plain VAAPI never
                // reaches this branch — `HdrTreatment::for_source` already
                // forces `SwTonemap` for it whenever a subtitle is burned in.
                let sub_upload =
                    "format=bgra,hwupload=derive_device=qsv:extra_hw_frames=64";
                let overlay_size = match (out_w, out_h) {
                    (Some(w), Some(h)) => format!(":w={w}:h={h}"),
                    _ => String::new(),
                };
                let overlay_qsv =
                    format!("overlay_qsv=eof_action=pass:repeatlast=0{overlay_size}");
                let main_video = build_vaapi_main_video(
                    accel,
                    treatment,
                    &tonemap,
                    params.max_width,
                    params.max_height,
                );
                format!(
                    "[0:{sub_idx}]{sub_preproc},{sub_upload}[sub];\
                     [0:v:0]{main_video}[vmain];\
                     [vmain][sub]{overlay_qsv}[v]"
                )
            } else {
                let main_video = build_cpu_main_video_filters(
                    build_scale_filter(params),
                    hdr,
                    do_sw_tonemap,
                    &params.tonemapping_algorithm,
                    params.tonemapping_desat,
                    params.tonemapping_peak,
                );
                build_cpu_overlay_complex(
                    sub_idx,
                    &sub_preproc,
                    &main_video,
                    hw_suffix.as_deref(),
                )
            };
            args.extend(["-filter_complex".into(), filter]);
            args.extend(["-map".into(), "[v]".into()]);
            if let Some(audio_idx) = params.audio_stream_index {
                args.extend(["-map".into(), format!("0:{}", audio_idx)]);
            } else {
                args.extend(["-map".into(), "0:a?".into()]);
            }
        }
    } else {
        // QSV/VAAPI scale on the GPU whenever frames are VAAPI-resident; only
        // the SW tonemap path decodes to CPU memory and needs the CPU scale filter.
        let scale_filter = if ffmpeg_video_codec != "copy" {
            if hw_gpu_resident(accel, treatment) {
                Some(build_vaapi_scale_filter(
                    treatment,
                    params.max_width,
                    params.max_height,
                ))
            } else {
                build_scale_filter(params)
            }
        } else {
            None
        };
        let vf = match (&scale_filter, &hw_suffix) {
            (Some(s), Some(suf)) => Some(format!("{s},{suf}")),
            (Some(s), None) => Some(s.clone()),
            (None, Some(suf)) if ffmpeg_video_codec != "copy" => Some(suf.clone()),
            _ => None,
        };
        let vf = if hdr && ffmpeg_video_codec != "copy" {
            if do_sw_tonemap {
                // Software tonemapx: CPU filter, output SDR. Rebuild filter chain
                // from scale + tonemapx (bypassing hw_suffix which carries hwupload
                // or format conversions that conflict with tonemapx).
                let algo = &params.tonemapping_algorithm;
                let desat = params.tonemapping_desat;
                let peak = params.tonemapping_peak;
                let out_fmt = if !accel.is_hw()
                    || accel.as_type() == HardwareAccelerationType::V4l2m2m
                {
                    "yuv420p"
                } else {
                    "nv12"
                };
                let tonemapx = format!(
                    "tonemapx=tonemap={algo}:desat={desat:.1}:peak={peak:.1}:t=bt709:m=bt709:p=bt709:format={out_fmt}"
                );
                let upload = if accel.as_type() == HardwareAccelerationType::Vaapi {
                    ",hwupload"
                } else {
                    ""
                };
                Some(match scale_filter.as_deref() {
                    Some(s) => format!("{s},{tonemapx}{upload}"),
                    None => format!("{tonemapx}{upload}"),
                })
            } else if hw_gpu_resident(accel, treatment) {
                // GPU-resident tone map/relabel (VPP, OpenCL, or BT.709
                // clamp) is already fully embedded in hw_suffix above.
                vf
            } else {
                // No tone mapping: rewrite colour metadata so clients treat output as SDR.
                let setparams =
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709";
                Some(match vf {
                    Some(f) => format!("{setparams},{f}"),
                    None => setparams.to_string(),
                })
            }
        } else {
            vf
        };
        if let Some(ref filter) = vf {
            args.extend(["-vf".into(), filter.clone()]);
        } else if let Some(audio_idx) = params.audio_stream_index {
            args.extend([
                "-map".into(),
                "0:v:0".into(),
                "-map".into(),
                format!("0:{}", audio_idx),
            ]);
        }
    }

    // Video codec
    args.extend(["-c:v".into(), ffmpeg_video_codec.clone()]);

    if ffmpeg_video_codec == "copy" {
        if is_hevc_copy {
            args.extend([
                "-tag:v".into(),
                hevc_copy_tag(
                    params
                        .hevc_copy_tag
                        .as_deref(),
                )
                .into(),
            ]);
            // Strip embedded Dolby Vision RPU NALs only when the source is actually DoVi;
            // dovi_rpu only supports hevc/av1 and will crash ffmpeg on any other codec.
            let is_dovi = matches!(
                params.source_video_range_type,
                Some(VideoRangeType::Dovi)
                    | Some(VideoRangeType::DoviWithHdr10)
                    | Some(VideoRangeType::DoviWithHlg)
                    | Some(VideoRangeType::DoviWithSdr)
            );
            if is_dovi {
                args.extend(["-bsf:v".into(), "dovi_rpu=strip=1".into()]);
            }
        }
    } else if is_hw {
        // HW encoders use bitrate control; CRF/preset/profile flags don't apply.
        // h264_nvenc needs an explicit 8-bit output: 10-bit frames fail, and
        // CUDA decode leaves frames in software memory so FFmpeg can convert
        // them to yuv420p before the encoder.
        if ffmpeg_video_codec == "h264_nvenc" {
            args.extend(["-pix_fmt".into(), "yuv420p".into()]);
        }
        if let Some(bitrate) = params.video_bitrate {
            args.extend(["-b:v".into(), bitrate.to_string()]);
        }
    } else if ffmpeg_video_codec == "libx264" {
        let preset = params
            .encoding_preset
            .unwrap_or_default()
            .to_string();
        args.extend([
            "-profile:v".into(),
            "high".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-crf".into(),
            params
                .h264_crf
                .to_string(),
            "-preset".into(),
            preset,
            "-tune".into(),
            "zerolatency".into(),
        ]);
        // Use client's max bitrate as a ceiling, not a CBR target.
        if let Some(bitrate) = params.video_bitrate {
            args.extend([
                "-maxrate".into(),
                bitrate.to_string(),
                "-bufsize".into(),
                (bitrate * 2).to_string(),
            ]);
        }
    } else if ffmpeg_video_codec == "libx265" {
        let preset = params
            .encoding_preset
            .unwrap_or_default()
            .to_string();
        args.extend([
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-crf".into(),
            params
                .h265_crf
                .to_string(),
            "-preset".into(),
            preset,
        ]);
        if let Some(bitrate) = params.video_bitrate {
            args.extend([
                "-maxrate".into(),
                bitrate.to_string(),
                "-bufsize".into(),
                (bitrate * 2).to_string(),
            ]);
        }
    } else if let Some(bitrate) = params.video_bitrate {
        args.extend(["-b:v".into(), bitrate.to_string()]);
    }

    // Audio codec
    args.extend(["-c:a".into(), ffmpeg_audio_codec.into()]);
    if ffmpeg_audio_codec == "copy" {
        // AAC streams from IPTV sources often use ADTS framing, which is not
        // valid inside MP4/fMP4 containers. Apply the reframing filter when copying.
        if params
            .source_audio_codec
            .as_deref()
            .and_then(|s| {
                s.parse::<AudioCodec>()
                    .ok()
            })
            .as_ref()
            .map(AudioCodec::needs_adts_reframe)
            .unwrap_or(false)
        {
            args.extend(["-bsf:a".into(), "aac_adtstoasc".into()]);
        }
    } else {
        let audio_bitrate = params
            .audio_bitrate
            .unwrap_or(128_000);
        args.extend(["-b:a".into(), audio_bitrate.to_string()]);
        if let Some(ch) = params.audio_channels {
            args.extend(["-ac".into(), ch.to_string()]);
        }
        if params.normalize_audio_loudness {
            args.extend(["-af".into(), "loudnorm=I=-14:TP=-1:LRA=11".into()]);
        }
    }

    // HLS output
    let playlist = params
        .output_dir
        .join("main.m3u8");
    let seg_ext = if is_hevc_copy { "m4s" } else { "ts" };
    let segment = params
        .output_dir
        .join(format!("segment_%05d.{}", seg_ext));

    let start_number =
        segment_start_index(params.start_time_ticks, params.segment_length);

    args.extend([
        "-f".into(),
        "hls".into(),
        "-hls_time".into(),
        params
            .segment_length
            .to_string(),
        "-start_number".into(),
        start_number.to_string(),
        "-hls_segment_filename".into(),
        segment
            .to_string_lossy()
            .into_owned(),
    ]);

    if params.is_live {
        // Sliding window: segments behind it are deleted (see
        // `live_cleanup_cutoff`), so an EVENT playlist would list missing
        // segments. omit_endlist: an ffmpeg exit must not end playback for the
        // client; the transcode loop ends the playlist itself when the source
        // turns out to end (`live_run_end`).
        //
        // A restarted ffmpeg (crash, HW→SW fallback) finds the previous run's
        // playlist: append_list continues it (next segment number,
        // EXT-X-DISCONTINUITY) instead of resetting it, which ExoPlayer
        // reports as PlaylistStuckException. Only on restarts, so a fresh
        // playlist doesn't start with a discontinuity. A source that sends a
        // backlog on connect appends that backlog again on every restart.
        let hls_flags = if playlist.exists() {
            "append_list+omit_endlist"
        } else {
            "omit_endlist"
        };
        args.extend([
            "-hls_list_size".into(),
            live_window_segments(params.segment_length).to_string(),
            "-hls_flags".into(),
            hls_flags.into(),
        ]);
    } else {
        // Keep all segments for the playback session.
        args.extend([
            "-hls_playlist_type".into(),
            "event".into(),
            "-hls_list_size".into(),
            "0".into(),
        ]);
    }

    if is_hevc_copy {
        // Apple HLS spec: HEVC must be delivered in fMP4/CMAF segments, not MPEG-TS.
        // Use just the filename for init; ffmpeg places it alongside the segments.
        //
        // -start_at_zero: shifts all timestamps so the first DTS = 0.
        // movflags=+frag_discont: writes each fragment's actual decode time into
        // MOOF::TRAF::TFDT so A/V sync is correct across discontinuities.
        args.extend([
            "-hls_segment_type".into(),
            "fmp4".into(),
            "-hls_fmp4_init_filename".into(),
            "init.mp4".into(),
            "-start_at_zero".into(),
            "-hls_segment_options".into(),
            "movflags=+frag_discont".into(),
        ]);
    }

    args.push(
        playlist
            .to_string_lossy()
            .into_owned(),
    );

    args
}

/// Spawn an ffmpeg process, drain its stderr to DEBUG logs, and wait for it
/// to finish (or be killed via `kill_rx`).  Returns the exit result and the
/// accumulated stderr text for error reporting.
async fn run_ffmpeg(
    args: Vec<String>,
    env_overrides: Vec<(String, String)>,
    kill_rx: tokio::sync::oneshot::Receiver<()>,
    monitor_stop_tx: tokio::sync::oneshot::Sender<()>,
    ffmpeg_pid_out: Arc<AtomicU32>,
    output_dir: std::path::PathBuf,
) -> (
    Option<std::result::Result<std::process::ExitStatus, std::io::Error>>,
    String,
) {
    let mut cmd = tokio::process::Command::new(ffmpeg_bin());
    cmd.hide_console();
    cmd.args(&args)
        .stderr(Stdio::piped());
    for (k, v) in env_overrides {
        cmd.env(k, v);
    }
    let child = cmd.spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            let _ = monitor_stop_tx.send(());
            return (Some(Err(e)), String::new());
        }
    };

    let pid = child
        .id()
        .unwrap_or(0);
    ffmpeg_pid_out.store(pid, Ordering::Relaxed);
    if pid > 0 {
        let _ = std::fs::write(output_dir.join(".pid"), pid.to_string());
    }
    let stderr = child
        .stderr
        .take();

    let (stderr_tx, stderr_rx) = tokio::sync::oneshot::channel::<String>();
    if let Some(stderr) = stderr {
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            let mut buf = String::new();
            while let Ok(Some(line)) = lines
                .next_line()
                .await
            {
                if !line.is_empty() {
                    debug!("ffmpeg: {}", line);
                    buf.push_str(&line);
                    buf.push('\n');
                }
            }
            let _ = stderr_tx.send(buf);
        });
    } else {
        let _ = stderr_tx.send(String::new());
    }

    let pid = ffmpeg_pid_out.load(Ordering::Relaxed);
    let result = tokio::select! {
        r = child.wait() => Some(r),
        _ = kill_rx => {
            // Resume if the buffer monitor paused ffmpeg so kill() can land.
            #[cfg(unix)]
            send_signal(pid, libc::SIGCONT);
            let _ = child.kill().await;
            let _ = child.wait().await;
            None
        }
    };

    let _ = monitor_stop_tx.send(());
    let stderr_out = stderr_rx
        .await
        .unwrap_or_default();
    (result, stderr_out)
}

/// How long a stop waits for the transcode loop to reap a killed ffmpeg.
pub(crate) const STOP_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Start an HLS transcode job by spawning ffmpeg.
/// If hardware-accelerated encoding fails, automatically retries once with
/// software encoding so a broken HW driver doesn't permanently block playback.
pub async fn start_transcode(
    session: Arc<RwLock<TranscodeSession>>,
    params: TranscodeParams,
) -> Result<()> {
    {
        let mut s = session
            .write()
            .await;
        // Stopped before it started: don't recreate the removed output dir.
        if s.stopped
            .is_cancelled()
        {
            return Ok(());
        }
        s.state = TranscodeState::Running;
        let _ = s
            .state_tx
            .send(TranscodeState::Running);
    }

    std::fs::create_dir_all(&params.output_dir)
        .map_err(|e| anyhow!("Failed to create output dir: {}", e))?;

    let session_clone = session.clone();
    tokio::spawn(async move {
        let mut params = params;
        let mut sw_fallback = false;
        let mut live_restarts = 0u32;
        const MAX_LIVE_RESTARTS: u32 = 10;
        let (stopped, live_activity) = {
            let s = session_clone
                .read()
                .await;
            (
                s.stopped
                    .clone(),
                s.live_activity
                    .clone(),
            )
        };
        // Ends the live maintenance task together with this loop.
        let ended = stopped.child_token();
        let _ended_on_exit = ended
            .clone()
            .drop_guard();
        let ffmpeg_pid = Arc::new(AtomicU32::new(0));
        let live_run = Arc::new(LiveRun::default());
        let start_index =
            segment_start_index(params.start_time_ticks, params.segment_length);
        if params.is_live {
            spawn_live_maintenance(
                session_clone.clone(),
                start_index,
                live_run.clone(),
                ffmpeg_pid.clone(),
                ended,
            );
        }

        loop {
            let args = build_hls_args(&params);
            debug!("ffmpeg args: {}", args.join(" "));
            if params.is_live {
                // Where this run starts writing (append_list continues after
                // the previous run), for `LiveRun::faster_than_realtime`.
                live_run.start(
                    highest_segment_index(&params.output_dir)
                        .map_or(start_index, |highest| highest + 1),
                );
            }
            let (kill_tx, kill_rx) = tokio::sync::oneshot::channel::<()>();
            let (monitor_stop_tx, monitor_stop_rx) =
                tokio::sync::oneshot::channel::<()>();

            let output_dir = {
                let mut s = session_clone
                    .write()
                    .await;
                // Stopped while no ffmpeg ran (no kill_tx to signal).
                if s.stopped
                    .is_cancelled()
                {
                    debug!(session_id = %s.id, "transcode stopped before ffmpeg (re)spawn");
                    s.wait_done
                        .notify_one();
                    break;
                }
                s.start_time_secs = params
                    .start_time_ticks
                    .map(|t| (t / 10_000_000) as u32)
                    .unwrap_or(0);
                s.kill_tx = Some(kill_tx);
                // Live sessions are throttled by `spawn_live_maintenance`, by
                // requested segments rather than reported positions.
                if !params.is_live {
                    spawn_buffer_monitor(
                        s.output_dir
                            .clone(),
                        s.segment_length,
                        segment_start_index(params.start_time_ticks, s.segment_length),
                        s.playback_offset_secs
                            .clone(),
                        ffmpeg_pid.clone(),
                        monitor_stop_rx,
                        s.id.clone(),
                    );
                }
                s.output_dir
                    .clone()
            };

            let env_overrides: Vec<(String, String)> = params
                .accelerator
                .env_overrides()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            let (result, stderr_out) = run_ffmpeg(
                args,
                env_overrides,
                kill_rx,
                monitor_stop_tx,
                ffmpeg_pid.clone(),
                output_dir,
            )
            .await;
            ffmpeg_pid.store(0, Ordering::Relaxed);

            let using_hw = params
                .accelerator
                .is_hw();

            // If HW accel caused the failure, retry once with software encoding.
            if !sw_fallback
                && using_hw
                && matches!(&result, Some(Ok(s)) if !s.success())
            {
                warn!(
                    accel = ?params.accelerator.as_type(),
                    stderr = stderr_out.trim(),
                    "HW-accelerated transcode failed — retrying with software encoding"
                );
                // Clean partial output so the retry starts fresh. Live keeps
                // it: the retry appends to the playlist the client is following.
                if !params.is_live {
                    let _ = std::fs::remove_dir_all(&params.output_dir);
                    let _ = std::fs::create_dir_all(&params.output_dir);
                }
                params.accelerator = Box::new(NoAccel);
                // VPP tonemap can't run without HW; preserve the user's intent by
                // activating software tonemapping so HDR content isn't washed out.
                if params.enable_vpp_tonemapping {
                    params.enable_tonemapping = true;
                }
                sw_fallback = true;
                continue;
            }

            // Decided before taking the session lock: it may ask the upstream.
            let source_ended = params.is_live
                && !stopped.is_cancelled()
                && live_source_ended(
                    result
                        .as_ref()
                        .and_then(|r| {
                            r.as_ref()
                                .ok()
                        }),
                    &stderr_out,
                    &params.input_url,
                )
                .await;

            let mut s = session_clone
                .write()
                .await;
            // A segment restart that gave up waiting for this run started a
            // newer one on this session: leave the session to it.
            if s.kill_tx
                .as_ref()
                .is_some_and(|kill_tx| !kill_tx.is_closed())
            {
                break;
            }
            s.kill_tx = None;

            // Live: restart on upstream failures, wait for a client that went
            // away, and end the playlist when the source ended.
            if params.is_live {
                let end = live_run_end(
                    stopped.is_cancelled(),
                    live_run
                        .idle_stopped
                        .swap(false, Ordering::Relaxed),
                    result.is_none(),
                    source_ended,
                );
                match end {
                    LiveRunEnd::Stopped => {
                        debug!(session_id = %s.id, "live stream killed by session stop");
                        s.wait_done
                            .notify_one();
                        break;
                    }
                    LiveRunEnd::Idle => {
                        let session_id =
                            s.id.clone();
                        drop(s);
                        if !wait_for_live_client(&live_activity, &stopped).await {
                            debug!(session_id = %session_id, "idle live stream stopped");
                            break;
                        }
                        info!(session_id = %session_id, "Live client is back — restarting ffmpeg");
                        continue;
                    }
                    LiveRunEnd::Ended => {
                        info!(session_id = %s.id, "Live source ended — ending the playlist");
                        if let Err(e) = end_live_playlist(&s.variant_playlist_path()) {
                            warn!(session_id = %s.id, error = %e, "failed to end the live playlist");
                        }
                        s.state = TranscodeState::Complete;
                        let _ = s
                            .state_tx
                            .send(TranscodeState::Complete);
                        s.wait_done
                            .notify_one();
                        break;
                    }
                    LiveRunEnd::Restart => {
                        let (status_str, stderr_str) = match &result {
                            Some(Err(e)) => ("error".to_string(), e.to_string()),
                            Some(Ok(status)) => (
                                format!("{status}"),
                                stderr_out
                                    .trim()
                                    .to_string(),
                            ),
                            None => ("killed".to_string(), String::new()),
                        };
                        if live_restarts < MAX_LIVE_RESTARTS {
                            live_restarts += 1;
                            warn!(
                                session_id = %s.id,
                                status = %status_str,
                                restart = live_restarts,
                                stderr = %stderr_str,
                                "live stream ffmpeg exited unexpectedly, restarting"
                            );
                            drop(s);
                            // A session stopped meanwhile gets no new ffmpeg.
                            tokio::select! {
                                _ = stopped.cancelled() => break,
                                _ = tokio::time::sleep(Duration::from_secs(3)) => continue,
                            }
                        } else {
                            let err_msg = format!(
                                "live stream ffmpeg exited after {MAX_LIVE_RESTARTS} restarts \
                                 (status {status_str}): {stderr_str}"
                            );
                            error!(session_id = %s.id, error = %err_msg, "Live stream failed");
                            s.state = TranscodeState::Error(err_msg.clone());
                            let _ = s
                                .state_tx
                                .send(TranscodeState::Error(err_msg));
                            s.wait_done
                                .notify_one();
                            break;
                        }
                    }
                }
            }

            match result {
                Some(Ok(status)) if status.success() => {
                    s.state = TranscodeState::Complete;
                    let _ = s
                        .state_tx
                        .send(TranscodeState::Complete);
                    info!(session_id = %s.id, sw_fallback, "Transcode completed successfully");
                }
                Some(Ok(status)) => {
                    let err_msg = format!(
                        "ffmpeg exited with status {}: {}",
                        status,
                        stderr_out.trim()
                    );
                    error!(session_id = %s.id, error = %err_msg, "Transcode failed");
                    s.state = TranscodeState::Error(err_msg.clone());
                    let _ = s
                        .state_tx
                        .send(TranscodeState::Error(err_msg));
                }
                Some(Err(e)) => {
                    let err_msg = format!("Failed to wait for ffmpeg: {}", e);
                    error!(session_id = %s.id, error = %err_msg, "Transcode error");
                    s.state = TranscodeState::Error(err_msg.clone());
                    let _ = s
                        .state_tx
                        .send(TranscodeState::Error(err_msg));
                }
                None => {
                    debug!(session_id = %s.id, "ffmpeg killed by session stop");
                }
            }

            s.wait_done
                .notify_one();
            break;
        }
    });

    Ok(())
}

/// Parameters for a progressive (non-HLS) transcode that streams to stdout.
pub struct ProgressiveTranscodeParams {
    pub input_url: String,
    pub container: String,   // "mp4", "ts", "mkv", "webm"
    pub video_codec: String, // "copy", "libx264", "libx265", "libvpx-vp9"
    pub audio_codec: String, // "copy", "aac", "libopus"
    pub start_time_ticks: Option<i64>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub video_bitrate: Option<u32>,
    pub audio_bitrate: Option<u32>,
    pub audio_channels: Option<u32>,
    pub audio_stream_index: Option<i32>,
    pub subtitle_stream_index: Option<i32>,
    pub burn_subtitle: bool,
    pub subtitle_width: Option<u32>,
    pub subtitle_height: Option<u32>,
    pub encoding_preset: Option<EncodingPreset>,
    pub source_video_codec: Option<String>,
    pub source_audio_codec: Option<String>,
    /// See `TranscodeParams::hevc_copy_tag`.
    pub hevc_copy_tag: Option<String>,
    pub accelerator: Box<dyn Accelerator>,
    pub source_video_range_type: Option<VideoRangeType>,
    pub enable_tonemapping: bool,
    pub enable_vpp_tonemapping: bool,
    pub tonemapping_algorithm: String,
    pub tonemapping_desat: f32,
    pub tonemapping_peak: f32,
    pub allow_hevc_encoding: bool,
    pub allow_av1_encoding: bool,
    pub h264_crf: u32,
    pub h265_crf: u32,
    /// Apply `loudnorm=I=-14:TP=-1:LRA=11` when transcoding audio. Has no effect
    /// when audio_codec is "copy". See EncodingOptions::normalize_audio_loudness.
    pub normalize_audio_loudness: bool,
}

/// Build the ffmpeg CLI args for a progressive transcode piped to stdout.
/// Container ffmpeg writes a progressive stream in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgressiveFormat {
    Mpegts,
    Webm,
    Matroska,
    Mp4,
}

impl ProgressiveFormat {
    /// The requested container, except that copied video into MP4 would need
    /// bitstream filters, so it is promoted to Matroska instead.
    pub(crate) fn for_request(container: &str, video_codec: &str) -> Self {
        let requested = match container {
            "ts" | "mpegts" => Self::Mpegts,
            "webm" => Self::Webm,
            "mkv" | "matroska" => Self::Matroska,
            _ => Self::Mp4,
        };
        if video_codec == "copy" && requested == Self::Mp4 {
            Self::Matroska
        } else {
            requested
        }
    }

    fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Mpegts => "mpegts",
            Self::Webm => "webm",
            Self::Matroska => "matroska",
            Self::Mp4 => "mp4",
        }
    }

    /// Whether source audio can be stream-copied into this container. MPEG-TS
    /// and MP4 follow the HLS segment rules; WebM only carries Opus and
    /// Vorbis. Unknown codecs are assumed copyable.
    pub(crate) fn can_copy_audio(self, source_audio_codec: Option<&str>) -> bool {
        match self {
            Self::Matroska => true,
            Self::Mp4 => SegmentContainer::Fmp4.can_copy_audio(source_audio_codec),
            Self::Mpegts => SegmentContainer::Ts.can_copy_audio(source_audio_codec),
            Self::Webm => match source_audio_codec.and_then(|s| {
                s.parse::<AudioCodec>()
                    .ok()
            }) {
                Some(AudioCodec::Opus | AudioCodec::Vorbis | AudioCodec::Other(_))
                | None => true,
                Some(_) => false,
            },
        }
    }

    /// Codec to encode audio to when the source can't be copied.
    pub(crate) fn fallback_audio_codec(self) -> &'static str {
        match self {
            Self::Webm => "opus",
            _ => "aac",
        }
    }
}

pub(crate) fn build_progressive_args(
    params: &ProgressiveTranscodeParams,
) -> Vec<String> {
    let accel = params
        .accelerator
        .as_ref();
    let is_hw = accel.is_hw();
    let hdr = is_hdr(
        params
            .source_video_range_type
            .as_ref(),
    );

    // The API layer must explicitly select a video encoder after checking
    // server and user permissions. Never turn a constrained copy request back
    // into a transcode here merely because SubtitleMethod=Encode was supplied.
    let burn_subtitle_filter = params.burn_subtitle
        && params
            .subtitle_stream_index
            .is_some_and(|index| index >= 0)
        && params.video_codec != "copy";

    let treatment = HdrTreatment::for_source(
        hdr,
        accel,
        params.enable_tonemapping,
        params.enable_vpp_tonemapping,
        burn_subtitle_filter,
    );
    let do_sw_tonemap = treatment == HdrTreatment::SwTonemap;

    let ffmpeg_video_codec = {
        let base = match params
            .video_codec
            .as_str()
        {
            "copy" => "copy",
            _ => "libx264",
        };
        if base != "copy" && is_hw {
            accel.encoder_name(base)
        } else {
            base.to_string()
        }
    };
    let ffmpeg_audio_codec = match params
        .audio_codec
        .as_str()
    {
        "copy" => "copy",
        "aac" => "aac",
        "libopus" | "opus" => "libopus",
        "mp3" => "libmp3lame",
        other => other,
    };

    let format = ProgressiveFormat::for_request(&params.container, &ffmpeg_video_codec)
        .ffmpeg_name();

    let mut args: Vec<String> = vec![
        "-v".into(),
        "error".into(),
        "-analyzeduration".into(),
        "5000000".into(),
        "-probesize".into(),
        "5000000".into(),
    ];
    args.extend(
        ffmpeg_http_input_args(&params.input_url)
            .iter()
            .map(|s| (*s).into()),
    );

    args.extend(
        accel.decode_input_args(
            params
                .source_video_codec
                .as_deref(),
            treatment,
            burn_subtitle_filter,
        ),
    );

    // Input seek (fast, before -i)
    if let Some(ticks) = params.start_time_ticks {
        let secs = ticks as f64 / 10_000_000.0;
        args.extend(["-ss".into(), format!("{:.6}", secs)]);
    }

    args.extend([
        "-i".into(),
        params
            .input_url
            .clone(),
    ]);

    let tonemap = TonemapOptions {
        algorithm: &params.tonemapping_algorithm,
        desat: params.tonemapping_desat,
        peak: params.tonemapping_peak,
    };
    let hw_suffix = accel.hw_filter_suffix(treatment, &tonemap);

    // Stream mapping
    let scale_filter = if ffmpeg_video_codec != "copy" {
        if hw_gpu_resident(accel, treatment) {
            Some(build_vaapi_scale_filter(
                treatment,
                params.max_width,
                params.max_height,
            ))
        } else {
            let tp = TranscodeParams {
                max_width: params.max_width,
                max_height: params.max_height,
                ..Default::default()
            };
            build_scale_filter(&tp)
        }
    } else {
        None
    };

    if burn_subtitle_filter {
        if let Some(sub_idx) = params.subtitle_stream_index {
            // Image subtitle (PGS/DVD): bitmap overlay via filter_complex.
            let (out_w, out_h) = (params.max_width, params.max_height);
            let sub_scale = match (out_w, out_h) {
                (Some(w), Some(h)) => format!("scale={w}:{h}:fast_bilinear"),
                (Some(w), None) => format!("scale={w}:-1:fast_bilinear"),
                (None, Some(h)) => format!("scale=-1:{h}:fast_bilinear"),
                _ => String::new(),
            };
            let sub_preproc = if sub_scale.is_empty() {
                "scale".to_string()
            } else {
                format!("scale,{sub_scale}")
            };

            let filter = if is_hw
                && accel.supports_gpu_resident_overlay()
                && treatment != HdrTreatment::SwTonemap
            {
                // QSV path: overlay_qsv keeps video in GPU memory throughout;
                // tone map / relabel runs on the main branch before the overlay.
                // QSV-only: see the equivalent comment in build_hls_args.
                let sub_upload =
                    "format=bgra,hwupload=derive_device=qsv:extra_hw_frames=64";
                let overlay_size = match (out_w, out_h) {
                    (Some(w), Some(h)) => format!(":w={w}:h={h}"),
                    _ => String::new(),
                };
                let overlay_qsv =
                    format!("overlay_qsv=eof_action=pass:repeatlast=0{overlay_size}");
                let main_video =
                    build_vaapi_main_video(accel, treatment, &tonemap, out_w, out_h);
                format!(
                    "[0:{sub_idx}]{sub_preproc},{sub_upload}[sub];\
                     [0:v:0]{main_video}[vmain];\
                     [vmain][sub]{overlay_qsv}[v]"
                )
            } else {
                let main_video = build_cpu_main_video_filters(
                    scale_filter.clone(),
                    hdr,
                    do_sw_tonemap,
                    &params.tonemapping_algorithm,
                    params.tonemapping_desat,
                    params.tonemapping_peak,
                );
                build_cpu_overlay_complex(
                    sub_idx,
                    &sub_preproc,
                    &main_video,
                    hw_suffix.as_deref(),
                )
            };
            args.extend(["-filter_complex".into(), filter]);
            args.extend(["-map".into(), "[v]".into()]);
            if let Some(audio_idx) = params.audio_stream_index {
                args.extend(["-map".into(), format!("0:{}", audio_idx)]);
            } else {
                args.extend(["-map".into(), "0:a?".into()]);
            }
        }
    } else {
        let vf = match (&scale_filter, &hw_suffix) {
            (Some(s), Some(suf)) => Some(format!("{s},{suf}")),
            (Some(s), None) => Some(s.clone()),
            (None, Some(suf)) if ffmpeg_video_codec != "copy" => Some(suf.clone()),
            _ => None,
        };
        let vf = if hdr && ffmpeg_video_codec != "copy" {
            if do_sw_tonemap {
                let algo = &params.tonemapping_algorithm;
                let desat = params.tonemapping_desat;
                let peak = params.tonemapping_peak;
                let out_fmt = if !accel.is_hw()
                    || accel.as_type() == HardwareAccelerationType::V4l2m2m
                {
                    "yuv420p"
                } else {
                    "nv12"
                };
                let tonemapx = format!(
                    "tonemapx=tonemap={algo}:desat={desat:.1}:peak={peak:.1}:t=bt709:m=bt709:p=bt709:format={out_fmt}"
                );
                let upload = if accel.as_type() == HardwareAccelerationType::Vaapi {
                    ",hwupload"
                } else {
                    ""
                };
                Some(match scale_filter.as_deref() {
                    Some(s) => format!("{s},{tonemapx}{upload}"),
                    None => format!("{tonemapx}{upload}"),
                })
            } else if hw_gpu_resident(accel, treatment) {
                vf
            } else {
                let setparams =
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709";
                Some(match vf {
                    Some(f) => format!("{setparams},{f}"),
                    None => setparams.to_string(),
                })
            }
        } else {
            vf
        };
        if let Some(ref filter) = vf {
            args.extend(["-vf".into(), filter.clone()]);
        } else if params
            .audio_stream_index
            .is_some()
            || params
                .subtitle_stream_index
                .is_some()
        {
            args.extend(["-map".into(), "0:v:0".into()]);
            if let Some(audio_idx) = params.audio_stream_index {
                args.extend(["-map".into(), format!("0:{}", audio_idx)]);
            } else {
                args.extend(["-map".into(), "0:a?".into()]);
            }
            if let Some(sub_idx) = params.subtitle_stream_index {
                if sub_idx >= 0 {
                    args.extend(["-map".into(), format!("0:{}?", sub_idx)]);
                }
            }
        }
    }

    // Video
    args.extend(["-c:v".into(), ffmpeg_video_codec.clone()]);
    if ffmpeg_video_codec == "copy" {
        // Apply the HEVC sample-entry tag the client asked for.
        if params
            .source_video_codec
            .as_deref()
            .and_then(|s| {
                s.parse::<VideoCodec>()
                    .ok()
            })
            .as_ref()
            .map(VideoCodec::is_hevc)
            .unwrap_or(false)
        {
            args.extend([
                "-tag:v".into(),
                hevc_copy_tag(
                    params
                        .hevc_copy_tag
                        .as_deref(),
                )
                .into(),
            ]);
        }
    } else if is_hw {
        // h264_nvenc needs an explicit 8-bit output: 10-bit frames fail, and
        // CUDA decode leaves frames in software memory so FFmpeg can convert
        // them to yuv420p before the encoder. The option applies with or
        // without a bitrate.
        if ffmpeg_video_codec == "h264_nvenc" {
            args.extend(["-pix_fmt".into(), "yuv420p".into()]);
        }
        if let Some(bitrate) = params.video_bitrate {
            args.extend(["-b:v".into(), bitrate.to_string()]);
        }
    } else if ffmpeg_video_codec == "libx264" {
        let preset = params
            .encoding_preset
            .unwrap_or_default()
            .to_string();
        args.extend([
            "-profile:v".into(),
            "high".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-crf".into(),
            params
                .h264_crf
                .to_string(),
            "-preset".into(),
            preset,
        ]);
        if let Some(bitrate) = params.video_bitrate {
            args.extend([
                "-maxrate".into(),
                bitrate.to_string(),
                "-bufsize".into(),
                (bitrate * 2).to_string(),
            ]);
        }
    } else if ffmpeg_video_codec == "libx265" {
        let preset = params
            .encoding_preset
            .unwrap_or_default()
            .to_string();
        args.extend([
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-crf".into(),
            params
                .h265_crf
                .to_string(),
            "-preset".into(),
            preset,
        ]);
        if let Some(bitrate) = params.video_bitrate {
            args.extend([
                "-maxrate".into(),
                bitrate.to_string(),
                "-bufsize".into(),
                (bitrate * 2).to_string(),
            ]);
        }
    } else if let Some(bitrate) = params.video_bitrate {
        args.extend(["-b:v".into(), bitrate.to_string()]);
    }

    // Audio
    args.extend(["-c:a".into(), ffmpeg_audio_codec.into()]);
    if ffmpeg_audio_codec == "copy" {
        if params
            .source_audio_codec
            .as_deref()
            .and_then(|s| {
                s.parse::<AudioCodec>()
                    .ok()
            })
            .as_ref()
            .map(AudioCodec::needs_adts_reframe)
            .unwrap_or(false)
        {
            args.extend(["-bsf:a".into(), "aac_adtstoasc".into()]);
        }
    } else {
        if let Some(bitrate) = params.audio_bitrate {
            args.extend(["-b:a".into(), bitrate.to_string()]);
        }
        if let Some(ch) = params.audio_channels {
            args.extend(["-ac".into(), ch.to_string()]);
        }
        if params.normalize_audio_loudness {
            args.extend(["-af".into(), "loudnorm=I=-14:TP=-1:LRA=11".into()]);
        }
    }

    // Format-specific flags
    args.extend(["-strict".into(), "unofficial".into()]);
    if format == "mp4" {
        args.extend([
            "-movflags".into(),
            "frag_keyframe+empty_moov+default_base_moof".into(),
        ]);
    }

    args.extend(["-f".into(), format.into(), "pipe:1".into()]);

    args
}

/// Start a progressive transcode that returns a readable byte stream.
pub fn start_progressive_transcode(
    params: ProgressiveTranscodeParams,
) -> Result<
    impl futures::Stream<Item = std::result::Result<bytes::Bytes, std::io::Error>>,
> {
    let args = build_progressive_args(&params);
    debug!("ffmpeg progressive args: {:?}", args);

    let env_overrides = params
        .accelerator
        .env_overrides();
    let mut cmd = tokio::process::Command::new(ffmpeg_bin());
    cmd.hide_console();
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env_overrides {
        cmd.env(k, v);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn ffmpeg: {}", e))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to capture ffmpeg stdout"))?;

    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to capture ffmpeg stderr"))?;

    // Log stderr line-by-line at DEBUG and reap child when done.
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines
            .next_line()
            .await
        {
            if !line.is_empty() {
                debug!("ffmpeg: {}", line);
            }
        }
        match child
            .wait()
            .await
        {
            Ok(status) if !status.success() => {
                if status.code() == Some(224) {
                    debug!(
                        "progressive ffmpeg exited after client disconnect: {}",
                        status
                    )
                } else {
                    error!("progressive ffmpeg exited: {}", status)
                }
            }
            Ok(status) => debug!("progressive ffmpeg exited: {}", status),
            Err(e) => error!("progressive ffmpeg wait error: {}", e),
        }
    });

    info!(
        "Starting progressive transcode (container={}, vcodec={}, acodec={})",
        params.container, params.video_codec, params.audio_codec
    );

    Ok(tokio_util::io::ReaderStream::new(stdout))
}

/// Generate the variant (child) HLS playlist server-side as a VOD playlist.
///
/// Lists ALL segments from time 0 to the end of the media so HLS.js can seek
/// to any position immediately. Each segment URL includes `runtimeTicks` and
/// `actualSegmentLengthTicks` query params so the segment handler knows the
/// cumulative position when it needs to restart FFmpeg.
pub fn generate_variant_playlist(
    session: &TranscodeSession,
    query_string: &str,
) -> String {
    let runtime_ticks = session.runtime_ticks;
    let segment_length = session.segment_length;
    let play_session_id = &session.id;
    let start_time_secs = session.start_time_secs;
    let use_fmp4 = session.use_fmp4();
    // fMP4 segments require HLS version 7; standard TS segments need version 6.
    let version = if use_fmp4 { 7u32 } else { 6u32 };
    let seg_ext = if use_fmp4 { "m4s" } else { "ts" };

    let extra_qs = if query_string.is_empty() {
        String::new()
    } else {
        format!("&{}", query_string)
    };

    if runtime_ticks <= 0 || segment_length == 0 {
        // Fallback: single long segment (shouldn't happen in practice)
        let mut buf = format!(
            "#EXTM3U\n\
             #EXT-X-PLAYLIST-TYPE:VOD\n\
             #EXT-X-VERSION:{version}\n\
             #EXT-X-TARGETDURATION:{seg}\n\
             #EXT-X-MEDIA-SEQUENCE:0\n",
            version = version,
            seg = segment_length,
        );
        if use_fmp4 {
            buf.push_str(&format!(
                "#EXT-X-MAP:URI=\"init.mp4?PlaySessionId={}\"\n",
                play_session_id
            ));
        }
        buf.push_str(&format!(
            "#EXTINF:{seg}.000000, nodesc\n\
             segment_00000.{ext}?PlaySessionId={psid}{qs}\n\
             #EXT-X-ENDLIST\n",
            seg = segment_length,
            ext = seg_ext,
            psid = play_session_id,
            qs = extra_qs,
        ));
        return buf;
    }

    let seg_length_ticks = (segment_length as i64)
        .to_ticks(TickUnit::Seconds)
        .unwrap_or(0);
    let whole_segments = runtime_ticks / seg_length_ticks;
    let remaining_ticks = runtime_ticks % seg_length_ticks;
    let total_segments = whole_segments + if remaining_ticks > 0 { 1 } else { 0 };

    let target_duration = segment_length + 1; // +1 to cover keyframe-boundary drift (HLS spec: must be >= any segment duration)

    let mut buf = String::with_capacity(total_segments as usize * 120);
    buf.push_str("#EXTM3U\n");
    buf.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
    buf.push_str(&format!("#EXT-X-VERSION:{}\n", version));
    buf.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_duration));
    buf.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
    // EXT-X-START is only useful for TS segments: for fMP4 the synthetic playlist
    // always starts at segment_00000 (no -start_number offset) and the Jellyfin
    // Web client drives seeking via video.currentTime. Emitting EXT-X-START for
    // fMP4 causes hls.js to attempt an in-buffer MSE seek on the very first
    // segment, which triggers a decode error in browsers playing HEVC 10-bit.
    if start_time_secs > 0 && !use_fmp4 {
        buf.push_str(&format!(
            "#EXT-X-START:TIME-OFFSET={:.6},PRECISE=YES\n",
            start_time_secs as f64
        ));
    }
    if use_fmp4 {
        buf.push_str(&format!(
            "#EXT-X-MAP:URI=\"init.mp4?PlaySessionId={}\"\n",
            play_session_id
        ));
    }

    let mut cumulative_ticks: i64 = 0;
    for i in 0..total_segments {
        let length_ticks = if i < whole_segments {
            seg_length_ticks
        } else {
            remaining_ticks
        };
        let length_secs = length_ticks as f64 / 10_000_000.0;

        buf.push_str(&format!("#EXTINF:{:.6}, nodesc\n", length_secs));
        buf.push_str(&format!(
            "segment_{:05}.{}?PlaySessionId={}&runtimeTicks={}&actualSegmentLengthTicks={}{}\n",
            i,
            seg_ext,
            play_session_id,
            cumulative_ticks,
            length_ticks,
            extra_qs,
        ));

        cumulative_ticks += length_ticks;
    }

    buf.push_str("#EXT-X-ENDLIST\n");
    buf
}

/// Generate the HEVC codec string for HLS CODECS attribute.
/// Matches Jellyfin's `HlsCodecStringHelpers.GetH265String()`.
fn hevc_hls_codec_string(
    tag: &str,
    profile: Option<&str>,
    level: Option<f64>,
) -> String {
    let profile_part = match profile {
        Some(p)
            if p.eq_ignore_ascii_case("main 10")
                || p.eq_ignore_ascii_case("main10") =>
        {
            "2.4"
        }
        _ => "1.4",
    };
    let level_val = level.unwrap_or(150.0) as i32;
    format!("{}.{}.L{}.B0", tag, profile_part, level_val)
}

/// Generate a master HLS playlist that references the variant playlist.
pub fn generate_master_playlist(session: &TranscodeSession) -> String {
    use remux_sdks::remux::VideoRangeType;

    let play_session_id = &session.id;

    let video_codec_str: String = match session
        .video_codec
        .as_str()
    {
        "copy" => {
            if session
                .source_video_codec
                .as_deref()
                .and_then(|s| {
                    s.parse::<VideoCodec>()
                        .ok()
                })
                .as_ref()
                .map(VideoCodec::is_hevc)
                .unwrap_or(false)
            {
                // Must name the fourcc build_hls_args actually wrote, or
                // clients that validate CODECS against the segments (Safari
                // via MSE) reject the stream.
                hevc_hls_codec_string(
                    hevc_copy_tag(
                        session
                            .hevc_copy_tag
                            .as_deref(),
                    ),
                    session
                        .source_video_profile
                        .as_deref(),
                    session.source_video_level,
                )
            } else {
                "avc1.640028".to_string()
            }
        }
        "h264" | "libx264" => "avc1.640028".to_string(),
        // Encoding produces MPEG-TS segments, which carry no sample entry at
        // all; hvc1 is the conventional RFC 6381 identifier there.
        "hevc" | "libx265" => hevc_hls_codec_string(
            "hvc1",
            session
                .source_video_profile
                .as_deref(),
            session.source_video_level,
        ),
        _ => "avc1.640028".to_string(),
    };
    let audio_codec_str = if session.audio_codec == "copy"
        && session
            .segment_container()
            .can_copy_audio(
                session
                    .source_audio_codec
                    .as_deref(),
            ) {
        // Use the actual source codec when copying so the CODECS attribute
        // matches the bitstream. Browsers that see "mp4a.40.2" but receive
        // eac3 will fail to initialize the audio decoder. Otherwise
        // build_hls_args encodes AAC.
        match session
            .source_audio_codec
            .as_deref()
            .and_then(|s| {
                s.parse::<AudioCodec>()
                    .ok()
            }) {
            Some(AudioCodec::Eac3) => "ec-3",
            Some(AudioCodec::Ac3) => "ac-3",
            Some(AudioCodec::Dts) => "dtsh",
            Some(AudioCodec::Flac) => "fLaC",
            _ => "mp4a.40.2",
        }
    } else {
        "mp4a.40.2"
    };
    let codecs = format!("{},{}", video_codec_str, audio_codec_str);

    // VIDEO-RANGE: only meaningful for copy-mode (passthrough) video. Transcoded output is SDR.
    let video_range_attr = if session.video_codec == "copy" {
        match session.source_video_range_type {
            Some(VideoRangeType::Hdr10)
            | Some(VideoRangeType::Hdr10Plus)
            | Some(VideoRangeType::DoviWithHdr10)
            | Some(VideoRangeType::Dovi) => ",VIDEO-RANGE=PQ",
            Some(VideoRangeType::Hlg) | Some(VideoRangeType::DoviWithHlg) => {
                ",VIDEO-RANGE=HLG"
            }
            _ => ",VIDEO-RANGE=SDR",
        }
    } else {
        ",VIDEO-RANGE=SDR"
    };

    let resolution_attr =
        match (session.source_video_width, session.source_video_height) {
            (Some(w), Some(h)) => format!(",RESOLUTION={}x{}", w, h),
            _ => String::new(),
        };

    let frame_rate_attr = match session.source_frame_rate {
        Some(fps) if fps > 0.0 => format!(",FRAME-RATE={:.3}", fps),
        _ => String::new(),
    };

    debug!(
        source_video_codec = ?session.source_video_codec,
        source_video_profile = ?session.source_video_profile,
        source_video_level = ?session.source_video_level,
        source_video_range_type = ?session.source_video_range_type,
        source_video_width = session.source_video_width,
        source_video_height = session.source_video_height,
        source_frame_rate = session.source_frame_rate,
        video_codec = session.video_codec.as_str(),
        codecs = codecs.as_str(),
        "generating master HLS playlist"
    );

    format!(
        "#EXTM3U\n\
         #EXT-X-VERSION:6\n\
         #EXT-X-INDEPENDENT-SEGMENTS\n\
         #EXT-X-STREAM-INF:BANDWIDTH=2000000,AVERAGE-BANDWIDTH=2000000,CODECS=\"{codecs}\"{video_range}{resolution}{frame_rate}\n\
         main.m3u8?PlaySessionId={play_session_id}\n",
        codecs = codecs,
        video_range = video_range_attr,
        resolution = resolution_attr,
        frame_rate = frame_rate_attr,
        play_session_id = play_session_id,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::hw_accel;
    use remux_sdks::remux::TranscodeReasons;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn args_contains(args: &[String], flag: &str) -> bool {
        args.iter()
            .any(|a| a == flag)
    }

    fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    }

    fn no_devices(_: &str) -> bool {
        false
    }
    fn all_devices(_: &str) -> bool {
        true
    }

    #[test]
    fn segment_index_parses_ts_and_m4s_and_rejects_non_segment_files() {
        assert_eq!(segment_index("segment_00042.ts"), Some(42));
        assert_eq!(segment_index("segment_00042.m4s"), Some(42));
        assert_eq!(segment_index("segment_00000.m4s"), Some(0));
        assert_eq!(segment_index("init.mp4"), None);
        assert_eq!(segment_index("playlist.m3u8"), None);
    }

    #[test]
    fn highest_segment_index_finds_the_max_across_m4s_segments() {
        // Regression test for #560: HEVC-copy sessions write .m4s segments,
        // and the buffer monitor must see them the same as .ts ones.
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "segment_00000.m4s",
            "segment_00003.m4s",
            "segment_00001.m4s",
        ] {
            std::fs::write(
                dir.path()
                    .join(name),
                b"",
            )
            .unwrap();
        }
        assert_eq!(
            highest_segment_index(
                &dir.path()
                    .to_path_buf()
            ),
            Some(3)
        );
    }

    #[test]
    fn highest_segment_index_finds_the_max_across_ts_segments() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["segment_00000.ts", "segment_00007.ts", "segment_00002.ts"] {
            std::fs::write(
                dir.path()
                    .join(name),
                b"",
            )
            .unwrap();
        }
        assert_eq!(
            highest_segment_index(
                &dir.path()
                    .to_path_buf()
            ),
            Some(7)
        );
    }

    #[test]
    fn highest_segment_index_is_none_for_an_empty_or_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            highest_segment_index(
                &dir.path()
                    .to_path_buf()
            ),
            None
        );
        assert_eq!(
            highest_segment_index(&PathBuf::from("/nonexistent/remux-test-dir")),
            None
        );
    }

    #[test]
    fn segment_start_index_matches_the_seek_offset() {
        // 1255s seek, 6s segments: floor(1255/6) = 209 — same formula
        // build_hls_args uses for -start_number.
        assert_eq!(segment_start_index(Some(1255 * 10_000_000), 6), 209);
        assert_eq!(segment_start_index(None, 6), 0);
        assert_eq!(segment_start_index(Some(0), 6), 0);
    }

    #[test]
    fn buffered_secs_from_segments_is_relative_to_the_seek_start_index() {
        // Regression test: a seek to 1255s starts ffmpeg at segment index
        // ~209 (segment names are absolute, offset by the seek position —
        // see segment_start_index), not 0. Only one segment has actually
        // been produced since the seek, so this must read as one
        // segment_length of buffered output, not 210 segments' worth.
        assert_eq!(buffered_secs_from_segments(Some(209), 209, 6), 6);
        assert_eq!(buffered_secs_from_segments(Some(212), 209, 6), 24);
        // No seek: start_index is 0, matches the old file-count behavior.
        assert_eq!(buffered_secs_from_segments(Some(3), 0, 6), 24);
        assert_eq!(buffered_secs_from_segments(None, 0, 6), 0);
        // Defensive: an index somehow below start_index never underflows.
        assert_eq!(buffered_secs_from_segments(Some(100), 209, 6), 6);
    }

    #[test]
    fn delete_old_segments_prunes_m4s_segments_below_cutoff() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "segment_00000.m4s",
            "segment_00001.m4s",
            "segment_00002.m4s",
        ] {
            std::fs::write(
                dir.path()
                    .join(name),
                b"",
            )
            .unwrap();
        }
        delete_old_segments(
            &dir.path()
                .to_path_buf(),
            2,
        );
        let remaining: std::collections::HashSet<String> =
            std::fs::read_dir(dir.path())
                .unwrap()
                .flatten()
                .map(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
        assert_eq!(
            remaining,
            ["segment_00002.m4s".to_string()]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn hw_none_when_no_hwaccels_listed() {
        assert_eq!(
            select_hw_accel("", all_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_none_when_devices_absent() {
        let hwaccels =
            "Hardware acceleration methods:\ncuda\nvaapi\nqsv\nv4l2m2m\nrkmpp\n";
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_nvenc_requires_cuda_and_device() {
        let hwaccels = "Hardware acceleration methods:\ncuda\n";
        // Device present
        assert_eq!(
            select_hw_accel(hwaccels, |p| p == "/dev/nvidia0", || None),
            HardwareAccelerationType::Nvenc
        );
        // Device absent
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_vaapi_requires_vaapi_and_device() {
        let hwaccels = "Hardware acceleration methods:\nvaapi\n";
        assert_eq!(
            select_hw_accel(hwaccels, |p| p == "/dev/dri/renderD128", || None),
            HardwareAccelerationType::Vaapi
        );
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_qsv_beats_vaapi_when_both_available() {
        let hwaccels = "Hardware acceleration methods:\nvaapi\nqsv\n";
        assert_eq!(
            select_hw_accel(
                hwaccels,
                |p| p == "/dev/dri/renderD128",
                || { Some("0x8086".to_string()) }
            ),
            HardwareAccelerationType::Qsv
        );
    }

    #[test]
    fn hw_qsv_requires_dri_device() {
        let hwaccels = "Hardware acceleration methods:\nqsv\n";
        assert_eq!(
            select_hw_accel(
                hwaccels,
                |p| p == "/dev/dri/renderD128",
                || { Some("0x8086".to_string()) }
            ),
            HardwareAccelerationType::Qsv
        );
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || Some("0x8086".to_string())),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_rkmpp_requires_mpp_service_device() {
        let hwaccels = "Hardware acceleration methods:\nrkmpp\n";
        assert_eq!(
            select_hw_accel(hwaccels, |p| p == "/dev/mpp_service", || None),
            HardwareAccelerationType::Rkmpp
        );
        // This was the Asahi Linux bug: rkmpp listed but device absent → None
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_v4l2m2m_requires_video0_device() {
        let hwaccels = "Hardware acceleration methods:\nv4l2m2m\n";
        assert_eq!(
            select_hw_accel(hwaccels, |p| p == "/dev/video0", || None),
            HardwareAccelerationType::V4l2m2m
        );
        assert_eq!(
            select_hw_accel(hwaccels, no_devices, || None),
            HardwareAccelerationType::None
        );
    }

    #[test]
    fn hw_videotoolbox_only_on_macos() {
        let hwaccels = "Hardware acceleration methods:\nvideotoolbox\n";
        let expected = if cfg!(target_os = "macos") {
            HardwareAccelerationType::VideoToolbox
        } else {
            HardwareAccelerationType::None
        };
        assert_eq!(select_hw_accel(hwaccels, all_devices, || None), expected);
    }

    #[test]
    fn hw_priority_nvenc_over_vaapi() {
        // When both are available nvenc wins
        let hwaccels = "Hardware acceleration methods:\ncuda\nvaapi\n";
        assert_eq!(
            select_hw_accel(hwaccels, all_devices, || None),
            HardwareAccelerationType::Nvenc
        );
    }

    #[test]
    fn hw_falls_through_to_rkmpp_when_others_absent() {
        let hwaccels =
            "Hardware acceleration methods:\ncuda\nvaapi\nqsv\nv4l2m2m\nrkmpp\n";
        // Only /dev/mpp_service present
        assert_eq!(
            select_hw_accel(hwaccels, |p| p == "/dev/mpp_service", || None),
            HardwareAccelerationType::Rkmpp
        );
    }

    fn default_hls(output_dir: PathBuf) -> TranscodeParams {
        TranscodeParams {
            input_url: "http://localhost/test.mkv".into(),
            output_dir,
            ..Default::default()
        }
    }

    fn default_progressive() -> ProgressiveTranscodeParams {
        ProgressiveTranscodeParams {
            input_url: "http://localhost/test.mkv".into(),
            container: "mp4".into(),
            video_codec: "copy".into(),
            audio_codec: "aac".into(),
            start_time_ticks: None,
            max_width: None,
            max_height: None,
            video_bitrate: None,
            audio_bitrate: None,
            audio_channels: None,
            audio_stream_index: None,
            subtitle_stream_index: None,
            burn_subtitle: false,
            subtitle_width: None,
            subtitle_height: None,
            encoding_preset: None,
            source_video_codec: None,
            source_audio_codec: None,
            hevc_copy_tag: None,
            accelerator: Box::new(NoAccel),
            source_video_range_type: None,
            enable_tonemapping: false,
            enable_vpp_tonemapping: false,
            tonemapping_algorithm: "hable".into(),
            tonemapping_desat: 0.0,
            tonemapping_peak: 0.0,
            allow_hevc_encoding: false,
            allow_av1_encoding: false,
            h264_crf: 23,
            h265_crf: 28,
            normalize_audio_loudness: false,
        }
    }

    #[test]
    fn hls_basic_copy() {
        let dir = PathBuf::from("/tmp/test_session");
        let args = build_hls_args(&default_hls(dir.clone()));

        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert_eq!(arg_after(&args, "-c:a"), Some("aac"));
        assert_eq!(arg_after(&args, "-f"), Some("hls"));
        assert!(
            !args_contains(&args, "-allowed_extensions"),
            "ordinary file inputs must retain ffmpeg's default extension policy"
        );
        // Default TS segments — no fmp4 flag
        assert!(!args_contains(&args, "-hls_segment_type"));
        // Playlist and segment paths
        assert!(
            args.iter()
                .any(|a| a.ends_with("main.m3u8"))
        );
        assert!(
            args.iter()
                .any(|a| a.contains("segment_") && a.ends_with(".ts"))
        );
    }

    #[test]
    fn hls_proxy_input_allows_extensionless_segments() {
        let args = build_hls_args(&TranscodeParams {
            input_url:
                "https://relay.example/hls?url=https%3A%2F%2Forigin.example%2Fplaylist"
                    .into(),
            ..default_hls(PathBuf::from("/tmp/test_extensionless_hls"))
        });

        assert_eq!(arg_after(&args, "-allowed_extensions"), Some("ALL"));
        assert_eq!(arg_after(&args, "-allowed_segment_extensions"), Some("ALL"));
        assert_eq!(arg_after(&args, "-extension_picky"), Some("0"));
    }

    #[test]
    fn hls_hevc_copy_uses_fmp4() {
        let dir = PathBuf::from("/tmp/test_hevc");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            source_video_codec: Some("hevc".into()),
            ..default_hls(dir)
        });

        assert_eq!(arg_after(&args, "-hls_segment_type"), Some("fmp4"));
        assert_eq!(
            arg_after(&args, "-hls_fmp4_init_filename"),
            Some("init.mp4")
        );
        assert_eq!(arg_after(&args, "-tag:v"), Some("hvc1"));
        assert_eq!(arg_after(&args, "-avoid_negative_ts"), Some("disabled"));
        assert!(args_contains(&args, "-start_at_zero"));
        assert_eq!(
            arg_after(&args, "-hls_segment_options"),
            Some("movflags=+frag_discont")
        );
        // No DoVi range type → dovi_rpu must NOT be injected (would crash on non-HEVC/AV1)
        assert!(
            !args
                .windows(2)
                .any(|w| w[0] == "-bsf:v" && w[1].contains("dovi_rpu"))
        );
        // Segments use .m4s extension
        assert!(
            args.iter()
                .any(|a| a.contains("segment_") && a.ends_with(".m4s"))
        );
    }

    #[test]
    fn hls_hevc_dovi_copy_strips_rpu() {
        let dir = PathBuf::from("/tmp/test_hevc_dovi");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            source_video_codec: Some("hevc".into()),
            source_video_range_type: Some(VideoRangeType::DoviWithHdr10),
            ..default_hls(dir)
        });
        assert_eq!(arg_after(&args, "-tag:v"), Some("hvc1"));
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-bsf:v" && w[1].contains("dovi_rpu")),
            "dovi_rpu bsf must be present for DoVi HEVC copy"
        );
    }

    #[test]
    fn hls_aac_copy_adds_adtstoasc() {
        let dir = PathBuf::from("/tmp/test_aac_copy");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "copy".into(),
            source_audio_codec: Some("aac".into()),
            ..default_hls(dir)
        });
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-bsf:a" && w[1] == "aac_adtstoasc"),
            "aac_adtstoasc bsf must be present when copying AAC audio"
        );
    }

    #[test]
    fn hls_aac_transcode_no_adtstoasc() {
        let dir = PathBuf::from("/tmp/test_aac_transcode");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "aac".into(),
            source_audio_codec: Some("aac".into()),
            ..default_hls(dir)
        });
        assert!(
            !args
                .windows(2)
                .any(|w| w[0] == "-bsf:a"),
            "aac_adtstoasc must NOT be added when re-encoding audio"
        );
    }

    #[test]
    fn hls_non_aac_copy_no_adtstoasc() {
        let dir = PathBuf::from("/tmp/test_ac3_copy");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "copy".into(),
            source_audio_codec: Some("ac3".into()),
            ..default_hls(dir)
        });
        assert!(
            !args
                .windows(2)
                .any(|w| w[0] == "-bsf:a"),
            "aac_adtstoasc must NOT be added for non-AAC audio"
        );
    }

    #[test]
    fn hls_hevc_copy_hvc1_tag_alias() {
        // "hvc1" codec string should also trigger fMP4 path
        let dir = PathBuf::from("/tmp/test_hvc1");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            source_video_codec: Some("hvc1".into()),
            ..default_hls(dir)
        });
        assert_eq!(arg_after(&args, "-hls_segment_type"), Some("fmp4"));
        assert_eq!(arg_after(&args, "-tag:v"), Some("hvc1"));
    }

    #[test]
    fn hls_libx264_transcode_flags() {
        let dir = PathBuf::from("/tmp/test_x264");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            ..default_hls(dir)
        });

        assert_eq!(arg_after(&args, "-c:v"), Some("libx264"));
        assert_eq!(arg_after(&args, "-crf"), Some("23"));
        assert_eq!(arg_after(&args, "-preset"), Some("ultrafast"));
        assert_eq!(arg_after(&args, "-profile:v"), Some("high"));
        assert_eq!(arg_after(&args, "-tune"), Some("zerolatency"));
        assert_eq!(arg_after(&args, "-pix_fmt"), Some("yuv420p"));
    }

    #[test]
    fn hls_libx264_custom_preset_and_bitrate() {
        let dir = PathBuf::from("/tmp/test_x264_br");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            encoding_preset: Some(EncodingPreset::Veryfast),
            video_bitrate: Some(4_000_000),
            ..default_hls(dir)
        });

        assert_eq!(arg_after(&args, "-preset"), Some("veryfast"));
        assert_eq!(arg_after(&args, "-maxrate"), Some("4000000"));
        assert_eq!(arg_after(&args, "-bufsize"), Some("8000000"));
    }

    #[test]
    fn hls_seek_offset_placed_before_input() {
        let dir = PathBuf::from("/tmp/test_seek");
        let ticks: i64 = 30i64
            .to_ticks(TickUnit::Seconds)
            .unwrap();
        let args = build_hls_args(&TranscodeParams {
            start_time_ticks: Some(ticks),
            ..default_hls(dir)
        });

        let ss_pos = args
            .iter()
            .position(|a| a == "-ss")
            .expect("-ss missing");
        let i_pos = args
            .iter()
            .position(|a| a == "-i")
            .expect("-i missing");
        assert!(ss_pos < i_pos, "-ss must come before -i");
        assert_eq!(args[ss_pos + 1], "30.000000");

        // start_number = floor(30 / 6) = 5
        assert_eq!(arg_after(&args, "-start_number"), Some("5"));
    }

    #[test]
    fn hls_no_seek_start_number_zero() {
        let dir = PathBuf::from("/tmp/test_noseek");
        let args = build_hls_args(&default_hls(dir));
        assert_eq!(arg_after(&args, "-start_number"), Some("0"));
        assert!(!args_contains(&args, "-ss"));
    }

    #[test]
    fn live_hls_writes_a_sliding_window_playlist() {
        let dir = tempfile::tempdir().unwrap();
        let params = TranscodeParams {
            is_live: true,
            start_time_ticks: Some(299 * 10_000_000),
            ..default_hls(
                dir.path()
                    .to_path_buf(),
            )
        };
        let args = build_hls_args(&params);
        // 300 s window of 6 s segments.
        assert_eq!(arg_after(&args, "-hls_list_size"), Some("50"));
        // Fresh session: no leading discontinuity from append_list.
        assert_eq!(arg_after(&args, "-hls_flags"), Some("omit_endlist"));
        assert!(!args_contains(&args, "-hls_playlist_type"));
        assert!(!args_contains(&args, "-ss"));
        assert_eq!(arg_after(&args, "-start_number"), Some("49"));
    }

    #[test]
    fn live_hls_restart_appends_to_the_existing_playlist() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path()
                .join("main.m3u8"),
            "#EXTM3U\n",
        )
        .unwrap();
        let params = TranscodeParams {
            is_live: true,
            start_time_ticks: Some(299 * 10_000_000),
            ..default_hls(
                dir.path()
                    .to_path_buf(),
            )
        };
        let args = build_hls_args(&params);
        assert_eq!(
            arg_after(&args, "-hls_flags"),
            Some("append_list+omit_endlist")
        );
        // Restarts reuse these params, so -start_number never exceeds the
        // playlist's MEDIA-SEQUENCE and append_list continues the numbering.
        assert_eq!(arg_after(&args, "-start_number"), Some("49"));
    }

    #[test]
    fn vod_hls_keeps_the_full_event_playlist() {
        let args = build_hls_args(&default_hls(PathBuf::from("/tmp/test_vod")));
        assert_eq!(arg_after(&args, "-hls_playlist_type"), Some("event"));
        assert_eq!(arg_after(&args, "-hls_list_size"), Some("0"));
        assert!(!args_contains(&args, "-hls_flags"));
    }

    #[test]
    fn live_window_covers_five_minutes() {
        assert_eq!(live_window_segments(6), 50);
        assert_eq!(live_window_segments(4), 75);
        assert_eq!(live_window_segments(1), 300);
    }

    #[test]
    fn live_cleanup_cutoff_trails_the_playlist_window() {
        // 50 listed segments plus 5 (30 s) for older playlist copies.
        assert_eq!(live_cleanup_cutoff(110, 6), 55);
        // Early in the session nothing is old enough.
        assert_eq!(live_cleanup_cutoff(54, 6), 0);
        assert_eq!(live_cleanup_cutoff(0, 6), 0);
    }

    #[test]
    fn live_cleanup_bounds_disk_use() {
        // A paused client with ffmpeg far ahead: only the window plus margin
        // stays on disk.
        let dir = tempfile::tempdir().unwrap();
        for idx in 0..120 {
            std::fs::write(
                dir.path()
                    .join(format!("segment_{idx:05}.ts")),
                b"x",
            )
            .unwrap();
        }
        let output_dir = dir
            .path()
            .to_path_buf();
        let highest = highest_segment_index(&output_dir).unwrap();
        delete_old_segments(&output_dir, live_cleanup_cutoff(highest, 6));
        let mut left: Vec<u32> = std::fs::read_dir(&output_dir)
            .unwrap()
            .filter_map(|e| {
                segment_index(
                    &e.unwrap()
                        .file_name()
                        .to_string_lossy(),
                )
            })
            .collect();
        left.sort_unstable();
        assert_eq!(left, (64..120).collect::<Vec<_>>());
    }

    #[test]
    fn vod_monitor_throttles_on_reported_position() {
        // Segment 49 on disk = 300 s buffered, nothing played: pause.
        assert_eq!(
            monitor_step(Some(49), 0, 6, 0, false, false).signal,
            Some(ThrottleSignal::Pause { ahead: 300 })
        );
        // Client caught up to 60 s: resume.
        assert_eq!(
            monitor_step(Some(49), 0, 6, 60, true, false).signal,
            Some(ThrottleSignal::Resume { ahead: 240 })
        );
        // Cleanup keeps 30 s behind the reported position.
        assert_eq!(
            monitor_step(Some(49), 0, 6, 60, false, true).delete_below,
            Some(5)
        );
    }

    #[test]
    fn a_run_faster_than_real_time_outran_the_clock() {
        let secs = Duration::from_secs;
        // A 330 s file copied in 2 s.
        assert!(outran_the_clock(0, Some(54), 6, secs(2)));
        // Live: real time plus the HLS demuxer's start 3 segments back.
        assert!(!outran_the_clock(0, Some(20), 6, secs(100)));
        assert!(!outran_the_clock(0, Some(2), 6, secs(1)));
        // A live server's 90 s backlog on connect, also after 10 minutes.
        assert!(!outran_the_clock(0, Some(14), 6, secs(1)));
        assert!(!outran_the_clock(0, Some(114), 6, secs(600)));
        // A restarted run that wrote nothing.
        assert!(!outran_the_clock(30, None, 6, secs(1)));
        assert!(!outran_the_clock(30, Some(29), 6, secs(1)));
        // A restarted run counts from its own first segment.
        assert!(!outran_the_clock(30, Some(64), 6, secs(150)));
        assert!(outran_the_clock(30, Some(64), 6, secs(10)));
    }

    #[test]
    fn live_run_latches_a_source_read_faster_than_real_time() {
        let run = LiveRun::default();
        run.start(10);
        // 18 s written just after the start: a live burst.
        assert!(!run.faster_than_realtime(Some(12), 6));
        // 330 s written at once.
        assert!(run.faster_than_realtime(Some(64), 6));
        // A restarted run reads the same source.
        run.start(65);
        assert!(run.faster_than_realtime(None, 6));
    }

    #[test]
    fn live_throttle_only_pauses_a_source_read_faster_than_real_time() {
        use super::ThrottleSignal::*;
        // (highest, requested, segment_length, paused, faster_than_realtime)
        // Real live with a paused client (or one reporting position 0): never
        // paused, however far ahead (#146).
        assert_eq!(live_throttle(Some(100), 20, 6, false, false), None);
        // A file: paused 150 s ahead of the newest requested segment.
        assert_eq!(
            live_throttle(Some(45), 20, 6, false, true),
            Some(Pause { ahead: 150 })
        );
        assert_eq!(live_throttle(Some(44), 20, 6, false, true), None);
        // Resumed once the client caught up by two segments.
        assert_eq!(live_throttle(Some(45), 22, 6, true, true), None);
        assert_eq!(
            live_throttle(Some(45), 23, 6, true, true),
            Some(Resume { ahead: 132 })
        );
        // A client ahead of ffmpeg, or no segments yet.
        assert_eq!(live_throttle(Some(45), 50, 6, false, true), None);
        assert_eq!(live_throttle(None, 0, 6, false, true), None);
        // Epoch-based start numbers don't overflow.
        assert_eq!(
            live_throttle(Some(u32::MAX), 0, 6, false, true),
            Some(Pause { ahead: u32::MAX })
        );
    }

    #[test]
    fn live_run_end_only_ends_a_source_that_ended() {
        use super::LiveRunEnd::*;
        // (stopped, idle_stopped, killed, source_ended)
        assert_eq!(live_run_end(true, false, true, false), Stopped);
        assert_eq!(live_run_end(true, false, false, true), Stopped);
        assert_eq!(live_run_end(false, false, false, true), Ended);
        // Any other exit restarts: upstream drops exit 0 too.
        assert_eq!(live_run_end(false, false, false, false), Restart);
        // Wait for the client, then restart.
        assert_eq!(live_run_end(false, true, true, false), Idle);
        assert_eq!(live_run_end(false, true, false, false), Idle);
        // The source ended while the idle stop raced its exit.
        assert_eq!(live_run_end(false, true, false, true), Ended);
        assert_eq!(live_run_end(false, false, true, false), Stopped);
    }

    #[test]
    fn stderr_input_errors_tell_drops_from_ends() {
        // Upstream drops (round-3 logs, ffmpeg 8 at `-v error`).
        for stderr in [
            "[http @ 0x55d0c8a1b2c0] Stream ends prematurely at 4705076, should be 18446744073709551615\n",
            "[in#0/mpegts @ 0x55d0c8a1b2c0] Error during demuxing: Input/output error\n",
            "[in#0/matroska,webm @ 0x5612] Error during demuxing: Connection reset by peer\n",
            "[tls @ 0x5612] Error in the pull function.\n",
            "[tcp @ 0x5612] Connection to tcp://iptv.example:80 failed: Connection refused\n",
            "[hls @ 0x5612] Error when loading first segment 'seg9.ts'\n",
        ] {
            assert!(stderr_reports_input_error(stderr), "{stderr}");
        }
        // A file read to its end logs nothing; decoder errors don't end it.
        for stderr in [
            "",
            "[h264 @ 0x5612] error while decoding MB 12 40, bytestream -7\n",
            "[aac @ 0x5612] channel element 0.0 is not allocated\n",
        ] {
            assert!(!stderr_reports_input_error(stderr), "{stderr}");
        }
    }

    #[test]
    fn declares_length_reads_range_responses() {
        // (status, Content-Range, Content-Length)
        assert!(declares_length(206, Some("bytes 0-0/345627810"), Some("1")));
        assert!(declares_length(200, None, Some("345627810")));
        // Live: unknown total, no length (chunked or close-delimited).
        assert!(!declares_length(206, Some("bytes 0-0/*"), Some("1")));
        assert!(!declares_length(200, None, None));
        assert!(!declares_length(302, None, Some("0")));
    }

    #[test]
    fn hls_content_types_are_recognised() {
        assert!(is_hls_content_type("application/vnd.apple.mpegurl"));
        assert!(is_hls_content_type("Application/X-MpegURL; charset=utf-8"));
        assert!(is_hls_content_type("audio/mpegurl"));
        assert!(!is_hls_content_type("video/mp2t"));
    }

    #[test]
    fn hls_playlist_end_reads_vod_live_and_master_playlists() {
        assert_eq!(
            hls_playlist_end(
                "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:VOD\n#EXTINF:6,\nseg0.ts\n#EXT-X-ENDLIST\n"
            ),
            HlsPlaylistEnd::Ends
        );
        assert_eq!(
            hls_playlist_end("#EXTM3U\n#EXTINF:6,\nseg0.ts\n#EXT-X-ENDLIST\n"),
            HlsPlaylistEnd::Ends
        );
        // Live, and EVENT without ENDLIST.
        assert_eq!(
            hls_playlist_end("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:9\n#EXTINF:6,\nseg9.ts\n"),
            HlsPlaylistEnd::Open
        );
        assert_eq!(
            hls_playlist_end(
                "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:EVENT\n#EXTINF:6,\nseg0.ts\n"
            ),
            HlsPlaylistEnd::Open
        );
        assert_eq!(
            hls_playlist_end(
                "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=800000\r\nlow/index.m3u8\r\n\
                 #EXT-X-STREAM-INF:BANDWIDTH=2000000\nhigh/index.m3u8\n"
            ),
            HlsPlaylistEnd::Variant("low/index.m3u8".into())
        );
    }

    #[tokio::test]
    async fn local_inputs_end_and_streaming_protocols_dont() {
        assert!(live_input_ends("/media/clips/promo.ts").await);
        assert!(live_input_ends("file:///media/clips/promo.ts").await);
        assert!(live_input_ends(r"C:\media\promo.ts").await);
        assert!(!live_input_ends("rtsp://camera.example/stream").await);
        assert!(!live_input_ends("udp://239.0.0.1:1234").await);
    }

    #[test]
    fn length_probe_retries_connection_limits_and_transport_errors() {
        // Transport error; connection limit; outage.
        for status in [None, Some(403), Some(429), Some(503), Some(509)] {
            assert!(length_probe_retries(status), "{status:?}");
        }
        // Answers about the resource itself.
        for status in [Some(200), Some(206), Some(404), Some(416), Some(500)] {
            assert!(!length_probe_retries(status), "{status:?}");
        }
        // All tries fit the budget, without counting request time.
        assert!(
            LENGTH_PROBE_BACKOFF * (LENGTH_PROBE_ATTEMPTS - 1) < LENGTH_PROBE_BUDGET
        );
    }

    #[tokio::test]
    async fn a_clean_exit_with_input_errors_or_a_failed_status_is_no_end() {
        let clean = std::process::ExitStatus::default();
        let local = "/media/clips/promo.ts";
        assert!(live_source_ended(Some(&clean), "", local).await);
        assert!(
            !live_source_ended(
                Some(&clean),
                "[in#0/mpegts @ 0x1] Error during demuxing: Input/output error\n",
                local,
            )
            .await
        );
        // Killed, or status != 0.
        assert!(!live_source_ended(None, "", local).await);
        // Real live over RTSP that closed cleanly.
        assert!(!live_source_ended(Some(&clean), "", "rtsp://camera.example/s").await);
    }

    #[test]
    fn end_live_playlist_appends_endlist_once() {
        let dir = tempfile::tempdir().unwrap();
        let playlist = dir
            .path()
            .join("main.m3u8");
        std::fs::write(
            &playlist,
            "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:5\n#EXTINF:6.0,\nsegment_00005.ts",
        )
        .unwrap();
        end_live_playlist(&playlist).unwrap();
        end_live_playlist(&playlist).unwrap();
        assert_eq!(
            std::fs::read_to_string(&playlist).unwrap(),
            "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:5\n#EXTINF:6.0,\nsegment_00005.ts\n#EXT-X-ENDLIST\n"
        );
    }

    fn hevc_session(video_codec: &str, tag: Option<&str>) -> TranscodeSession {
        TranscodeSession {
            id: "play-session".into(),
            user_id: Uuid::nil(),
            item_id: Uuid::nil(),
            media_source_id: Uuid::nil(),
            output_dir: PathBuf::from("/tmp/test_master"),
            input_url: "http://example.invalid/video".into(),
            state: TranscodeState::Running,
            state_tx: Arc::new(tokio::sync::watch::channel(TranscodeState::Running).0),
            created_at: std::time::Instant::now(),
            video_codec: video_codec.into(),
            audio_codec: "aac".into(),
            audio_stream_index: None,
            subtitle_stream_index: None,
            burn_subtitle: false,
            segment_length: 6,
            transcode_reasons: TranscodeReasons::default(),
            kill_tx: None,
            wait_done: Arc::new(tokio::sync::Notify::new()),
            stopped: Default::default(),
            requests: Default::default(),
            live_activity: Default::default(),
            last_segment_index: Arc::new(AtomicU32::new(0)),
            start_time_secs: 0,
            playback_offset_secs: Arc::new(AtomicU32::new(0)),
            runtime_ticks: 120i64
                .to_ticks(TickUnit::Seconds)
                .unwrap(),
            is_live: false,
            source_video_codec: Some("hevc".into()),
            source_audio_codec: Some("aac".into()),
            hevc_copy_tag: tag.map(str::to_string),
            source_video_profile: Some("Main 10".into()),
            source_video_level: Some(120.0),
            source_video_range_type: None,
            source_video_width: None,
            source_video_height: None,
            source_frame_rate: None,
            video_bitrate: None,
            hardware_acceleration_type: None,
        }
    }

    #[test]
    fn hls_hevc_copy_uses_the_tag_the_client_asked_for() {
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            source_video_codec: Some("hevc".into()),
            hevc_copy_tag: Some("hev1".into()),
            ..default_hls(PathBuf::from("/tmp/test_hev1"))
        });
        assert_eq!(arg_after(&args, "-tag:v"), Some("hev1"));
    }

    #[test]
    fn hls_hevc_copy_defaults_to_hvc1_when_client_said_nothing() {
        // Unknown -> today's behaviour, so nothing that plays now regresses.
        for tag in [None, Some("hvc1")] {
            let args = build_hls_args(&TranscodeParams {
                video_codec: "copy".into(),
                source_video_codec: Some("hevc".into()),
                hevc_copy_tag: tag.map(str::to_string),
                ..default_hls(PathBuf::from("/tmp/test_hvc1_default"))
            });
            assert_eq!(arg_after(&args, "-tag:v"), Some("hvc1"), "tag={tag:?}");
        }
    }

    #[test]
    fn progressive_hevc_copy_uses_the_tag_the_client_asked_for() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            source_video_codec: Some("hevc".into()),
            hevc_copy_tag: Some("hev1".into()),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-tag:v"), Some("hev1"));
    }

    #[test]
    fn master_playlist_codecs_matches_the_sample_entry_written_on_copy() {
        // CODECS has to name the fourcc build_hls_args wrote, or Safari
        // validates the manifest against the segments and refuses.
        let playlist = generate_master_playlist(&hevc_session("copy", Some("hev1")));
        assert!(
            playlist.contains("hev1.2.4.L120.B0"),
            "expected hev1 CODECS: {playlist}"
        );
        assert!(!playlist.contains("hvc1."), "playlist: {playlist}");
    }

    #[test]
    fn master_playlist_codecs_stays_hvc1_when_client_said_nothing() {
        for tag in [None, Some("hvc1")] {
            let playlist = generate_master_playlist(&hevc_session("copy", tag));
            assert!(
                playlist.contains("hvc1.2.4.L120.B0"),
                "expected hvc1 CODECS for {tag:?}: {playlist}"
            );
        }
    }

    #[test]
    fn hls_audio_copy_follows_the_segment_container() {
        // (source video, source audio, expected -c:a, expected fMP4)
        let cases = [
            ("hevc", "flac", "copy", true),
            ("HEVC", "flac", "copy", true),
            ("H.265", "flac", "copy", true),
            ("hevc", "truehd", "aac", true),
            ("hevc", "pcm_s16le", "aac", true),
            ("h264", "flac", "aac", false),
            ("h264", "truehd", "aac", false),
            ("h264", "ac3", "copy", false),
        ];
        for (video, audio, expected_audio, fmp4) in cases {
            let args = build_hls_args(&TranscodeParams {
                video_codec: "copy".into(),
                audio_codec: "copy".into(),
                source_video_codec: Some(video.into()),
                source_audio_codec: Some(audio.into()),
                ..default_hls(PathBuf::from("/tmp/test_audio_copy"))
            });
            assert_eq!(
                arg_after(&args, "-c:a"),
                Some(expected_audio),
                "{video} + {audio}"
            );
            assert_eq!(
                arg_after(&args, "-hls_segment_type") == Some("fmp4"),
                fmp4,
                "{video} + {audio}"
            );

            let mut session = hevc_session("copy", None);
            session.source_video_codec = Some(video.into());
            session.audio_codec = "copy".into();
            session.source_audio_codec = Some(audio.into());
            assert_eq!(session.use_fmp4(), fmp4, "{video} + {audio}");
        }
    }

    #[test]
    fn progressive_format_promotes_copied_mp4_to_matroska() {
        assert_eq!(
            ProgressiveFormat::for_request("mp4", "copy"),
            ProgressiveFormat::Matroska
        );
        assert_eq!(
            ProgressiveFormat::for_request("mp4", "libx264"),
            ProgressiveFormat::Mp4
        );
        assert_eq!(
            ProgressiveFormat::for_request("ts", "copy"),
            ProgressiveFormat::Mpegts
        );
    }

    #[test]
    fn progressive_audio_copy_follows_the_container() {
        use ProgressiveFormat::*;
        // (container, source audio, can copy)
        let cases = [
            (Matroska, "flac", true),
            (Matroska, "truehd", true),
            (Mp4, "flac", true),
            (Mp4, "truehd", false),
            (Mpegts, "flac", false),
            (Mpegts, "ac3", true),
            (Webm, "opus", true),
            (Webm, "aac", false),
            (Webm, "flac", false),
        ];
        for (format, audio, expected) in cases {
            assert_eq!(
                format.can_copy_audio(Some(audio)),
                expected,
                "{format:?} + {audio}"
            );
        }
        assert_eq!(Webm.fallback_audio_codec(), "opus");
        assert_eq!(Mpegts.fallback_audio_codec(), "aac");
    }

    #[test]
    fn master_playlist_names_copied_flac_and_transcoded_truehd() {
        let mut session = hevc_session("copy", None);
        session.audio_codec = "copy".into();
        session.source_audio_codec = Some("flac".into());
        let playlist = generate_master_playlist(&session);
        assert!(playlist.contains(",fLaC\""), "playlist: {playlist}");

        session.source_audio_codec = Some("truehd".into());
        let playlist = generate_master_playlist(&session);
        assert!(playlist.contains(",mp4a.40.2\""), "playlist: {playlist}");
    }

    #[test]
    fn master_playlist_codecs_stays_hvc1_when_encoding_hevc() {
        // The encode path emits MPEG-TS, which has no sample entry at all, so
        // the source-side tag must not leak into CODECS.
        let playlist = generate_master_playlist(&hevc_session("hevc", Some("hev1")));
        assert!(
            playlist.contains("hvc1.2.4.L120.B0"),
            "expected hvc1 CODECS: {playlist}"
        );
        assert!(!playlist.contains("hev1."), "playlist: {playlist}");
    }

    #[test]
    fn resumed_vod_playlist_advertises_start_offset_and_full_seek_map() {
        let session = TranscodeSession {
            id: "play-session".into(),
            user_id: Uuid::nil(),
            item_id: Uuid::nil(),
            media_source_id: Uuid::nil(),
            output_dir: PathBuf::from("/tmp/test_playlist"),
            input_url: "http://example.invalid/video".into(),
            state: TranscodeState::Running,
            state_tx: Arc::new(tokio::sync::watch::channel(TranscodeState::Running).0),
            created_at: std::time::Instant::now(),
            video_codec: "copy".into(),
            audio_codec: "aac".into(),
            audio_stream_index: None,
            subtitle_stream_index: None,
            burn_subtitle: false,
            segment_length: 6,
            transcode_reasons: TranscodeReasons::default(),
            kill_tx: None,
            wait_done: Arc::new(tokio::sync::Notify::new()),
            stopped: Default::default(),
            requests: Default::default(),
            live_activity: Default::default(),
            last_segment_index: Arc::new(AtomicU32::new(0)),
            start_time_secs: 30,
            playback_offset_secs: Arc::new(AtomicU32::new(0)),
            runtime_ticks: 120i64
                .to_ticks(TickUnit::Seconds)
                .unwrap(),
            is_live: false,
            source_video_codec: Some("h264".into()),
            source_audio_codec: Some("aac".into()),
            hevc_copy_tag: None,
            source_video_profile: None,
            source_video_level: None,
            source_video_range_type: None,
            source_video_width: None,
            source_video_height: None,
            source_frame_rate: None,
            video_bitrate: None,
            hardware_acceleration_type: None,
        };

        let playlist = generate_variant_playlist(&session, "");

        assert!(playlist.contains("#EXT-X-START:TIME-OFFSET=30.000000,PRECISE=YES"));
        assert!(playlist.contains("segment_00000.ts?PlaySessionId=play-session&runtimeTicks=0&actualSegmentLengthTicks=60000000"));
        assert!(playlist.contains("segment_00005.ts?PlaySessionId=play-session&runtimeTicks=300000000&actualSegmentLengthTicks=60000000"));
    }

    #[test]
    fn hls_scale_filter_both_dimensions() {
        let dir = PathBuf::from("/tmp/test_scale");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            max_width: Some(1920),
            max_height: Some(1080),
            ..default_hls(dir)
        });

        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("min(1920,iw)"), "vf: {vf}");
        assert!(vf.contains("min(1080,ih)"), "vf: {vf}");
        assert!(
            vf.contains("force_original_aspect_ratio=decrease"),
            "vf: {vf}"
        );
    }

    #[test]
    fn hls_scale_filter_width_only() {
        let dir = PathBuf::from("/tmp/test_scale_w");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            max_width: Some(1280),
            ..default_hls(dir)
        });
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("min(1280,iw)"), "vf: {vf}");
        assert!(vf.contains(":-2"), "vf: {vf}");
    }

    #[test]
    fn hls_audio_bitrate_and_channels() {
        let dir = PathBuf::from("/tmp/test_audio");
        let args = build_hls_args(&TranscodeParams {
            audio_bitrate: Some(192_000),
            audio_channels: Some(2),
            ..default_hls(dir)
        });

        assert_eq!(arg_after(&args, "-b:a"), Some("192000"));
        assert_eq!(arg_after(&args, "-ac"), Some("2"));
    }

    #[test]
    fn hls_audio_copy_no_bitrate_flags() {
        let dir = PathBuf::from("/tmp/test_acopy");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "copy".into(),
            ..default_hls(dir)
        });
        assert_eq!(arg_after(&args, "-c:a"), Some("copy"));
        assert!(
            !args_contains(&args, "-b:a"),
            "must not set -b:a when copying audio"
        );
        assert!(
            !args_contains(&args, "-ac"),
            "must not set -ac when copying audio"
        );
    }

    #[test]
    fn hls_subtitle_burn_uses_selected_encoder_and_filter_complex() {
        let dir = PathBuf::from("/tmp/test_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_eq!(arg_after(&args, "-c:v"), Some("libx264"));
        // filter_complex with overlay
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(fc.contains("overlay"), "filter_complex: {fc}");
        assert!(
            fc.contains("[0:2]"),
            "filter_complex should reference sub stream: {fc}"
        );
        // map [v] output label
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-map" && w[1] == "[v]")
        );
    }

    #[test]
    fn hls_subtitle_burn_cannot_override_video_copy() {
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(PathBuf::from("/tmp/test_sub_copy"))
        });

        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert!(!args_contains(&args, "-filter_complex"));
    }

    #[test]
    fn hls_subtitle_burn_with_scale_in_filter_complex() {
        let dir = PathBuf::from("/tmp/test_sub_scale");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(3),
            max_width: Some(1280),
            max_height: Some(720),
            ..default_hls(dir)
        });

        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains("scale=1280:720:fast_bilinear"),
            "sub scale: {fc}"
        );
        assert!(fc.contains("min(1280,iw)"), "video scale: {fc}");
        assert!(fc.contains("overlay"), "overlay: {fc}");
    }

    /// Exactly one 8-bit output constraint, after the input and before the muxer.
    fn assert_h264_nvenc_yuv420p(args: &[String]) {
        assert_eq!(arg_after(args, "-c:v"), Some("h264_nvenc"));
        assert_eq!(
            args.iter()
                .filter(|a| *a == "-pix_fmt")
                .count(),
            1,
            "expected exactly one -pix_fmt: {args:?}"
        );
        assert_eq!(arg_after(args, "-pix_fmt"), Some("yuv420p"));
        let i_pos = args
            .iter()
            .position(|a| a == "-i")
            .expect("-i missing");
        let pix_pos = args
            .iter()
            .position(|a| a == "-pix_fmt")
            .expect("-pix_fmt missing");
        let out_pos = args
            .iter()
            .position(|a| a == "-f")
            .expect("-f missing");
        assert!(
            i_pos + 1 < pix_pos && pix_pos < out_pos,
            "-pix_fmt must follow the input and precede the output: {args:?}"
        );
    }

    #[test]
    fn hls_nvenc_hardware_accel() {
        let dir = PathBuf::from("/tmp/test_nvenc");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            enable_tonemapping: false,
            ..default_hls(dir)
        });

        // Input flag before -i
        let hwaccel_pos = args
            .iter()
            .position(|a| a == "-hwaccel")
            .expect("-hwaccel missing");
        let i_pos = args
            .iter()
            .position(|a| a == "-i")
            .expect("-i missing");
        assert!(hwaccel_pos < i_pos);
        assert_eq!(args[hwaccel_pos + 1], "cuda");
        // Unknown range, no scale: the encoder option is the only format constraint.
        assert_h264_nvenc_yuv420p(&args);
        assert!(!args_contains(&args, "-vf"));
        assert!(!args_contains(&args, "-filter_complex"));
    }

    #[test]
    fn hls_nvenc_hdr10_without_tonemap_pins_yuv420p() {
        for bitrate in [None, Some(5_000_000)] {
            let args = build_hls_args(&TranscodeParams {
                video_codec: "libx264".into(),
                accelerator: Box::new(hw_accel::Nvenc),
                source_video_range_type: Some(VideoRangeType::Hdr10),
                enable_tonemapping: false,
                video_bitrate: bitrate,
                ..default_hls(PathBuf::from("/tmp/test_nvenc_hdr_clamp"))
            });
            assert_h264_nvenc_yuv420p(&args);
            assert_eq!(
                arg_after(&args, "-vf"),
                Some(
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
                )
            );
            let expected = bitrate.map(|rate| rate.to_string());
            assert_eq!(arg_after(&args, "-b:v"), expected.as_deref());
        }
    }

    #[test]
    fn hls_nvenc_sw_tonemap_and_subtitle_burn_pin_yuv420p() {
        let tonemap = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            ..default_hls(PathBuf::from("/tmp/test_nvenc_tonemap"))
        });
        assert_h264_nvenc_yuv420p(&tonemap);
        assert_eq!(
            arg_after(&tonemap, "-vf"),
            Some(
                "tonemapx=tonemap=hable:desat=0.0:peak=0.0:t=bt709:m=bt709:p=bt709:format=nv12"
            )
        );

        let burned = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(PathBuf::from("/tmp/test_nvenc_tonemap_sub"))
        });
        assert_h264_nvenc_yuv420p(&burned);
        let fc =
            arg_after(&burned, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains(
                "tonemapx=tonemap=hable:desat=0.0:peak=0.0:t=bt709:m=bt709:p=bt709:format=yuv420p"
            ),
            "{fc}"
        );
        assert!(fc.contains("]overlay="), "{fc}");
    }

    #[test]
    fn hls_nvenc_stream_copy_omits_pix_fmt() {
        let args = build_hls_args(&TranscodeParams {
            video_codec: "copy".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            video_bitrate: Some(4_000_000),
            ..default_hls(PathBuf::from("/tmp/test_nvenc_copy"))
        });
        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    #[test]
    fn videotoolbox_decode_decision_uses_av1_hardware_capability() {
        let vt_no_av1 = hw_accel::VideoToolbox {
            av1_hw_decode: false,
        };
        let vt_av1 = hw_accel::VideoToolbox {
            av1_hw_decode: true,
        };

        // AV1 without hw decode → sw decode
        assert!(vt_no_av1.requires_software_decode(Some("av1"), false));
        // AV1 with hw decode → hw decode ok
        assert!(!vt_av1.requires_software_decode(Some("av1"), false));
        // HDR always forces sw decode regardless of av1 hw cap
        assert!(vt_av1.requires_software_decode(Some("h264"), true));
        // Non-HDR h264 with no av1 cap → hw decode ok
        assert!(!vt_no_av1.requires_software_decode(Some("h264"), false));
    }

    #[test]
    fn hls_videotoolbox_software_decode_keeps_hardware_encode() {
        let dir = PathBuf::from("/tmp/test_videotoolbox_hdr");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "h264".into(),
            source_video_codec: Some("hevc".into()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            accelerator: Box::new(hw_accel::VideoToolbox {
                av1_hw_decode: false,
            }),
            ..default_hls(dir)
        });

        assert!(
            !args
                .iter()
                .any(|arg| arg == "-hwaccel"),
            "HDR input must stay in system memory for CPU tone mapping: {args:?}"
        );
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_videotoolbox"));
    }

    #[test]
    fn hls_vaapi_hardware_accel() {
        let dir = PathBuf::from("/tmp/test_vaapi");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Vaapi {
                device: "/dev/dri/renderD128".into(),
                driver: "iHD".into(),
                opencl: false,
            }),
            ..default_hls(dir)
        });

        // init_hw_device with driver= instead of legacy -vaapi_device
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-init_hw_device"
                    && w[1].contains("vaapi=va:")
                    && w[1].contains("/dev/dri/renderD128")
                    && w[1].contains("driver=iHD")),
            "expected init_hw_device vaapi with iHD driver, got: {args:?}"
        );
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-filter_hw_device" && w[1] == "va"),
            "expected -filter_hw_device va"
        );
        // legacy -vaapi_device must NOT appear
        assert!(
            !args
                .windows(2)
                .any(|w| w[0] == "-vaapi_device")
        );
        // Encoder remapped
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_vaapi"));
    }

    #[test]
    fn hls_vaapi_vpp_tonemap_with_subtitle_falls_back_to_sw_tonemap() {
        // VAAPI has no GPU-resident overlay filter (unlike QSV's overlay_qsv):
        // the CPU `overlay` filter that composites the subtitle outputs plain
        // yuv420p frames, which tonemap_vaapi can't consume (issue #378).
        // Burning a subtitle in on VAAPI must still fall back to software
        // tone mapping instead of silently clamping HDR to SDR metadata with
        // no actual tone map.
        let dir = PathBuf::from("/tmp/test_vaapi_vpp_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Vaapi {
                device: "/dev/dri/renderD128".into(),
                driver: "iHD".into(),
                opencl: false,
            }),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains("tonemapx"),
            "must fall back to SW tonemapx when VPP+subtitle on VAAPI: {fc}"
        );
        assert!(
            !fc.contains("tonemap_vaapi"),
            "tonemap_vaapi is incompatible with the CPU overlay's yuv420p output: {fc}"
        );
        assert!(
            fc.contains("overlay"),
            "must still burn the subtitle in: {fc}"
        );
    }

    /// `HdrTreatment::for_source` only routes *tone-mapping* treatments away
    /// from GPU residency for a CPU-overlay burn-in (`SwTonemap`) — an SDR
    /// source burning a subtitle in gets plain `Sdr` treatment, which has
    /// nothing to do with tone mapping. Regression test: `Vaapi::input_args`
    /// requesting real hardware decode must not leak into this path, since
    /// the CPU `overlay` filter that composites the subtitle reads `0:v:0`
    /// directly with no `hwdownload` and cannot consume VAAPI-resident frames.
    #[test]
    fn hls_vaapi_sdr_subtitle_burn_stays_on_cpu() {
        let dir = PathBuf::from("/tmp/test_vaapi_sdr_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi()),
            source_video_range_type: Some(VideoRangeType::Sdr),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_vaapi_sw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains("]overlay="),
            "CPU overlay compositing the subtitle: {fc}"
        );
    }

    /// Same bug, HDR variant: no tone-mapping toggle enabled means `Clamp`
    /// treatment, which — like `Sdr` — is not `SwTonemap` and so was still
    /// requesting VAAPI-resident hardware decode feeding the CPU overlay.
    #[test]
    fn hls_vaapi_hdr_clamp_subtitle_burn_stays_on_cpu() {
        let dir = PathBuf::from("/tmp/test_vaapi_clamp_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_vaapi_sw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains("]overlay="),
            "CPU overlay compositing the subtitle: {fc}"
        );
    }

    #[test]
    fn hls_codec_list_defaults_to_h264() {
        let dir = PathBuf::from("/tmp/test_codec_list");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "av1,hevc,vp9,h264".into(),
            ..default_hls(dir)
        });
        assert_eq!(arg_after(&args, "-c:v"), Some("libx264"));
    }

    #[test]
    fn hls_vaapi_no_driver_omits_driver_option() {
        let dir = PathBuf::from("/tmp/test_vaapi_amd");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Vaapi {
                device: "/dev/dri/renderD128".into(),
                driver: String::new(), // AMD / unknown: no driver=
                opencl: false,
            }),
            ..default_hls(dir)
        });

        let init_dev = args
            .windows(2)
            .find(|w| w[0] == "-init_hw_device")
            .expect("init_hw_device missing");
        assert!(
            !init_dev[1].contains("driver="),
            "unexpected driver= in {}",
            init_dev[1]
        );
        assert!(init_dev[1].contains("vaapi=va:"));
    }

    #[test]
    fn hls_qsv_hardware_accel() {
        let dir = PathBuf::from("/tmp/test_qsv");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(hw_accel::Qsv {
                vaapi_device: "/dev/dri/renderD128".into(),
                vaapi_driver: "iHD".into(),
                opencl: false,
            }),
            ..default_hls(dir)
        });

        // VAAPI device init with iHD driver
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-init_hw_device"
                    && w[1].starts_with("vaapi=va:")
                    && w[1].contains("driver=iHD")),
            "expected VAAPI init with iHD: {args:?}"
        );
        // QSV device derived from VAAPI
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-init_hw_device" && w[1] == "qsv=qs@va"),
            "expected qsv=qs@va init"
        );
        // filter_hw_device points at QSV
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-filter_hw_device" && w[1] == "qs"),
            "expected -filter_hw_device qs"
        );
        // VAAPI hwaccel (not qsv) for decode
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-hwaccel" && w[1] == "vaapi"),
            "expected -hwaccel vaapi"
        );
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-hwaccel_output_format" && w[1] == "vaapi"),
            "expected -hwaccel_output_format vaapi"
        );
        // Encoder remapped to QSV
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_qsv"));
        // scale_vaapi in -vf
        let vf = arg_after(&args, "-vf").expect("-vf missing for QSV");
        assert!(
            vf.contains("scale_vaapi"),
            "expected scale_vaapi in vf: {vf}"
        );
        assert!(
            vf.contains("hwmap=derive_device=qsv"),
            "expected hwmap in vf: {vf}"
        );
        assert!(vf.contains("format=qsv"), "expected format=qsv in vf: {vf}");
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    fn qsv() -> hw_accel::Qsv {
        hw_accel::Qsv {
            vaapi_device: "/dev/dri/renderD128".into(),
            vaapi_driver: "iHD".into(),
            opencl: false,
        }
    }

    fn vaapi() -> hw_accel::Vaapi {
        hw_accel::Vaapi {
            device: "/dev/dri/renderD128".into(),
            driver: "iHD".into(),
            opencl: false,
        }
    }

    #[test]
    fn hls_qsv_vpp_tonemap_without_subtitle_uses_tonemap_vaapi() {
        // Baseline: VPP tonemap without subtitle burn uses tonemap_vaapi in -vf
        // (not SW tonemap), and decode stays hardware end to end.
        let dir = PathBuf::from("/tmp/test_qsv_vpp_nosub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            ..default_hls(dir)
        });

        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(
            vf.contains("tonemap_vaapi"),
            "expected tonemap_vaapi in vf: {vf}"
        );
        assert_p010_into_tonemap(vf);
        assert!(
            !vf.contains("tonemapx"),
            "should not use SW tonemapx when VPP available: {vf}"
        );
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-hwaccel_output_format" && w[1] == "vaapi"),
            "decode should stay hardware when VPP tonemap runs uninterrupted"
        );
    }

    fn has_pair(args: &[String], k: &str, v: &str) -> bool {
        args.windows(2)
            .any(|w| w[0] == k && w[1] == v)
    }

    /// `tonemap_vaapi` must be fed the 10-bit surface; an NV12 scale in front
    /// of it produces a solid cyan picture.
    fn assert_p010_into_tonemap(chain: &str) {
        let tm = chain
            .find("tonemap_vaapi")
            .expect("tonemap_vaapi missing");
        let scale = chain[..tm]
            .rfind("scale_vaapi=")
            .expect("scale_vaapi must precede tonemap_vaapi");
        let scale_args = &chain[scale..tm];
        assert!(
            scale_args.contains("format=p010"),
            "scale_vaapi before tonemap_vaapi must output p010: {chain}"
        );
        assert!(
            !scale_args.contains("format=nv12"),
            "8-bit NV12 must not reach tonemap_vaapi: {chain}"
        );
        assert!(
            chain[tm..].contains("tonemap_vaapi=format=nv12"),
            "tone map must still emit NV12 for the encoder: {chain}"
        );
    }

    fn assert_qsv_hw_decode(args: &[String]) {
        assert!(
            has_pair(args, "-hwaccel", "vaapi")
                && has_pair(args, "-hwaccel_output_format", "vaapi"),
            "QSV must hardware-decode when the graph stays on the GPU: {args:?}"
        );
    }

    fn assert_qsv_sw_decode(args: &[String]) {
        assert!(
            !has_pair(args, "-hwaccel_output_format", "vaapi"),
            "QSV must software-decode when the graph runs CPU filters: {args:?}"
        );
        assert!(
            has_pair(args, "-init_hw_device", "qsv=qs@va"),
            "QSV encoder still needs the device chain: {args:?}"
        );
    }

    /// Asserts the GPU-resident overlay graph: subtitle uploaded to QSV,
    /// main branch on VAAPI/QSV surfaces, `overlay_qsv` composite, and no
    /// CPU `overlay`/`format=yuv420p` anywhere.
    fn assert_qsv_gpu_overlay(fc: &str) {
        assert!(
            fc.contains("hwupload=derive_device=qsv"),
            "subtitle must be uploaded to QSV: {fc}"
        );
        assert!(
            fc.contains("scale_vaapi"),
            "main branch must scale on GPU: {fc}"
        );
        assert!(
            fc.contains("derive_device=qsv") && fc.contains(",format=qsv"),
            "main branch must map into QSV memory: {fc}"
        );
        assert!(
            fc.contains("overlay_qsv="),
            "must composite with overlay_qsv: {fc}"
        );
        assert!(
            !fc.contains("]overlay="),
            "CPU overlay must not appear in the GPU graph: {fc}"
        );
        assert!(
            !fc.contains("format=yuv420p") && !fc.contains("tonemapx"),
            "no CPU colour filters in the GPU graph: {fc}"
        );
        let main_end = fc
            .find("[vmain]")
            .expect("main branch label");
        let overlay_at = fc
            .find("overlay_qsv=")
            .unwrap();
        assert!(
            main_end < overlay_at,
            "main-branch colour handling must precede the overlay: {fc}"
        );
    }

    #[test]
    fn hls_qsv_vpp_tonemap_with_subtitle_stays_on_gpu() {
        // Issue #378 originally forced software decode + tonemapx here because
        // tonemap_vaapi cannot consume CPU overlay output. With overlay_qsv the
        // overlay follows the tone map on the GPU, so the whole graph stays
        // hardware-resident.
        let dir = PathBuf::from("/tmp/test_qsv_vpp_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            tonemapping_desat: 0.0,
            tonemapping_peak: 0.0,
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert!(fc.contains("tonemap_vaapi"), "expected VPP tone map: {fc}");
        assert_p010_into_tonemap(fc);
        assert!(
            fc.find("tonemap_vaapi")
                .unwrap()
                < fc.find("overlay_qsv")
                    .unwrap(),
            "tone map must run before the overlay: {fc}"
        );
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_qsv"));
    }

    #[test]
    fn hls_qsv_hdr_clamp_with_subtitle_stays_on_gpu() {
        // Both tone-mapping toggles off (the default): HDR is clamped to SDR
        // via a BT.709 relabel on the GPU, and the overlay is overlay_qsv.
        let dir = PathBuf::from("/tmp/test_qsv_clamp_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            burn_subtitle: true,
            subtitle_stream_index: Some(5),
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert!(
            fc.contains("scale_vaapi=format=nv12"),
            "10-bit source must be converted to nv12 on the GPU: {fc}"
        );
        assert!(
            fc.contains(
                "format=qsv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
            ),
            "BT.709 relabel must follow the hwmap: {fc}"
        );
        assert!(!fc.contains("tonemap_vaapi"), "no tone map requested: {fc}");
    }

    #[test]
    fn hls_qsv_hdr_clamp_without_subtitle_uses_hw_decode() {
        // The decode gate is shared with the subtitle path: an HDR source with
        // no tone mapping and no burn-in must also stay on the GPU.
        let dir = PathBuf::from("/tmp/test_qsv_clamp_nosub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            max_width: Some(1920),
            max_height: Some(1080),
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.starts_with("scale_vaapi="), "GPU scale first: {vf}");
        assert!(
            vf.ends_with(
                "hwmap=derive_device=qsv,format=qsv,\
                 setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
            ),
            "hwmap then BT.709 relabel: {vf}"
        );
        assert_eq!(
            vf.matches("setparams")
                .count(),
            1,
            "setparams must not be prepended a second time: {vf}"
        );
        assert!(
            !vf.contains("format=nv12,") || vf.contains("scale_vaapi"),
            "{vf}"
        );
        assert!(!vf.contains("tonemapx"), "{vf}");
    }

    #[test]
    fn hls_qsv_sw_tonemap_with_subtitle_keeps_cpu_graph() {
        // Explicit software tonemapx is the one remaining CPU path on QSV:
        // software decode, CPU overlay, format=nv12 into the QSV encoder.
        let dir = PathBuf::from("/tmp/test_qsv_sw_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_qsv_sw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(fc.contains("tonemapx"), "explicit SW tone map: {fc}");
        assert!(fc.contains("]overlay="), "CPU overlay: {fc}");
        assert!(fc.contains("format=nv12"), "nv12 into QSV encoder: {fc}");
        assert!(
            !fc.contains("overlay_qsv") && !fc.contains("hwmap"),
            "no GPU filters on CPU frames: {fc}"
        );
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_qsv"));
    }

    #[test]
    fn progressive_qsv_vpp_tonemap_with_subtitle_stays_on_gpu() {
        // Same as the HLS case via the progressive (stdout) path, which has its
        // own copy of the graph construction.
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_progressive()
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert!(fc.contains("tonemap_vaapi"), "expected VPP tone map: {fc}");
        assert_p010_into_tonemap(fc);
    }

    #[test]
    fn hls_qsv_vpp_tonemap_with_resolution_cap_scales_to_p010() {
        let dir = PathBuf::from("/tmp/test_qsv_vpp_cap");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::DoviWithHdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            max_width: Some(1920),
            max_height: Some(1080),
            ..default_hls(dir)
        });

        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("min(1920,iw)"), "cap must be applied: {vf}");
        assert_p010_into_tonemap(vf);
    }

    #[test]
    fn progressive_qsv_vpp_tonemap_without_subtitle_scales_to_p010() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            ..default_progressive()
        });

        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_p010_into_tonemap(vf);
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    fn qsv_opencl() -> hw_accel::Qsv {
        hw_accel::Qsv {
            opencl: true,
            ..qsv()
        }
    }

    /// OpenCL tone map on GPU-resident frames: P010 scale, map into OpenCL,
    /// tone map, map back through VAAPI into QSV.
    fn assert_ocl_chain(chain: &str) {
        let map_in = chain
            .find("hwmap=derive_device=opencl:mode=read")
            .expect("must map into OpenCL");
        let scale = chain[..map_in]
            .rfind("scale_vaapi=")
            .expect("scale_vaapi must precede the OpenCL map");
        assert!(
            chain[scale..map_in].contains("format=p010"),
            "OpenCL tone map needs the 10-bit surface: {chain}"
        );
        let tm = chain
            .find("tonemap_opencl=")
            .expect("tonemap_opencl missing");
        assert!(map_in < tm, "{chain}");
        assert!(
            chain[tm..].contains(
                "hwmap=derive_device=qsv:mode=write:reverse=1:extra_hw_frames=16,format=qsv"
            ),
            "must return to QSV memory for the encoder: {chain}"
        );
        assert!(
            !chain.contains("tonemapx") && !chain.contains("tonemap_vaapi"),
            "{chain}"
        );
    }

    #[test]
    fn hls_qsv_tonemapping_uses_opencl_on_gpu() {
        let dir = PathBuf::from("/tmp/test_qsv_ocl");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv_opencl()),
            source_video_range_type: Some(VideoRangeType::DoviWithHdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            tonemapping_peak: 100.0,
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        assert!(
            has_pair(&args, "-init_hw_device", "opencl=ocl@va"),
            "{args:?}"
        );
        assert!(has_pair(&args, "-filter_hw_device", "ocl"), "{args:?}");
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_ocl_chain(vf);
        assert!(vf.contains("tonemap=bt2390:peak=100"), "{vf}");
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_qsv"));
    }

    #[test]
    fn hls_qsv_opencl_tonemap_with_subtitle_stays_on_gpu() {
        let dir = PathBuf::from("/tmp/test_qsv_ocl_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv_opencl()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert_ocl_chain(fc);
        assert!(
            fc.find("tonemap_opencl")
                .unwrap()
                < fc.find("overlay_qsv")
                    .unwrap(),
            "tone map must run before the overlay: {fc}"
        );
    }

    #[test]
    fn progressive_qsv_tonemapping_uses_opencl_on_gpu() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv_opencl()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            ..default_progressive()
        });

        assert_qsv_hw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_ocl_chain(vf);
    }

    #[test]
    fn hls_qsv_tonemapping_without_opencl_falls_back_to_tonemapx() {
        let dir = PathBuf::from("/tmp/test_qsv_no_ocl");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            ..default_hls(dir)
        });

        assert_qsv_sw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("tonemapx"), "{vf}");
        assert!(!vf.contains("tonemap_opencl"), "{vf}");
    }

    fn assert_vaapi_hw_decode(args: &[String]) {
        assert!(
            has_pair(args, "-hwaccel", "vaapi")
                && has_pair(args, "-hwaccel_output_format", "vaapi"),
            "VAAPI must hardware-decode when the graph stays on the GPU: {args:?}"
        );
    }

    fn assert_vaapi_sw_decode(args: &[String]) {
        assert!(
            !has_pair(args, "-hwaccel_output_format", "vaapi"),
            "VAAPI must software-decode when the graph runs CPU filters: {args:?}"
        );
        assert!(
            has_pair(args, "-filter_hw_device", "va"),
            "VAAPI encoder still needs the device chain: {args:?}"
        );
    }

    fn vaapi_opencl() -> hw_accel::Vaapi {
        hw_accel::Vaapi {
            opencl: true,
            ..vaapi()
        }
    }

    /// Like `assert_ocl_chain`, but for plain VAAPI: the reverse map lands
    /// directly back in VAAPI memory (no QSV re-hop — the encoder is already
    /// native VAAPI).
    fn assert_vaapi_ocl_chain(chain: &str) {
        let map_in = chain
            .find("hwmap=derive_device=opencl:mode=read")
            .expect("must map into OpenCL");
        let scale = chain[..map_in]
            .rfind("scale_vaapi=")
            .expect("scale_vaapi must precede the OpenCL map");
        assert!(
            chain[scale..map_in].contains("format=p010"),
            "OpenCL tone map needs the 10-bit surface: {chain}"
        );
        let tm = chain
            .find("tonemap_opencl=")
            .expect("tonemap_opencl missing");
        assert!(map_in < tm, "{chain}");
        assert!(
            chain[tm..].contains(
                "hwmap=derive_device=vaapi:mode=write:reverse=1,format=vaapi"
            ),
            "must return to VAAPI memory for the encoder: {chain}"
        );
        assert!(
            !chain.contains("tonemapx") && !chain.contains("tonemap_vaapi"),
            "{chain}"
        );
    }

    #[test]
    fn hls_vaapi_vpp_tonemap_without_subtitle_scales_to_p010() {
        let dir = PathBuf::from("/tmp/test_vaapi_vpp_nosub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            ..default_hls(dir)
        });

        assert_vaapi_hw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("tonemap_vaapi"), "expected VPP tone map: {vf}");
        assert_p010_into_tonemap(vf);
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_vaapi"));
    }

    #[test]
    fn hls_vaapi_tonemapping_uses_opencl_on_gpu() {
        let dir = PathBuf::from("/tmp/test_vaapi_ocl");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi_opencl()),
            source_video_range_type: Some(VideoRangeType::DoviWithHdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            tonemapping_peak: 100.0,
            ..default_hls(dir)
        });

        assert_vaapi_hw_decode(&args);
        assert!(
            has_pair(&args, "-init_hw_device", "opencl=ocl@va"),
            "{args:?}"
        );
        assert!(has_pair(&args, "-filter_hw_device", "ocl"), "{args:?}");
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_vaapi_ocl_chain(vf);
        assert!(vf.contains("tonemap=bt2390:peak=100"), "{vf}");
        assert_eq!(arg_after(&args, "-c:v"), Some("h264_vaapi"));
    }

    #[test]
    fn hls_vaapi_opencl_tonemap_with_subtitle_falls_back_to_sw_tonemap() {
        // VAAPI has no GPU-resident overlay, so unlike QSV a burn-in still
        // blocks OpenCL tone mapping (see `vaapi_tone_mapping_runs_on_opencl_when_available`
        // in hw_accel.rs) and falls back to software tone mapping + CPU overlay.
        let dir = PathBuf::from("/tmp/test_vaapi_ocl_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi_opencl()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(fc.contains("tonemapx"), "{fc}");
        assert!(!fc.contains("tonemap_opencl"), "{fc}");
        assert!(fc.contains("]overlay="), "CPU overlay: {fc}");
    }

    #[test]
    fn progressive_vaapi_tonemapping_uses_opencl_on_gpu() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi_opencl()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            ..default_progressive()
        });

        assert_vaapi_hw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_vaapi_ocl_chain(vf);
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    #[test]
    fn hls_vaapi_tonemapping_without_opencl_falls_back_to_tonemapx() {
        let dir = PathBuf::from("/tmp/test_vaapi_no_ocl");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "bt2390".into(),
            ..default_hls(dir)
        });

        assert_vaapi_sw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(vf.contains("tonemapx"), "{vf}");
        assert!(!vf.contains("tonemap_opencl"), "{vf}");
    }

    #[test]
    fn hls_vaapi_sdr_uses_hw_decode_with_no_extra_hop() {
        // Plain SDR VAAPI transcode: real GPU decode + scale, no VPP/OpenCL/
        // clamp suffix needed since a native h264_vaapi encoder consumes
        // VAAPI-format frames directly (unlike QSV, which always needs a
        // final hwmap into QSV memory).
        let dir = PathBuf::from("/tmp/test_vaapi_sdr");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(vaapi()),
            ..default_hls(dir)
        });

        assert_vaapi_hw_decode(&args);
        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert_eq!(vf, "scale_vaapi=format=nv12:extra_hw_frames=24");
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    #[test]
    fn hls_qsv_hdr_clamp_keeps_nv12_scale() {
        // No tone map: the relabel path truncates to 8-bit on purpose.
        let dir = PathBuf::from("/tmp/test_qsv_clamp_nv12");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            ..default_hls(dir)
        });

        let vf = arg_after(&args, "-vf").expect("-vf missing");
        assert!(
            vf.contains("scale_vaapi=format=nv12"),
            "clamp stays NV12: {vf}"
        );
        assert!(!vf.contains("tonemap_vaapi"), "clamp has no tone map: {vf}");
    }

    #[test]
    fn progressive_qsv_hdr_clamp_with_subtitle_stays_on_gpu() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_progressive()
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert!(fc.contains("setparams=color_primaries=bt709"), "{fc}");
    }

    #[test]
    fn progressive_qsv_sw_tonemap_with_subtitle_keeps_cpu_graph() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_progressive()
        });

        assert_qsv_sw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(fc.contains("tonemapx") && fc.contains("]overlay="), "{fc}");
        assert!(!fc.contains("overlay_qsv"), "{fc}");
    }

    #[test]
    fn hls_qsv_sdr_subtitle_overlay_unchanged() {
        // SDR + burn-in on QSV: the pre-existing overlay_qsv graph, no colour
        // handling at all.
        let dir = PathBuf::from("/tmp/test_qsv_sdr_sub");
        let args = build_hls_args(&TranscodeParams {
            video_codec: "libx264".into(),
            accelerator: Box::new(qsv()),
            source_video_range_type: Some(VideoRangeType::Sdr),
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_hls(dir)
        });

        assert_qsv_hw_decode(&args);
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert_qsv_gpu_overlay(fc);
        assert!(
            !fc.contains("setparams") && !fc.contains("tonemap"),
            "SDR needs no colour handling: {fc}"
        );
    }

    #[test]
    fn progressive_basic_copy() {
        let args = build_progressive_args(&default_progressive());

        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert_eq!(arg_after(&args, "-c:a"), Some("aac"));
        // Copy into mp4 is promoted to matroska to avoid bsf issues
        assert_eq!(arg_after(&args, "-f"), Some("matroska"));
        assert!(args.last() == Some(&"pipe:1".to_string()));
    }

    #[test]
    fn progressive_mp4_transcode_uses_mp4_format() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            container: "mp4".into(),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-f"), Some("mp4"));
        assert!(args_contains(&args, "-movflags"));
    }

    #[test]
    fn progressive_ts_container() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            container: "ts".into(),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-f"), Some("mpegts"));
    }

    #[test]
    fn progressive_seek_before_input() {
        let ticks: i64 = 60i64
            .to_ticks(TickUnit::Seconds)
            .unwrap();
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            start_time_ticks: Some(ticks),
            ..default_progressive()
        });
        let ss_pos = args
            .iter()
            .position(|a| a == "-ss")
            .expect("-ss missing");
        let i_pos = args
            .iter()
            .position(|a| a == "-i")
            .expect("-i missing");
        assert!(ss_pos < i_pos);
        assert_eq!(args[ss_pos + 1], "60.000000");
    }

    #[test]
    fn progressive_hevc_copy_adds_hvc1_tag() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            source_video_codec: Some("hevc".into()),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-tag:v"), Some("hvc1"));
    }

    #[test]
    fn progressive_nvenc_remaps_encoder() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            container: "mp4".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            source_video_range_type: Some(VideoRangeType::Sdr),
            enable_tonemapping: false,
            ..default_progressive()
        });
        assert_h264_nvenc_yuv420p(&args);
        assert!(!args_contains(&args, "-vf"));
        assert!(!args_contains(&args, "-filter_complex"));
    }

    #[test]
    fn progressive_nvenc_hdr10_without_tonemap_pins_yuv420p() {
        for bitrate in [None, Some(5_000_000)] {
            let args = build_progressive_args(&ProgressiveTranscodeParams {
                video_codec: "libx264".into(),
                container: "mp4".into(),
                accelerator: Box::new(hw_accel::Nvenc),
                source_video_range_type: Some(VideoRangeType::Hdr10),
                enable_tonemapping: false,
                video_bitrate: bitrate,
                ..default_progressive()
            });
            assert_h264_nvenc_yuv420p(&args);
            assert_eq!(
                arg_after(&args, "-vf"),
                Some(
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
                )
            );
            let expected = bitrate.map(|rate| rate.to_string());
            assert_eq!(arg_after(&args, "-b:v"), expected.as_deref());
        }
    }

    #[test]
    fn progressive_nvenc_sw_tonemap_and_subtitle_burn_pin_yuv420p() {
        let tonemap = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            container: "mp4".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            ..default_progressive()
        });
        assert_h264_nvenc_yuv420p(&tonemap);
        assert_eq!(
            arg_after(&tonemap, "-vf"),
            Some(
                "tonemapx=tonemap=hable:desat=0.0:peak=0.0:t=bt709:m=bt709:p=bt709:format=nv12"
            )
        );

        let burned = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            container: "mp4".into(),
            accelerator: Box::new(hw_accel::Nvenc),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_tonemapping: true,
            burn_subtitle: true,
            subtitle_stream_index: Some(2),
            ..default_progressive()
        });
        assert_h264_nvenc_yuv420p(&burned);
        let fc =
            arg_after(&burned, "-filter_complex").expect("-filter_complex missing");
        assert!(
            fc.contains(
                "tonemapx=tonemap=hable:desat=0.0:peak=0.0:t=bt709:m=bt709:p=bt709:format=yuv420p"
            ),
            "{fc}"
        );
        assert!(fc.contains("]overlay="), "{fc}");
    }

    #[test]
    fn progressive_nvenc_stream_copy_omits_pix_fmt() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            accelerator: Box::new(hw_accel::Nvenc),
            video_bitrate: Some(4_000_000),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert!(!args_contains(&args, "-pix_fmt"));
    }

    #[test]
    fn progressive_subtitle_burn_filter_complex() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "libx264".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(1),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-c:v"), Some("libx264"));
        let fc = arg_after(&args, "-filter_complex").expect("-filter_complex missing");
        assert!(fc.contains("overlay"), "fc: {fc}");
    }

    #[test]
    fn progressive_subtitle_burn_cannot_override_video_copy() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            video_codec: "copy".into(),
            burn_subtitle: true,
            subtitle_stream_index: Some(1),
            ..default_progressive()
        });

        assert_eq!(arg_after(&args, "-c:v"), Some("copy"));
        assert!(!args_contains(&args, "-filter_complex"));
    }

    #[test]
    fn progressive_audio_channels_and_bitrate() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            audio_codec: "aac".into(),
            audio_bitrate: Some(256_000),
            audio_channels: Some(6),
            ..default_progressive()
        });
        assert_eq!(arg_after(&args, "-b:a"), Some("256000"));
        assert_eq!(arg_after(&args, "-ac"), Some("6"));
    }

    #[test]
    fn progressive_reconnect_flags_present() {
        let args = build_progressive_args(&default_progressive());
        assert!(args_contains(&args, "-reconnect"));
        assert!(args_contains(&args, "-reconnect_streamed"));
        assert!(args_contains(&args, "-reconnect_delay_max"));
        assert!(!args_contains(&args, "-reconnect_at_eof"));
    }

    #[test]
    fn ffmpeg_http_input_args_depend_on_protocol_and_container() {
        for url in [
            "http://127.0.0.1:8080/torrents/1/stream/0",
            "https://cdn.example.com/video.mkv",
        ] {
            assert!(
                !ffmpeg_http_input_args(url).is_empty(),
                "expected reconnect args for {url}"
            );
            assert!(
                !ffmpeg_http_input_args(url).contains(&"-http_persistent"),
                "-http_persistent is an HLS-demuxer-only option, rejected for {url}"
            );
        }
        for url in [
            "/media/tv/show/episode.mkv",
            r"C:\media\tv\show\episode.mkv",
            r"\\server\media\show\episode.mkv",
            "file:///media/tv/show/episode.mkv",
            "rtsp://camera.example.com/live",
        ] {
            assert!(
                ffmpeg_http_input_args(url).is_empty(),
                "expected no reconnect args for {url}"
            );
        }

        let hls = ffmpeg_http_input_args(
            "https://relay.example/api/manifest?url=https%3A%2F%2Forigin.example%2Flive.m3u8",
        );
        assert_eq!(hls, ["-http_persistent", "0"]);

        // A generic relay endpoint is not necessarily HLS. DASH inputs still
        // need the normal reconnect options.
        let dash = ffmpeg_http_input_args(
            "https://relay.example/hls?url=https%3A%2F%2Forigin.example%2Flive.mpd",
        );
        assert!(dash.contains(&"-reconnect"));
    }

    #[test]
    fn progressive_local_input_omits_reconnect_flags() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            input_url: "/media/tv/show/episode.mkv".into(),
            ..default_progressive()
        });
        assert!(!args_contains(&args, "-reconnect"));
        assert!(!args_contains(&args, "-reconnect_at_eof"));
        assert!(!args_contains(&args, "-reconnect_streamed"));
        assert!(!args_contains(&args, "-reconnect_delay_max"));
    }

    #[test]
    fn hls_http_input_gets_reconnect_flags() {
        let args = build_hls_args(&default_hls(PathBuf::from("/tmp/test_session")));
        assert!(args_contains(&args, "-reconnect"));
        assert!(args_contains(&args, "-reconnect_streamed"));
        assert!(args_contains(&args, "-reconnect_delay_max"));
        assert!(!args_contains(&args, "-reconnect_at_eof"));
    }

    #[test]
    fn hls_local_input_omits_reconnect_flags() {
        let args = build_hls_args(&TranscodeParams {
            input_url: "/media/tv/show/episode.mkv".into(),
            ..default_hls(PathBuf::from("/tmp/test_session"))
        });
        assert!(!args_contains(&args, "-reconnect"));
        assert!(!args_contains(&args, "-reconnect_at_eof"));
        assert!(!args_contains(&args, "-reconnect_streamed"));
        assert!(!args_contains(&args, "-reconnect_delay_max"));
    }

    #[test]
    fn hls_loudnorm_added_when_transcoding_audio() {
        let dir = PathBuf::from("/tmp/test_session");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "aac".into(),
            normalize_audio_loudness: true,
            ..default_hls(dir)
        });
        assert!(args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    #[test]
    fn hls_loudnorm_absent_when_disabled() {
        let dir = PathBuf::from("/tmp/test_session");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "aac".into(),
            normalize_audio_loudness: false,
            ..default_hls(dir)
        });
        assert!(!args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    #[test]
    fn hls_loudnorm_absent_when_audio_copy() {
        let dir = PathBuf::from("/tmp/test_session");
        let args = build_hls_args(&TranscodeParams {
            audio_codec: "copy".into(),
            normalize_audio_loudness: true,
            ..default_hls(dir)
        });
        assert!(!args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    #[test]
    fn progressive_loudnorm_added_when_transcoding_audio() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            audio_codec: "aac".into(),
            normalize_audio_loudness: true,
            ..default_progressive()
        });
        assert!(args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    #[test]
    fn progressive_loudnorm_absent_when_disabled() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            audio_codec: "aac".into(),
            normalize_audio_loudness: false,
            ..default_progressive()
        });
        assert!(!args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    #[test]
    fn progressive_loudnorm_absent_when_audio_copy() {
        let args = build_progressive_args(&ProgressiveTranscodeParams {
            audio_codec: "copy".into(),
            normalize_audio_loudness: true,
            ..default_progressive()
        });
        assert!(!args_contains(&args, "loudnorm=I=-14:TP=-1:LRA=11"));
    }

    /// Manual debug tool — prints the real ffmpeg command line(s) for the
    /// issue #520 repro (4K HEVC Main10 HDR10 + PGS subtitle burn on QSV) so
    /// they can be copy-pasted and run by hand against real Arc/QSV hardware,
    /// without needing a live server request or a device profile to hit this
    /// code path. Override via env vars; defaults match the issue report.
    ///
    /// cargo test -p remux-server --lib print_qsv_hdr_subtitle_ffmpeg_cmd -- --ignored --nocapture
    #[test]
    #[ignore = "manual debug tool — prints an ffmpeg command line to run by hand"]
    fn print_qsv_hdr_subtitle_ffmpeg_cmd() {
        let input = std::env::var("FFMPEG_TEST_INPUT")
            .unwrap_or_else(|_| "/path/to/source.mkv".to_string());
        let output_dir = std::env::var("FFMPEG_TEST_OUTPUT_DIR")
            .unwrap_or_else(|_| "/tmp/qsv_hdr_sub_test".to_string());
        let sub_index: i32 = std::env::var("FFMPEG_TEST_SUB_INDEX")
            .ok()
            .and_then(|s| {
                s.parse()
                    .ok()
            })
            .unwrap_or(3);
        let vaapi_device = std::env::var("FFMPEG_TEST_VAAPI_DEVICE")
            .unwrap_or_else(|_| "/dev/dri/renderD128".to_string());
        let vaapi_driver = std::env::var("FFMPEG_TEST_VAAPI_DRIVER")
            .unwrap_or_else(|_| "iHD".to_string());

        let qsv_accel = || {
            Box::new(hw_accel::Qsv {
                vaapi_device: vaapi_device.clone(),
                vaapi_driver: vaapi_driver.clone(),
                opencl: false,
            }) as Box<dyn Accelerator>
        };

        let hls_args = build_hls_args(&TranscodeParams {
            input_url: input.clone(),
            output_dir: PathBuf::from(&output_dir),
            video_codec: "libx264".into(),
            audio_codec: "aac".into(),
            accelerator: qsv_accel(),
            source_video_codec: Some("hevc".into()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            tonemapping_desat: 0.0,
            tonemapping_peak: 100.0,
            burn_subtitle: true,
            subtitle_stream_index: Some(sub_index),
            ..default_hls(PathBuf::from(&output_dir))
        });

        let progressive_args = build_progressive_args(&ProgressiveTranscodeParams {
            input_url: input,
            container: "mp4".into(),
            video_codec: "libx264".into(),
            audio_codec: "aac".into(),
            accelerator: qsv_accel(),
            source_video_codec: Some("hevc".into()),
            source_video_range_type: Some(VideoRangeType::Hdr10),
            enable_vpp_tonemapping: true,
            tonemapping_algorithm: "hable".into(),
            tonemapping_desat: 0.0,
            tonemapping_peak: 100.0,
            burn_subtitle: true,
            subtitle_stream_index: Some(sub_index),
            ..default_progressive()
        });

        println!("\n=== HLS (writes segments under {output_dir}) ===");
        println!("mkdir -p {output_dir} && ffmpeg {}", hls_args.join(" "));

        println!("\n=== Progressive (pipes to stdout, redirect to a file) ===");
        println!(
            "ffmpeg {} > /tmp/qsv_hdr_sub_test.mp4",
            progressive_args.join(" ")
        );

        println!(
            "\nOverride with FFMPEG_TEST_INPUT / FFMPEG_TEST_OUTPUT_DIR / \
             FFMPEG_TEST_SUB_INDEX / FFMPEG_TEST_VAAPI_DEVICE / FFMPEG_TEST_VAAPI_DRIVER"
        );
    }
}

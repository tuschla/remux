use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{
    common, db,
    db::auth,
    playback::{
        decision::PlaybackPermissions,
        session::{TranscodeSession, longest_segment},
    },
};
use remux_sdks::remux::{PlayMethod, PlaybackInfo, QueueItem};

/// Held per play session while its transcode is created, so a stop or dir
/// removal never misses one still being created.
pub(crate) static TRANSCODE_CREATE_LOCKS: crate::keyed_lock::KeyedLock<String> =
    crate::keyed_lock::KeyedLock::new();

/// How long to wait for a killed ffmpeg to be reaped before warning, and
/// for `stop_all` as a whole.
const STOP_WAIT: Duration = Duration::from_secs(10);

/// How a stop treats the transcode's ffmpeg.
#[derive(Clone, Copy)]
pub enum StopMode {
    /// Return at once; ffmpeg is reaped and its dir removed in the background.
    Detached,
    /// Wait for the reap: the caller reuses the dir or exits.
    Reaped,
}

#[derive(Clone)]
pub struct PlaybackSession {
    pub play_session_id: String,
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub media_source_id: Option<String>,
    pub device_id: String,
    pub client_name: String,
    pub position_ticks: i64,
    pub can_seek: bool,
    pub is_paused: bool,
    pub last_paused_at: Option<DateTime<Utc>>,
    pub is_muted: bool,
    pub volume_level: Option<i32>,
    pub audio_stream_index: Option<i32>,
    pub subtitle_stream_index: Option<i32>,
    pub play_method: Option<String>,
    /// What a stream endpoint actually served, which also set `play_method`;
    /// client reports must not overwrite it.
    served: Option<ServedPlayback>,
    pub now_playing_queue: Option<Vec<QueueItem>>,
    pub playlist_item_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    /// When the client last reported this play session (start, progress,
    /// ping); `None` for a transcode stub.
    pub last_report: Option<std::time::Instant>,
    /// Active transcode session owned by this playback session, if any.
    pub transcode: Option<Arc<tokio::sync::RwLock<TranscodeSession>>>,
    /// Stream group UUID that the selected source belongs to, if any.
    pub group_id: Option<Uuid>,
    /// Kind of the item being played, used to populate NowPlayingItem in session broadcasts.
    pub item_kind: Option<db::MediaKind>,
    /// Torrent sources selected for this playback. Their leases are released
    /// with the session, after any active HTTP response has finished reading.
    torrents: Vec<Arc<crate::torrent::TorrentLease>>,
}

impl PlaybackSession {
    /// What a stream endpoint served for this session, if one has recorded it.
    pub fn served(&self) -> Option<&ServedPlayback> {
        self.served
            .as_ref()
    }

    /// Whether the client receives the source video without re-encoding.
    pub fn video_is_copied(&self) -> bool {
        match &self.served {
            // A stream endpoint recorded what it sent: that is the answer.
            Some(served) => served.video_copied(),
            // Nothing recorded: believe the client, unless it says it is
            // being transcoded.
            None => {
                self.play_method
                    .as_deref()
                    != Some("Transcode")
            }
        }
    }

    /// False until a start report has claimed this session (a transcode
    /// stub); progress reports may still have set its item.
    fn is_reported(&self) -> bool {
        !self
            .user_id
            .is_nil()
    }

    /// Whether the client still plays this session's transcode although
    /// another of its item was created `at`: it reported the session since,
    /// or last reported it paused (a paused player requests nothing but
    /// keeps reporting). A retry's predecessor gets neither.
    fn still_played(&self, at: std::time::Instant) -> bool {
        self.is_paused
            || self
                .last_report
                .is_some_and(|reported| reported >= at)
    }
}

/// Which of a device's playback `sessions` `/Sessions` shows, and the
/// transcode serving it. Reported sessions rank first; a stub no report has
/// given an item is never shown, as Jellyfin shows nothing before the start
/// report. If the client retried a reported transcode under a new
/// PlaySessionId without a new start report, the retry's stub serves it.
pub(crate) async fn now_playing<'a>(
    sessions: &[&'a PlaybackSession],
) -> Option<(
    &'a PlaybackSession,
    Option<Arc<tokio::sync::RwLock<TranscodeSession>>>,
)> {
    let shown = sessions
        .iter()
        .copied()
        .filter(|s| {
            s.is_reported()
                || !s
                    .item_id
                    .is_nil()
        })
        .max_by_key(|s| {
            (
                s.is_reported(),
                s.transcode
                    .is_some(),
                s.last_activity,
            )
        })?;
    if shown
        .transcode
        .is_some()
        || shown
            .play_method
            .as_deref()
            != Some("Transcode")
    {
        return Some((
            shown,
            shown
                .transcode
                .clone(),
        ));
    }
    for stub in sessions
        .iter()
        .filter(|s| !s.is_reported())
    {
        let Some(ts) = &stub.transcode else {
            continue;
        };
        let t = ts
            .read()
            .await;
        if !t
            .stopped
            .is_cancelled()
            && t.item_id == shown.item_id
            && t.user_id == shown.user_id
        {
            return Some((shown, Some(ts.clone())));
        }
    }
    Some((shown, None))
}

/// What a stream endpoint actually served for a playback session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServedPlayback {
    /// The source bytes, unchanged.
    Direct,
    /// FFmpeg output: how each of its tracks was produced.
    Ffmpeg {
        video: FfmpegTrack,
        audio: FfmpegTrack,
    },
}

impl ServedPlayback {
    /// Any FFmpeg output — remux included — is `Transcode` in Jellyfin's
    /// vocabulary; clients tell Remux / Direct Stream apart via
    /// `TranscodingInfo.IsVideoDirect` / `IsAudioDirect`.
    ///
    /// Unchanged source bytes are recorded as `DirectPlay`: remux only serves
    /// the static route when direct play is offered, so a client's
    /// `DirectStream` report is superseded by what the stream endpoint served.
    pub fn play_method(&self) -> PlayMethod {
        match self {
            Self::Direct => PlayMethod::DirectPlay,
            Self::Ffmpeg { .. } => PlayMethod::Transcode,
        }
    }

    /// Whether the client decodes the source video itself.
    pub fn video_copied(&self) -> bool {
        match self {
            Self::Direct => true,
            Self::Ffmpeg { video, .. } => video.copied,
        }
    }
}

/// One track (video or audio) of FFmpeg output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegTrack {
    /// Codec the client receives: the source's when copied, the encoder
    /// target otherwise. `None` when the source has no such track —
    /// clients read a missing video codec as audio-only output.
    pub codec: Option<String>,
    /// Stream-copied from the source rather than re-encoded.
    pub copied: bool,
}

impl FfmpegTrack {
    /// `output` is FFmpeg's codec argument: `copy` or an encoder target.
    pub fn new(output: &str, source_codec: Option<&str>) -> Self {
        if output.eq_ignore_ascii_case("copy") {
            Self {
                codec: source_codec.map(str::to_string),
                copied: true,
            }
        } else {
            Self {
                codec: Some(output.to_string()),
                copied: false,
            }
        }
    }
}

#[derive(Clone)]
pub struct PlaybackSessionManager {
    sessions: Arc<DashMap<String, PlaybackSession>>,
    // Requests can arrive before the client's playback-start report. Hold
    // their leases and the play method they actually served briefly, then
    // move them into the PlaybackSession on start.
    pending_torrents:
        Arc<DashMap<String, (DateTime<Utc>, Vec<Arc<crate::torrent::TorrentLease>>)>>,
    pending_play_methods: Arc<DashMap<String, (DateTime<Utc>, ServedPlayback)>>,
    // Makes that handoff atomic: a request cannot be stranded in pending
    // while playback-start inserts its session. True once `stop_all` began:
    // no transcode is attached after.
    handoff: Arc<std::sync::Mutex<bool>>,
    base_dir: PathBuf,
}

impl PlaybackSessionManager {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        let base_dir = base_dir.into();
        let _ = std::fs::create_dir_all(&base_dir);
        Self {
            sessions: Arc::new(DashMap::new()),
            pending_torrents: Arc::new(DashMap::new()),
            pending_play_methods: Arc::new(DashMap::new()),
            handoff: Arc::new(std::sync::Mutex::new(false)),
            base_dir,
        }
    }

    /// Handle a `POST /sessions/playing` report.
    ///
    /// Enforces the per-user session limit, resolves the optional StreamGroup
    /// source, builds and inserts the `PlaybackSession`, and emits the playback-
    /// start log line (skipped for transcode — the HLS handler logs that after
    /// it has codec/bitrate/reason details).
    pub async fn start(
        &self,
        db: &sqlx::SqlitePool,
        auth_session: &auth::AuthSession,
        data: &PlaybackInfo,
    ) -> anyhow::Result<()> {
        let play_session_id = data
            .play_session_id
            .clone()
            .unwrap_or_else(|| {
                common::get_uuid()
                    .as_simple()
                    .to_string()
            });

        // A method already recorded by a stream endpoint overrides this in
        // `insert`; until then, don't trust a method the server can't use.
        let client_play_method = constrain_client_play_method(
            db,
            &auth_session.user,
            data.play_method
                .clone(),
        )
        .await;

        // Enforce per-user concurrent-stream limit.
        let max_sessions = auth_session
            .user
            .policy
            .as_ref()
            .map(|p| p.max_active_sessions)
            .unwrap_or(0);
        if max_sessions > 0 {
            // Exclude the caller's own device: insert() will replace any existing
            // session for that device, so it doesn't consume an extra slot.
            let current = self.count_for_user(
                auth_session
                    .user
                    .id,
                Some(
                    &auth_session
                        .device
                        .id,
                ),
            );
            if current >= max_sessions as usize {
                return Err(anyhow::anyhow!("Stream limit reached")
                    .context("Maximum concurrent streams reached"));
            }
        }

        let item_id = data.item_id;
        if item_id.is_nil() {
            warn!(
                client = %auth_session.device.app_name,
                "PlaybackStart missing item_id, skipping session creation"
            );
            return Ok(());
        }

        // If the client selected a StreamGroup source, record its group UUID.
        let group_id: Option<Uuid> = if let Some(ref sid) = data.media_source_id {
            if let Ok(uid) = sid.parse::<Uuid>() {
                db::Media::get_by_id(db, &uid)
                    .await
                    .ok()
                    .flatten()
                    .filter(|m| m.kind == db::MediaKind::StreamGroup)
                    .map(|_| uid)
            } else {
                None
            }
        } else {
            None
        };

        let item = db::Media::get_by_id(db, &item_id)
            .await
            .ok()
            .flatten();
        let item_kind = item
            .as_ref()
            .map(|m| {
                m.kind
                    .clone()
            });

        // Jellyfin sets LastPlayedDate on playback start (never on a bare
        // stop) so DatePlayed-sorted lists — Continue Watching, the unified
        // Next Up feed — can rank an in-progress item by recency even before
        // it's ever been fully watched or reported any progress.
        if let Some(ref media) = item {
            if let Ok(mut state) =
                db::UserMediaState::get_or_new(db, &auth_session.user, media).await
            {
                state.last_played_at = Some(Utc::now().naive_utc());
                let _ = state
                    .save(db)
                    .await;
            }
        }

        let ps = PlaybackSession {
            play_session_id: play_session_id.clone(),
            user_id: auth_session
                .user
                .id,
            item_id,
            media_source_id: data
                .media_source_id
                .clone(),
            device_id: auth_session
                .device
                .id
                .clone(),
            client_name: auth_session
                .device
                .app_name
                .clone(),
            position_ticks: data
                .position_ticks
                .unwrap_or(0),
            can_seek: data.can_seek,
            is_paused: data.is_paused,
            last_paused_at: if data.is_paused {
                Some(Utc::now())
            } else {
                None
            },
            is_muted: data.is_muted,
            volume_level: data.volume_level,
            audio_stream_index: data.audio_stream_index,
            subtitle_stream_index: data.subtitle_stream_index,
            play_method: client_play_method
                .as_ref()
                .map(|m| m.to_string()),
            served: None,
            now_playing_queue: data
                .now_playing_queue
                .clone(),
            playlist_item_id: data
                .playlist_item_id
                .clone(),
            started_at: Utc::now(),
            last_activity: Utc::now(),
            last_report: Some(std::time::Instant::now()),
            transcode: None,
            group_id,
            item_kind,
            torrents: Vec::new(),
        };

        self.insert(ps)
            .await;

        let effective_play_method = self
            .get(&play_session_id)
            .and_then(|session| {
                session
                    .play_method
                    .and_then(|method| {
                        method
                            .parse()
                            .ok()
                    })
            });

        // For transcode sessions, master_hls_video fires the info log once it
        // has full codec/bitrate/reasons info. For direct play/stream, log here.
        let is_transcode = matches!(effective_play_method, Some(PlayMethod::Transcode));
        if !is_transcode {
            // Best-effort: fetch media title and source path for the log line.
            let media_title = db::Media::get_by_id(db, &item_id)
                .await
                .ok()
                .flatten()
                .map(|m| m.title)
                .unwrap_or_default();

            let (source_title, source_path) =
                if let Some(ref sid) = data.media_source_id {
                    if let Ok(source_uuid) = sid.parse::<Uuid>() {
                        let m = db::Media::get_by_id(db, &source_uuid)
                            .await
                            .ok()
                            .flatten();
                        (
                            m.as_ref()
                                .map(|m| {
                                    m.title
                                        .clone()
                                }),
                            m.and_then(|m| {
                                m.stream_info
                                    .map(|si| si.descriptor)
                            }),
                        )
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

            let log_session_id = play_session_id
                .trim_start_matches("audio-")
                .trim_start_matches("video-");
            let position_secs = data
                .position_ticks
                .unwrap_or(0)
                / 10_000_000;
            info!(
                play_session_id = log_session_id,
                %item_id,
                title = %media_title,
                source = ?source_title,
                path = ?source_path,
                user = %auth_session.user.username,
                client = %auth_session.device.app_name,
                play_method = ?effective_play_method,
                audio_stream = ?data.audio_stream_index,
                subtitle_stream = ?data.subtitle_stream_index,
                position_secs,
                "▶ Playback started"
            );
        }

        Ok(())
    }

    /// Handle a `POST /sessions/playing/progress` report.
    ///
    /// Updates the in-memory session state, notifies the transcode buffer
    /// monitor, persists the playback position to the DB, and logs stream-
    /// selection changes.
    pub async fn progress(
        &self,
        db: &sqlx::SqlitePool,
        user: &db::User,
        psid: &str,
        data: &PlaybackInfo,
    ) -> anyhow::Result<()> {
        let ps_snapshot = self.get(psid);
        let ps = match ps_snapshot.as_ref() {
            Some(ps) => ps,
            None => return Ok(()),
        };

        // If the session has no valid item and the progress report can't supply one,
        // it's a ghost (e.g. an unclaimed transcode stub). Evict it instead of
        // keeping it alive via last_activity.
        if ps
            .item_id
            .is_nil()
            && data
                .item_id
                .is_nil()
        {
            self.stop(psid, StopMode::Detached)
                .await;
            return Ok(());
        }

        let item_id = if !data
            .item_id
            .is_nil()
        {
            data.item_id
        } else {
            ps.item_id
        };
        // A method recorded by a stream endpoint reflects what is actually
        // served; clients (jellyfin-web) re-send theirs on every report.
        let client_play_method = if ps
            .served
            .is_some()
        {
            None
        } else {
            constrain_client_play_method(
                db,
                user,
                data.play_method
                    .clone(),
            )
            .await
        };

        // Detect encode-parameter changes and log them once.
        // We ignore pause/unpause — those are not encode changes.
        let audio_changed = data
            .audio_stream_index
            .is_some()
            && data.audio_stream_index != ps.audio_stream_index;
        let subtitle_changed = data
            .subtitle_stream_index
            .is_some()
            && data.subtitle_stream_index != ps.subtitle_stream_index;
        let method_changed = client_play_method.is_some()
            && client_play_method
                .as_ref()
                .map(|m| m.to_string())
                != ps.play_method;
        if audio_changed || subtitle_changed || method_changed {
            info!(
                play_session_id = psid.trim_start_matches("audio-").trim_start_matches("video-"),
                item_id = %item_id,
                user = %user.username,
                audio_stream = if audio_changed {
                    format!("{:?} → {:?}", ps.audio_stream_index, data.audio_stream_index)
                } else {
                    format!("{:?}", ps.audio_stream_index)
                },
                subtitle_stream = if subtitle_changed {
                    format!("{:?} → {:?}", ps.subtitle_stream_index, data.subtitle_stream_index)
                } else {
                    format!("{:?}", ps.subtitle_stream_index)
                },
                play_method = if method_changed {
                    format!("{:?} → {:?}", ps.play_method, client_play_method)
                } else {
                    format!("{:?}", ps.play_method)
                },
                "⟳ Playback params changed"
            );
        }

        self.update(psid, |ps| {
            if !data
                .item_id
                .is_nil()
            {
                ps.item_id = data.item_id;
            }
            ps.position_ticks = data
                .position_ticks
                .unwrap_or(ps.position_ticks);
            if data.is_paused && !ps.is_paused {
                ps.last_paused_at = Some(Utc::now());
            } else if !data.is_paused {
                ps.last_paused_at = None;
            }
            ps.is_paused = data.is_paused;
            ps.is_muted = data.is_muted;
            ps.volume_level = data
                .volume_level
                .or(ps.volume_level);
            ps.audio_stream_index = data
                .audio_stream_index
                .or(ps.audio_stream_index);
            ps.subtitle_stream_index = data
                .subtitle_stream_index
                .or(ps.subtitle_stream_index);
            // Re-check under the update: a stream endpoint may have recorded
            // a method since the snapshot.
            if let Some(ref method) = client_play_method {
                if ps
                    .served
                    .is_none()
                {
                    ps.play_method = Some(method.to_string());
                }
            }
            ps.last_activity = Utc::now();
            ps.last_report = Some(std::time::Instant::now());
        });

        // Update transcode buffer monitor with actual playback position.
        if let Some(position_ticks) = data.position_ticks {
            if let Some(ref ts_lock) = ps.transcode {
                if let Ok(ts) = ts_lock.try_read() {
                    let position_secs = (position_ticks / 10_000_000) as u32;
                    let offset = position_secs.saturating_sub(ts.start_time_secs);
                    ts.playback_offset_secs
                        .store(offset, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // Persist position to DB (no watched-threshold check on progress).
        let position_ticks = data
            .position_ticks
            .unwrap_or(ps.position_ticks);
        if let Ok(Some(media)) = db::Media::get_by_id(db, &item_id).await {
            let cfg = user
                .configuration
                .as_ref()
                .map(|c| {
                    c.0.clone()
                })
                .unwrap_or_default();

            let audio_idx = if cfg.remember_audio_selections {
                data.audio_stream_index
                    .or(ps.audio_stream_index)
                    .map(|x| x as i64)
            } else {
                None
            };
            let subtitle_idx = if cfg.remember_subtitle_selections {
                data.subtitle_stream_index
                    .or(ps.subtitle_stream_index)
                    .map(|x| x as i64)
            } else {
                None
            };

            db::UserMediaState::update_playback(
                db,
                user,
                &media,
                position_ticks,
                audio_idx,
                subtitle_idx,
                media.runtime,
            )
            .await?;
        }

        Ok(())
    }

    /// Handle a `POST /sessions/playing/stopped` report.
    ///
    /// Removes the playback session (stopping any active transcode), persists
    /// the final position to the DB (with the 90 % watched-mark check), and
    /// emits an info log line.
    ///
    /// Returns whether this stop crossed the watched threshold.
    pub async fn stopped(
        &self,
        db: &sqlx::SqlitePool,
        user: &db::User,
        psid: &str,
        data: &PlaybackInfo,
    ) -> anyhow::Result<bool> {
        // Wait out a transcode being created for this session, so the stop
        // finds it.
        let create = TRANSCODE_CREATE_LOCKS
            .lock_if_exists(&psid.to_string())
            .await;
        let ps = self
            .stop(psid, StopMode::Detached)
            .await;
        drop(create);

        let item_id = Some(data.item_id)
            .filter(|id| !id.is_nil())
            .or_else(|| {
                ps.as_ref()
                    .map(|s| s.item_id)
            });
        let final_ticks = data
            .position_ticks
            .or_else(|| {
                ps.as_ref()
                    .map(|s| s.position_ticks)
            });

        let played = Self::persist_stop(db, user, item_id, final_ticks).await?;

        let position_source = if data
            .position_ticks
            .is_some()
        {
            "report"
        } else if ps.is_some() {
            "session"
        } else {
            "missing"
        };
        info!(
            play_session_id = psid,
            position_source,
            position_ticks = ?final_ticks,
            position_available = final_ticks.is_some(),
            "Playback stopped"
        );
        Ok(played)
    }

    /// Persist the final position of a stop report (with the 90 % watched-mark
    /// check) for `item_id`, if it names a known media row. A stop without an
    /// authoritative position is a no-op so it cannot overwrite a saved
    /// position with zero.
    ///
    /// Shared by session-backed stops and stop reports that arrive without any
    /// play session: some clients never call
    /// `/sessions/playing` or `/sessions/playing/progress` and their only
    /// report is the stop, carrying the item id and position but no
    /// `PlaySessionId`. Jellyfin persists that report regardless of session
    /// state, so dropping it would lose Continue Watching for those clients.
    ///
    /// Returns whether this stop crossed the watched threshold.
    pub async fn persist_stop(
        db: &sqlx::SqlitePool,
        user: &db::User,
        item_id: Option<uuid::Uuid>,
        final_ticks: Option<i64>,
    ) -> anyhow::Result<bool> {
        let Some(item_id) = item_id else {
            return Ok(false);
        };
        let Some(final_ticks) = final_ticks else {
            return Ok(false);
        };
        let Ok(Some(media)) = db::Media::get_by_id(db, &item_id).await else {
            return Ok(false);
        };
        db::UserMediaState::update_playback(
            db,
            user,
            &media,
            final_ticks,
            None, // don't overwrite stream selections on stop
            None,
            media.runtime, // Some(runtime) triggers watched-threshold check
        )
        .await
    }

    /// Read the last persisted playback position for an item. This is used
    /// when a stop arrives after the in-memory session has been evicted and
    /// the client omitted its final position.
    pub async fn persisted_position_ticks(
        db: &sqlx::SqlitePool,
        user: &db::User,
        item_id: uuid::Uuid,
    ) -> Option<i64> {
        let media = db::Media::get_by_id(db, &item_id)
            .await
            .ok()??;
        let state = db::UserMediaState::get_or_new(db, user, &media)
            .await
            .ok()?;
        Some(
            state
                .playback_position
                .saturating_mul(10_000_000),
        )
    }

    /// Insert (or replace) a playback session, preserving any transcode that was
    /// pre-attached before `report_playback_start` fired.
    /// Removes stale sessions for the same device so `get_sessions` always
    /// finds the most recent playback.
    pub async fn insert(&self, mut session: PlaybackSession) {
        // Remove any previous session for this device (different play_session_id).
        // Unreported transcode stubs are left alone: a start report without
        // (or with another) PlaySessionId must not kill the stream it reports.
        if !session
            .device_id
            .is_empty()
        {
            let stale: Vec<String> = self
                .sessions
                .iter()
                .filter(|e| {
                    e.value()
                        .device_id
                        == session.device_id
                        && e.value()
                            .is_reported()
                        && e.key() != &session.play_session_id
                })
                .map(|e| {
                    e.key()
                        .clone()
                })
                .collect();
            for id in stale {
                self.stop(&id, StopMode::Detached)
                    .await;
            }
        }
        let _handoff = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        if let Some((_, (_, pending))) = self
            .pending_torrents
            .remove(&session.play_session_id)
        {
            retain_leases(&mut session.torrents, pending);
        }
        if let Some(existing) = self
            .sessions
            .get(&session.play_session_id)
        {
            if session
                .served
                .is_none()
            {
                session.served = existing
                    .served
                    .clone();
                if session
                    .served
                    .is_some()
                {
                    session.play_method = existing
                        .play_method
                        .clone();
                }
            }
            retain_leases(
                &mut session.torrents,
                existing
                    .torrents
                    .clone(),
            );
        }
        if let Some((_, (_, served))) = self
            .pending_play_methods
            .remove(&session.play_session_id)
        {
            session.play_method = Some(
                served
                    .play_method()
                    .to_string(),
            );
            session.served = Some(served);
        }
        // Take a pre-attached transcode while replacing the entry, so one
        // attached during the awaits above is not dropped with its ffmpeg.
        let entry = self
            .sessions
            .entry(
                session
                    .play_session_id
                    .clone(),
            );
        if let dashmap::mapref::entry::Entry::Occupied(existing) = &entry
            && session
                .transcode
                .is_none()
        {
            session.transcode = existing
                .get()
                .transcode
                .clone();
        }
        entry.insert(session);
    }

    /// Return a clone of the session, if it exists.
    pub fn get(&self, id: &str) -> Option<PlaybackSession> {
        self.sessions
            .get(id)
            .map(|e| {
                e.value()
                    .clone()
            })
    }

    /// Return a clone of the transcode session attached to this playback session.
    pub fn get_transcode(
        &self,
        id: &str,
    ) -> Option<Arc<tokio::sync::RwLock<TranscodeSession>>> {
        self.sessions
            .get(id)?
            .value()
            .transcode
            .clone()
    }

    /// Return clones of all active sessions.
    pub fn get_all(&self) -> Vec<PlaybackSession> {
        self.sessions
            .iter()
            .map(|e| {
                e.value()
                    .clone()
            })
            .collect()
    }

    /// Return the most recently active reported session for a device.
    /// Used as a fallback when the client omits PlaySessionId (e.g. DirectPlay).
    pub fn get_by_device(&self, device_id: &str) -> Option<PlaybackSession> {
        self.sessions
            .iter()
            .filter(|e| {
                e.value()
                    .device_id
                    == device_id
                    && e.value()
                        .is_reported()
            })
            .max_by_key(|e| {
                e.value()
                    .last_activity
            })
            .map(|e| {
                e.value()
                    .clone()
            })
    }

    /// Count active sessions for a user, optionally excluding sessions from a
    /// specific device. Excluding the caller's device is correct when checking
    /// before `insert()`, since insert() replaces any existing session for that
    /// device and it shouldn't count toward the limit.
    pub fn count_for_user(&self, user_id: Uuid, exclude_device: Option<&str>) -> usize {
        self.sessions
            .iter()
            .filter(|e| {
                let s = e.value();
                s.user_id == user_id
                    && exclude_device.map_or(true, |d| s.device_id != d)
            })
            .count()
    }

    /// Update a session in-place via a closure.
    pub fn update<F: FnOnce(&mut PlaybackSession)>(&self, id: &str, f: F) {
        if let Some(mut entry) = self
            .sessions
            .get_mut(id)
        {
            f(entry.value_mut());
        }
    }

    /// Record the playback method selected by a server stream endpoint.
    /// Update a claimed client session immediately; if no session exists or
    /// only an unclaimed HLS stub exists, keep the method pending for `start`.
    pub fn record_server_play_method(&self, id: &str, served: ServedPlayback) {
        let _handoff = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        // HLS may attach an unclaimed transcode stub (nil user/item) before
        // the playback-start report; keep the method pending for that too.
        let claimed = self
            .sessions
            .get_mut(id)
            .is_some_and(|mut session| {
                session.play_method = Some(
                    served
                        .play_method()
                        .to_string(),
                );
                session.served = Some(served.clone());
                !(session
                    .user_id
                    .is_nil()
                    && session
                        .item_id
                        .is_nil())
            });
        if claimed {
            self.pending_play_methods
                .remove(id);
        } else {
            self.pending_play_methods
                .insert(id.to_string(), (Utc::now(), served));
        }
    }

    /// Update `last_activity` on the session.
    pub fn ping(&self, id: &str) {
        self.update(id, |s| s.last_activity = Utc::now());
    }

    /// The client's playback ping: counts as a report (see `last_report`).
    pub fn ping_reported(&self, id: &str) {
        self.update(id, |s| {
            s.last_activity = Utc::now();
            s.last_report = Some(std::time::Instant::now());
        });
    }

    /// Attach a transcode session. If no playback session exists yet (the client
    /// calls master.m3u8 before POST /sessions/playing), a stub is inserted so the
    /// transcode isn't lost. `insert` will later overwrite the stub fields while
    /// preserving the transcode.
    ///
    /// The stub records the requesting device. Returns false, attaching
    /// nothing, once shutdown has begun (`stop_all`).
    #[must_use]
    pub fn attach_transcode(
        &self,
        id: &str,
        device_id: &str,
        ts: Arc<tokio::sync::RwLock<TranscodeSession>>,
    ) -> bool {
        let closed = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        if *closed {
            return false;
        }
        if let Some(mut entry) = self
            .sessions
            .get_mut(id)
        {
            entry
                .value_mut()
                .transcode = Some(ts);
        } else {
            self.sessions
                .insert(
                    id.to_string(),
                    PlaybackSession {
                        play_session_id: id.to_string(),
                        transcode: Some(ts),
                        user_id: Uuid::nil(),
                        item_id: Uuid::nil(),
                        media_source_id: None,
                        device_id: device_id.to_string(),
                        client_name: String::new(),
                        position_ticks: 0,
                        can_seek: true,
                        is_paused: false,
                        last_paused_at: None,
                        is_muted: false,
                        volume_level: None,
                        audio_stream_index: None,
                        subtitle_stream_index: None,
                        play_method: None,
                        served: None,
                        now_playing_queue: None,
                        playlist_item_id: None,
                        started_at: Utc::now(),
                        last_activity: Utc::now(),
                        last_report: None,
                        group_id: None,
                        item_kind: None,
                        torrents: Vec::new(),
                    },
                );
        }
        true
    }

    /// Stop transcodes their client left for a newer play session of the
    /// same item: a retry or stream switch under a new PlaySessionId, often
    /// without DELETE /Videos/ActiveEncodings. Transcodes are grouped by
    /// device, user and item. One is stopped once its client has neither
    /// requested it (`TranscodeSession::left_for`) nor reported it played
    /// (`PlaybackSession::still_played`) since the newest other of its group
    /// was created, and it has been quiet for twice its longest segment
    /// (`TranscodeSession::abandoned`).
    ///
    /// Browser tabs share a DeviceId; a playing tab requests after the other
    /// started, and a paused one keeps reporting.
    pub async fn supersede_abandoned_transcodes(&self, now: std::time::Instant) {
        let attached: Vec<_> = self
            .sessions
            .iter()
            .filter_map(|e| {
                let s = e.value();
                if s.device_id
                    .is_empty()
                {
                    return None;
                }
                Some((
                    e.key()
                        .clone(),
                    s.device_id
                        .clone(),
                    s.transcode
                        .clone()?,
                ))
            })
            .collect();
        let mut groups: HashMap<_, Vec<_>> = HashMap::new();
        for (id, device_id, ts) in attached {
            let (key, created_at) = {
                let s = ts
                    .read()
                    .await;
                ((device_id, s.user_id, s.item_id), s.created_at)
            };
            groups
                .entry(key)
                .or_default()
                .push((id, ts, created_at));
        }
        for group in groups.into_values() {
            for (id, ts, _) in &group {
                let Some(newest_other) = group
                    .iter()
                    .filter(|(other, _, _)| other != id)
                    .map(|(_, _, created_at)| *created_at)
                    .max()
                else {
                    continue;
                };
                let playlist = {
                    let s = ts
                        .read()
                        .await;
                    if !s.left_for(newest_other) {
                        continue;
                    }
                    s.variant_playlist_path()
                };
                let longest = tokio::fs::read_to_string(playlist)
                    .await
                    .map(|playlist| longest_segment(&playlist))
                    .unwrap_or_default();
                if ts
                    .read()
                    .await
                    .abandoned(now, longest)
                {
                    self.stop_superseded(id, ts, newest_other)
                        .await;
                }
            }
        }
    }

    /// Stop `ts` unless a report since `newest_other` was created claims it
    /// or `id` no longer holds it (a seek restarted the transcode), both
    /// checked in the same map access as the take. An unreported stub goes
    /// entirely; a reported session keeps its playback state.
    async fn stop_superseded(
        &self,
        id: &str,
        ts: &Arc<tokio::sync::RwLock<TranscodeSession>>,
        newest_other: std::time::Instant,
    ) {
        let holds = |s: &PlaybackSession| {
            !s.still_played(newest_other)
                && s.transcode
                    .as_ref()
                    .is_some_and(|t| Arc::ptr_eq(t, ts))
        };
        let handoff = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        let taken = if let Some((_, stub)) = self
            .sessions
            .remove_if(id, |_, s| !s.is_reported() && holds(s))
        {
            self.pending_torrents
                .remove(id);
            self.pending_play_methods
                .remove(id);
            stub.transcode
        } else {
            self.sessions
                .get_mut(id)
                .filter(|e| holds(e.value()))
                .and_then(|mut e| {
                    e.value_mut()
                        .transcode
                        .take()
                })
        };
        drop(handoff);
        let Some(ts) = taken else {
            return;
        };
        info!(
            play_session_id = %id,
            "Stopping transcode abandoned for a newer play session of the item"
        );
        self.kill_transcode(ts, StopMode::Detached)
            .await;
    }

    /// Stop and remove the transcode from a session, keeping the session.
    pub async fn stop_transcode(&self, id: &str, mode: StopMode) {
        let ts = self
            .sessions
            .get_mut(id)
            .and_then(|mut e| {
                e.value_mut()
                    .transcode
                    .take()
            });
        if let Some(ts) = ts {
            self.kill_transcode(ts, mode)
                .await;
        }
    }

    /// Stop the transcode (if any) and remove the playback session entirely.
    /// Returns the removed session so callers can read final position/item data.
    pub async fn stop(&self, id: &str, mode: StopMode) -> Option<PlaybackSession> {
        // Also release references captured before a playback-start report.
        let _handoff = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        self.pending_torrents
            .remove(id);
        self.pending_play_methods
            .remove(id);
        let (_, session) = self
            .sessions
            .remove(id)?;
        drop(_handoff);
        if let Some(ts) = session
            .transcode
            .clone()
        {
            self.kill_transcode(ts, mode)
                .await;
        }
        Some(session)
    }

    /// Capture the source selected by the stream handler, never resolve it
    /// again at stop time (a group's candidates may have changed by then).
    ///
    /// Once playback has started, the lease belongs directly to its session.
    /// Before that report arrives, it is held briefly and handed off by insert.
    pub async fn retain_torrent(
        &self,
        id: &str,
        torrent: &Arc<crate::torrent::TorrentManager>,
        hash: &str,
    ) {
        let lease = torrent
            .acquire(hash)
            .await;
        let _handoff = self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned");
        if let Some(mut session) = self
            .sessions
            .get_mut(id)
        {
            retain_lease(&mut session.torrents, lease);
        } else {
            let mut entry = self
                .pending_torrents
                .entry(id.to_string())
                .or_insert_with(|| (Utc::now(), Vec::new()));
            entry.0 = Utc::now();
            retain_lease(&mut entry.1, lease);
        }
    }

    /// Path where a given HLS segment lives on disk (used for disk-based recovery).
    pub fn segment_path(&self, play_session_id: &str, segment_id: &str) -> PathBuf {
        let session_dir = self
            .base_dir
            .join(play_session_id);

        if let Ok(entries) = std::fs::read_dir(&session_dir) {
            let mut latest_dir: Option<(PathBuf, std::time::SystemTime)> = None;
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_dir() {
                        let modified = metadata
                            .modified()
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        if latest_dir
                            .as_ref()
                            .map_or(true, |(_, max)| modified > *max)
                        {
                            latest_dir = Some((entry.path(), modified));
                        }
                    }
                }
            }
            if let Some((dir, _)) = latest_dir {
                return dir.join(format!("{}.ts", segment_id));
            }
        }

        session_dir.join(format!("{}.ts", segment_id))
    }

    pub fn base_dir(&self) -> &std::path::Path {
        &self.base_dir
    }

    pub fn active_session_ids(&self) -> Vec<String> {
        self.sessions
            .iter()
            .map(|e| {
                e.key()
                    .clone()
            })
            .collect()
    }

    /// Server shutdown: refuse new transcodes, stop every session, then
    /// remove all transcode dirs, within `STOP_WAIT` in total.
    pub async fn stop_all(&self) {
        let ids = {
            let mut closed = self
                .handoff
                .lock()
                .expect("handoff lock is not poisoned");
            *closed = true;
            self.active_session_ids()
        };
        let base_dir = self
            .base_dir
            .clone();
        let stop = async {
            futures::future::join_all(
                ids.iter()
                    .map(|id| self.stop(id, StopMode::Reaped)),
            )
            .await;
            // Also the dirs of transcodes stopped earlier, whose background
            // removal the exit would cut short.
            let _ = tokio::task::spawn_blocking(move || {
                for entry in std::fs::read_dir(base_dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            })
            .await;
        };
        if tokio::time::timeout(STOP_WAIT, stop)
            .await
            .is_err()
        {
            warn!("timed out stopping playback sessions");
        }
    }

    /// Undo `stop_all`'s refusal of new transcodes, for a process that keeps
    /// serving after all (a restart that failed to spawn its successor).
    pub fn reopen(&self) {
        *self
            .handoff
            .lock()
            .expect("handoff lock is not poisoned") = false;
    }

    /// Spawn a background task that stops abandoned transcodes every
    /// `interval` (see `supersede_abandoned_transcodes`).
    pub fn spawn_supersede_task(self, interval: Duration) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker
                    .tick()
                    .await;
                self.supersede_abandoned_transcodes(std::time::Instant::now())
                    .await;
            }
        })
    }

    /// Spawn a background task that reaps sessions idle longer than `max_age`.
    pub fn spawn_cleanup_task(
        self,
        interval: Duration,
        max_age: Duration,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker
                    .tick()
                    .await;
                let cutoff = Utc::now()
                    - chrono::Duration::from_std(max_age).unwrap_or_default();
                let stale: Vec<String> = self
                    .sessions
                    .iter()
                    .filter(|e| {
                        e.value()
                            .last_activity
                            < cutoff
                    })
                    .map(|e| {
                        e.key()
                            .clone()
                    })
                    .collect();
                for id in stale {
                    info!("Cleaning up idle session: {}", id);
                    self.stop(&id, StopMode::Detached)
                        .await;
                }
                // Requests that never received a playback-start report must
                // not retain a torrent forever. Active responses still own
                // their own reference when this pending entry expires.
                let _handoff = self
                    .handoff
                    .lock()
                    .expect("handoff lock is not poisoned");
                self.pending_torrents
                    .retain(|_, (seen, _)| *seen >= cutoff);
                self.pending_play_methods
                    .retain(|_, (seen, _)| *seen >= cutoff);
            }
        })
    }

    /// Mark a transcode stopped and tell ffmpeg to die. Unless `Reaped`, the
    /// wait for its reap (seconds if a thread is in ext4 writeback) and its
    /// dir removal run detached.
    async fn kill_transcode(
        &self,
        ts: Arc<tokio::sync::RwLock<TranscodeSession>>,
        mode: StopMode,
    ) {
        let (id, kill_tx, wait_done, output_dir) = {
            let mut s = ts
                .write()
                .await;
            s.stopped
                .cancel();
            (
                s.id.clone(),
                s.kill_tx
                    .take(),
                s.wait_done
                    .clone(),
                s.output_dir
                    .clone(),
            )
        };
        // A failed send means that run's ffmpeg is already gone.
        let killed = kill_tx.is_some_and(|kill_tx| {
            kill_tx
                .send(())
                .is_ok()
        });
        // Waits for the reap however long it takes: reusing or removing the
        // dir while ffmpeg still writes to it would leave it half removed.
        let reaped = async move {
            if killed {
                let notified = wait_done.notified();
                tokio::pin!(notified);
                if tokio::time::timeout(STOP_WAIT, &mut notified)
                    .await
                    .is_err()
                {
                    warn!(play_session_id = %id, "transcode did not stop in time");
                    notified.await;
                }
            }
            id
        };
        if matches!(mode, StopMode::Reaped) {
            reaped.await;
            return;
        }
        let sessions = self.clone();
        tokio::spawn(async move {
            let id = reaped.await;
            // Under the create lock, so a transcode the play session starts
            // again cannot be attached to the dir while it is removed.
            let _create = TRANSCODE_CREATE_LOCKS
                .lock(id.clone())
                .await;
            if sessions
                .get_transcode(&id)
                .is_none()
            {
                let _ = tokio::task::spawn_blocking(move || {
                    std::fs::remove_dir_all(output_dir)
                })
                .await;
            }
        });
    }
}

fn retain_leases(
    target: &mut Vec<Arc<crate::torrent::TorrentLease>>,
    leases: Vec<Arc<crate::torrent::TorrentLease>>,
) {
    for lease in leases {
        retain_lease(target, lease);
    }
}

fn retain_lease(
    target: &mut Vec<Arc<crate::torrent::TorrentLease>>,
    lease: Arc<crate::torrent::TorrentLease>,
) {
    if !target
        .iter()
        .any(|existing| Arc::ptr_eq(existing, &lease))
    {
        target.push(lease);
    }
}

/// Clamp a client-reported play method to what this user's permissions allow,
/// so a session never records processing the server would refuse to do.
async fn constrain_client_play_method(
    db: &sqlx::SqlitePool,
    user: &db::User,
    method: Option<PlayMethod>,
) -> Option<PlayMethod> {
    let method = method?;
    let encoding = db::Settings::get_encoding_config(db)
        .await
        .unwrap_or_default();
    Some(
        PlaybackPermissions::for_user(&encoding, Some(user))
            .constrain_reported_method(method),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_track_names_the_codec_the_client_receives() {
        assert_eq!(
            FfmpegTrack::new("copy", Some("hevc")),
            FfmpegTrack {
                codec: Some("hevc".to_string()),
                copied: true,
            }
        );
        assert_eq!(
            FfmpegTrack::new("h264", Some("hevc")),
            FfmpegTrack {
                codec: Some("h264".to_string()),
                copied: false,
            }
        );
        // No source video: nothing to name, so clients see audio-only output.
        assert_eq!(
            FfmpegTrack::new("copy", None),
            FfmpegTrack {
                codec: None,
                copied: true,
            }
        );
        // A remux is still FFmpeg output, but its video is the source's.
        let remux = ServedPlayback::Ffmpeg {
            video: FfmpegTrack::new("copy", Some("hevc")),
            audio: FfmpegTrack::new("copy", Some("eac3")),
        };
        assert_eq!(remux.play_method(), PlayMethod::Transcode);
        assert!(remux.video_copied());
    }

    fn transcoded() -> ServedPlayback {
        ServedPlayback::Ffmpeg {
            video: FfmpegTrack::new("h264", Some("hevc")),
            audio: FfmpegTrack::new("aac", Some("dts")),
        }
    }

    fn playback_session(id: &str, method: PlayMethod) -> PlaybackSession {
        PlaybackSession {
            play_session_id: id.to_string(),
            user_id: Uuid::nil(),
            item_id: Uuid::new_v4(),
            media_source_id: None,
            device_id: String::new(),
            client_name: String::new(),
            position_ticks: 0,
            can_seek: true,
            is_paused: false,
            last_paused_at: None,
            is_muted: false,
            volume_level: None,
            audio_stream_index: None,
            subtitle_stream_index: None,
            play_method: Some(method.to_string()),
            served: None,
            now_playing_queue: None,
            playlist_item_id: None,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            last_report: None,
            transcode: None,
            group_id: None,
            item_kind: None,
            torrents: Vec::new(),
        }
    }

    #[tokio::test]
    async fn play_method_handoff_works_when_server_records_first() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let id = "server-first";

        sessions.record_server_play_method(id, transcoded());
        sessions
            .insert(playback_session(id, PlayMethod::DirectPlay))
            .await;

        assert_eq!(
            sessions
                .get(id)
                .and_then(|session| session.play_method),
            Some(PlayMethod::Transcode.to_string())
        );
        assert!(
            sessions
                .get(id)
                .is_some_and(|session| session.served == Some(transcoded()))
        );
        assert!(
            !sessions
                .pending_play_methods
                .contains_key(id)
        );

        sessions
            .insert(playback_session(id, PlayMethod::DirectPlay))
            .await;
        assert_eq!(
            sessions
                .get(id)
                .and_then(|session| session.play_method),
            Some(PlayMethod::Transcode.to_string())
        );
        assert!(
            sessions
                .get(id)
                .is_some_and(|session| session.served == Some(transcoded()))
        );
    }

    #[tokio::test]
    async fn play_method_handoff_works_when_client_records_first() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let id = "client-first";

        sessions
            .insert(playback_session(id, PlayMethod::DirectPlay))
            .await;
        sessions.record_server_play_method(id, ServedPlayback::Direct);
        sessions.record_server_play_method(id, transcoded());

        assert_eq!(
            sessions
                .get(id)
                .and_then(|session| session.play_method),
            Some(PlayMethod::Transcode.to_string())
        );
        assert!(
            sessions
                .get(id)
                .is_some_and(|session| session.served == Some(transcoded()))
        );
        assert!(
            !sessions
                .pending_play_methods
                .contains_key(id)
        );
    }

    #[tokio::test]
    async fn play_method_handoff_survives_an_unclaimed_transcode_stub() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let id = "transcode-stub-first";
        let mut stub = playback_session(id, PlayMethod::DirectPlay);
        stub.user_id = Uuid::nil();
        stub.item_id = Uuid::nil();

        sessions
            .insert(stub)
            .await;
        sessions.record_server_play_method(id, transcoded());
        assert!(
            sessions
                .pending_play_methods
                .contains_key(id)
        );

        sessions
            .insert(playback_session(id, PlayMethod::DirectPlay))
            .await;
        assert_eq!(
            sessions
                .get(id)
                .and_then(|session| session.play_method),
            Some(PlayMethod::Transcode.to_string())
        );
        assert!(
            !sessions
                .pending_play_methods
                .contains_key(id)
        );
    }

    /// A stopped transcode's dir is removed by a background task.
    async fn removed(path: PathBuf) -> bool {
        for _ in 0..500 {
            if !path.exists() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    fn transcode_of(
        base: &std::path::Path,
        id: &str,
        user_id: Uuid,
        item_id: Uuid,
    ) -> Arc<tokio::sync::RwLock<TranscodeSession>> {
        TranscodeSession::new(
            id.to_string(),
            user_id,
            item_id,
            item_id,
            String::new(),
            base.join(id),
            "copy".to_string(),
            "aac".to_string(),
            None,
            None,
            false,
            6,
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
        )
    }

    /// When a sweep checks: long after the transcodes of a test were created
    /// (tests can't advance `Instant`).
    fn later() -> std::time::Instant {
        std::time::Instant::now() + Duration::from_secs(60)
    }

    /// `ts`'s client last requested a segment or playlist `at`.
    async fn requested(
        ts: &Arc<tokio::sync::RwLock<TranscodeSession>>,
        at: std::time::Instant,
    ) {
        ts.read()
            .await
            .requests
            .requested_at(at);
    }

    /// `ts` was created `at`.
    async fn created(
        ts: &Arc<tokio::sync::RwLock<TranscodeSession>>,
        at: std::time::Instant,
    ) {
        ts.write()
            .await
            .created_at = at;
    }

    async fn stopped(ts: &Arc<tokio::sync::RwLock<TranscodeSession>>) -> bool {
        ts.read()
            .await
            .stopped
            .is_cancelled()
    }

    #[tokio::test]
    async fn a_retry_supersedes_the_transcodes_it_abandoned() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let now = later();

        // The first attempt was reported as started, and reported progress
        // until the client retried; its retries never were.
        let mut reported = playback_session("reported", PlayMethod::Transcode);
        reported.user_id = user;
        reported.device_id = "phone".to_string();
        reported.last_report = Some(now - Duration::from_secs(25));
        sessions
            .insert(reported)
            .await;
        let reported_ts = transcode_of(temp.path(), "reported", user, item);
        created(&reported_ts, now - Duration::from_secs(30)).await;
        requested(&reported_ts, now - Duration::from_secs(25)).await;
        assert!(sessions.attach_transcode("reported", "phone", reported_ts.clone()));
        // Never requested: the client failed on its master playlist.
        let abandoned = transcode_of(temp.path(), "retry-1", user, item);
        created(&abandoned, now - Duration::from_secs(24)).await;
        assert!(sessions.attach_transcode("retry-1", "phone", abandoned.clone()));
        // The attempt the client plays.
        let playing = transcode_of(temp.path(), "retry-2", user, item);
        created(&playing, now - Duration::from_secs(20)).await;
        requested(&playing, now - Duration::from_secs(1)).await;
        assert!(sessions.attach_transcode("retry-2", "phone", playing));
        // Abandoned too, but alone in their groups: nothing replaced them.
        let kept = [
            // Another device on the item (live TV with several viewers).
            ("other-device", "tv", user, item),
            // Another user on the device and item.
            ("other-user", "phone", Uuid::new_v4(), item),
            // Another item on the device (a music client preloading).
            ("other-item", "phone", user, Uuid::new_v4()),
        ];
        for (id, device, kept_user, kept_item) in kept {
            let ts = transcode_of(temp.path(), id, kept_user, kept_item);
            assert!(sessions.attach_transcode(id, device, ts));
        }

        sessions
            .supersede_abandoned_transcodes(now)
            .await;

        // An abandoned stub goes entirely.
        assert!(
            sessions
                .get("retry-1")
                .is_none()
        );
        assert!(stopped(&abandoned).await);
        assert!(
            removed(
                temp.path()
                    .join("retry-1")
            )
            .await
        );
        // A reported session keeps its playback state, not its transcode.
        assert!(
            sessions
                .get("reported")
                .is_some()
        );
        assert!(stopped(&reported_ts).await);
        assert!(
            sessions
                .get_transcode("reported")
                .is_none()
        );
        for id in ["retry-2", "other-device", "other-user", "other-item"] {
            assert!(
                sessions
                    .get_transcode(id)
                    .is_some(),
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_paused_player_next_to_a_playing_one_keeps_its_transcode() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let now = later();
        // Browser tabs share their DeviceId. Tab 1 has been paused for 40 s
        // (requesting nothing with a full buffer) while tab 2, opened after
        // tab 1, plays.
        let tab1 = transcode_of(temp.path(), "tab-1", user, item);
        created(&tab1, now - Duration::from_secs(55)).await;
        assert!(sessions.attach_transcode("tab-1", "browser", tab1.clone()));
        let tab2 = transcode_of(temp.path(), "tab-2", user, item);
        created(&tab2, now - Duration::from_secs(50)).await;
        requested(&tab2, now - Duration::from_secs(1)).await;
        assert!(sessions.attach_transcode("tab-2", "browser", tab2.clone()));
        requested(&tab1, now - Duration::from_secs(40)).await;

        sessions
            .supersede_abandoned_transcodes(now)
            .await;

        for (id, ts) in [("tab-1", &tab1), ("tab-2", &tab2)] {
            assert!(
                sessions
                    .get_transcode(id)
                    .is_some(),
                "{id}"
            );
            assert!(!stopped(ts).await, "{id}");
        }
    }

    #[tokio::test]
    async fn a_paused_player_keeps_its_transcode_when_another_switches_streams() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let now = later();
        // Tab 1 reported a pause 45 s ago and has requested nothing since.
        let mut paused = playback_session("tab-1", PlayMethod::Transcode);
        paused.user_id = user;
        paused.item_id = item;
        paused.device_id = "browser".to_string();
        paused.is_paused = true;
        paused.last_report = Some(now - Duration::from_secs(45));
        sessions
            .insert(paused)
            .await;
        let tab1 = transcode_of(temp.path(), "tab-1", user, item);
        created(&tab1, now - Duration::from_secs(60)).await;
        requested(&tab1, now - Duration::from_secs(50)).await;
        assert!(sessions.attach_transcode("tab-1", "browser", tab1.clone()));
        // Then tab 2 switched its audio track: a new transcode, which tab 1
        // never requested after, as a retry's predecessor wouldn't either.
        let switched = transcode_of(temp.path(), "tab-2-switched", user, item);
        created(&switched, now - Duration::from_secs(40)).await;
        requested(&switched, now - Duration::from_secs(1)).await;
        assert!(sessions.attach_transcode("tab-2-switched", "browser", switched));

        sessions
            .supersede_abandoned_transcodes(now)
            .await;
        assert!(!stopped(&tab1).await);
        assert!(
            sessions
                .get_transcode("tab-1")
                .is_some()
        );

        // Not paused and not reported since, it is taken for a retry's
        // predecessor.
        sessions.update("tab-1", |s| s.is_paused = false);
        sessions
            .supersede_abandoned_transcodes(now)
            .await;
        assert!(stopped(&tab1).await);
        assert!(
            sessions
                .get("tab-1")
                .is_some(),
            "the reported session keeps its playback state"
        );
    }

    #[tokio::test]
    async fn a_resumed_player_keeps_its_transcode_until_it_requests_again() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let start = later();
        let at = |secs: u64| start + Duration::from_secs(secs);
        // Tab 1 pauses at 0 s with a full buffer, so requests nothing more.
        let mut paused = playback_session("tab-1", PlayMethod::Transcode);
        paused.user_id = user;
        paused.item_id = item;
        paused.device_id = "browser".to_string();
        paused.is_paused = true;
        paused.last_report = Some(at(0));
        sessions
            .insert(paused)
            .await;
        let tab1 = transcode_of(temp.path(), "tab-1", user, item);
        created(&tab1, start - Duration::from_secs(30)).await;
        requested(&tab1, at(0)).await;
        assert!(sessions.attach_transcode("tab-1", "browser", tab1.clone()));
        // Tab 2 switches its audio track at 10 s: a new transcode.
        let switched = transcode_of(temp.path(), "tab-2-switched", user, item);
        created(&switched, at(10)).await;
        assert!(sessions.attach_transcode(
            "tab-2-switched",
            "browser",
            switched.clone()
        ));

        // Sweeps every 2 s. Tab 1 resumes at 45 s (a report) and requests
        // its next segment 8 s later, when its buffer runs low.
        for t in (12..=70).step_by(2) {
            requested(&switched, at(t)).await;
            if t == 46 {
                sessions.update("tab-1", |s| {
                    s.is_paused = false;
                    s.last_report = Some(at(45));
                });
            }
            if t == 54 {
                requested(&tab1, at(53)).await;
            }
            sessions
                .supersede_abandoned_transcodes(at(t + 1))
                .await;
            assert!(!stopped(&tab1).await, "{t}");
        }
        assert!(
            sessions
                .get_transcode("tab-1")
                .is_some()
        );
    }

    #[tokio::test]
    async fn two_players_with_long_segments_keep_their_transcodes() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let start = later();
        let at = |secs: u64| start + Duration::from_secs(secs);
        // Two tabs copy a video with 16 s keyframe intervals, so each
        // requests a segment every 16 s although 6 s segments were asked
        // for; tab 2 opens 10 s after tab 1, while tab 1 waits 6 s more for
        // its next request.
        let tab1 = transcode_of(temp.path(), "tab-1", user, item);
        created(&tab1, at(0)).await;
        requested(&tab1, at(0)).await;
        assert!(sessions.attach_transcode("tab-1", "browser", tab1.clone()));
        let tab2 = transcode_of(temp.path(), "tab-2", user, item);
        created(&tab2, at(10)).await;
        requested(&tab2, at(10)).await;
        assert!(sessions.attach_transcode("tab-2", "browser", tab2.clone()));
        for id in ["tab-1", "tab-2"] {
            std::fs::write(
                temp.path()
                    .join(id)
                    .join("main.m3u8"),
                "#EXTM3U\n#EXTINF:16.016000,\nsegment_00000.ts\n",
            )
            .unwrap();
        }

        // Sweeps every 2 s for 150 s.
        for t in (10..=160).step_by(2) {
            if t % 16 == 0 {
                requested(&tab1, at(t)).await;
            }
            if t > 10 && (t - 10) % 16 == 0 {
                requested(&tab2, at(t)).await;
            }
            sessions
                .supersede_abandoned_transcodes(at(t + 1))
                .await;
        }

        for (id, ts) in [("tab-1", &tab1), ("tab-2", &tab2)] {
            assert!(!stopped(ts).await, "{id}");
        }
    }

    #[tokio::test]
    async fn of_transcodes_created_at_once_the_unplayed_one_is_superseded() {
        for played_id in ["a", "b"] {
            let temp = tempfile::tempdir().unwrap();
            let sessions = PlaybackSessionManager::new(temp.path());
            let user = Uuid::new_v4();
            let item = Uuid::new_v4();
            let now = later();
            // jellyfin-web retries twice at once (video error and hls.js
            // fatal error): two masters attach, in either order; the client
            // plays one.
            let mut transcodes = Vec::new();
            for id in ["a", "b"] {
                let ts = transcode_of(temp.path(), id, user, item);
                created(&ts, now - Duration::from_secs(20)).await;
                if id == played_id {
                    requested(&ts, now - Duration::from_secs(1)).await;
                }
                assert!(sessions.attach_transcode(id, "browser", ts.clone()));
                transcodes.push((id, ts));
            }

            sessions
                .supersede_abandoned_transcodes(now)
                .await;

            for (id, ts) in transcodes {
                let played = id == played_id;
                assert_eq!(
                    sessions
                        .get_transcode(id)
                        .is_some(),
                    played,
                    "{id}"
                );
                assert_eq!(stopped(&ts).await, !played, "{id}");
            }
        }
    }

    #[tokio::test]
    async fn a_retry_loop_keeps_only_the_attempts_requested_lately() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let now = later();
        // A client retrying every 2 s under a new PlaySessionId, without
        // DELETE /Videos/ActiveEncodings: each attempt is requested from its
        // creation until the next one is created, then never again.
        let ids: Vec<String> = (0..10)
            .map(|k| format!("attempt-{k}"))
            .collect();
        for (k, id) in ids
            .iter()
            .enumerate()
        {
            let ts = transcode_of(temp.path(), id, user, item);
            let last_request = now - Duration::from_secs(2 * k as u64);
            created(&ts, last_request - Duration::from_secs(1)).await;
            requested(&ts, last_request).await;
            assert!(sessions.attach_transcode(id, "phone", ts));
        }

        sessions
            .supersede_abandoned_transcodes(now)
            .await;

        // Abandoned after 12 s, twice the 6 s segment length: only the
        // attempts of the last 12 s remain.
        let remaining: Vec<_> = ids
            .iter()
            .filter(|id| {
                sessions
                    .get_transcode(id)
                    .is_some()
            })
            .collect();
        assert_eq!(
            remaining,
            ids[..6]
                .iter()
                .collect::<Vec<_>>()
        );
    }

    /// The play session `/Sessions` shows for `device` and the play session
    /// of the transcode it shows.
    async fn shown(
        sessions: &PlaybackSessionManager,
        device: &str,
    ) -> Option<(String, Option<String>)> {
        let all = sessions.get_all();
        let device_sessions: Vec<_> = all
            .iter()
            .filter(|s| s.device_id == device)
            .collect();
        let (ps, ts) = now_playing(&device_sessions).await?;
        let ts_id = match ts {
            Some(ts) => Some(
                ts.read()
                    .await
                    .id
                    .clone(),
            ),
            None => None,
        };
        Some((
            ps.play_session_id
                .clone(),
            ts_id,
        ))
    }

    #[tokio::test]
    async fn sessions_never_show_an_unreported_transcode_stub() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let stub = transcode_of(temp.path(), "stub", user, item);
        assert!(sessions.attach_transcode("stub", "phone", stub));

        // Before a start report nothing plays, as on Jellyfin.
        assert_eq!(shown(&sessions, "phone").await, None);

        // A direct play reported while the stub is still around.
        let mut direct = playback_session("direct", PlayMethod::DirectPlay);
        direct.user_id = user;
        direct.item_id = item;
        direct.device_id = "phone".to_string();
        sessions
            .insert(direct)
            .await;
        assert_eq!(
            shown(&sessions, "phone").await,
            Some(("direct".to_string(), None))
        );
    }

    #[tokio::test]
    async fn a_retry_without_a_start_report_shows_the_reported_item() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let mut reported = playback_session("p1", PlayMethod::Transcode);
        reported.user_id = user;
        reported.item_id = item;
        reported.device_id = "tv".to_string();
        sessions
            .insert(reported)
            .await;
        let p1 = transcode_of(temp.path(), "p1", user, item);
        assert!(sessions.attach_transcode("p1", "tv", p1));
        // The retry under a new PlaySessionId, which sends no start report,
        // and a stub of another item.
        let p2 = transcode_of(temp.path(), "p2", user, item);
        assert!(sessions.attach_transcode("p2", "tv", p2));
        let other = transcode_of(temp.path(), "other", user, Uuid::new_v4());
        assert!(sessions.attach_transcode("other", "tv", other));

        assert_eq!(
            shown(&sessions, "tv").await,
            Some(("p1".to_string(), Some("p1".to_string())))
        );
        // p1's transcode superseded: the retry's serves the reported item.
        sessions
            .stop_transcode("p1", StopMode::Detached)
            .await;
        assert_eq!(
            shown(&sessions, "tv").await,
            Some(("p1".to_string(), Some("p2".to_string())))
        );
        sessions
            .stop("p2", StopMode::Detached)
            .await;
        assert_eq!(shown(&sessions, "tv").await, Some(("p1".to_string(), None)));
    }

    #[tokio::test]
    async fn a_start_report_without_the_play_session_id_keeps_the_devices_stub() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode_of(temp.path(), "hls", Uuid::nil(), Uuid::new_v4());
        assert!(sessions.attach_transcode("hls", "phone", ts.clone()));

        // `start` makes up a PlaySessionId when the report has none.
        let mut report = playback_session("generated", PlayMethod::Transcode);
        report.user_id = Uuid::new_v4();
        report.device_id = "phone".to_string();
        sessions
            .insert(report)
            .await;

        assert!(
            sessions
                .get_transcode("hls")
                .is_some()
        );
        assert!(!stopped(&ts).await);
        assert_eq!(
            sessions
                .get_by_device("phone")
                .map(|s| s.play_session_id),
            Some("generated".to_string())
        );
    }

    #[tokio::test]
    async fn a_stop_before_the_first_spawn_prevents_it() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode_of(temp.path(), "early", Uuid::nil(), Uuid::new_v4());
        assert!(sessions.attach_transcode("early", "phone", ts.clone()));

        sessions
            .stop("early", StopMode::Detached)
            .await;
        crate::playback::engine::start_transcode(
            ts.clone(),
            crate::playback::engine::TranscodeParams {
                output_dir: temp
                    .path()
                    .join("early"),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(
            ts.read()
                .await
                .state,
            crate::playback::session::TranscodeState::Starting,
            "no ffmpeg was started"
        );
        assert!(
            removed(
                temp.path()
                    .join("early")
            )
            .await
        );
    }

    #[tokio::test]
    async fn a_stop_before_the_loop_spawns_ends_it() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode_of(temp.path(), "live", Uuid::nil(), Uuid::new_v4());
        assert!(sessions.attach_transcode("live", "tv", ts.clone()));
        let wait_done = ts
            .read()
            .await
            .wait_done
            .clone();
        crate::playback::engine::start_transcode(
            ts.clone(),
            crate::playback::engine::TranscodeParams {
                output_dir: temp
                    .path()
                    .join("live"),
                is_live: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // The loop task hasn't run yet (single-threaded runtime). Live, so a
        // spawned ffmpeg would restart instead of ending the loop.
        sessions
            .stop("live", StopMode::Detached)
            .await;
        tokio::time::timeout(Duration::from_secs(5), wait_done.notified())
            .await
            .expect("the loop ended without spawning ffmpeg");
        assert!(
            removed(
                temp.path()
                    .join("live")
            )
            .await
        );
    }

    #[tokio::test]
    async fn stop_all_stops_every_transcode_and_refuses_new_ones_until_reopened() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode_of(temp.path(), "running", Uuid::nil(), Uuid::new_v4());
        assert!(sessions.attach_transcode("running", "phone", ts.clone()));
        // A transcode stopped earlier whose background removal hasn't run.
        let earlier = temp
            .path()
            .join("earlier");
        std::fs::create_dir_all(&earlier).unwrap();

        sessions
            .stop_all()
            .await;

        assert!(stopped(&ts).await);
        assert!(
            sessions
                .get("running")
                .is_none()
        );
        assert!(
            !temp
                .path()
                .join("running")
                .exists()
        );
        assert!(!earlier.exists());
        let late = transcode_of(temp.path(), "late", Uuid::nil(), Uuid::new_v4());
        assert!(!sessions.attach_transcode("late", "phone", late));
        assert!(
            sessions
                .get("late")
                .is_none()
        );

        // As /System/Restart does when the new process fails to spawn.
        sessions.reopen();
        let ts = transcode_of(temp.path(), "after", Uuid::nil(), Uuid::new_v4());
        assert!(sessions.attach_transcode("after", "phone", ts));
    }

    #[tokio::test]
    async fn a_supersede_returns_at_once_and_cleans_up_after_the_reap() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let item = Uuid::new_v4();
        let now = later();
        let old = transcode_of(temp.path(), "old", Uuid::nil(), item);
        // A running ffmpeg whose loop never reports it reaped, like one with
        // a thread stuck in the kernel after SIGKILL.
        let (kill_tx, mut kill_rx) = tokio::sync::oneshot::channel();
        old.write()
            .await
            .kill_tx = Some(kill_tx);
        assert!(sessions.attach_transcode("old", "phone", old.clone()));
        let new = transcode_of(temp.path(), "new", Uuid::nil(), item);
        requested(&new, now).await;
        assert!(sessions.attach_transcode("new", "phone", new));

        tokio::time::timeout(
            Duration::from_secs(1),
            sessions.supersede_abandoned_transcodes(now),
        )
        .await
        .expect("supersede returned without waiting");

        assert!(
            kill_rx
                .try_recv()
                .is_ok(),
            "ffmpeg was told to die"
        );
        assert!(stopped(&old).await);
        assert!(
            sessions
                .get("old")
                .is_none()
        );

        // The dir stays while ffmpeg may still write to it...
        let dir = temp
            .path()
            .join("old");
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(dir.exists());
        // ...and goes once the loop reports ffmpeg reaped.
        old.read()
            .await
            .wait_done
            .notify_one();
        assert!(removed(dir).await);
    }

    #[tokio::test]
    async fn a_supersede_spares_the_transcode_a_seek_put_in_place() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let item = Uuid::new_v4();
        let abandoned = transcode_of(temp.path(), "seek", Uuid::nil(), item);
        assert!(sessions.attach_transcode("seek", "phone", abandoned.clone()));
        // A seek restarted the play session's transcode after the sweep
        // found the old one abandoned.
        let restarted = transcode_of(temp.path(), "seek", Uuid::nil(), item);
        assert!(sessions.attach_transcode("seek", "phone", restarted.clone()));

        sessions
            .stop_superseded("seek", &abandoned, std::time::Instant::now())
            .await;

        assert!(!stopped(&restarted).await);
        assert!(
            sessions
                .get_transcode("seek")
                .is_some_and(|ts| Arc::ptr_eq(&ts, &restarted))
        );
    }
}

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::{path::PathBuf, sync::Arc, time::Duration};
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

/// The concrete stream behind a playback report's source, which a reported
/// audio track index points into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayedStream {
    pub id: Uuid,
    /// The track PlaybackInfo had the stream play unless asked for another;
    /// `None` when no PlaybackInfo answer for the report is known.
    pub automatic_audio: Option<i64>,
}

/// Per play session: held while creating its transcode, so stops and dir
/// removal never miss one still being created.
pub(crate) static TRANSCODE_CREATE_LOCKS: crate::keyed_lock::KeyedLock<String> =
    crate::keyed_lock::KeyedLock::new();

/// How a stop treats the transcode's ffmpeg.
#[derive(Clone, Copy)]
pub enum StopMode {
    /// Return at once; ffmpeg is reaped and its dir removed in the background.
    Detached,
    /// Wait (bounded) for the reap: the caller reuses the dir or exits.
    Reaped,
}

#[derive(Clone)]
pub struct PlaybackSession {
    pub play_session_id: String,
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub media_source_id: Option<String>,
    /// Concrete stream PlaybackInfo selected behind `media_source_id`.
    pub played_stream: Option<PlayedStream>,
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
    /// When the start report came, on the clock transcode requests are
    /// timed with (see `end_left_playbacks`).
    reported_at: std::time::Instant,
    pub last_activity: DateTime<Utc>,
    /// When the client last reported this play session (start, progress,
    /// pause, resume, ping); `None` for a transcode stub.
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

    /// False for a transcode stub no playback report has claimed yet.
    pub(crate) fn is_reported(&self) -> bool {
        !self
            .user_id
            .is_nil()
    }

    /// Whether the client still plays this session's transcode in spite of
    /// one created `at` for its item on its device: it reported the session
    /// since, or last reported it paused (a paused player requests nothing,
    /// and reports again within 10-15 s). A retry's predecessor does
    /// neither: the client left it, reports included, for the retry.
    fn still_played(&self, at: std::time::Instant) -> bool {
        self.is_paused
            || self
                .last_report
                .is_some_and(|reported| reported >= at)
    }
}

/// Which of a device's playback `sessions` `/Sessions` shows, and the
/// transcode serving it. Reported sessions come first; an unreported
/// transcode stub has no item to show (Jellyfin shows none before the start
/// report) and is never shown. A reported transcode whose client retried
/// under a new PlaySessionId without a new start report is served by the
/// retry's stub.
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
        if !t.stopped && t.item_id == shown.item_id && t.user_id == shown.user_id {
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
    // while playback-start inserts its session. Its value is `closed`: set by
    // `stop_all` (shutdown), after which no transcode is attached.
    handoff: Arc<std::sync::Mutex<bool>>,
    /// The latest start report of each (device, user), kept after its
    /// playback stopped: `end_left_playbacks` measures the older playbacks
    /// against it. Pruned once the owner has no older playback left.
    latest_starts: Arc<DashMap<(String, Uuid), std::time::Instant>>,
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
            latest_starts: Arc::new(DashMap::new()),
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
        played_stream: Option<PlayedStream>,
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
            // Exclude the caller's own device: it counts as one stream, the
            // one starting now, whatever else it still has in flight.
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
            played_stream,
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
            reported_at: std::time::Instant::now(),
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
    /// selection changes. `reported_stream` is the stream PlaybackInfo
    /// selected for the report's source, used when the session has none.
    pub async fn progress(
        &self,
        db: &sqlx::SqlitePool,
        user: &db::User,
        psid: &str,
        data: &PlaybackInfo,
        reported_stream: Option<PlayedStream>,
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

            // Remembering off forgets the stored pick, as Jellyfin does, so
            // switching it back on does not revive a pick from before.
            let audio = if cfg.remember_audio_selections {
                data.audio_stream_index
                    .or(ps.audio_stream_index)
                    .map_or(db::TrackMemory::Keep, |x| {
                        let index = x as i64;
                        let played = ps
                            .played_stream
                            .filter(|_| ps.item_id == item_id)
                            .or(reported_stream);
                        db::TrackMemory::Remember(db::AudioPick {
                            index,
                            stream_id: played.map(|p| p.id),
                            chosen: played
                                .and_then(|p| p.automatic_audio)
                                .is_some_and(|automatic| automatic != index),
                        })
                    })
            } else {
                db::TrackMemory::Forget
            };
            let subtitle = if cfg.remember_subtitle_selections {
                data.subtitle_stream_index
                    .or(ps.subtitle_stream_index)
                    .map_or(db::TrackMemory::Keep, |x| {
                        db::TrackMemory::Remember(x as i64)
                    })
            } else {
                db::TrackMemory::Forget
            };

            db::UserMediaState::update_playback(
                db,
                user,
                &media,
                position_ticks,
                audio,
                subtitle,
                media.runtime,
            )
            .await?;
        }

        Ok(())
    }

    /// Handle a `POST /sessions/playing/stopped` report.
    ///
    /// Removes the playback session (stopping any active transcode) and what
    /// the report leaves behind on `device_id` (`stop_left_behind`), persists
    /// the final position to the DB (with the 90 % watched-mark check), and
    /// emits an info log line.
    ///
    /// Returns whether this stop crossed the watched threshold.
    pub async fn stopped(
        &self,
        db: &sqlx::SqlitePool,
        user: &db::User,
        device_id: &str,
        psid: &str,
        data: &PlaybackInfo,
    ) -> anyhow::Result<bool> {
        // Stop a transcode still being created for this session too.
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
        if let Some(item_id) = item_id {
            self.stop_left_behind(
                device_id,
                user.id,
                item_id,
                psid,
                data.play_session_id
                    .is_none(),
            )
            .await;
        }
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
            db::TrackMemory::Keep, // don't overwrite stream selections on stop
            db::TrackMemory::Keep,
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
    /// Ends the user's earlier sessions of the same item on the device that
    /// have no transcode: the client restarted that playback under a new play
    /// session. The others, whatever their item, are ended by
    /// `end_left_playbacks` once their client has left them.
    pub async fn insert(&self, mut session: PlaybackSession) {
        // Only the same user's: another user signed in on the device has a
        // playback of their own. Without a transcode there are no requests
        // to tell a playback left behind from another tab's (jellyfin-web
        // tabs share their DeviceId), so only the same item's is replaced.
        // Unreported transcode stubs are left alone: a start report without
        // (or with another) PlaySessionId must not kill the stream it reports.
        if !session
            .device_id
            .is_empty()
        {
            if session.is_reported() {
                self.latest_starts
                    .entry((
                        session
                            .device_id
                            .clone(),
                        session.user_id,
                    ))
                    .and_modify(|at| *at = (*at).max(session.reported_at))
                    .or_insert(session.reported_at);
            }
            let stale: Vec<String> = self
                .sessions
                .iter()
                .filter(|e| {
                    let s = e.value();
                    s.device_id == session.device_id
                        && s.is_reported()
                        && s.user_id == session.user_id
                        && s.item_id == session.item_id
                        && s.transcode
                            .is_none()
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

    /// Return `user_id`'s most recently active reported session on a device.
    /// Used as a fallback when the client omits PlaySessionId (e.g. DirectPlay).
    /// Another user signed in with the same DeviceId has sessions of their own.
    pub fn get_by_device(
        &self,
        device_id: &str,
        user_id: Uuid,
    ) -> Option<PlaybackSession> {
        self.sessions
            .iter()
            .filter(|e| {
                let s = e.value();
                s.device_id == device_id && s.user_id == user_id && s.is_reported()
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

    /// `get_by_device` for a request whose user isn't known (no token): the
    /// device's session, if only one user has reported sessions on it.
    pub fn get_by_device_of_its_only_user(
        &self,
        device_id: &str,
    ) -> Option<PlaybackSession> {
        let user_id = {
            let mut users = self
                .sessions
                .iter()
                .filter(|e| {
                    e.value()
                        .device_id
                        == device_id
                        && e.value()
                            .is_reported()
                })
                .map(|e| {
                    e.value()
                        .user_id
                });
            let user_id = users.next()?;
            if users.any(|other| other != user_id) {
                return None;
            }
            user_id
        };
        self.get_by_device(device_id, user_id)
    }

    /// Count active sessions for a user, optionally excluding sessions from a
    /// specific device. Excluding the caller's device is correct when checking
    /// before `insert()`: as with a Jellyfin session, a device counts as one
    /// stream, the one it starts now.
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

    /// A client's ping of its play session: a report, see `ping`.
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
                        played_stream: None,
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
                        reported_at: std::time::Instant::now(),
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

    /// Stop transcodes their client abandoned for a newer play session (a
    /// retry or stream switch under a new PlaySessionId, often without
    /// DELETE /Videos/ActiveEncodings). Within each group of one device's
    /// transcodes of one item for one user, one is stopped once, since the
    /// newest other of the group was created, its client requested nothing
    /// (`TranscodeSession::left_for`) nor reported its play session as
    /// played (`PlaybackSession::still_played`), and once it requested
    /// nothing for two of the segments ffmpeg cut
    /// (`TranscodeSession::abandoned_with_segments_of`). A retry's
    /// predecessor goes quiet before the retry exists. Two players of the
    /// item on a device (browser tabs share their DeviceId) each request
    /// after the other started, and a paused one, which requests nothing,
    /// still reports: jellyfin-web every 10 s, Android TV every 15 s, and
    /// on resume, before its buffer runs low enough to request again; until
    /// its first report after the other one switched streams, its last
    /// report's pause keeps it. Unreported stubs are removed entirely.
    ///
    /// A client retrying every `r` s, each attempt stopped `w` + up to 2 s
    /// (the sweep interval) after its last request, leaves at most
    /// ⌈(w + 2 s) / r⌉ + 1 transcodes running: `w` is 12 s at 6 s segments,
    /// 20 s for a copied video with 10 s keyframe intervals (twice the
    /// longest segment, so two players of such a video don't stop each
    /// other). Its last attempt stays until the idle reaper.
    pub async fn supersede_abandoned_transcodes(&self, now: std::time::Instant) {
        let attached: Vec<_> = self
            .sessions
            .iter()
            .filter(|e| {
                !e.value()
                    .device_id
                    .is_empty()
            })
            .filter_map(|e| {
                Some((
                    e.key()
                        .clone(),
                    e.value()
                        .device_id
                        .clone(),
                    e.value()
                        .transcode
                        .clone()?,
                ))
            })
            .collect();
        let mut groups: std::collections::HashMap<_, Vec<_>> =
            std::collections::HashMap::new();
        for (id, device_id, ts) in attached {
            let (key, created_at) = {
                let s = ts
                    .read()
                    .await;
                if s.stopped {
                    continue;
                }
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
                // Checked last: a request may have come in meanwhile.
                let playlist = {
                    let s = ts
                        .read()
                        .await;
                    if !s.abandoned(now) || !s.left_for(newest_other) {
                        continue;
                    }
                    s.variant_playlist_path()
                };
                // Segments ffmpeg cut longer than requested (a copied
                // video's keyframe intervals) are requested less often.
                let longest = tokio::fs::read_to_string(playlist)
                    .await
                    .map(|playlist| longest_segment(&playlist))
                    .unwrap_or_default();
                if !ts
                    .read()
                    .await
                    .abandoned_with_segments_of(now, longest)
                    || self
                        .sessions
                        .get(id)
                        .is_some_and(|s| s.still_played(newest_other))
                {
                    continue;
                }
                info!(
                    play_session_id = %id,
                    "Stopping transcode abandoned for a newer play session of the item"
                );
                self.stop_superseded(id, ts, newest_other)
                    .await;
            }
        }
    }

    /// Stop `ts` if `id` still holds it (a seek may have replaced it): an
    /// unreported stub goes entirely, a reported session keeps its playback
    /// state.
    async fn stop_superseded(
        &self,
        id: &str,
        ts: &Arc<tokio::sync::RwLock<TranscodeSession>>,
        newest_other: std::time::Instant,
    ) {
        // A report sent meanwhile keeps it too.
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
                .and_then(|mut e| {
                    if holds(e.value()) {
                        e.value_mut()
                            .transcode
                            .take()
                    } else {
                        None
                    }
                })
        };
        drop(handoff);
        if let Some(ts) = taken {
            self.kill_transcode(ts, StopMode::Detached)
                .await;
        }
    }

    /// End the reported playbacks their client left for a newer one of its
    /// user on the device without a stop report: a closed tab, then a new
    /// one; a channel zap of a client that skips Stopped. Whatever its item,
    /// a playback is left once its client sent no report on it since the
    /// newer one's start report (which may have stopped since), nor for
    /// `REPORTS_LEFT_AFTER`, and, with a transcode, made no request of it
    /// since then, nor for two of the segments ffmpeg cut (see
    /// `left_for_playback`). Another tab playing on the shared DeviceId
    /// reports after that start, paused too, so it is kept. The session
    /// goes too: it would show as now playing after the newer playback
    /// stops, and count as a stream, until the idle reaper.
    pub async fn end_left_playbacks(&self, now: std::time::Instant) {
        let reported: Vec<_> = self
            .sessions
            .iter()
            .filter(|e| {
                e.value()
                    .is_reported()
                    && !e
                        .value()
                        .device_id
                        .is_empty()
            })
            .map(|e| {
                let s = e.value();
                (
                    e.key()
                        .clone(),
                    (
                        s.device_id
                            .clone(),
                        s.user_id,
                    ),
                    s.reported_at,
                    s.last_report,
                    s.transcode
                        .clone(),
                )
            })
            .collect();
        // A start is only needed while an older playback of its owner is left.
        self.latest_starts
            .retain(|owner, latest| {
                reported
                    .iter()
                    .any(|(_, other_owner, reported_at, _, _)| {
                        other_owner == owner && reported_at < latest
                    })
            });
        for (id, owner, reported_at, last_report, ts) in &reported {
            let newer: Vec<_> = reported
                .iter()
                .filter(|(other, other_owner, other_reported_at, _, _)| {
                    other != id
                        && other_owner == owner
                        && other_reported_at > reported_at
                })
                .map(|(_, _, other_reported_at, _, _)| *other_reported_at)
                .chain(
                    // Kept after the playback it started stopped.
                    self.latest_starts
                        .get(owner)
                        .map(|latest| *latest)
                        .filter(|latest| latest > reported_at),
                )
                .collect();
            // A paused player stops requesting segments, so a transcode is
            // measured from the first playback started after it: a tab that
            // requested since then is still there. Reports go on while
            // paused, so without a transcode the latest start will do.
            let since = if ts.is_some() {
                newer
                    .iter()
                    .min()
            } else {
                newer
                    .iter()
                    .max()
            };
            let Some(&since) = since else {
                continue;
            };
            // Its client also stops reporting it: a paused player requests
            // no segments, but goes on reporting.
            if !reports_left(*last_report, since, now) {
                continue;
            }
            if let Some(ts) = ts
                && !left_for_playback(ts, since, now).await
            {
                continue;
            }
            // A seek restarts the session's transcode under this lock; a
            // report, or a transcode, may have come in meanwhile.
            let _create = TRANSCODE_CREATE_LOCKS
                .lock_if_exists(id)
                .await;
            let still_left = self
                .sessions
                .get(id)
                .is_some_and(|s| {
                    let same_transcode = match (&s.transcode, ts) {
                        (Some(current), Some(ts)) => Arc::ptr_eq(current, ts),
                        (None, None) => true,
                        _ => false,
                    };
                    same_transcode && reports_left(s.last_report, since, now)
                });
            if still_left {
                self.end_left_playback(id)
                    .await;
            }
        }
    }

    async fn end_left_playback(&self, id: &str) {
        info!(
            play_session_id = %id,
            "Ending playback the device left for a newer one of its user"
        );
        self.stop(id, StopMode::Detached)
            .await;
    }

    /// End what a stop report of `item_id` by `user_id` leaves behind on
    /// `device_id` under other play sessions than `play_session_id`.
    ///
    /// That is the user's reported sessions of the item whose ffmpeg stream was
    /// already stopped (superseded, or DELETE ActiveEncodings): the client
    /// switched streams under a new play session, and reports on that one.
    /// With a live transcode, or never served by ffmpeg, a session may be
    /// another tab still playing the item, so it is kept.
    ///
    /// `unnamed`: the report carried no PlaySessionId, so it was matched to
    /// the device's reported session; the transcodes it played are the
    /// device's unclaimed ones of the item.
    async fn stop_left_behind(
        &self,
        device_id: &str,
        user_id: Uuid,
        item_id: Uuid,
        play_session_id: &str,
        unnamed: bool,
    ) {
        if device_id.is_empty() {
            return;
        }
        let abandoned: Vec<String> = self
            .sessions
            .iter()
            .filter(|e| {
                let s = e.value();
                e.key() != play_session_id
                    && s.device_id == device_id
                    && s.is_reported()
                    && s.user_id == user_id
                    && s.item_id == item_id
                    && matches!(s.served, Some(ServedPlayback::Ffmpeg { .. }))
                    && s.transcode
                        .is_none()
            })
            .map(|e| {
                e.key()
                    .clone()
            })
            .collect();
        for id in abandoned {
            // A seek restarts its session's transcode under this lock: once
            // it is ours, a seeking session has its transcode again.
            let _create = TRANSCODE_CREATE_LOCKS
                .lock_if_exists(&id)
                .await;
            if self
                .sessions
                .get(&id)
                .is_some_and(|s| {
                    s.transcode
                        .is_none()
                })
            {
                info!(
                    play_session_id = %id,
                    stopped_play_session_id = play_session_id,
                    "Ending play session the device switched streams away from"
                );
                self.stop(&id, StopMode::Detached)
                    .await;
            }
        }
        if unnamed {
            self.stop_unclaimed_transcodes(device_id, user_id, item_id)
                .await;
        }
    }

    /// Stop `user_id`'s transcodes of `item_id` on `device_id` that no playback
    /// report has claimed, for a stop report without PlaySessionId: its
    /// client's reports went to a generated play session, never to the
    /// transcode it played. Returns whether any was stopped.
    pub async fn stop_unclaimed_transcodes(
        &self,
        device_id: &str,
        user_id: Uuid,
        item_id: Uuid,
    ) -> bool {
        if device_id.is_empty() {
            return false;
        }
        let stubs: Vec<_> = self
            .sessions
            .iter()
            .filter(|e| {
                e.value()
                    .device_id
                    == device_id
                    && !e
                        .value()
                        .is_reported()
            })
            .filter_map(|e| {
                Some((
                    e.key()
                        .clone(),
                    e.value()
                        .transcode
                        .clone()?,
                ))
            })
            .collect();
        let mut stopped = false;
        for (id, ts) in stubs {
            let of_the_stop = {
                let t = ts
                    .read()
                    .await;
                t.user_id == user_id && t.item_id == item_id
            };
            if !of_the_stop {
                continue;
            }
            info!(
                play_session_id = %id,
                %item_id,
                "Stopping unclaimed transcode of the item stopped with no play session"
            );
            // Stop a transcode still being created for it too.
            let _create = TRANSCODE_CREATE_LOCKS
                .lock_if_exists(&id)
                .await;
            stopped |= self
                .stop(&id, StopMode::Detached)
                .await
                .is_some();
        }
        stopped
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
    /// remove all transcode dirs, bounded by `STOP_WAIT` in total.
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
        if tokio::time::timeout(crate::playback::engine::STOP_WAIT, stop)
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

    /// Spawn a background task that ends left playbacks and stops abandoned
    /// transcodes every `interval` (see `end_left_playbacks` and
    /// `supersede_abandoned_transcodes`).
    pub fn spawn_supersede_task(self, interval: Duration) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker
                    .tick()
                    .await;
                let now = std::time::Instant::now();
                self.end_left_playbacks(now)
                    .await;
                self.supersede_abandoned_transcodes(now)
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

    /// Mark a transcode stopped and tell ffmpeg to die. Unless `Reaped`, its
    /// reap (seconds if a thread is in ext4 writeback) and dir removal run
    /// detached.
    async fn kill_transcode(
        &self,
        ts: Arc<tokio::sync::RwLock<TranscodeSession>>,
        mode: StopMode,
    ) {
        let wait = matches!(mode, StopMode::Reaped);
        let (id, kill_tx, wait_done, output_dir) = {
            let mut s = ts
                .write()
                .await;
            s.stopped = true;
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
        let reaped = async move {
            if killed {
                let notified = wait_done.notified();
                tokio::pin!(notified);
                if tokio::time::timeout(
                    crate::playback::engine::STOP_WAIT,
                    &mut notified,
                )
                .await
                .is_err()
                {
                    warn!(play_session_id = %id, "transcode did not stop in time");
                    // Detached: removing the dir while ffmpeg still writes
                    // to it would leave it half removed.
                    if !wait {
                        notified.await;
                    }
                }
            }
            id
        };
        if wait {
            // The caller restarts or exits; it owns the dir.
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

/// How long a playback goes without a report before its client has left
/// it: Jellyfin clients report progress every 10 s (jellyfin-web) to 15 s
/// (Android TV), paused too; this leaves room for a missed one.
const REPORTS_LEFT_AFTER: Duration = Duration::from_secs(30);

/// Whether a client last reporting on a playback at `last_report` (never:
/// `None`) left it for one begun at `since`, as of `now`.
fn reports_left(
    last_report: Option<std::time::Instant>,
    since: std::time::Instant,
    now: std::time::Instant,
) -> bool {
    last_report.is_none_or(|last_report| {
        last_report < since
            && now.saturating_duration_since(last_report) >= REPORTS_LEFT_AFTER
    })
}

/// Whether `ts`'s client left it for a playback begun at `since`: no request
/// started since then, and none for two of the segments ffmpeg cut, which
/// with a copied video end at the source's keyframes, so may be longer than
/// requested.
async fn left_for_playback(
    ts: &Arc<tokio::sync::RwLock<TranscodeSession>>,
    since: std::time::Instant,
    now: std::time::Instant,
) -> bool {
    let playlist = {
        let s = ts
            .read()
            .await;
        if s.stopped || !s.left_for(since) || !s.abandoned(now) {
            return false;
        }
        s.variant_playlist_path()
    };
    let longest = tokio::fs::read_to_string(playlist)
        .await
        .map(|playlist| longest_segment(&playlist))
        .unwrap_or_default();
    ts.read()
        .await
        .abandoned_with_segments_of(now, longest)
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
            played_stream: None,
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
            reported_at: std::time::Instant::now(),
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

    fn attach(
        sessions: &PlaybackSessionManager,
        id: &str,
        device_id: &str,
        ts: Arc<tokio::sync::RwLock<TranscodeSession>>,
    ) -> bool {
        sessions.attach_transcode(id, device_id, ts)
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

    fn transcode(
        base: &std::path::Path,
        id: &str,
        item_id: Uuid,
    ) -> Arc<tokio::sync::RwLock<TranscodeSession>> {
        transcode_of(base, id, Uuid::nil(), item_id)
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
        assert!(attach(&sessions, "reported", "phone", reported_ts.clone()));
        // Never requested: the client failed on its master playlist.
        let abandoned = transcode_of(temp.path(), "retry-1", user, item);
        created(&abandoned, now - Duration::from_secs(24)).await;
        assert!(attach(&sessions, "retry-1", "phone", abandoned.clone()));
        // The attempt the client plays.
        let playing = transcode_of(temp.path(), "retry-2", user, item);
        created(&playing, now - Duration::from_secs(20)).await;
        requested(&playing, now - Duration::from_secs(1)).await;
        assert!(attach(&sessions, "retry-2", "phone", playing));
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
            assert!(attach(&sessions, id, device, ts));
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
        assert!(
            abandoned
                .read()
                .await
                .stopped
        );
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
        assert!(
            reported_ts
                .read()
                .await
                .stopped
        );
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
        assert!(attach(&sessions, "tab-1", "browser", tab1.clone()));
        let tab2 = transcode_of(temp.path(), "tab-2", user, item);
        created(&tab2, now - Duration::from_secs(50)).await;
        requested(&tab2, now - Duration::from_secs(1)).await;
        assert!(attach(&sessions, "tab-2", "browser", tab2.clone()));
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
            assert!(
                !ts.read()
                    .await
                    .stopped,
                "{id}"
            );
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
        assert!(attach(&sessions, "tab-1", "browser", tab1.clone()));
        // Then tab 2 switched its audio track: a new transcode, which tab 1
        // never requested after, as a retry's predecessor wouldn't either.
        let switched = transcode_of(temp.path(), "tab-2-switched", user, item);
        created(&switched, now - Duration::from_secs(40)).await;
        requested(&switched, now - Duration::from_secs(1)).await;
        assert!(attach(&sessions, "tab-2-switched", "browser", switched));

        sessions
            .supersede_abandoned_transcodes(now)
            .await;
        assert!(
            !tab1
                .read()
                .await
                .stopped
        );
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
        assert!(
            tab1.read()
                .await
                .stopped
        );
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
        assert!(attach(&sessions, "tab-1", "browser", tab1.clone()));
        // Tab 2 switches its audio track at 10 s: a new transcode.
        let switched = transcode_of(temp.path(), "tab-2-switched", user, item);
        created(&switched, at(10)).await;
        assert!(attach(
            &sessions,
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
            assert!(
                !tab1
                    .read()
                    .await
                    .stopped,
                "{t}"
            );
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
        assert!(attach(&sessions, "tab-1", "browser", tab1.clone()));
        let tab2 = transcode_of(temp.path(), "tab-2", user, item);
        created(&tab2, at(10)).await;
        requested(&tab2, at(10)).await;
        assert!(attach(&sessions, "tab-2", "browser", tab2.clone()));
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
            assert!(
                !ts.read()
                    .await
                    .stopped,
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_transcode_waiting_for_a_segment_is_not_superseded() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        let now = later();
        // A universal audio request mints a PlaySessionId: a second request
        // for the track still playing starts a second transcode.
        let first = transcode_of(temp.path(), "first", user, item);
        assert!(attach(&sessions, "first", "phone", first.clone()));
        // Its last request is long past, but still waits for ffmpeg.
        let request = first
            .read()
            .await
            .requests
            .begin();
        let second = transcode_of(temp.path(), "second", user, item);
        requested(&second, now - Duration::from_secs(1)).await;
        assert!(attach(&sessions, "second", "phone", second));

        sessions
            .supersede_abandoned_transcodes(now)
            .await;

        assert!(
            !first
                .read()
                .await
                .stopped
        );
        for id in ["first", "second"] {
            assert!(
                sessions
                    .get_transcode(id)
                    .is_some(),
                "{id}"
            );
        }
        drop(request);
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
                assert!(attach(&sessions, id, "browser", ts.clone()));
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
                assert_eq!(
                    ts.read()
                        .await
                        .stopped,
                    !played,
                    "{id}"
                );
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
            assert!(attach(&sessions, id, "phone", ts));
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
        assert!(attach(&sessions, "stub", "phone", stub));

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
        assert!(attach(&sessions, "p1", "tv", p1));
        // The retry under a new PlaySessionId, which sends no start report,
        // and a stub of another item.
        let p2 = transcode_of(temp.path(), "p2", user, item);
        assert!(attach(&sessions, "p2", "tv", p2));
        let other = transcode_of(temp.path(), "other", user, Uuid::new_v4());
        assert!(attach(&sessions, "other", "tv", other));

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
        let ts = transcode(temp.path(), "hls", Uuid::new_v4());
        assert!(attach(&sessions, "hls", "phone", ts.clone()));

        // `start` makes up a PlaySessionId when the report has none.
        let user = Uuid::new_v4();
        let mut report = playback_session("generated", PlayMethod::Transcode);
        report.user_id = user;
        report.device_id = "phone".to_string();
        sessions
            .insert(report)
            .await;

        assert!(
            sessions
                .get_transcode("hls")
                .is_some()
        );
        assert!(
            !ts.read()
                .await
                .stopped
        );
        assert_eq!(
            sessions
                .get_by_device("phone", user)
                .map(|s| s.play_session_id),
            Some("generated".to_string())
        );
        // Another user signed in on the device has no session there.
        assert!(
            sessions
                .get_by_device("phone", Uuid::new_v4())
                .is_none()
        );
    }

    fn reported(id: &str, user: Uuid, device: &str, item: Uuid) -> PlaybackSession {
        let mut session = playback_session(id, PlayMethod::Transcode);
        session.user_id = user;
        session.device_id = device.to_string();
        session.item_id = item;
        session
    }

    /// A reported session served by ffmpeg, its transcode not attached.
    fn streamed(id: &str, user: Uuid, item: Uuid) -> PlaybackSession {
        let mut session = reported(id, user, "browser", item);
        session.served = Some(transcoded());
        session
    }

    /// Put a session in place without `insert` replacing any.
    fn put(sessions: &PlaybackSessionManager, session: PlaybackSession) {
        sessions
            .sessions
            .insert(
                session
                    .play_session_id
                    .clone(),
                session,
            );
    }

    #[tokio::test]
    async fn a_start_report_replaces_only_the_users_untranscoded_session_of_the_item() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let (user, other_user) = (Uuid::new_v4(), Uuid::new_v4());
        let (item, other_item) = (Uuid::new_v4(), Uuid::new_v4());
        // A tab transcoding the item: `end_left_playbacks` judges it.
        sessions
            .insert(reported("transcoded", user, "browser", item))
            .await;
        let transcoded_ts = transcode_of(temp.path(), "transcoded", user, item);
        assert!(attach(
            &sessions,
            "transcoded",
            "browser",
            transcoded_ts.clone()
        ));
        // Direct played: the same item's, a second tab's of another item on
        // the shared DeviceId, and another user's signed in on the device.
        for (id, session_user, session_item) in [
            ("direct", user, item),
            ("other-item", user, other_item),
            ("other-user", other_user, item),
        ] {
            sessions
                .insert(reported(id, session_user, "browser", session_item))
                .await;
        }

        // The item restarted under a new play session.
        sessions
            .insert(reported("restarted", user, "browser", item))
            .await;

        assert!(
            sessions
                .get("direct")
                .is_none()
        );
        for id in ["transcoded", "other-item", "other-user", "restarted"] {
            assert!(
                sessions
                    .get(id)
                    .is_some(),
                "{id}"
            );
        }
        assert!(
            !transcoded_ts
                .read()
                .await
                .stopped
        );
    }

    #[tokio::test]
    async fn a_playback_left_for_a_newer_one_of_its_user_ends_whatever_its_item() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let (user, other_user) = (Uuid::new_v4(), Uuid::new_v4());
        let start = std::time::Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        // (id, device, user, started, last request, last report): each its
        // own item.
        let playbacks = [
            // Closed (or zapped away from) before the newer ones started.
            ("left", "browser", user, 0, 0, 0),
            // Another tab, which requested after the ones after it started
            // (its reports were lost).
            ("requesting-tab", "browser", user, 1, 30, 0),
            // Another tab, paused before "newer" started: no requests, but
            // it goes on reporting.
            ("paused-tab", "browser", user, 2, 5, 95),
            ("newer", "browser", user, 20, 99, 99),
            // Another user's on the device, and the user's on another one.
            ("other-user", "browser", other_user, 0, 0, 0),
            ("other-device", "tv", user, 0, 0, 0),
        ];
        let mut transcodes = Vec::new();
        for (id, device, session_user, started, last_request, last_report) in playbacks
        {
            let mut session = reported(id, session_user, device, Uuid::new_v4());
            session.reported_at = at(started);
            session.last_report = Some(at(last_report));
            let item = session.item_id;
            put(&sessions, session);
            let ts = transcode_of(temp.path(), id, session_user, item);
            created(&ts, at(started)).await;
            requested(&ts, at(last_request)).await;
            assert!(attach(&sessions, id, device, ts.clone()));
            transcodes.push((id, ts));
        }

        sessions
            .end_left_playbacks(at(100))
            .await;

        for (id, ts) in transcodes {
            let left = id == "left";
            assert_eq!(
                sessions
                    .get(id)
                    .is_none(),
                left,
                "{id}"
            );
            assert_eq!(
                ts.read()
                    .await
                    .stopped,
                left,
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_playback_left_for_a_newer_one_is_kept_until_abandoned() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let start = std::time::Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        // A channel zap: the last request and report a second before the
        // next channel.
        for (id, started, last) in [("ch1", 0, 9), ("ch2", 10, 30)] {
            let mut session = reported(id, user, "tv", Uuid::new_v4());
            session.reported_at = at(started);
            session.last_report = Some(at(last));
            let item = session.item_id;
            put(&sessions, session);
            let ts = transcode_of(temp.path(), id, user, item);
            created(&ts, at(started)).await;
            requested(&ts, at(last)).await;
            assert!(attach(&sessions, id, "tv", ts));
        }

        // 11 s without a request, under two 6 s segments: maybe a stall.
        // Then 12 s, abandoned, but 12 s without a report: maybe a slow
        // network. Then 29 s.
        for secs in [20, 21, 38] {
            sessions
                .end_left_playbacks(at(secs))
                .await;
            assert!(
                sessions
                    .get_transcode("ch1")
                    .is_some(),
                "{secs}"
            );
        }

        sessions
            .end_left_playbacks(at(39))
            .await;
        assert!(
            sessions
                .get("ch1")
                .is_none()
        );
        assert!(
            sessions
                .get_transcode("ch2")
                .is_some()
        );
    }

    #[tokio::test]
    async fn a_direct_play_left_for_a_newer_one_ends_once_it_stops_reporting() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let start = std::time::Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        // (id, started, last report), each its own item, none transcoded.
        for (id, started, last_report) in [
            // The app closed 5 s in; the newer playback started at 20 s.
            ("left", 0, 5),
            // Another tab, still reporting.
            ("reporting-tab", 1, 30),
            ("newer", 20, 30),
        ] {
            let mut session = reported(id, user, "browser", Uuid::new_v4());
            session.served = Some(ServedPlayback::Direct);
            session.reported_at = at(started);
            session.last_report = Some(at(last_report));
            put(&sessions, session);
        }

        // 29 s without a report: maybe a slow network.
        sessions
            .end_left_playbacks(at(34))
            .await;
        assert!(
            sessions
                .get("left")
                .is_some()
        );

        sessions
            .end_left_playbacks(at(35))
            .await;
        for (id, kept) in [("left", false), ("reporting-tab", true), ("newer", true)] {
            assert_eq!(
                sessions
                    .get(id)
                    .is_some(),
                kept,
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_playback_left_for_one_stopped_since_still_ends() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let start = std::time::Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        let item = Uuid::new_v4();
        let mut left = reported("left", user, "tv", item);
        left.reported_at = at(0);
        left.last_report = Some(at(9));
        put(&sessions, left);
        let left_ts = transcode_of(temp.path(), "left", user, item);
        created(&left_ts, at(0)).await;
        requested(&left_ts, at(9)).await;
        assert!(attach(&sessions, "left", "tv", left_ts.clone()));
        // A newer playback starts at 10 s and stops at 16 s, before the
        // left one counts as abandoned.
        let mut newer = reported("newer", user, "tv", Uuid::new_v4());
        newer.reported_at = at(10);
        sessions
            .insert(newer)
            .await;
        sessions
            .stop("newer", StopMode::Detached)
            .await;

        sessions
            .end_left_playbacks(at(39))
            .await;
        assert!(
            sessions
                .get("left")
                .is_none()
        );
        assert!(
            left_ts
                .read()
                .await
                .stopped
        );
        // Nothing older is left to measure against that start.
        sessions
            .end_left_playbacks(at(41))
            .await;
        assert!(
            sessions
                .latest_starts
                .is_empty()
        );
    }

    #[tokio::test]
    async fn a_request_without_user_finds_the_session_of_the_devices_only_user() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        put(&sessions, reported("first", user, "tv", Uuid::new_v4()));
        put(&sessions, reported("second", user, "tv", Uuid::new_v4()));
        assert!(
            sessions
                .get_by_device_of_its_only_user("tv")
                .is_some()
        );
        // A transcode stub has no user to tell apart.
        assert!(attach(
            &sessions,
            "stub",
            "tv",
            transcode(temp.path(), "stub", Uuid::new_v4())
        ));
        assert!(
            sessions
                .get_by_device_of_its_only_user("tv")
                .is_some()
        );

        // Two users signed in on the device: whose request is it?
        put(
            &sessions,
            reported("other", Uuid::new_v4(), "tv", Uuid::new_v4()),
        );
        assert!(
            sessions
                .get_by_device_of_its_only_user("tv")
                .is_none()
        );
        assert!(
            sessions
                .get_by_device_of_its_only_user("phone")
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_stop_ends_the_users_session_the_device_switched_streams_away_from() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let (user, other_user) = (Uuid::new_v4(), Uuid::new_v4());
        let (item, other_item) = (Uuid::new_v4(), Uuid::new_v4());
        // Before the stream switch: its transcode was superseded, and the
        // client reports on its new play session since.
        put(&sessions, streamed("switched-from", user, item));
        // Another tab still transcoding the item.
        put(&sessions, streamed("tab", user, item));
        assert!(attach(
            &sessions,
            "tab",
            "browser",
            transcode(temp.path(), "tab", item)
        ));
        // Another tab direct playing it: no transcode to lose.
        let mut direct = reported("direct", user, "browser", item);
        direct.served = Some(ServedPlayback::Direct);
        put(&sessions, direct);
        put(&sessions, streamed("other-user", other_user, item));
        put(&sessions, streamed("other-item", user, other_item));
        let mut other_device = streamed("other-device", user, item);
        other_device.device_id = "tv".to_string();
        put(&sessions, other_device);

        sessions
            .stop_left_behind("browser", user, item, "switched-to", false)
            .await;

        assert!(
            sessions
                .get("switched-from")
                .is_none()
        );
        for id in ["tab", "direct", "other-user", "other-item", "other-device"] {
            assert!(
                sessions
                    .get(id)
                    .is_some(),
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_session_between_the_transcodes_of_a_seek_is_not_ended() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        // A seek stopped its transcode and is creating the next one.
        put(&sessions, streamed("seeking", user, item));
        let creating = TRANSCODE_CREATE_LOCKS
            .lock("seeking".to_string())
            .await;
        let stop = tokio::spawn({
            let sessions = sessions.clone();
            async move {
                sessions
                    .stop_left_behind("browser", user, item, "other-tab", false)
                    .await
            }
        });
        // Let the stop find the session without a transcode and wait.
        tokio::task::yield_now().await;
        assert!(attach(
            &sessions,
            "seeking",
            "browser",
            transcode(temp.path(), "seeking", item)
        ));
        drop(creating);
        stop.await
            .unwrap();

        assert!(
            sessions
                .get_transcode("seeking")
                .is_some()
        );
    }

    #[tokio::test]
    async fn a_stop_without_play_session_stops_the_unclaimed_transcode_of_the_item() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let user = Uuid::new_v4();
        let item = Uuid::new_v4();
        // Reports without PlaySessionId went to a generated session; the
        // stream they were about is a transcode no report claimed.
        put(&sessions, reported("generated", user, "browser", item));
        let played = transcode_of(temp.path(), "hls", user, item);
        assert!(attach(&sessions, "hls", "browser", played.clone()));
        let kept = [
            ("other-item", "browser", user, Uuid::new_v4()),
            ("other-device", "tv", user, item),
            // Another user signed in on the device, playing the item too.
            ("other-user", "browser", Uuid::new_v4(), item),
        ];
        for (id, device, kept_user, kept_item) in kept {
            assert!(attach(
                &sessions,
                id,
                device,
                transcode_of(temp.path(), id, kept_user, kept_item)
            ));
        }

        // A report naming its play session says nothing about other streams.
        sessions
            .stop_left_behind("browser", user, item, "generated", false)
            .await;
        assert!(
            sessions
                .get_transcode("hls")
                .is_some()
        );

        sessions
            .stop_left_behind("browser", user, item, "generated", true)
            .await;
        assert!(
            sessions
                .get("hls")
                .is_none()
        );
        assert!(
            played
                .read()
                .await
                .stopped
        );
        assert!(
            removed(
                temp.path()
                    .join("hls")
            )
            .await
        );
        for (id, _, _, _) in kept {
            assert!(
                sessions
                    .get_transcode(id)
                    .is_some(),
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn a_stop_before_the_first_spawn_prevents_it() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode(temp.path(), "early", Uuid::new_v4());
        assert!(attach(&sessions, "early", "phone", ts.clone()));

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
        let ts = transcode(temp.path(), "live", Uuid::new_v4());
        assert!(attach(&sessions, "live", "tv", ts.clone()));
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
    async fn stop_all_stops_every_transcode_and_refuses_new_ones() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let ts = transcode(temp.path(), "running", Uuid::new_v4());
        assert!(attach(&sessions, "running", "phone", ts.clone()));
        // A transcode stopped earlier whose background removal hasn't run.
        let earlier = temp
            .path()
            .join("earlier");
        std::fs::create_dir_all(&earlier).unwrap();

        sessions
            .stop_all()
            .await;

        assert!(
            ts.read()
                .await
                .stopped
        );
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
        let late = transcode(temp.path(), "late", Uuid::new_v4());
        assert!(!attach(&sessions, "late", "phone", late));
        assert!(
            sessions
                .get("late")
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_stop_during_transcode_creation_stops_the_new_transcode() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());

        // master.m3u8 for X is creating a transcode when DELETE
        // ActiveEncodings for X arrives.
        let creating = TRANSCODE_CREATE_LOCKS
            .lock("x".to_string())
            .await;
        let stop = tokio::spawn({
            let sessions = sessions.clone();
            // As DELETE ActiveEncodings does.
            async move {
                let _create = TRANSCODE_CREATE_LOCKS
                    .lock_if_exists(&"x".to_string())
                    .await;
                sessions
                    .stop_transcode("x", StopMode::Detached)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!stop.is_finished(), "the stop waits for the creation");
        let ts = transcode(temp.path(), "x", Uuid::new_v4());
        assert!(attach(&sessions, "x", "phone", ts.clone()));
        drop(creating);
        stop.await
            .unwrap();
        assert!(
            ts.read()
                .await
                .stopped
        );
        assert!(
            sessions
                .get_transcode("x")
                .is_none()
        );

        // A later request reusing X is not affected.
        let ts = transcode(temp.path(), "x", Uuid::new_v4());
        assert!(attach(&sessions, "x", "phone", ts));
        assert!(
            sessions
                .get_transcode("x")
                .is_some()
        );
    }

    #[tokio::test]
    async fn a_supersede_returns_at_once_and_cleans_up_after_the_reap() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        let item = Uuid::new_v4();
        let now = later();
        let old = transcode(temp.path(), "old", item);
        // A running ffmpeg whose loop never reports it reaped, like one with
        // a thread stuck in the kernel after SIGKILL.
        let (kill_tx, mut kill_rx) = tokio::sync::oneshot::channel();
        old.write()
            .await
            .kill_tx = Some(kill_tx);
        assert!(attach(&sessions, "old", "phone", old.clone()));
        let new = transcode(temp.path(), "new", item);
        requested(&new, now).await;
        assert!(attach(&sessions, "new", "phone", new));

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
        assert!(
            old.read()
                .await
                .stopped
        );
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
        let abandoned = transcode(temp.path(), "seek", item);
        assert!(attach(&sessions, "seek", "phone", abandoned.clone()));
        // A seek restarted the play session's transcode after the sweep
        // found the old one abandoned.
        let restarted = transcode(temp.path(), "seek", item);
        assert!(attach(&sessions, "seek", "phone", restarted.clone()));

        sessions
            .stop_superseded("seek", &abandoned, std::time::Instant::now())
            .await;

        assert!(
            !restarted
                .read()
                .await
                .stopped
        );
        assert!(
            sessions
                .get_transcode("seek")
                .is_some_and(|ts| Arc::ptr_eq(&ts, &restarted))
        );
    }

    #[tokio::test]
    async fn a_failed_restart_reopens_transcoding() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = PlaybackSessionManager::new(temp.path());
        sessions
            .stop_all()
            .await;
        let refused = transcode(temp.path(), "refused", Uuid::new_v4());
        assert!(!attach(&sessions, "refused", "phone", refused));

        // As /System/Restart does when the new process fails to spawn.
        sessions.reopen();

        let ts = transcode(temp.path(), "after", Uuid::new_v4());
        assert!(attach(&sessions, "after", "phone", ts));
        assert!(
            sessions
                .get_transcode("after")
                .is_some()
        );
    }
}

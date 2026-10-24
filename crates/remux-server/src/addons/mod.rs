//! Unified addon abstraction. Each addon kind declares which resources ×
//! media types it serves; user-added instances are rows in the `addons` table.

pub mod addon;
pub mod betterposters;
pub mod deezer;
pub mod eclipse;
pub mod introdb;
pub mod iptv;
pub mod lrclib;
pub mod media_tracker;
pub mod opendal;
pub mod probe;
pub mod squid;
pub mod stremio;
pub mod tmdb;
pub mod torznab;
pub mod ytdlp;

use anyhow::{Result, anyhow};
use arc_swap::ArcSwap;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use sqlx::SqlitePool;
use std::{
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::keyed_lock::KeyedLock;
use libc;
use tracing::{Instrument, debug, error, info, trace, warn};
use uuid::Uuid;

use crate::{
    AppContext, api,
    common::{ItemProgress, ProgressReporter},
    db, sdks,
    services::MediaResolveService,
};
pub use addon::{Addon, CatalogState, set_user_addon_override, user_addon_override};
use remux_sdks::remuxdb;

pub use remux_sdks::remux::AddonPresetRef;
use remux_sdks::remux::{LyricDto, MediaSegments, RemoteLyricInfoDto};

pub use remux_sdks::{
    remux::{
        AddonCatalogDto, AddonDto, AddonMetadata, AddonOption, AddonOptionType,
        AddonSelectOption, CreateAddonRequest, MediaKind, UpdateAddonCatalogRequest,
        UpdateAddonRequest,
    },
    stremio::ResourceType,
};

#[derive(Debug, Clone)]
pub struct CatalogInfo {
    pub provider_catalog_id: String,
    pub name: String,
    /// Whether this catalog should be enabled by default (before the user changes it).
    pub default_enabled: bool,
    /// Default per-catalog item limit (before the user changes it).
    pub default_max_items: Option<i64>,
    /// Media kind for auto-created collections backed by this catalog.
    pub collection_media_kind: Option<db::CollectionMediaKind>,
    /// The MediaKind of items this specific catalog yields.
    pub media_kind: Option<db::MediaKind>,
}

impl CatalogInfo {
    pub fn new(
        provider_catalog_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            provider_catalog_id: provider_catalog_id.into(),
            name: name.into(),
            default_enabled: false,
            default_max_items: None,
            collection_media_kind: None,
            media_kind: None,
        }
    }
}

/// A `CatalogInfo` merged with its persisted `CatalogState` override (if any) —
/// the single, fully-resolved view of a catalog that callers should use. Avoids
/// every caller re-implementing "use the stored override, else fall back to the
/// provider's declared default" itself.
#[derive(Debug, Clone)]
pub struct ResolvedCatalog {
    pub provider_catalog_id: String,
    /// Full "addon:{addon_id}:{provider_catalog_id}" id, usable with `make_catalog_stream()`.
    pub catalog_id: String,
    /// Deterministic collection id for this catalog's `media_relations` membership.
    pub collection_id: Uuid,
    pub name: String,
    pub media_kind: Option<db::MediaKind>,
    pub collection_media_kind: Option<db::CollectionMediaKind>,
    pub enabled: bool,
    pub max_items: Option<i64>,
    pub tags: Vec<String>,
}

#[async_trait]
pub trait RemoteMediaStream: Send + Sync {
    async fn stream(
        &self,
        ctx: &AppContext,
    ) -> Result<Pin<Box<dyn Stream<Item = db::Media> + Send>>>;
}

#[derive(Debug)]
pub struct LyricSearchRequest {
    pub title: String,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub duration_secs: Option<f64>,
}

/// Save relation links that were deferred onto `media.relations` by `apply_meta`.
/// Must be called after `db::Media::upsert` so `left_media_id` FK constraints are satisfied.
pub(crate) async fn save_pending_relations(ctx: &AppContext, items: &[db::Media]) {
    // TMDB ID is the canonical key for person rows.  Name-keyed person stubs
    // (produced by Stremio/Jellyfin addons when no TMDB ID is available) must NOT
    // be persisted — the TMDB addon will insert them with the correct TMDB-keyed UUID
    // when it enriches the parent movie/series.  Storing them now would create
    // duplicate rows alongside any existing TMDB-keyed row for the same person.
    let name_keyed_person_ids: std::collections::HashSet<Uuid> = items
        .iter()
        .filter_map(|m| {
            m.relations
                .as_ref()
        })
        .flatten()
        .filter(|(_, m)| {
            m.kind == db::MediaKind::Person
                && m.external_ids
                    .tmdb
                    .is_none()
        })
        .map(|(_, m)| m.id)
        .collect();

    // One batched upsert for all relation media (persons/genres) across the whole slice —
    // avoids opening a separate transaction per item (N items → N transactions otherwise).
    let all_rel_media: Vec<db::Media> = items
        .iter()
        .filter_map(|m| {
            m.relations
                .as_ref()
        })
        .flatten()
        .map(|(_, m)| m.clone())
        .filter(|m| !name_keyed_person_ids.contains(&m.id))
        .collect();
    if !all_rel_media.is_empty() {
        if let Err(e) = db::Media::upsert(&ctx.db, &all_rel_media).await {
            warn!(error = %e, "failed to upsert relation media batch");
        }
    }

    // Collect items that have relations, then batch delete + batch upsert
    // (replaces N×delete + N×upsert with 1 delete + 1 upsert).
    let items_with_rels: Vec<&db::Media> = items
        .iter()
        .filter(|m| {
            m.relations
                .as_ref()
                .map_or(false, |r| !r.is_empty())
        })
        .collect();
    if items_with_rels.is_empty() {
        return;
    }

    let all_ids: Vec<uuid::Uuid> = items_with_rels
        .iter()
        .map(|m| m.id)
        .collect();
    // Always use m.id as left_media_id — relations may have been built against a
    // temporary UUID (e.g. before IMDB resolution in stremio_search) that was later
    // recomputed to the stable UUID. m.id is the authoritative current identity.
    let all_rels: Vec<db::MediaRelation> = items_with_rels
        .iter()
        .flat_map(|m| {
            let current_id = m.id;
            m.relations
                .as_ref()
                .unwrap()
                .iter()
                .map(move |(r, _)| db::MediaRelation {
                    left_media_id: current_id,
                    ..r.clone()
                })
        })
        // Don't link relations that point to name-keyed person stubs.
        .filter(|r| !name_keyed_person_ids.contains(&r.right_media_id))
        .collect();

    // Fetch existing relations and only write the delta — avoids the WAL pressure
    // of a full delete+reinsert on steady-state runs where nothing changed.
    let existing = db::MediaRelation::get_by_left_ids(&ctx.db, &all_ids)
        .await
        .unwrap_or_default();

    // `apply_meta` deliberately omits provider genre relations when Genres is
    // locked, and provider Person (cast/crew) relations when Cast is locked.
    // If another relation type is still pending, the replacement logic below
    // must not interpret those omitted genres/people as deletions. Resolve
    // the existing right-hand media kinds once and retain genre links for
    // every item whose Genres field is locked, and person links for every
    // item whose Cast field is locked.
    let genre_locked_ids: std::collections::HashSet<Uuid> = items_with_rels
        .iter()
        .filter(|item| item.is_field_locked(&db::MetadataField::Genres))
        .map(|item| item.id)
        .collect();
    let cast_locked_ids: std::collections::HashSet<Uuid> = items_with_rels
        .iter()
        .filter(|item| item.is_field_locked(&db::MetadataField::Cast))
        .map(|item| item.id)
        .collect();
    let locked_relation_right_ids: Vec<Uuid> = existing
        .iter()
        .filter(|relation| {
            genre_locked_ids.contains(&relation.left_media_id)
                || cast_locked_ids.contains(&relation.left_media_id)
        })
        .map(|relation| relation.right_media_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    // `None` means "failed to resolve; preserve everything for locked items"
    // (fail safe, since resolving genre/person kinds requires a DB round-trip).
    let locked_right_kinds: Option<std::collections::HashMap<Uuid, db::MediaKind>> =
        if locked_relation_right_ids.is_empty() {
            Some(std::collections::HashMap::new())
        } else {
            match db::Media::get_by_ids(&ctx.db, &locked_relation_right_ids).await {
                Ok(media) => Some(
                    media
                        .into_iter()
                        .map(|right| (right.id, right.kind))
                        .collect(),
                ),
                Err(e) => {
                    warn!(error = %e, "failed to resolve locked relation kinds; preserving all locked-item relations");
                    None
                }
            }
        };

    type RelKey = (Uuid, Uuid, Option<db::RelationRole>);

    let existing_map: std::collections::HashMap<RelKey, &db::MediaRelation> = existing
        .iter()
        .map(|r| ((r.left_media_id, r.right_media_id, r.role), r))
        .collect();

    let desired_keys: std::collections::HashSet<RelKey> = all_rels
        .iter()
        .map(|r| (r.left_media_id, r.right_media_id, r.role))
        .collect();

    let to_delete: Vec<Uuid> = existing
        .iter()
        .filter(|r| {
            if desired_keys.contains(&(r.left_media_id, r.right_media_id, r.role)) {
                return false;
            }
            let is_genre_locked = genre_locked_ids.contains(&r.left_media_id);
            let is_cast_locked = cast_locked_ids.contains(&r.left_media_id);
            if !is_genre_locked && !is_cast_locked {
                return true;
            }
            match &locked_right_kinds {
                Some(kinds) => match kinds.get(&r.right_media_id) {
                    Some(db::MediaKind::Genre | db::MediaKind::MusicGenre) => {
                        !is_genre_locked
                    }
                    Some(db::MediaKind::Person) => !is_cast_locked,
                    _ => true,
                },
                None => false,
            }
        })
        .map(|r| r.relation_id)
        .collect();

    let to_upsert: Vec<db::MediaRelation> = all_rels
        .into_iter()
        .filter(|r| {
            let key = (r.left_media_id, r.right_media_id, r.role);
            match existing_map.get(&key) {
                None => true,
                Some(ex) => ex.weight != r.weight || ex.character != r.character,
            }
        })
        .collect();

    if !to_delete.is_empty() {
        db::MediaRelation::delete_by_ids(&ctx.db, &to_delete)
            .await
            .ok();
    }
    if !to_upsert.is_empty() {
        if let Err(e) = db::MediaRelation::upsert(&ctx.db, &to_upsert).await {
            warn!(error = %e, "failed to upsert relations batch");
        }
    }
}

/// Persist `provider:` tags collected from meta addons. Only `provider:`-prefixed
/// tags are touched — user-set tags with other prefixes are left intact.
pub(crate) async fn save_pending_tags(ctx: &AppContext, items: &[db::Media]) {
    // Collect (media_id, tag) pairs for all items with provider tags.
    let mut rows: Vec<(uuid::Uuid, &str)> = Vec::new();
    let mut ids_with_tags: Vec<uuid::Uuid> = Vec::new();
    for item in items {
        let provider_tags: Vec<&str> = item
            .tags
            .iter()
            .filter(|t| t.starts_with("provider:"))
            .map(String::as_str)
            .collect();
        if provider_tags.is_empty() {
            continue;
        }
        ids_with_tags.push(item.id);
        for tag in provider_tags {
            rows.push((item.id, tag));
        }
    }

    if rows.is_empty() {
        return;
    }

    // One DELETE for all affected media IDs, one batch INSERT for all tags.
    let mut delete_qb = sqlx::QueryBuilder::new(
        "DELETE FROM media_tags WHERE tag LIKE 'provider:%' AND media_id IN (",
    );
    let mut sep = delete_qb.separated(", ");
    for id in &ids_with_tags {
        sep.push_bind(id);
    }
    delete_qb.push(")");

    let mut insert_qb =
        sqlx::QueryBuilder::new("INSERT OR IGNORE INTO media_tags (media_id, tag) ");
    insert_qb.push_values(&rows, |mut b, (id, tag)| {
        b.push_bind(id)
            .push_bind(tag);
    });

    if let Err(e) = delete_qb
        .build()
        .execute(&ctx.db)
        .await
    {
        warn!(error = %e, "failed to clear provider tags");
        return;
    }
    if let Err(e) = insert_qb
        .build()
        .execute(&ctx.db)
        .await
    {
        warn!(error = %e, "failed to insert provider tags");
    }
}

pub(crate) fn merge_media(target: &mut db::Media, source: &db::Media, replace: bool) {
    use remux_utils::merge_option;

    if !target.is_field_locked(&db::MetadataField::Name)
        && (replace
            || target
                .title
                .is_empty())
        && !source
            .title
            .is_empty()
    {
        target.title = source
            .title
            .clone();
    }

    if !target.is_field_locked(&db::MetadataField::Overview) {
        merge_option(&mut target.description, &source.description, replace);
    }
    merge_option(&mut target.released_at, &source.released_at, replace);
    if !target.is_field_locked(&db::MetadataField::Runtime) {
        merge_option(&mut target.runtime, &source.runtime, replace);
    }
    merge_option(
        &mut target.rating_audience,
        &source.rating_audience,
        replace,
    );
    if !target.is_field_locked(&db::MetadataField::OfficialRating) {
        merge_option(&mut target.certification, &source.certification, replace);
        merge_option(
            &mut target.certification_age,
            &source.certification_age,
            replace,
        );
    }
    if !target.is_field_locked(&db::MetadataField::ProductionLocations) {
        merge_option(&mut target.country, &source.country, replace);
    }
    merge_option(
        &mut target.original_language,
        &source.original_language,
        replace,
    );
    merge_option(&mut target.trailers, &source.trailers, replace);
    merge_option(
        &mut target.digital_released_at,
        &source.digital_released_at,
        replace,
    );
    merge_option(&mut target.status, &source.status, replace);
    merge_option(&mut target.end_date, &source.end_date, replace);
    merge_option(&mut target.idx, &source.idx, replace);
    merge_option(&mut target.parent_idx, &source.parent_idx, replace);

    target
        .external_ids
        .merge(&source.external_ids, replace);
    if let Some(source_ratings) = &source.external_ratings {
        target
            .external_ratings
            .get_or_insert_default()
            .merge(source_ratings, replace);
        merge_option(&mut target.rating_audience, &source.rating_audience, true);
    }
}

/// Default display title for a season: Jellyfin calls season 0 "Specials"
/// (a per-library-configurable name there; a fixed default here).
pub(crate) fn season_title(idx: i64) -> String {
    if idx == 0 {
        "Specials".to_string()
    } else {
        format!("Season {idx}")
    }
}

pub(crate) fn apply_title_format(media: &mut db::Media) {
    if media.kind == db::MediaKind::Season
        && !media.is_field_locked(&db::MetadataField::Name)
    {
        media.title = season_title(
            media
                .idx
                .unwrap_or(1),
        );
    }
    if media.kind == db::MediaKind::Episode {
        // Stored episode titles are kept clean: Jellyfin clients render the
        // season/episode from IndexNumber/ParentIndexNumber, so embedding a
        // "SxxExx - " prefix here double-prints it. Any prefix that slipped in
        // from a raw source is stripped so this stays the single invariant for
        // episode titles.
        if let Some(stripped) = strip_episode_title_prefix(&media.title) {
            media.title = stripped;
        }
    }
}

/// Remove a leading `S<season>E<episode> - ` / `E<episode> - ` prefix (optionally
/// space-separated, e.g. `S01 E07`) from an episode title. Returns `None` when
/// there's nothing to strip.
fn strip_episode_title_prefix(title: &str) -> Option<String> {
    let t = title.trim_start();
    let mut after = t;

    if after.starts_with('S') {
        after = take_digits(&after[1..]);
        after = after.trim_start();
        if !after.starts_with('E') {
            return None;
        }
    } else if !after.starts_with('E') {
        return None;
    }
    after = take_digits(&after[1..]);
    after = after.trim_start();

    match after.strip_prefix('-') {
        Some(rest) => {
            let rest = rest.trim_start();
            (!rest.is_empty()).then(|| rest.to_string())
        }
        None => None,
    }
}

/// The leading run of ASCII digits.
fn take_digits(s: &str) -> &str {
    let n = s
        .as_bytes()
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .count();
    &s[n..]
}

fn series_is_active(status: &Option<db::MediaStatus>) -> bool {
    !matches!(
        status,
        Some(db::MediaStatus::Ended) | Some(db::MediaStatus::Unreleased)
    )
}

/// Whether an episode is still recent (or upcoming) enough to be re-fetched on
/// every pass instead of only until it has been refreshed once.
///
/// The row coming out of a tree fetch is freshly parsed, so it usually has no
/// `digital_released_at` (TMDB episodes only carry their air date in
/// `released_at`). Look at the stored digital date first, then the fetched one,
/// then the air date; an episode with no date at all is treated as active.
fn episode_in_active_window(child: &db::Media) -> bool {
    episode_in_active_window_with(child, None)
}

fn episode_in_active_window_with(
    child: &db::Media,
    stored_digital: Option<chrono::NaiveDateTime>,
) -> bool {
    match stored_digital
        .or(child.digital_released_at)
        .or(child.released_at)
    {
        None => true,
        Some(dt) => {
            let cutoff = chrono::Utc::now().naive_utc() - chrono::Duration::days(180);
            dt > cutoff
        }
    }
}

fn child_refresh_force(
    force_refresh: bool,
    in_active_window: bool,
    child: &db::Media,
) -> Option<bool> {
    if force_refresh || in_active_window {
        Some(true)
    } else if child
        .refreshed_at
        .is_none()
    {
        Some(false)
    } else {
        None
    }
}

fn apply_meta(media: &mut db::Media, mut patch: db::Media, replace: bool) {
    // Merge images onto the in-memory struct; db::Media::upsert persists them via
    // sync_from_media after the media row is committed, avoiding FK violations.
    if !patch
        .images
        .is_empty()
    {
        use remux_utils::merge_vec;
        let patch_images = std::mem::take(&mut patch.images);
        let imgs = &mut media.images;
        merge_vec(&mut imgs.primary, patch_images.primary, replace);
        merge_vec(&mut imgs.backdrop, patch_images.backdrop, replace);
        merge_vec(&mut imgs.logo, patch_images.logo, replace);
        merge_vec(&mut imgs.thumb, patch_images.thumb, replace);
    }

    if !patch
        .tags
        .is_empty()
        && !media.is_field_locked(&db::MetadataField::Tags)
    {
        media
            .tags
            .extend(std::mem::take(&mut patch.tags));
        media
            .tags
            .sort_unstable();
        media
            .tags
            .dedup();
    }

    merge_media(media, &patch, replace);

    if let Some(relations) = patch.relations {
        if !relations.is_empty()
            && matches!(
                media.kind,
                db::MediaKind::Movie
                    | db::MediaKind::Series
                    | db::MediaKind::Episode
                    | db::MediaKind::Album
            )
        {
            let pending: Vec<(db::MediaRelation, db::Media)> = relations
                .into_iter()
                .filter(|(_, right_media)| match right_media.kind {
                    db::MediaKind::Person => {
                        !media.is_field_locked(&db::MetadataField::Cast)
                    }
                    db::MediaKind::Genre | db::MediaKind::MusicGenre => {
                        !media.is_field_locked(&db::MetadataField::Genres)
                    }
                    _ => true,
                })
                .collect();
            if !pending.is_empty() {
                match &mut media.relations {
                    Some(existing) => existing.extend(pending),
                    None => media.relations = Some(pending),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
impl From<crate::stream::StreamInfo> for db::Media {
    fn from(si: crate::stream::StreamInfo) -> Self {
        let title = si
            .name
            .clone()
            .or_else(|| {
                si.description
                    .clone()
            })
            .unwrap_or_default();
        let probe_data = si
            .probe_data
            .clone();
        db::Media {
            kind: db::MediaKind::Stream,
            title,
            stream_info: Some(si),
            probe_data,
            ..Default::default()
        }
    }
}

// Preset registry
// ---------------------------------------------------------------------------

pub struct AddonPresetRegistration(pub fn() -> Box<dyn AddonPreset>);
inventory::collect!(AddonPresetRegistration);

pub(super) fn make_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("remux-server/1.0")
        .build()
        .expect("failed to build HTTP client")
}

pub fn registered_presets() -> Vec<Box<dyn AddonPreset>> {
    inventory::iter::<AddonPresetRegistration>
        .into_iter()
        .map(|r| (r.0)())
        .collect()
}

// ---------------------------------------------------------------------------
// AddonPreset trait — kind descriptor + factory
// ---------------------------------------------------------------------------

pub trait AddonPreset: Send + Sync {
    fn id(&self) -> &'static str;
    fn metadata(&self) -> AddonMetadata;
    fn from_cfg(
        &self,
        addon_id: Uuid,
        cfg: &serde_json::Value,
        config: &crate::Config,
    ) -> Result<AddonCapabilities>;

    /// Transform the config before it is persisted to the DB.
    /// Use this to convert inline secrets into file references, strip write-only fields, etc.
    /// The default is a no-op.
    fn normalize_cfg(
        &self,
        cfg: serde_json::Value,
        _config: &crate::Config,
    ) -> Result<serde_json::Value> {
        Ok(cfg)
    }
}

// ---------------------------------------------------------------------------
// AddonKind — lean identity + manifest trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait AddonKind: Send + Sync {
    fn id(&self) -> &'static str;

    /// Returns `Ok(Some((resources, types)))` when the addon can determine its
    /// own capabilities (e.g. by fetching a remote manifest).
    /// Returns `Ok(None)` to signal "no override — caller should fall back to
    /// the preset's `metadata().supported_*`".
    /// Returns `Err` when a required remote fetch fails and the addon cannot
    /// be used (the error propagates to the API caller).
    async fn available_info(
        &self,
    ) -> Result<
        Option<(
            Vec<remux_sdks::stremio::ResourceRef>,
            Vec<remux_sdks::stremio::MediaType>,
        )>,
    > {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Capability traits
// ---------------------------------------------------------------------------

#[async_trait]
pub trait IndexAddon: Send + Sync {
    async fn refresh_index(
        &self,
        ctx: &AppContext,
        addon: &Addon,
        progress: ProgressReporter,
    ) -> Result<()>;
    async fn purge_index(&self, ctx: &AppContext, addon: &Addon) -> Result<()>;

    /// Best-available estimate of how many items this addon's index holds —
    /// used to size `RefreshLibrary`'s overall progress total before this
    /// addon's own `refresh_index` has run, and again afterward to correct
    /// that estimate against the real count. `None` when nothing is known
    /// yet (e.g. this addon has never completed a scan).
    async fn index_estimate(&self, _ctx: &AppContext, _addon: &Addon) -> Option<usize> {
        None
    }
}

#[async_trait]
pub trait CatalogAddon: Send + Sync {
    async fn catalog_list(&self, ctx: &AppContext) -> Result<Vec<CatalogInfo>>;
    async fn catalog_stream(
        &self,
        ctx: &AppContext,
        local_id: &str,
    ) -> Result<Option<Pin<Box<dyn Stream<Item = db::Media> + Send>>>>;
}

#[derive(Debug, Clone, Default)]
pub struct ImageFetchOptions {
    pub image_type: Option<api::ImageType>,
    pub include_all_languages: bool,
}

#[async_trait]
pub trait MetaAddon: Send + Sync {
    async fn supports(&self, media: &db::Media) -> bool;
    /// Fetch metadata for `media` and return a partial `db::Media` patch.
    /// Only the fields the addon knows about need to be populated; the caller
    /// merges the patch into the existing record via `merge_media`.
    /// Populate `patch.images` for images, `patch.relations` for people/genres.
    async fn meta_fetch(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        config: &api::ServerConfiguration,
    ) -> Result<Option<db::Media>>;
    /// Called after all items for a given meta_id have been processed.
    /// Addons can use this to evict per-series caches they built during the run.
    fn on_series_done(&self, _meta_id: &str) {}
    /// Called when `meta_fetch` was cancelled by the caller's own timeout
    /// rather than returning an `Err` on its own. `tokio::time::timeout`
    /// drops the in-flight future instead of letting it run to completion,
    /// so the addon's own error-handling code inside `meta_fetch` never runs
    /// and never gets a chance to remember the failure — this is the only
    /// place that can tell the addon a timeout happened for `media`.
    fn on_meta_fetch_timeout(&self, _media: &db::Media) {}
    /// Time left before this addon's shared upstream cooldown (see
    /// `SharedRateLimit`) clears, or `Duration::ZERO` if it has none or isn't
    /// currently blocked. `refresh_meta` checks this before starting a
    /// timeout-bounded `meta_fetch` call: a cooldown longer than the timeout
    /// would otherwise consume the entire budget just waiting, and the call
    /// gets killed before ever sending a request — indistinguishable from a
    /// genuinely hung addon, but really just the rate limiter working as
    /// intended.
    async fn rate_limit_cooldown(&self) -> Duration {
        Duration::ZERO
    }
    /// Fetch remote image candidates for manual image selection in the UI.
    async fn images_fetch(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        options: ImageFetchOptions,
    ) -> Result<Vec<crate::api::RemoteImageInfo>> {
        Ok(vec![])
    }
}

#[async_trait]
pub trait TreeAddon: Send + Sync {
    fn supports(&self, root: &db::Media) -> bool;
    async fn get_children(
        &self,
        root: &db::Media,
        ctx: &AppContext,
    ) -> Result<Option<Vec<db::Media>>>;
    /// Time left before this addon's shared upstream cooldown clears. See
    /// `MetaAddon::rate_limit_cooldown` — the same reasoning applies here:
    /// `get_direct_children` bounds this call with a timeout, and without
    /// this check a long cooldown would consume that whole budget waiting,
    /// indistinguishable from a genuinely hung tree fetch.
    async fn rate_limit_cooldown(&self) -> Duration {
        Duration::ZERO
    }
}

#[async_trait]
pub trait SearchAddon: Send + Sync {
    async fn search_supports(&self, kind: &db::MediaKind) -> bool;
    async fn search(
        &self,
        kind: &db::MediaKind,
        query: &str,
        limit: usize,
        ctx: &AppContext,
    ) -> Result<Option<Vec<db::Media>>>;
}

#[derive(Clone)]
pub struct SubtitleInfo {
    pub id: String,
    pub url: Option<crate::stream::StreamDescriptor>,
    pub lang: Option<String>,
    pub is_forced: bool,
    pub is_hi: bool,
    /// Release name supplied by the subtitle provider, if any.
    pub filename: Option<String>,
    pub from_trusted: Option<bool>,
    pub ai_translated: Option<bool>,
}

/// The release being played, so an addon can return subtitles made for that
/// exact file instead of a generic list for the title. All fields are optional;
/// addons that can't use them ignore the query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubtitleQuery {
    pub filename: Option<String>,
    pub video_size: Option<i64>,
}

impl SubtitleQuery {
    fn cache_suffix(&self) -> String {
        if self == &Self::default() {
            return String::new();
        }
        format!(
            ":{}:{}",
            self.filename
                .as_deref()
                .unwrap_or(""),
            self.video_size
                .map_or(String::new(), |size| size.to_string())
        )
    }
}

#[async_trait]
pub trait SubtitleAddon: Send + Sync {
    fn supports(&self, media: &db::Media) -> bool;
    async fn subtitle_fetch(
        &self,
        media: &db::Media,
        query: &SubtitleQuery,
        db: &SqlitePool,
    ) -> Result<Vec<SubtitleInfo>>;
}

#[async_trait]
pub trait StreamAddon: Send + Sync {
    fn supports(&self, media: &db::Media) -> bool;
    async fn get_streams(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        id_prefixes: Option<&[String]>,
    ) -> Result<Vec<crate::stream::StreamInfo>>;
    /// Serve bytes for a stream that requires this addon's config (e.g. credentials).
    /// Only called when `StreamDescriptor::addon_id()` points to this addon.
    async fn serve_stream(
        &self,
        descriptor: &crate::stream::StreamDescriptor,
        headers: &axum::http::HeaderMap,
    ) -> axum_anyhow::ApiResult<axum::response::Response> {
        Err(axum_anyhow::ApiError::builder()
            .status(axum::http::StatusCode::BAD_REQUEST)
            .title("stream")
            .detail("serve_stream not implemented for this addon")
            .build())
    }
}

#[async_trait]
pub trait SegmentAddon: Send + Sync {
    fn supports(&self, media: &db::Media) -> bool;
    async fn segment_fetch(
        &self,
        media: &db::Media,
        ctx: &AppContext,
    ) -> Result<MediaSegments>;
}

#[async_trait]
pub trait LyricAddon: Send + Sync {
    fn provider_id(&self) -> String;
    async fn lyric_fetch(&self, req: &LyricSearchRequest) -> Result<Option<LyricDto>>;
    async fn lyric_search(
        &self,
        req: &LyricSearchRequest,
    ) -> Result<Vec<RemoteLyricInfoDto>>;
    async fn lyric_get_by_id(&self, id: &str) -> Result<Option<LyricDto>>;
}

// ---------------------------------------------------------------------------
// AddonCapabilities — produced by AddonPreset::from_cfg
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct AddonCapabilities {
    pub metadata: AddonMetadata,
    /// The live manifest could not be fetched when the addon was loaded, so
    /// `metadata` is the preset's static fallback. Such an addon re-checks its
    /// manifest per request (see `AddonRuntime::live_id_prefixes`) and is
    /// skipped while that still fails.
    pub manifest_unreachable: bool,
    pub kind: Option<Arc<dyn AddonKind>>,
    pub catalog: Option<Arc<dyn CatalogAddon>>,
    pub meta: Option<Arc<dyn MetaAddon>>,
    pub stream: Option<Arc<dyn StreamAddon>>,
    pub search: Option<Arc<dyn SearchAddon>>,
    pub subtitle: Option<Arc<dyn SubtitleAddon>>,
    pub tree: Option<Arc<dyn TreeAddon>>,
    pub segment: Option<Arc<dyn SegmentAddon>>,
    pub lyric: Option<Arc<dyn LyricAddon>>,
    pub index: Option<Arc<dyn IndexAddon>>,
    pub media_tracker: Option<Arc<dyn media_tracker::MediaTrackerAddon>>,
}

// ---------------------------------------------------------------------------
// AddonRuntime — one entry in the service Vec
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AddonRuntime {
    pub row: Addon,
    pub caps: AddonCapabilities,
}

impl std::ops::Deref for AddonRuntime {
    type Target = AddonCapabilities;
    fn deref(&self) -> &Self::Target {
        &self.caps
    }
}

impl AddonRuntime {
    /// Fetches this addon's live catalog list and merges in its persisted
    /// per-catalog overrides (enabled/max_items/tags). Catalogs without a
    /// stored override fall back to the provider's own declared defaults.
    pub async fn resolve_catalogs(
        &self,
        ctx: &AppContext,
    ) -> Result<Vec<ResolvedCatalog>> {
        let Some(catalog) = self
            .catalog
            .as_ref()
        else {
            return Ok(vec![]);
        };
        let available = catalog
            .catalog_list(ctx)
            .await?;
        let states = self
            .row
            .catalog_states();
        let addon_id = self
            .row
            .id;
        Ok(available
            .into_iter()
            .map(|info| {
                let state = states.get(&info.provider_catalog_id);
                ResolvedCatalog {
                    catalog_id: make_media_id(addon_id, &info.provider_catalog_id),
                    collection_id: Uuid::new_v5(
                        &addon_id,
                        info.provider_catalog_id
                            .as_bytes(),
                    ),
                    enabled: state
                        .map(|s| s.enabled)
                        .unwrap_or(info.default_enabled),
                    max_items: state
                        .and_then(|s| s.max_items)
                        .or(info.default_max_items),
                    tags: state
                        .map(|s| {
                            s.tags
                                .clone()
                        })
                        .unwrap_or_default(),
                    provider_catalog_id: info.provider_catalog_id,
                    name: info.name,
                    media_kind: info.media_kind,
                    collection_media_kind: info.collection_media_kind,
                }
            })
            .collect())
    }

    pub fn supports_type(&self, kind: &db::MediaKind) -> bool {
        // Manifest types (live metadata) are the authoritative upper bound.
        // "Series" in a type list covers Episode and Season too (Stremio model).
        let mt: Vec<db::MediaKind> = self
            .caps
            .metadata
            .supported_types
            .iter()
            .cloned()
            .map(db::MediaKind::from)
            .collect();
        if !mt.is_empty() && !kind_in_type_list(kind, &mt) {
            return false;
        }
        self.row
            .types
            .is_empty()
            || kind_in_type_list(
                kind,
                &self
                    .row
                    .types,
            )
    }

    /// Returns the `idPrefixes` declared for a resource in the live manifest metadata,
    /// or `None` if the resource has no prefix restriction.
    fn resource_id_prefixes(&self, kind: &ResourceType) -> Option<&[String]> {
        self.caps
            .metadata
            .supported_resources
            .iter()
            .find(|r| &r.name == kind)
            .and_then(|r| {
                r.id_prefixes
                    .as_deref()
            })
    }

    /// The id prefixes `kind` is limited to, or `Err` when the addon's manifest
    /// couldn't be loaded and it should be skipped. An addon whose manifest
    /// failed at load is asked again here; it answers from its cache, and
    /// after a failure only retries once per cooldown.
    async fn live_id_prefixes(
        &self,
        kind: &ResourceType,
    ) -> Result<Option<std::borrow::Cow<'_, [String]>>> {
        if !self
            .caps
            .manifest_unreachable
        {
            return Ok(self
                .resource_id_prefixes(kind)
                .map(std::borrow::Cow::Borrowed));
        }
        let info = match &self
            .caps
            .kind
        {
            Some(k) => {
                k.available_info()
                    .await?
            }
            None => None,
        };
        let Some((resources, _)) = info else {
            return Ok(None);
        };
        let resource = resources
            .iter()
            .find(|r| &r.name == kind)
            .ok_or_else(|| anyhow!("manifest does not declare {kind}"))?;
        Ok(resource
            .id_prefixes
            .clone()
            .map(std::borrow::Cow::Owned))
    }
}

/// Returns true when `runtime` should run for the given user context.
///
/// `override_ids = None`        → no override active; only `is_default` addons run (background
///                                tasks and users whose addon list hasn't been customised).
/// `override_ids = Some(ids)`   → user has a custom addon list; only those addon IDs run,
///                                plus system addons which always run unconditionally.
fn user_scoped(runtime: &AddonRuntime, override_ids: Option<&[Uuid]>) -> bool {
    if runtime
        .row
        .system
    {
        return true;
    }
    match override_ids {
        None => {
            runtime
                .row
                .is_default
        }
        Some(ids) => ids.contains(
            &runtime
                .row
                .id,
        ),
    }
}

/// Returns `None` for a manifest type with no recognized `MediaKind`
/// equivalent (e.g. fankai's "anime") instead of collapsing it to `Movie`
/// like the default `From<stremio::MediaType>` impl does — that would make
/// `supports_type` believe an anime/series-only addon serves only movies,
/// excluding it from `addons_for::<dyn StreamAddon>` for every
/// Series/Season/Episode lookup.
pub(crate) fn recognized_manifest_media_kind(
    t: sdks::stremio::MediaType,
) -> Option<sdks::remux::MediaKind> {
    use sdks::{remux::MediaKind as MK, stremio::MediaType as MT};
    Some(match t {
        MT::Movie => MK::Movie,
        MT::Series => MK::Series,
        MT::Tv | MT::Channel => MK::TvChannel,
        MT::Album => MK::Album,
        MT::Artist => MK::Artist,
        MT::Track => MK::Track,
        MT::Events => MK::TvChannel,
        MT::Other(s) => match s.as_str() {
            t if crate::db::is_live_tv_type_alias(t) => MK::TvChannel,
            "episode" => MK::Episode,
            "season" => MK::Season,
            "person" => MK::Person,
            "genre" => MK::Genre,
            "studio" => MK::Studio,
            "collection" => MK::Collection,
            "folder" => MK::Folder,
            "stream" => MK::Stream,
            "playlist" => MK::Playlist,
            _ => return None,
        },
    })
}

fn kind_in_type_list(kind: &db::MediaKind, list: &[db::MediaKind]) -> bool {
    list.contains(kind)
        || (matches!(kind, db::MediaKind::Episode | db::MediaKind::Season)
            && list.contains(&db::MediaKind::Series))
}

pub(crate) type ManifestInfo = (
    Vec<remux_sdks::stremio::ResourceRef>,
    Vec<remux_sdks::stremio::MediaType>,
);

const MANIFEST_FAILURE_COOLDOWN: Duration = Duration::from_secs(30);

enum ManifestState {
    Empty,
    Loaded(ManifestInfo),
    Failed { at: Instant, error: String },
}

/// Remembers an addon's manifest once it loaded, and after a failure answers
/// from the failure until the cooldown passes, so an unreachable addon is
/// retried at most once per cooldown however often it is asked.
pub(crate) struct ManifestCache {
    state: tokio::sync::Mutex<ManifestState>,
}

impl Default for ManifestCache {
    fn default() -> Self {
        Self {
            state: tokio::sync::Mutex::new(ManifestState::Empty),
        }
    }
}

impl ManifestCache {
    pub(crate) async fn get_or_fetch<F, Fut>(&self, fetch: F) -> Result<ManifestInfo>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<ManifestInfo>>,
    {
        let mut state = self
            .state
            .lock()
            .await;
        match &*state {
            ManifestState::Loaded(info) => return Ok(info.clone()),
            ManifestState::Failed { at, error }
                if at.elapsed() < MANIFEST_FAILURE_COOLDOWN =>
            {
                return Err(anyhow!("{error}"));
            }
            _ => {}
        }
        match fetch().await {
            Ok(info) => {
                *state = ManifestState::Loaded(info.clone());
                Ok(info)
            }
            Err(e) => {
                *state = ManifestState::Failed {
                    at: Instant::now(),
                    error: format!("{e:#}"),
                };
                Err(e)
            }
        }
    }
}

fn apply_manifest_info(
    caps: &mut AddonCapabilities,
    resource_refs: Vec<remux_sdks::stremio::ResourceRef>,
    raw_types: Vec<remux_sdks::stremio::MediaType>,
) {
    caps.metadata
        .supported_resources = resource_refs;
    if !raw_types.is_empty() {
        caps.metadata
            .supported_types = raw_types
            .into_iter()
            .filter_map(recognized_manifest_media_kind)
            .collect();
    }
}

// ---------------------------------------------------------------------------
// AddonService
// ---------------------------------------------------------------------------

const MANIFEST_FETCH_CONCURRENCY: usize = 25;

/// Most TMDB `/find` lookups in flight across all searches; queued ones wait
/// in arrival order. That bounds concurrency, not rate: at ~30–50 ms an
/// answer, 10 allow a few hundred requests a second against TMDB's ~40.
/// Measured live: up to 31 lookups a search, a typed query peaking at 152
/// TMDB requests in one second, no 429. Cached answers skip this queue.
const SEARCH_FIND_CONCURRENCY: usize = 10;
/// Most lookups queued for a permit across all searches; past it a search
/// asks about no more. About what 10 permits answer within one 300 ms budget
/// at ~40 ms each. Unbounded, a 35 s `/find` stall queued 1,253 lookups for
/// long-gone keystrokes, which fresh searches then waited behind.
const SEARCH_FIND_QUEUE_MAX: usize = 64;
/// How long one `/find` may take before its permit is given back. The shared
/// 30 s request timeout let 10 lookups stalled on TMDB hold every permit,
/// so fresh searches got no dedupe for ~30 s after TMDB recovered; their
/// budget is 300 ms anyway.
const SEARCH_FIND_REQUEST_TIMEOUT: Duration = Duration::from_secs(3);

/// Bounds the `/find` lookups of all searches.
struct SearchFindLimits {
    /// Lookups in flight.
    permits: Arc<tokio::sync::Semaphore>,
    /// Lookups queued for a permit.
    queue: Arc<tokio::sync::Semaphore>,
    /// How long one lookup may take.
    request_timeout: Duration,
}

impl SearchFindLimits {
    fn new(permits: usize, queue: usize) -> Self {
        Self {
            permits: Arc::new(tokio::sync::Semaphore::new(permits)),
            queue: Arc::new(tokio::sync::Semaphore::new(queue)),
            request_timeout: SEARCH_FIND_REQUEST_TIMEOUT,
        }
    }
}

static SEARCH_FIND_LIMITS: std::sync::LazyLock<SearchFindLimits> =
    std::sync::LazyLock::new(|| {
        SearchFindLimits::new(SEARCH_FIND_CONCURRENCY, SEARCH_FIND_QUEUE_MAX)
    });
/// How long a search waits on its `/find` lookups.
#[cfg(not(test))]
const SEARCH_FIND_BUDGET: Duration = Duration::from_millis(300);
/// Generous, so a loaded test host still sees every lookup land.
#[cfg(test)]
const SEARCH_FIND_BUDGET: Duration = Duration::from_secs(5);
/// Outlives any one `/find`; the lookup clears it sooner.
const SEARCH_FIND_RUNNING_TTL: Duration = Duration::from_secs(60);

/// Most imdb ids `SEARCH_FIND_ANSWERS` holds, at a few dozen bytes each.
const SEARCH_FIND_ANSWERS_MAX: u64 = 20_000;
/// TMDB's `/find` answers for search dedupe, by imdb id. Held apart from the
/// shared response cache: a library refresh fills that with season and
/// details responses, which evicted 94% of these answers live and left
/// searches to ask again within the budget.
static SEARCH_FIND_ANSWERS: std::sync::LazyLock<remux_utils::Store> =
    std::sync::LazyLock::new(|| remux_utils::Store::new(SEARCH_FIND_ANSWERS_MAX));

/// The tmdb ids TMDB's `/find` places an imdb id as.
#[derive(Clone, Copy, Debug, Default)]
struct SearchFindAnswer {
    movie: Option<i64>,
    tv: Option<i64>,
}

impl SearchFindAnswer {
    /// Keeps `found` in `SEARCH_FIND_ANSWERS` for as long as the response
    /// cache would.
    fn remember(imdb: &str, found: &sdks::tmdb::FindByIdResponse) -> Self {
        let answer = Self {
            movie: MediaResolveService::find_first_of_kind(found, false),
            tv: MediaResolveService::find_first_of_kind(found, true),
        };
        SEARCH_FIND_ANSWERS.save(
            imdb,
            answer,
            crate::services::resolve::find_cache_ttl(found),
        );
        answer
    }

    fn tmdb_id(self, is_tv: bool) -> Option<i64> {
        if is_tv { self.tv } else { self.movie }
    }
}

/// A running `/find`'s answer: the tmdb id, or `None` when TMDB doesn't place
/// the imdb id as the searched kind or the lookup failed.
type SearchFind = futures::future::Shared<tokio::sync::oneshot::Receiver<Option<i64>>>;
/// What the store holds of a `SearchFind`, so the handles left are the
/// lookup's own and those of the searches waiting on it.
type WeakSearchFind =
    futures::future::WeakShared<tokio::sync::oneshot::Receiver<Option<i64>>>;

/// Store key holding the running `/find` for `imdb` as `kind`, which
/// overlapping searches (keystrokes) join instead of sending it again.
fn search_find_running_key(kind: &db::MediaKind, imdb: &str) -> String {
    format!("search_find_running:{kind}:{imdb}")
}

/// `tt` and digits, as imdb ids are. An addon's id is spliced into the
/// `/find` url path as is, so `tt1/../../movie/550` asked for another
/// endpoint and `tt1375666?x=1` got Inception's answer.
fn is_imdb_id(id: &str) -> bool {
    id.strip_prefix("tt")
        .is_some_and(|digits| {
            !digits.is_empty()
                && digits
                    .bytes()
                    .all(|b| b.is_ascii_digit())
        })
}

/// Lowercased letters and digits: "The Wire!" and "the wire" match.
fn title_key(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The imdb ids of a Movie/Series search, deduped by
/// `dedupe_holding_find_candidates_apart`, worth asking TMDB's `/find` about,
/// so a Cinemeta-style copy (imdb id alone) can match the TMDB copy of the
/// same title (tmdb id alone). Only `is_find_candidate` results, and only
/// when a tmdb-only result is there to match, or for one sharing its imdb id
/// with a result that has a tmdb id: the final dedupe merges those through
/// the imdb id, and only the candidate's own tmdb id shows whether they are
/// one title. A TMDB copy absorbed into such a result by the first dedupe,
/// or none at all, otherwise left an imdb-only Interstellar hidden behind an
/// Inception copy carrying its imdb id.
///
/// Merges drop no more results ahead of a result than there are imdb-only
/// results ahead of it, so an imdb-only result can make the returned `limit`
/// only while fewer than `limit` other results precede it; all of those are
/// asked about. Ones the final dedupe would merge or titled like a tmdb-only
/// result go first, being the likeliest twins should the budget run out.
fn search_find_candidates<'a>(
    kind: &db::MediaKind,
    results: &'a [db::Media],
    limit: usize,
) -> Vec<&'a db::NonEmptyString> {
    let mut tmdb_titles = std::collections::HashSet::new();
    // Imdb ids of results that also have a tmdb id.
    let mut linked_imdbs = std::collections::HashSet::new();
    let mut candidates: Vec<(&db::NonEmptyString, String)> = Vec::new();
    let mut others = 0;
    for m in results {
        let ids = &m.external_ids;
        if is_find_candidate(kind, m) {
            // Deduped among themselves, so each imdb id shows up once.
            if let Some(imdb) = &ids.imdb
                && others < limit
            {
                candidates.push((imdb, title_key(&m.title)));
            }
            continue;
        }
        if m.kind == *kind
            && ids
                .tmdb
                .is_some()
        {
            match &ids.imdb {
                Some(imdb) => {
                    linked_imdbs.insert(imdb);
                }
                None => {
                    tmdb_titles.insert(title_key(&m.title));
                }
            }
        }
        others += 1;
    }
    if tmdb_titles.is_empty() {
        candidates.retain(|(imdb, _)| linked_imdbs.contains(imdb));
    }
    // Stable: position order within each group.
    candidates.sort_by_key(|(imdb, title)| {
        !(linked_imdbs.contains(imdb) || tmdb_titles.contains(title))
    });
    candidates
        .into_iter()
        .map(|(imdb, _)| imdb)
        .collect()
}

/// Whether `/find` may give `m` a tmdb id in a search for `kind`: an imdb id
/// alone, well-formed (see `is_imdb_id`). The answer is the searched kind's
/// tmdb id, so only for results of that kind: a movie-typed meta in a Series
/// search took The Wire's tv id 1438 and opened as the movie 1438.
fn is_find_candidate(kind: &db::MediaKind, m: &db::Media) -> bool {
    let ids = &m.external_ids;
    m.kind == *kind
        && ids
            .tmdb
            .is_none()
        && ids
            .imdb
            .as_ref()
            .is_some_and(|imdb| is_imdb_id(imdb))
}

/// `dedupe_by_identity`, except `/find` candidates merge only with each
/// other. One sharing just its imdb id with a result that has a tmdb id may
/// be another title under a wrong imdb id, which only its own `/find` answer
/// tells: merged into an Inception copy carrying Interstellar's imdb id, an
/// imdb-only Interstellar was never asked about and stayed hidden. The final
/// dedupe, after the fill, groups them by what TMDB answered.
fn dedupe_holding_find_candidates_apart(
    kind: &db::MediaKind,
    results: Vec<db::Media>,
) -> Vec<db::Media> {
    let mut positions: [Vec<usize>; 2] = Default::default();
    let mut parts: [Vec<db::Media>; 2] = Default::default();
    for (i, m) in results
        .into_iter()
        .enumerate()
    {
        let part = usize::from(is_find_candidate(kind, &m));
        positions[part].push(i);
        parts[part].push(m);
    }
    let mut kept: Vec<(usize, db::Media)> = Vec::new();
    for (positions, part) in positions
        .into_iter()
        .zip(parts)
    {
        // Items with one id always group, and a group keeps its first item,
        // so a kept item is the first with its id.
        let mut first: HashMap<Uuid, usize> = HashMap::with_capacity(part.len());
        for (&i, m) in positions
            .iter()
            .zip(&part)
        {
            first
                .entry(m.id)
                .or_insert(i);
        }
        kept.extend(
            db::Media::dedupe_by_identity(part)
                .into_iter()
                .map(|m| (first[&m.id], m)),
        );
    }
    kept.sort_unstable_by_key(|(i, _)| *i);
    kept.into_iter()
        .map(|(_, m)| m)
        .collect()
}

/// TMDB's `(imdb, tmdb)` mapping for the `search_find_candidates` of a
/// Movie/Series search. TMDB is asked rather than trusting an addon's cross
/// id: Cinemeta's `moviedb_id` names the wrong title for some items.
///
/// Answers already known, misses included, are taken as is: from
/// `SEARCH_FIND_ANSWERS`, else from the response cache, which also holds
/// those of opened items. The rest are queued for a permit of `limits` while
/// the queue has room; those landing within `budget` are used. A lookup that
/// got its permit runs on past the budget, so a slow one still lands for the
/// next keystroke, and one another search already started is joined, not
/// resent. One still queued once no search waits on it is dropped unsent.
async fn tmdb_ids_via_find(
    kind: &db::MediaKind,
    results: &[db::Media],
    limit: usize,
    budget: Duration,
    limits: &SearchFindLimits,
    ctx: &AppContext,
) -> Vec<(db::NonEmptyString, i64)> {
    let deadline = tokio::time::Instant::now() + budget;
    let is_tv = *kind == db::MediaKind::Series;
    let mut found = Vec::new();
    let mut unknown = Vec::new();
    for imdb in search_find_candidates(kind, results, limit) {
        match SEARCH_FIND_ANSWERS.get::<SearchFindAnswer>(imdb.as_str()) {
            Some(answer) => found.extend(
                answer
                    .tmdb_id(is_tv)
                    .map(|tmdb| (imdb.clone(), tmdb)),
            ),
            None => unknown.push(imdb),
        }
    }
    if unknown.is_empty() {
        return found;
    }
    let Ok(Some(client)) = tokio::time::timeout_at(
        deadline,
        crate::common::tmdb_search_find_client(
            &ctx.db,
            &ctx.config
                .tmdb_base_url,
        ),
    )
    .await
    else {
        return found;
    };
    let mut owned = Vec::new();
    let mut lookups = futures::stream::FuturesUnordered::new();
    for imdb in unknown {
        // Never queued behind lookups that still have to ask TMDB.
        if let Some(cached) =
            MediaResolveService::cached_find_by(imdb.to_string(), "imdb_id", &client)
        {
            found.extend(
                SearchFindAnswer::remember(imdb, &cached)
                    .tmdb_id(is_tv)
                    .map(|tmdb| (imdb.clone(), tmdb)),
            );
            continue;
        }
        let key = search_find_running_key(kind, imdb);
        let answer = loop {
            if let Some(running) = ctx
                .store
                .get::<WeakSearchFind>(&key)
            {
                if let Some(running) = running.upgrade() {
                    break Some(running);
                }
                // Left by a lookup that ended without clearing it.
                ctx.store
                    .delete(key.as_str());
            }
            let Ok(slot) = Arc::clone(&limits.queue).try_acquire_owned() else {
                break None;
            };
            let (tx, rx) = tokio::sync::oneshot::channel();
            let rx: SearchFind = futures::FutureExt::shared(rx);
            let Some(weak) = rx.downgrade() else {
                break None;
            };
            // Claims the key atomically.
            if ctx
                .store
                .insert(&key, weak, SEARCH_FIND_RUNNING_TTL)
            {
                owned.push((imdb.clone(), key, tx, slot, rx.clone()));
                break Some(rx);
            }
        };
        if let Some(answer) = answer {
            lookups.push(async move { (imdb, answer.await) });
        }
    }
    if !owned.is_empty() {
        let store = ctx
            .store
            .clone();
        let permits = Arc::clone(&limits.permits);
        let request_timeout = limits.request_timeout;
        // Detached, so lookups outliving the budget still land.
        tokio::spawn(async move {
            // Permits are taken in candidate order: title matches first.
            for (imdb, key, tx, slot, running) in owned {
                let Ok(permit) = Arc::clone(&permits)
                    .acquire_owned()
                    .await
                else {
                    return;
                };
                drop(slot);
                // `running` is the only handle left: every search waiting on
                // it ran out of budget while it was queued. After a stall,
                // sending those held up fresh searches' lookups for ~35 s.
                if running.strong_count() == Some(1) {
                    store.delete(key);
                    continue;
                }
                let client = client.clone();
                let store = store.clone();
                tokio::spawn(async move {
                    // Adds no load while TMDB is throttling the server.
                    tokio::time::sleep(
                        crate::common::tmdb_rate_limit()
                            .remaining_cooldown()
                            .await,
                    )
                    .await;
                    // A timed-out lookup is no answer: nothing is cached or
                    // remembered, so the next search asks again.
                    let answer = tokio::time::timeout(
                        request_timeout,
                        MediaResolveService::find_by(
                            imdb.to_string(),
                            "imdb_id",
                            &client,
                        ),
                    )
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .map(|found| SearchFindAnswer::remember(&imdb, &found));
                    drop(permit);
                    store.delete(key);
                    let _ = tx.send(answer.and_then(|a| a.tmdb_id(is_tv)));
                    // Joinable until now, when the answer is remembered.
                    drop(running);
                });
            }
        });
    }
    while let Ok(Some((imdb, answer))) =
        tokio::time::timeout_at(deadline, lookups.next()).await
    {
        if let Ok(Some(tmdb)) = answer {
            found.push((imdb.clone(), tmdb));
        }
    }
    found
}

/// A Movie/Series search's `results` with the tmdb ids TMDB's `/find` gives
/// their imdb-only copies filled in, ready for the final dedupe; deduped
/// first, so the lookups know which results can make `limit`.
async fn link_via_find(
    kind: &db::MediaKind,
    results: Vec<db::Media>,
    limit: usize,
    ctx: &AppContext,
) -> Vec<db::Media> {
    let mut results = dedupe_holding_find_candidates_apart(kind, results);
    let found = tmdb_ids_via_find(
        kind,
        &results,
        limit,
        SEARCH_FIND_BUDGET,
        &SEARCH_FIND_LIMITS,
        ctx,
    )
    .await;
    // Both copies get the id the other carries, so either matches a stored
    // row by whichever id that row has.
    let mut linked = Vec::new();
    for (imdb, tmdb) in found {
        for (i, m) in results
            .iter_mut()
            .enumerate()
        {
            // `/find` placed the imdb id as the searched kind only.
            if m.kind != *kind {
                continue;
            }
            let ids = &mut m.external_ids;
            if ids
                .tmdb
                .is_none()
                && ids
                    .imdb
                    .as_ref()
                    == Some(&imdb)
            {
                ids.tmdb = Some(tmdb);
            } else if ids
                .imdb
                .is_none()
                && ids.tmdb == Some(tmdb)
            {
                ids.imdb = Some(imdb.clone());
            } else {
                continue;
            }
            linked.push(i);
        }
    }
    // Adopted again: the first adoption only knew each copy's own ids, so a
    // stored imdb-only row hid behind the TMDB copy listed before it once the
    // two were linked.
    if !linked.is_empty() {
        let mut relinked: Vec<db::Media> = linked
            .iter()
            .map(|&i| std::mem::take(&mut results[i]))
            .collect();
        db::Media::adopt_existing_rows(&ctx.db, &mut relinked).await;
        for (i, m) in linked
            .into_iter()
            .zip(relinked)
        {
            results[i] = m;
        }
    }
    results
}

#[derive(Clone)]
pub struct AddonService {
    inner: Arc<ArcSwap<Vec<AddonRuntime>>>,
}

#[async_trait]
trait PickCap<T: ?Sized + Send + Sync> {
    async fn pick(&self, media: &db::Media) -> bool;
}

#[async_trait]
impl PickCap<dyn MetaAddon> for AddonRuntime {
    async fn pick(&self, media: &db::Media) -> bool {
        if !self
            .row
            .resources
            .contains(&ResourceType::Meta)
        {
            return false;
        }
        let Some(cap) = self
            .caps
            .meta
            .as_ref()
        else {
            return false;
        };
        let Ok(prefixes) = self
            .live_id_prefixes(&ResourceType::Meta)
            .await
        else {
            return false;
        };
        if let Some(prefixes) = prefixes.as_deref() {
            let gp_ext = media
                .grandparent
                .as_deref()
                .map(|gp| &gp.external_ids);
            let candidates = media.candidate_ids(gp_ext);
            if candidates.is_empty() {
                return false;
            }
            return candidates
                .iter()
                .any(|id| {
                    prefixes
                        .iter()
                        .any(|p| id.starts_with(p.as_str()))
                });
        }
        cap.supports(media)
            .await
    }
}

#[async_trait]
impl PickCap<dyn StreamAddon> for AddonRuntime {
    async fn pick(&self, media: &db::Media) -> bool {
        if !self
            .row
            .resources
            .contains(&ResourceType::Stream)
        {
            return false;
        }
        let Ok(prefixes) = self
            .live_id_prefixes(&ResourceType::Stream)
            .await
        else {
            return false;
        };
        if let Some(prefixes) = prefixes.as_deref() {
            let gp_ext = media
                .grandparent
                .as_deref()
                .map(|gp| &gp.external_ids);
            let candidates = media.candidate_ids(gp_ext);
            if candidates.is_empty() {
                return false;
            }
            return candidates
                .iter()
                .any(|id| {
                    prefixes
                        .iter()
                        .any(|p| id.starts_with(p.as_str()))
                });
        }
        match self
            .caps
            .stream
            .as_ref()
        {
            Some(cap) => cap.supports(media),
            None => false,
        }
    }
}

#[async_trait]
impl PickCap<dyn SubtitleAddon> for AddonRuntime {
    async fn pick(&self, media: &db::Media) -> bool {
        if !self
            .row
            .resources
            .contains(&ResourceType::Subtitles)
        {
            return false;
        }
        let Ok(prefixes) = self
            .live_id_prefixes(&ResourceType::Subtitles)
            .await
        else {
            return false;
        };
        if let Some(prefixes) = prefixes.as_deref() {
            let gp_ext = media
                .grandparent
                .as_deref()
                .map(|gp| &gp.external_ids);
            let candidates = media.candidate_ids(gp_ext);
            if candidates.is_empty() {
                return false;
            }
            return candidates
                .iter()
                .any(|id| {
                    prefixes
                        .iter()
                        .any(|p| id.starts_with(p.as_str()))
                });
        }
        match self
            .caps
            .subtitle
            .as_ref()
        {
            Some(cap) => cap.supports(media),
            None => false,
        }
    }
}

impl AddonService {
    async fn addons_for<T>(
        &self,
        media: &db::Media,
        db: &SqlitePool,
        user_id: Option<Uuid>,
    ) -> Vec<AddonRuntime>
    where
        T: ?Sized + Send + Sync + 'static,
        AddonRuntime: PickCap<T>,
    {
        let override_ids = match user_id {
            Some(uid) => match addon::user_addon_override(db, uid).await {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "failed to load user addon override");
                    return Vec::new();
                }
            },
            None => None,
        };
        let all = self
            .inner
            .load();
        let candidates: Vec<&AddonRuntime> = all
            .iter()
            .filter(|r| r.supports_type(&media.kind))
            .filter(|r| user_scoped(r, override_ids.as_deref()))
            .collect();
        let picks = futures::future::join_all(
            candidates
                .iter()
                .map(|r| PickCap::<T>::pick(*r, media)),
        )
        .await;
        let mut out: Vec<AddonRuntime> = candidates
            .into_iter()
            .zip(picks)
            .filter_map(|(r, picked)| picked.then(|| r.clone()))
            .collect();
        if let Some(ids) = &override_ids {
            out.sort_by_key(|r| {
                ids.iter()
                    .position(|id| {
                        *id == r
                            .row
                            .id
                    })
                    .unwrap_or(usize::MAX)
            });
        }
        out
    }

    /// The media tracker capability of one enabled addon, if it has one. Returns
    /// `None` once an addon is disabled or deleted, which is why queued
    /// deliveries treat that as permanent rather than retrying forever.
    pub fn media_tracker_for(
        &self,
        addon_id: Uuid,
    ) -> Option<Arc<dyn media_tracker::MediaTrackerAddon>> {
        self.inner
            .load()
            .iter()
            .find(|r| {
                r.row
                    .id
                    == addon_id
                    && r.row
                        .enabled
            })
            .and_then(|r| {
                r.caps
                    .media_tracker
                    .clone()
            })
    }

    /// Whether any enabled addon can track at all. Nothing can be connected to
    /// an addon that is not installed, so this answers without a query.
    pub fn has_media_tracker(&self) -> bool {
        self.inner
            .load()
            .iter()
            .any(|r| {
                r.row
                    .enabled
                    && r.caps
                        .media_tracker
                        .is_some()
            })
    }

    /// Swaps the live runtime list. Test-only: the real list is built from
    /// `registered_presets()`, which has no way to carry a stub addon, so
    /// without this seam the delivery path can only be exercised by shipping a
    /// provider.
    #[cfg(test)]
    pub fn replace_runtimes_for_test(&self, runtimes: Vec<AddonRuntime>) {
        self.inner
            .store(Arc::new(runtimes));
    }

    pub async fn list_for_user(
        &self,
        db: &SqlitePool,
        user_id: Option<Uuid>,
    ) -> Vec<AddonRuntime> {
        let override_ids = match user_id {
            Some(uid) => match addon::user_addon_override(db, uid).await {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "failed to load user addon override");
                    return Vec::new();
                }
            },
            None => None,
        };
        let mut out: Vec<AddonRuntime> = self
            .inner
            .load()
            .iter()
            .filter(|r| user_scoped(r, override_ids.as_deref()))
            .cloned()
            .collect();
        if let Some(ids) = &override_ids {
            out.sort_by_key(|r| {
                ids.iter()
                    .position(|id| {
                        *id == r
                            .row
                            .id
                    })
                    .unwrap_or(usize::MAX)
            });
        }
        out
    }

    pub async fn from_db(db: &SqlitePool, config: &crate::Config) -> Result<Self> {
        let runtimes = Self::load_runtimes(db, config).await?;
        Ok(Self {
            inner: Arc::new(ArcSwap::from_pointee(runtimes)),
        })
    }

    async fn load_runtimes(
        db: &SqlitePool,
        config: &crate::Config,
    ) -> Result<Vec<AddonRuntime>> {
        let presets = registered_presets();
        let addons = Addon::list(db).await?;
        let mut pending = Vec::new();

        for mut addon in addons
            .into_iter()
            .filter(|a| a.enabled)
        {
            let Some(preset) = presets
                .iter()
                .find(|p| {
                    p.id()
                        == addon
                            .preset
                            .kind
                })
            else {
                warn!(
                    addon_id = %addon.id,
                    kind = %addon.preset.kind,
                    "skipping addon with unknown preset kind"
                );
                continue;
            };
            match preset.from_cfg(
                addon.id,
                addon
                    .preset
                    .config
                    .expose(),
                config,
            ) {
                Ok(mut caps) => {
                    // Start with the preset's static metadata, then upgrade with live manifest data.
                    caps.metadata = preset.metadata();
                    pending.push((addon, caps));
                }
                Err(e) => warn!(
                    addon_id = %addon.id,
                    kind = %addon.preset.kind,
                    error = %e,
                    "failed to instantiate addon"
                ),
            }
        }

        // Manifests are fetched concurrently so N stalled addons cost one
        // timeout window, not N.
        let runtimes = futures::future::join_all(
            pending
                .into_iter()
                .map(|(addon, mut caps)| async move {
                    if let Some(kind) = caps
                        .kind
                        .clone()
                    {
                        match kind
                            .available_info()
                            .await
                        {
                            Ok(Some((resource_refs, raw_types))) => {
                                apply_manifest_info(
                                    &mut caps,
                                    resource_refs,
                                    raw_types,
                                );
                            }
                            Ok(None) => {}
                            Err(e) => {
                                caps.manifest_unreachable = true;
                                warn!(
                                    addon_id = %addon.id,
                                    name = %addon.name,
                                    error = %e,
                                    "failed to fetch addon manifest at load time"
                                );
                            }
                        }
                    }
                    AddonRuntime { row: addon, caps }
                }),
        )
        .await;
        Ok(runtimes)
    }

    pub async fn reload(&self, db: &SqlitePool, config: &crate::Config) -> Result<()> {
        let runtimes = Self::load_runtimes(db, config).await?;
        self.inner
            .store(Arc::new(runtimes));
        Ok(())
    }

    pub fn list(&self) -> arc_swap::Guard<Arc<Vec<AddonRuntime>>> {
        self.inner
            .load()
    }

    pub fn get(&self, id: Uuid) -> Option<AddonRuntime> {
        self.inner
            .load()
            .iter()
            .find(|r| {
                r.row
                    .id
                    == id
            })
            .cloned()
    }

    pub fn catalog_addons(&self) -> Vec<AddonRuntime> {
        self.inner
            .load()
            .iter()
            .filter(|r| {
                r.catalog
                    .is_some()
            })
            .cloned()
            .collect()
    }

    /// Returns `(addon, catalogs)` pairs for every catalog-capable addon that could
    /// produce any of `kinds`, with each addon's catalog list already filtered down to
    /// catalogs whose own `media_kind` is one of `kinds`. Addons are pre-filtered via
    /// `supports_type` as a cheap upper-bound check before calling `catalog_list()`
    /// (which may hit the network); per-addon listing errors are logged and skipped.
    pub async fn catalogs_for_kinds(
        &self,
        ctx: &AppContext,
        kinds: &[db::MediaKind],
    ) -> Vec<(AddonRuntime, Vec<ResolvedCatalog>)> {
        let mut out = Vec::new();
        for runtime in self
            .catalog_addons()
            .into_iter()
            .filter(|r| {
                kinds
                    .iter()
                    .any(|k| r.supports_type(k))
            })
        {
            let addon_id = runtime
                .row
                .id;
            let resolved = match runtime
                .resolve_catalogs(ctx)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    warn!(addon = %addon_id, error = %e, "failed to list addon catalogs, skipping");
                    continue;
                }
            };
            if resolved.is_empty() {
                continue;
            }
            out.push((runtime, resolved));
        }
        out
    }

    pub async fn purge_indexes(&self, ctx: &AppContext) -> Result<()> {
        let addons: Vec<AddonRuntime> = self
            .inner
            .load()
            .iter()
            .cloned()
            .collect();
        for runtime in &addons {
            if let Some(index) = &runtime.index {
                if let Err(e) = index
                    .purge_index(ctx, &runtime.row)
                    .await
                {
                    warn!(addon = %runtime.row.name, error = %e, "purge_index failed");
                }
            }
        }
        Ok(())
    }

    /// Rough per-addon item-count guess when an addon has never been scanned
    /// before (so `IndexAddon::index_estimate` has nothing to go on yet) —
    /// just enough to size the progress total sensibly; corrected against the
    /// real count as soon as that addon's own scan completes.
    const INDEX_ESTIMATE_FALLBACK: usize = 250;

    fn indexable_addons(&self) -> Vec<AddonRuntime> {
        self.inner
            .load()
            .iter()
            .filter(|r| {
                r.row
                    .enabled
                    && r.index
                        .is_some()
            })
            .cloned()
            .collect()
    }

    /// Sum of each indexable addon's best-known item count (its last real
    /// scan size, or a fallback guess when it's never been scanned) — sizes
    /// `RefreshLibrary`'s overall progress total before `refresh_indexes` has
    /// actually run.
    pub async fn estimate_index_items(&self, ctx: &AppContext) -> usize {
        let mut total = 0usize;
        for runtime in self.indexable_addons() {
            let Some(index) = &runtime.index else {
                continue;
            };
            total += index
                .index_estimate(ctx, &runtime.row)
                .await
                .unwrap_or(Self::INDEX_ESTIMATE_FALLBACK);
        }
        total
    }

    /// Refreshes every enabled addon's index, weighting each addon's slice of
    /// `item_progress` by its own item-count estimate (correcting that
    /// estimate against the real count once its scan completes) rather than
    /// splitting the range evenly — an addon that's disabled or has nothing to
    /// do no longer eats an equal share of the bar regardless of its actual
    /// size. Returns the real total item count indexed, for the caller to use
    /// as the base offset for the next phase.
    pub async fn refresh_indexes(
        &self,
        ctx: &AppContext,
        item_progress: &ItemProgress,
        base: usize,
    ) -> Result<usize> {
        let addons = self.indexable_addons();
        info!(addons = addons.len(), "starting index refresh");
        let start = std::time::Instant::now();
        let mut offset = base;
        for runtime in &addons {
            let Some(index) = &runtime.index else {
                continue;
            };
            let estimate = index
                .index_estimate(ctx, &runtime.row)
                .await
                .unwrap_or(Self::INDEX_ESTIMATE_FALLBACK);
            let sub = item_progress.child(offset, estimate);
            if let Err(e) = index
                .refresh_index(ctx, &runtime.row, sub.clone())
                .await
            {
                warn!(addon = %runtime.row.name, error = %e, "refresh_index failed");
            }
            // Always finish this addon's own slice, whether it succeeded or
            // failed — otherwise a failure can leave the bar sitting at this
            // addon's starting point until unrelated later work moves it.
            sub.set(100.0);
            let actual = index
                .index_estimate(ctx, &runtime.row)
                .await
                .unwrap_or(estimate);
            item_progress.adjust_total(actual as i64 - estimate as i64);
            offset += actual;
        }
        info!(addons = addons.len(), elapsed = ?start.elapsed(), items = offset - base, "index refresh complete");
        Ok(offset - base)
    }

    pub fn get_catalog(&self, id: Uuid) -> Option<Arc<dyn CatalogAddon>> {
        self.inner
            .load()
            .iter()
            .find(|r| {
                r.row
                    .id
                    == id
            })
            .and_then(|r| {
                r.catalog
                    .clone()
            })
    }

    /// Return the tags configured for a specific catalog within an addon.
    pub fn catalog_tags(&self, addon_uuid: &str, local_cat_id: &str) -> Vec<String> {
        let Ok(id) = Uuid::parse_str(addon_uuid) else {
            return vec![];
        };
        self.inner
            .load()
            .iter()
            .find(|r| {
                r.row
                    .id
                    == id
            })
            .map(|r| {
                r.row
                    .catalog_states()
                    .get(local_cat_id)
                    .map(|s| {
                        s.tags
                            .clone()
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }

    pub fn make_catalog_stream(
        &self,
        media_id: &str,
    ) -> Option<Box<dyn RemoteMediaStream>> {
        let rest = media_id.strip_prefix("addon:")?;
        let (uuid_str, local_id) = rest.split_once(':')?;
        let id = Uuid::parse_str(uuid_str).ok()?;
        let addon = self
            .inner
            .load()
            .iter()
            .find(|r| {
                r.row
                    .id
                    == id
            })
            .and_then(|r| {
                r.catalog
                    .clone()
            })?;
        Some(Box::new(AddonCatalogStream {
            addon,
            local_id: local_id.to_string(),
        }))
    }

    #[tracing::instrument(level = "debug", target = "remux_server::metadata_refresh", skip_all, fields(id = %media.id, title = %media.title, kind = %media.kind, force_refresh))]
    pub async fn refresh_meta(
        &self,
        media: &mut db::Media,
        ctx: &AppContext,
        force_refresh: bool,
        config: &api::ServerConfiguration,
    ) -> Result<()> {
        media
            .grandparent(&ctx.db)
            .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "grandparent_lookup"))
            .await
            .ok();

        // Fill in whatever external ids we can before any addon runs, so
        // every addon in this batch sees the fuller id set rather than each
        // doing its own partial, addon-specific resolution.
        //
        // Seasons and episodes already carry the TMDB identity needed by their
        // metadata providers. Do not turn a metadata tree refresh into a
        // per-child external-ID enrichment job; that remains available to
        // explicit callers of `resolve_external_ids` when it is actually needed.
        let resolves_external_ids =
            !matches!(media.kind, db::MediaKind::Season | db::MediaKind::Episode);
        if resolves_external_ids {
            MediaResolveService::resolve_external_ids(media, ctx, false)
                .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "resolve_external_ids"))
                .await;
        }

        let applicable = self
            .addons_for::<dyn MetaAddon>(media, &ctx.db, None)
            .await;

        trace!(
            target: "remux_server::metadata_refresh",
            id = %media.id,
            title = %media.title,
            kind = %media.kind,
            addons = %applicable.iter().map(|r| r.row.name.as_str()).collect::<Vec<_>>().join(", "),
            "metadata refresh addons selected"
        );

        if applicable.is_empty() {
            return Ok(());
        }

        // Cap on a single addon's `meta_fetch` call, below. Some addons
        // (observed: AIO, proxying to a third-party `aiometadata` backend)
        // hang up to their own ~30s upstream timeout under load; without
        // this, one bad addon stalls the whole item even though the other
        // addons in the same fan-out already finished.
        let addon_fetch_timeout = config.addon_fetch_timeout();
        let media_ref: &db::Media = media;
        let results = async {
        futures::future::join_all(
            applicable
                .iter()
                .map(|r| {
                    let addon = r
                        .row
                        .name
                        .clone();
                    let span = tracing::debug_span!(target: "remux_server::metadata_refresh", "addon_meta_fetch", addon = %addon);
                    async move {
                        let meta_addon = r
                            .meta
                            .as_ref()
                            .unwrap();
                        // A shared upstream cooldown (see `SharedRateLimit`) that outlasts
                        // our own timeout would otherwise consume the entire budget just
                        // waiting for it to clear, dying before a request is ever sent —
                        // indistinguishable from a genuinely hung addon, but really just
                        // the rate limiter doing its job. Skip the attempt entirely rather
                        // than let that masquerade as a failure.
                        let cooldown = meta_addon
                            .rate_limit_cooldown()
                            .await;
                        if cooldown >= addon_fetch_timeout {
                            debug!(
                                addon = %addon,
                                cooldown = ?cooldown,
                                "skipping addon fetch: shared rate limit cooldown exceeds timeout"
                            );
                            return Ok(None);
                        }
                        fetch_meta_with_retry(
                            meta_addon.as_ref(),
                            &addon,
                            media_ref,
                            ctx,
                            config,
                            addon_fetch_timeout,
                        )
                        .await
                    }
                    .instrument(span)
                }),
        )
        .await
        }
        .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "addon_meta_fetch_all", addons = applicable.len()))
        .await;

        // Accumulate all addon patches into a fresh empty object so the
        // highest-priority addon (first in list, lowest priority number) wins
        // each field — later addons only fill gaps. The real `media` stays
        // untouched until the combined result is applied once at the end,
        // where `force_refresh` controls whether existing values are replaced.
        let mut combined: Option<db::Media> = None;
        for (r, result) in applicable
            .iter()
            .zip(results)
        {
            match result {
                Ok(Some(patch)) => {
                    let acc = combined.get_or_insert_with(db::Media::default);
                    apply_meta(acc, patch, false);
                }
                Ok(None) => {}
                Err(e) => {
                    error!(addon = %r.row.name, error = %e, "meta addon error")
                }
            }
        }
        if let Some(combined) = combined {
            apply_meta(media, combined, force_refresh);
        }

        // Apply SxxExx / "Season N" title formatting once, after all patches are merged.
        // Calling it inside apply_meta would re-apply the prefix on every patch.
        apply_title_format(media);

        // External IDs resolved by meta enrichment may now match a row already
        // in the DB under a different id (e.g. discovered via a different
        // addon first) — adopt it rather than drifting into a duplicate.
        // Season/Episode identity is anchored to parent_id, not external IDs —
        // skip them.
        if matches!(media.kind, db::MediaKind::Movie | db::MediaKind::Series) {
            if let Some(existing_id) =
                db::Media::find_existing_id_by_ext(&ctx.db, media)
                    .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "find_existing_id_by_ext"))
                    .await
            {
                media.id = existing_id;
            }
        }
        if media.kind == db::MediaKind::Person {
            if let Some(tmdb_id) = media
                .external_ids
                .tmdb
            {
                media.id = crate::common::stable_media_uuid(
                    &db::MediaKind::Person,
                    &tmdb_id.to_string(),
                );
            }
        }

        media.refreshed_at = Some(chrono::Utc::now().naive_utc());

        Ok(())
    }

    pub fn get_tree(
        &self,
        root: db::Media,
        ctx: &AppContext,
    ) -> impl futures::Stream<Item = db::Media> + 'static {
        let svc = self.clone();
        let ctx = ctx.clone();
        async_stream::stream! {
            let mut seen = std::collections::HashSet::new();
            seen.insert(root.id);
            let root_title = root.title.clone();
            let root_id = root.id;
            let mut queue = vec![root];
            let mut total_yielded = 0usize;

            while let Some(node) = queue.pop() {
                let applicable: Vec<Arc<dyn TreeAddon>> = svc
                    .inner
                    .load()
                    .iter()
                    .filter_map(|r| {
                        if !r
                            .tree
                            .as_ref()
                            .map(|t| t.supports(&node))
                            .unwrap_or(false)
                        {
                            return None;
                        }
                        if let Some(prefixes) = r.resource_id_prefixes(&ResourceType::Meta)
                        {
                            let gp_ext = node
                                .grandparent
                                .as_deref()
                                .map(|gp| &gp.external_ids);
                            let candidates = node.candidate_ids(gp_ext);
                            if candidates.is_empty()
                                || !candidates
                                    .iter()
                                    .any(|id| prefixes.iter().any(|p| id.starts_with(p.as_str())))
                            {
                                return None;
                            }
                        }
                        r.tree
                            .as_ref()
                            .cloned()
                    })
                    .collect();

                for addon in &applicable {
                    match addon
                        .get_children(&node, &ctx)
                        .await
                    {
                        Ok(Some(children)) if !children.is_empty() => {
                            for child in children {
                                if seen.insert(child.id) {
                                    let is_leaf = matches!(
                                        child.kind,
                                        db::MediaKind::Episode | db::MediaKind::Track
                                    );
                                    if !is_leaf {
                                        queue.push(child.clone());
                                    }
                                    total_yielded += 1;
                                    yield child;
                                }
                            }
                            break;
                        }
                        Ok(_) => continue,
                        Err(e) => {
                            debug!(id = %node.id, error = %e, "get_children failed");
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Processes each item's metadata/tree and upserts it, returning the map of
    /// `original_id -> final_id` for every item that was passed in.
    ///
    /// The two ids differ whenever `process_meta_item_inner` adopts an existing
    /// row's UUID via `find_existing_id_by_ext` (dedup by external ID) instead of
    /// writing under the caller-computed id. Callers that recorded anything keyed
    /// on the pre-call id (catalog membership rows, stale-member diffs, etc.) MUST
    /// remap through this before using it — the row was upserted under `final_id`,
    /// not `original_id`.
    #[tracing::instrument(level = "debug", target = "remux_server::metadata_refresh", skip_all, fields(items = media.len(), force_refresh))]
    pub async fn process_meta_batch(
        &self,
        media: Vec<db::Media>,
        ctx: &AppContext,
        force_refresh: bool,
        on_item_done: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Result<HashMap<Uuid, Uuid>> {
        use futures::StreamExt;

        let config = db::Settings::get_config_or_default(&ctx.db).await;
        // Clamp before casting: a persisted/API-set 0 or negative value would
        // otherwise stall `buffer_unordered` (0) or wrap around to near
        // `usize::MAX` (negative), not just fail to throttle.
        let concurrency = config
            .meta_concurrency
            .max(1) as usize;
        trace!(
            target: "remux_server::metadata_refresh",
            items = media.len(),
            force_refresh,
            concurrency,
            "processing metadata batch"
        );
        let config = Arc::new(config);
        // Shared across this whole batch — top-level items, and every season/
        // episode any of them refreshes — so nested fan-out inside a single
        // item's own tree walk can't multiply past this budget. See
        // `process_meta_item_inner` for where seasons/episodes acquire from it.
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));

        let svc = self.clone();
        let ctx_owned = ctx.clone();

        let mut stream = std::pin::pin!(
            futures::stream::iter(media)
                .map(move |m| {
                    let svc = svc.clone();
                    let ctx = ctx_owned.clone();
                    let cfg = Arc::clone(&config);
                    let sem = Arc::clone(&semaphore);
                    let original_id = m.id;
                    async move {
                        let final_id = svc
                            .process_meta_item(m, ctx, force_refresh, cfg, sem)
                            .await;
                        (original_id, final_id)
                    }
                })
                .buffer_unordered(concurrency)
        );

        let mut id_map = HashMap::new();
        while let Some((original_id, final_id)) = stream
            .next()
            .await
        {
            if let Some(ref f) = on_item_done {
                f();
            }
            id_map.insert(original_id, final_id);
        }

        Ok(id_map)
    }

    /// Fetch the direct children of `node` from the first applicable tree addon.
    /// Returns an empty vec if no addon supports this node or none return children.
    #[tracing::instrument(level = "debug", target = "remux_server::metadata_refresh", skip_all, fields(node_id = %node.id, node_kind = %node.kind))]
    async fn get_direct_children(
        &self,
        node: &db::Media,
        ctx: &AppContext,
        config: &api::ServerConfiguration,
    ) -> Vec<db::Media> {
        let fetch_timeout = config.addon_fetch_timeout();
        let applicable: Vec<Arc<dyn TreeAddon>> = self
            .inner
            .load()
            .iter()
            .filter_map(|r| {
                if !r
                    .row
                    .resources
                    .contains(&ResourceType::Meta)
                {
                    return None;
                }
                if !r
                    .tree
                    .as_ref()
                    .map(|t| t.supports(node))
                    .unwrap_or(false)
                {
                    return None;
                }
                if let Some(prefixes) = r.resource_id_prefixes(&ResourceType::Meta) {
                    let gp_ext = node
                        .grandparent
                        .as_deref()
                        .map(|gp| &gp.external_ids);
                    let candidates = node.candidate_ids(gp_ext);
                    if candidates.is_empty()
                        || !candidates
                            .iter()
                            .any(|id| {
                                prefixes
                                    .iter()
                                    .any(|p| id.starts_with(p.as_str()))
                            })
                    {
                        return None;
                    }
                }
                r.tree
                    .as_ref()
                    .cloned()
            })
            .collect();

        for addon in &applicable {
            let cooldown = addon
                .rate_limit_cooldown()
                .await;
            if cooldown >= fetch_timeout {
                debug!(
                    id = %node.id,
                    cooldown = ?cooldown,
                    "skipping tree fetch: shared rate limit cooldown exceeds timeout"
                );
                continue;
            }
            let result = tokio::time::timeout(
                fetch_timeout,
                addon
                    .get_children(node, ctx)
                    .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "addon_get_children")),
            )
            .await
            .unwrap_or_else(|_| {
                Err(anyhow!(
                    "addon get_children timed out after {:?}",
                    fetch_timeout
                ))
            });
            match result {
                Ok(Some(children)) if !children.is_empty() => {
                    debug!(children = children.len(), "get_direct_children fetched");
                    return children;
                }
                Ok(_) => continue,
                Err(e) => debug!(error = %e, "get_children failed"),
            }
        }
        vec![]
    }

    /// Load a map of `(kind_str, idx) → existing_uuid` for all direct children
    /// of `parent_id` that have a non-null idx. Used to adopt existing UUIDs
    /// for children arriving with a different computed UUID.
    /// Maps `(kind, idx)` -> `(id, refreshed_at)` for existing children of `parent_id`.
    /// Callers must adopt *both* fields when re-matching a freshly-parsed child from
    /// an addon to an existing row — the addon has no concept of `refreshed_at`, so a
    /// child that only adopts `id` always looks unrefreshed to `child_refresh_force`,
    /// forcing every episode to be refetched on every single pass.
    async fn child_uuid_map(
        &self,
        db: &sqlx::SqlitePool,
        parent_id: Uuid,
    ) -> std::collections::HashMap<(String, i64), (Uuid, Option<chrono::NaiveDateTime>)>
    {
        sqlx::query_as::<_, (String, Option<i64>, Uuid, Option<chrono::NaiveDateTime>)>(
            "SELECT CAST(kind AS TEXT), idx, id, refreshed_at FROM media WHERE parent_id = ? AND idx IS NOT NULL",
        )
        .bind(parent_id)
        .fetch_all(db)
        .await
        .inspect_err(|e| error!(parent_id = %parent_id, error = %e, "child_uuid_map query failed"))
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, idx, id, refreshed_at)| idx.map(|i| ((k, i), (id, refreshed_at))))
        .collect()
    }

    pub(crate) async fn process_meta_item(
        &self,
        media: db::Media,
        ctx: AppContext,
        force_refresh: bool,
        config: Arc<api::ServerConfiguration>,
        semaphore: Arc<tokio::sync::Semaphore>,
    ) -> Uuid {
        self.process_meta_item_inner(media, ctx, force_refresh, config, semaphore)
            .await
    }

    /// Total elapsed here (via span close) covers the whole tree walk for one
    /// root item — root refresh plus every child/grandchild — see the
    /// `children_refresh`/`children_upsert`/`grandchildren`/
    /// `grandchildren_refresh`/`grandchildren_upsert` spans below for the
    /// breakdown.
    #[tracing::instrument(level = "debug", target = "remux_server::metadata_refresh", skip_all, fields(id = %media.id, title = %media.title, kind = %media.kind, force_refresh))]
    async fn process_meta_item_inner(
        &self,
        mut media: db::Media,
        ctx: AppContext,
        force_refresh: bool,
        config: Arc<api::ServerConfiguration>,
        semaphore: Arc<tokio::sync::Semaphore>,
    ) -> Uuid {
        use futures::StreamExt;

        // Bounds how many season/episode tasks are polled concurrently within
        // this one item's own tree walk; actual network concurrency is capped
        // by `semaphore` regardless, so this only needs to be "large enough
        // to not artificially serialize" — reusing the same configured value
        // keeps it consistent with the outer batch's own concurrency knob.
        let concurrency = config
            .meta_concurrency
            .max(1) as usize;

        let original_id = media.id;

        let root_refresh_result = {
            let _permit = semaphore
                .acquire()
                .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "root_permit_wait"))
                .await
                .expect("semaphore is never closed");
            self.refresh_meta(&mut media, &ctx, force_refresh, &config)
                .await
        };
        if let Err(e) = root_refresh_result {
            warn!(id = %media.id, error = %e, "failed to refresh metadata, keeping as-is");
            if let Err(e) = db::Media::upsert(&ctx.db, &[media.clone()]).await {
                error!(id = %media.id, error = %e, "failed to upsert media");
            } else {
                save_pending_relations(&ctx, &[media.clone()]).await;
                save_pending_tags(&ctx, &[media.clone()]).await;
            }
            // Evict per-series caches/failure markers here too, not just on
            // the success paths below — `medias_cache` and `failed` are
            // scoped to the addon's own lifetime, not one refresh run, so a
            // series that hits this path and never reaches the eviction call
            // stays cached (or permanently blacklisted) across every future
            // refresh until the server restarts.
            self.notify_series_done(&media);
            return media.id;
        }

        // If this Person's ID was rewritten (name-keyed → tmdb-keyed) by refresh_meta,
        // delete the stale name-keyed row so it doesn't linger as a duplicate.
        if media.kind == db::MediaKind::Person && media.id != original_id {
            if let Err(e) = db::Media::delete(&ctx.db, &original_id).await {
                warn!(
                    old_id = %original_id,
                    new_id = %media.id,
                    error = %e,
                    "failed to delete stale name-keyed person row"
                );
            }
        }

        // Resolve actual UUID: adopt an existing DB row that shares any external ID.
        // Cascades stale parent_id / grandparent_id references in existing children
        // before adopting, so those rows stay attached to the correct parent UUID.
        let computed_id = media.id;
        let mut root_was_remapped = false;
        if let Some(existing_id) =
            db::Media::find_existing_id_by_ext(&ctx.db, &media).await
        {
            if existing_id != computed_id {
                if let Err(e) = db::Media::cascade_update_parent_refs(
                    &ctx.db,
                    computed_id,
                    existing_id,
                )
                .await
                {
                    warn!(old = %computed_id, new = %existing_id, error = %e,
                        "cascade_update_parent_refs failed");
                }
                media.id = existing_id;
                root_was_remapped = true;
            }
        }
        // Upsert root. Images are held back and attached separately below:
        // `Media::upsert` inserts `media_images` rows keyed on the item's own
        // `id` in the *same* transaction as the root row's insert, but a
        // per-kind external-id unique index violation (see the migration
        // that added them) can redirect that root insert onto a different
        // existing row via `ON CONFLICT DO UPDATE` — leaving an images
        // insert keyed on an id that was never actually written, which
        // fails the deferred `media_images.media_id` foreign key. Inserting
        // images after the id below is confirmed avoids that entirely.
        let pending_images = std::mem::take(&mut media.images);
        if let Err(e) = db::Media::upsert(&ctx.db, &[media.clone()]).await {
            // A unique-index violation here (rather than a plain PK conflict)
            // means one of `media`'s external IDs is already owned by a
            // *different* row than the one we just adopted — the ambiguous-
            // match resolver can only redirect onto one row, so a second,
            // disjoint match is left dangling. Merge that row into ours and
            // retry once instead of failing identically on every future
            // refresh.
            let is_unique_violation = matches!(
                e.downcast_ref::<sqlx::Error>(),
                Some(sqlx::Error::Database(db_err)) if db_err.is_unique_violation()
            );
            let merged = is_unique_violation
                && db::Media::merge_conflicting_duplicate(&ctx.db, &media).await;
            if !merged {
                error!(id = %media.id, error = %e, "failed to upsert root media");
                self.notify_series_done(&media);
                return media.id;
            }
            if let Err(e2) = db::Media::upsert(&ctx.db, &[media.clone()]).await {
                error!(id = %media.id, error = %e2,
                    "failed to upsert root media after duplicate merge");
                self.notify_series_done(&media);
                return media.id;
            }
        }

        // The upsert above may have silently landed on a different row than
        // `media.id`: `Media::upsert`'s `ON CONFLICT DO UPDATE` has no
        // conflict target, so SQLite fires it for ANY unique index
        // violation, not just the primary key — including the per-kind
        // external-id unique indexes. If another concurrent task committed
        // a row under a different id with the same external id between our
        // dedup check above and this upsert, our own insert gets silently
        // redirected onto that row instead of failing or creating a
        // duplicate. Re-check now (authoritative, since our own write just
        // committed) and correct our bookkeeping before anything downstream
        // (season/episode trees, catalog relations) keys off the wrong id.
        if let Some(existing_id) =
            db::Media::find_existing_id_by_ext(&ctx.db, &media).await
        {
            if existing_id != media.id {
                if let Err(e) = db::Media::cascade_update_parent_refs(
                    &ctx.db,
                    media.id,
                    existing_id,
                )
                .await
                {
                    warn!(old = %media.id, new = %existing_id, error = %e,
                        "cascade_update_parent_refs failed after post-upsert id correction");
                }
                media.id = existing_id;
                root_was_remapped = true;
            }
        }
        let actual_root_id = media.id;

        if !pending_images.is_empty() {
            media.images = pending_images;
            if let Err(e) = db::Media::upsert(&ctx.db, &[media.clone()]).await {
                warn!(id = %actual_root_id, error = %e, "failed to attach images to root media");
            }
        }

        // Build in-memory grandparent stub so children's refresh_meta calls can read
        // the series TMDB/IMDB ID and genres without hitting the DB.
        let gp_stub = {
            let mut gp = db::Media::default();
            gp.id = actual_root_id;
            gp.external_ids = media
                .external_ids
                .clone();
            if let Some(rels) = media
                .relations
                .as_ref()
            {
                let genre_rels: Vec<(db::MediaRelation, db::Media)> = rels
                    .iter()
                    .filter(|(_, m)| m.kind == db::MediaKind::Genre)
                    .cloned()
                    .collect();
                if !genre_rels.is_empty() {
                    gp.relations = Some(genre_rels);
                }
            }
            gp
        };
        // Shared as one Arc, not deep-cloned per child: every level-1 and
        // level-2 child below gets its own `.grandparent`, and cloning a
        // `Media` (including any embedded genre relations) into each of
        // potentially thousands of children is exactly what made large-tree
        // refreshes memory-heavy.
        let gp_stub = Arc::new(gp_stub);

        db::UserMediaState::remap_orphaned_for(&ctx.db, &[media.clone()]).await;
        save_pending_relations(&ctx, &[media.clone()]).await;
        save_pending_tags(&ctx, &[media.clone()]).await;

        let is_continuing = series_is_active(&media.status);

        // Level 1: direct children (Seasons, Albums, etc.) — see
        // `get_direct_children`'s own span for fetch timing.
        let raw_level1 = self
            .get_direct_children(&media, &ctx, &config)
            .await;
        if raw_level1.is_empty() {
            self.notify_series_done(&media);
            return actual_root_id;
        }

        // Always load existing level-1 UUIDs by (kind, idx) position. This lets us adopt
        // the stored UUID for any child whose UUID scheme changed (e.g. old series_imdb-
        // anchored → new parent_id-anchored). Also handles root-remap cases.
        let existing_l1 = self
            .child_uuid_map(&ctx.db, actual_root_id)
            .await;

        // Load ALL grandchild UUIDs (and refreshed_at) in one query keyed by
        // (parent_id, kind, idx). Avoids one query per season (O(n_seasons) → O(1)
        // queries). refreshed_at must travel with the id — see child_uuid_map's
        // doc comment for why leaving it behind defeats child_refresh_force.
        let existing_l2: std::collections::HashMap<
            (Uuid, String, i64),
            (
                Uuid,
                Option<chrono::NaiveDateTime>,
                Option<chrono::NaiveDateTime>,
            ),
        > = sqlx::query_as::<
            _,
            (
                Uuid,
                String,
                Option<i64>,
                Uuid,
                Option<chrono::NaiveDateTime>,
                Option<chrono::NaiveDateTime>,
            ),
        >(
            "SELECT parent_id, CAST(kind AS TEXT), idx, id, refreshed_at, digital_released_at
             FROM media WHERE grandparent_id = ? AND idx IS NOT NULL",
        )
        .bind(actual_root_id)
        .fetch_all(&ctx.db)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(pid, k, idx, id, refreshed_at, digital)| {
            idx.map(|i| ((pid, k, i), (id, refreshed_at, digital)))
        })
        .collect();

        // Seasons/albums fan out concurrently — each one's own `refresh_meta`
        // still throttles through `semaphore` (shared for the whole batch),
        // so this can't multiply past the configured budget the way an
        // independent per-level concurrency cap would.
        let level1: Vec<db::Media> = async {
        futures::stream::iter(raw_level1)
            .map(|mut child| {
                let svc = self.clone();
                let ctx = ctx.clone();
                let config = Arc::clone(&config);
                let semaphore = Arc::clone(&semaphore);
                let gp_stub = gp_stub.clone();
                let existing_l1 = &existing_l1;
                async move {
                    child.parent_id = Some(actual_root_id);
                    child.grandparent = Some(gp_stub);

                    // Adopt the existing DB UUID (and refreshed_at) for this (kind, idx)
                    // position if found. The new child UUID may differ from what's stored
                    // (due to UUID scheme changes or root remapping) — adopting the stored
                    // UUID avoids duplicate rows and keeps grandchild parent_id references
                    // intact. `refreshed_at` must also be adopted: this `child` was just
                    // freshly parsed from the addon's raw response, which has no concept
                    // of it, so leaving it unset makes `child_refresh_force` below treat
                    // an already-refreshed child as brand new every single pass.
                    if let Some(idx) = child.idx {
                        let key = (
                            child
                                .kind
                                .to_string(),
                            idx,
                        );
                        if let Some(&(existing_id, existing_refreshed_at)) =
                            existing_l1.get(&key)
                        {
                            child.refreshed_at = existing_refreshed_at;
                            if existing_id != child.id {
                                // When the root was remapped, cascade any references to the new
                                // child UUID (which was never in the DB) before adopting.
                                if root_was_remapped {
                                    if let Err(e) = db::Media::cascade_update_parent_refs(
                                        &ctx.db,
                                        child.id,
                                        existing_id,
                                    )
                                    .await
                                    {
                                        warn!(old = %child.id, new = %existing_id, error = %e,
                                            "cascade for child failed");
                                    }
                                }
                                child.id = existing_id;
                            }
                        }
                    }

                    let in_active_window = is_continuing
                        && matches!(child.kind, db::MediaKind::Episode)
                        && episode_in_active_window(&child);
                    if let Some(effective_force) =
                        child_refresh_force(force_refresh, in_active_window, &child)
                    {
                        let _permit = semaphore
                            .acquire()
                            .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "permit_wait"))
                            .await
                            .expect("semaphore is never closed");
                        if let Err(e) = svc
                            .refresh_meta(&mut child, &ctx, effective_force, &config)
                            .await
                        {
                            warn!(id = %child.id, error = %e, "failed to refresh child meta");
                        }
                    }
                    child
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await
        }
        .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "children_refresh"))
        .await;

        let mut level1_ok: Vec<&db::Media> = Vec::with_capacity(level1.len());
        async {
            for chunk in level1.chunks(db::CHUNK_SIZE) {
                if let Err(e) = db::Media::upsert(&ctx.db, chunk).await {
                    error!(error = %e, "failed to upsert children");
                } else {
                    db::UserMediaState::remap_orphaned_for(&ctx.db, chunk).await;
                    save_pending_relations(&ctx, chunk).await;
                    save_pending_tags(&ctx, chunk).await;
                    level1_ok.extend(chunk);
                }
            }
        }
        .instrument(tracing::debug_span!(
            target: "remux_server::metadata_refresh",
            "children_upsert",
            children = level1.len()
        ))
        .await;

        // Level 2: grandchildren (Episodes, Tracks, etc.) — one fetch+upsert
        // per level-1 child. Only process children whose level-1 upsert
        // succeeded to avoid orphaned rows. Different seasons' episode
        // batches fan out concurrently for the same reason level 1 does;
        // `semaphore` still bounds the real cost regardless.
        //
        // Each season's fetch/refresh/write are already their own spans
        // (`get_direct_children`, `refresh_meta`, `level2_write` below) —
        // `level2_season` just gives them a common per-season parent so a
        // trace viewer groups one season's work together.
        futures::stream::iter(level1_ok)
            .for_each_concurrent(concurrency, |child| {
                let svc = self.clone();
                let ctx = ctx.clone();
                let config = Arc::clone(&config);
                let semaphore = Arc::clone(&semaphore);
                let gp_stub = gp_stub.clone();
                let existing_l2 = &existing_l2;
                async move {
                    // Created here (first poll), not synchronously in the
                    // `for_each_concurrent` closure — a season stuck waiting
                    // for a concurrency slot before this future's first poll
                    // must not have that queue time counted as if it were
                    // this span's own idle time.
                    let child_span = tracing::debug_span!(target: "remux_server::metadata_refresh", "grandchildren", child_id = %child.id);
                    async move {
                    let actual_child_id = child.id;
                    let raw_level2 = svc
                        .get_direct_children(child, &ctx, &config)
                        .await;
                    if raw_level2.is_empty() {
                        return;
                    }

                    let episode_count = raw_level2.len();
                    let mut level2: Vec<db::Media> = Vec::with_capacity(episode_count);
                    async {
                    for mut gc in raw_level2 {
                        gc.parent_id = Some(actual_child_id);
                        gc.grandparent_id = Some(actual_root_id);
                        gc.grandparent = Some(gp_stub.clone());

                        // Adopt existing UUID + refreshed_at from the pre-loaded grandchild
                        // map. `gc` is freshly parsed from the addon's raw response, which
                        // has no concept of refreshed_at — without adopting it here too,
                        // child_refresh_force below always treats this episode as never
                        // refreshed, refetching it on every single pass.
                        let mut stored_digital = None;
                        if let Some(idx) = gc.idx {
                            let key = (
                                actual_child_id,
                                gc.kind
                                    .to_string(),
                                idx,
                            );
                            if let Some(&(
                                existing_id,
                                existing_refreshed_at,
                                existing_digital,
                            )) = existing_l2.get(&key)
                            {
                                gc.id = existing_id;
                                gc.refreshed_at = existing_refreshed_at;
                                stored_digital = existing_digital;
                            }
                        }

                        let in_active_window = is_continuing
                            && matches!(gc.kind, db::MediaKind::Episode)
                            && episode_in_active_window_with(&gc, stored_digital);
                        if let Some(effective_force) =
                            child_refresh_force(force_refresh, in_active_window, &gc)
                        {
                            let _permit = semaphore
                                .acquire()
                                .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "permit_wait"))
                                .await
                                .expect("semaphore is never closed");
                            if let Err(e) = svc
                                .refresh_meta(&mut gc, &ctx, effective_force, &config)
                                .await
                            {
                                warn!(id = %gc.id, error = %e, "failed to refresh grandchild meta");
                            }
                        }
                        level2.push(gc);
                    }
                    }
                    .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "grandchildren_refresh", grandchildren = episode_count))
                    .await;

                    async {
                        for chunk in level2.chunks(db::CHUNK_SIZE) {
                            if let Err(e) = db::Media::upsert(&ctx.db, chunk).await {
                                error!(error = %e, "failed to upsert grandchildren");
                            } else {
                                db::UserMediaState::remap_orphaned_for(&ctx.db, chunk).await;
                                save_pending_relations(&ctx, chunk).await;
                                save_pending_tags(&ctx, chunk).await;
                            }
                        }
                    }
                    .instrument(tracing::debug_span!(target: "remux_server::metadata_refresh", "grandchildren_upsert", grandchildren = level2.len()))
                    .await;
                    }
                    .instrument(child_span)
                    .await
                }
            })
            .await;

        self.notify_series_done(&media);
        actual_root_id
    }

    fn notify_series_done(&self, media: &db::Media) {
        if let Some(meta_id) = media
            .external_ids
            .stremio_lookup_id()
        {
            for r in self
                .inner
                .load()
                .iter()
            {
                if let Some(ref meta_addon) = r.meta {
                    meta_addon.on_series_done(&meta_id);
                }
            }
        }
    }

    pub async fn search(
        &self,
        kind: &db::MediaKind,
        query: &str,
        limit: usize,
        ctx: &AppContext,
        user_id: Option<Uuid>,
    ) -> Result<Vec<db::Media>> {
        let override_ids = match user_id {
            Some(uid) => addon::user_addon_override(&ctx.db, uid)
                .await
                .unwrap_or(None),
            None => None,
        };
        let addons: Vec<AddonRuntime> = self
            .inner
            .load()
            .iter()
            .filter(|r| {
                r.supports_type(kind)
                    && r.row
                        .resources
                        .contains(&ResourceType::Search)
                    && r.search
                        .is_some()
                    && user_scoped(r, override_ids.as_deref())
            })
            .cloned()
            .collect();

        // Ask every eligible addon concurrently and merge, so an empty answer
        // (Stremio addons always answer) can't hide the others' results.
        // Every addon gets the full configured timeout, even once others have
        // answered, so which results show up doesn't depend on timing races.
        let config = db::Settings::get_config_or_default(&ctx.db).await;
        let addon_fetch_timeout = config.addon_fetch_timeout();
        let mut searches = Vec::with_capacity(addons.len());
        for r in &addons {
            if let Some(search) = &r.search
                && search
                    .search_supports(kind)
                    .await
            {
                searches.push((r, search));
            }
        }
        // `join_all` keeps priority order; on a duplicate the first
        // (higher-priority) copy wins below.
        let per_addon: Vec<Vec<db::Media>> = futures::future::join_all(
            searches
                .iter()
                .map(|(r, search)| async move {
                    let answer = search.search(kind, query, limit, ctx);
                    match tokio::time::timeout(addon_fetch_timeout, answer).await {
                        Ok(Ok(results)) => return results.unwrap_or_default(),
                        // `{:#}`: adapters wrap the upstream error in context, so the
                        // cause is only in the chain.
                        Ok(Err(e)) => {
                            warn!(addon = %r.row.name, query, error = %format!("{e:#}"), "search addon error")
                        }
                        Err(_) => warn!(addon = %r.row.name, "search addon timed out"),
                    }
                    vec![]
                }),
        )
        .await;
        let answered = per_addon
            .iter()
            .filter(|r| !r.is_empty())
            .count();
        let mut results: Vec<db::Media> = per_addon
            .into_iter()
            .flatten()
            .collect();
        // Adopt before deduping: two addons' copies of a stored item end up
        // with the same id even when they carry different external ids.
        db::Media::adopt_existing_rows(&ctx.db, &mut results).await;
        // An imdb-only copy and a tmdb-only copy only match once TMDB maps
        // one onto the other. One addon doesn't list a title twice.
        if answered >= 2 && matches!(kind, db::MediaKind::Movie | db::MediaKind::Series)
        {
            results = link_via_find(kind, results, limit, ctx).await;
        }
        // Dedupe before truncating, so duplicates don't eat into the limit.
        let mut merged = db::Media::dedupe_by_identity(results);
        merged.truncate(limit);
        for m in &merged {
            ctx.store
                .save(m.id.to_string(), m.clone(), Duration::from_secs(3600));
        }
        Ok(merged)
    }

    #[tracing::instrument(skip_all, fields(title = %media.title, kind = %media.kind))]
    pub async fn fetch_images(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        options: ImageFetchOptions,
    ) -> Result<Vec<crate::api::RemoteImageInfo>> {
        let addons = self
            .addons_for::<dyn MetaAddon>(media, &ctx.db, None)
            .await;

        let mut out = Vec::new();
        for r in addons {
            match r
                .meta
                .as_ref()
                .unwrap()
                .images_fetch(media, ctx, options.clone())
                .await
            {
                Ok(images) => out.extend(images),
                Err(e) => {
                    warn!(addon = %r.row.name, error = %e, "images_fetch failed")
                }
            }
        }
        Ok(out)
    }

    /// Subtitle addon calls are slow network round-trips with no coalescing
    /// of their own (unlike `refresh_streams`' TTL + `STREAM_LOCKS`), so an
    /// Items detail fetch racing a PlaybackInfo call for the same item used
    /// to each pay the full addon fetch independently and concurrently.
    /// Cache the result briefly and serialize concurrent callers the same way.
    /// `query` asks addons for subtitles made for one release; the default
    /// query asks for the title's generic list. Each distinct query is cached
    /// on its own.
    #[tracing::instrument(skip_all, fields(title = %media.title, kind = %media.kind))]
    pub async fn fetch_subtitles(
        &self,
        media: &mut db::Media,
        query: &SubtitleQuery,
        ctx: &AppContext,
        background: bool,
        user_id: Option<Uuid>,
    ) -> Vec<SubtitleInfo> {
        const SUBTITLES_TTL: Duration = Duration::from_secs(5 * 60);
        const SUBTITLE_ADDON_TIMEOUT: Duration = Duration::from_secs(5);
        static SUBTITLE_LOCKS: KeyedLock<String> = KeyedLock::new();

        if media.kind == db::MediaKind::Episode {
            media
                .grandparent(&ctx.db)
                .await
                .ok();
        }

        let cache_key = format!(
            "addon-subtitles:{}:{}{}",
            media.id,
            user_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "anon".to_string()),
            query.cache_suffix()
        );
        if let Some(cached) = ctx
            .store
            .get::<Vec<SubtitleInfo>>(&cache_key)
        {
            return (*cached).clone();
        }

        let _guard = SUBTITLE_LOCKS
            .lock(cache_key.clone())
            .await;
        // Re-check after acquiring the lock — another task may have just
        // populated the cache while this one was waiting.
        if let Some(cached) = ctx
            .store
            .get::<Vec<SubtitleInfo>>(&cache_key)
        {
            return (*cached).clone();
        }

        let addons = self
            .addons_for::<dyn SubtitleAddon>(media, &ctx.db, user_id)
            .await;

        debug!(count = addons.len(), "subtitle addons matched");
        let instant = Instant::now();
        // Addons are queried concurrently so the slowest one sets the wait
        // instead of the sum, and each is capped so one hung addon can't
        // stall playback. Results keep addon order.
        let media_ref: &db::Media = media;
        let fetched = futures::future::join_all(
            addons
                .iter()
                .map(|r| async move {
                    debug!(addon = %r.row.name, "fetching subtitles from addon");
                    let result = tokio::time::timeout(
                        SUBTITLE_ADDON_TIMEOUT,
                        r.subtitle
                            .as_ref()
                            .unwrap()
                            .subtitle_fetch(media_ref, query, &ctx.db),
                    )
                    .await;
                    (r, result)
                }),
        )
        .await;
        let mut subs = vec![];
        for (r, result) in fetched {
            match result {
                Ok(Ok(s)) => {
                    debug!(addon = %r.row.name, count = s.len(), "subtitle addon returned results");
                    subs.extend(s);
                }
                Ok(Err(e)) => {
                    warn!(addon = %r.row.name, error = %e, "subtitle addon failed")
                }
                Err(_) => {
                    warn!(addon = %r.row.name, timeout = ?SUBTITLE_ADDON_TIMEOUT, "subtitle addon timed out")
                }
            }
        }
        if background {
            debug!(subs = subs.len(), addons = addons.len(), elapsed = ?instant.elapsed(), "subtitles fetched");
        } else {
            info!(subs = subs.len(), addons = addons.len(), elapsed = ?instant.elapsed(), "subtitles fetched");
        }
        // An addon skipped while its manifest is unreachable would be missing
        // from this result until the cache expires, even once it recovers.
        let mut skipped_unreachable = false;
        for r in self
            .inner
            .load()
            .iter()
            .filter(|r| {
                r.caps
                    .manifest_unreachable
                    && r.row
                        .resources
                        .contains(&ResourceType::Subtitles)
            })
        {
            if r.live_id_prefixes(&ResourceType::Subtitles)
                .await
                .is_err()
            {
                skipped_unreachable = true;
                break;
            }
        }
        if !skipped_unreachable {
            ctx.store
                .save(cache_key, subs.clone(), SUBTITLES_TTL);
        }
        subs
    }

    pub async fn get_streams(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        user_id: Option<Uuid>,
    ) -> Result<Vec<db::Media>> {
        let addons = self
            .addons_for::<dyn StreamAddon>(media, &ctx.db, user_id)
            .await;

        debug!(
            media_id = %media.id,
            media_kind = ?media.kind,
            addon_count = addons.len(),
            "resolving streams"
        );

        let tasks: Vec<_> = addons
            .into_iter()
            .map(|r| async move {
                let name = &r.row.name;
                let t = std::time::Instant::now();
                let id_prefixes = match r
                    .live_id_prefixes(&ResourceType::Stream)
                    .await
                {
                    Ok(p) => p.map(|p| p.into_owned()),
                    Err(e) => {
                        warn!(addon = %name, error = %e, "stream addon manifest unavailable");
                        return vec![];
                    }
                };
                match r
                    .stream
                    .as_ref()
                    .unwrap()
                    .get_streams(media, ctx, id_prefixes.as_deref())
                    .await
                {
                    Ok(mut streams) => {
                        let elapsed = t.elapsed();
                        if streams.is_empty() {
                            debug!(addon = %name, ?elapsed, "addon: no streams");
                        } else {
                            let sf = &r.row.service_filter;
                            if !sf.is_empty() {
                                streams.retain(|s| {
                                    let service_match = s.service_id
                                        .as_deref()
                                        .map(|id| sf.iter().any(|f| f.eq_ignore_ascii_case(id)))
                                        .unwrap_or(false);
                                    let addon_match = s.stream_addon
                                        .as_deref()
                                        .map(|a| sf.iter().any(|f| f.eq_ignore_ascii_case(a)))
                                        .unwrap_or(false);
                                    service_match || addon_match
                                });
                            }
                            debug!(addon = %name, count = streams.len(), ?elapsed, "addon: streams found");
                            let addon_id = r.row.id;
                            for s in &mut streams {
                                s.source = Some(name.clone());
                                s.addon_id = Some(addon_id);
                            }
                        }
                        streams
                    }
                    Err(e) => {
                        warn!(addon = %name, error = %e, elapsed = ?t.elapsed(), "stream addon failed");
                        vec![]
                    }
                }
            })
            .collect();
        let all: Vec<db::Media> = futures::future::join_all(tasks)
            .await
            .into_iter()
            .flatten()
            .map(db::Media::from)
            .collect();
        Ok(all)
    }

    fn stream_dedup_key(s: &db::Media) -> Option<String> {
        let si = s
            .stream_info
            .as_ref()?;
        match &si.descriptor {
            crate::stream::StreamDescriptor::Torrent {
                info_hash,
                file_hint,
                file_idx,
                ..
            } => {
                let file_key = file_hint
                    .as_deref()
                    .or(si
                        .filename
                        .as_deref())
                    .filter(|name| !name.is_empty())
                    .map(str::to_ascii_lowercase)
                    .unwrap_or_else(|| {
                        file_idx
                            .map(|index| format!("#{index}"))
                            .unwrap_or_default()
                    });
                Some(format!("torrent:{}:{file_key}", info_hash.to_lowercase()))
            }
            crate::stream::StreamDescriptor::Http { url, .. } => {
                let filename = si
                    .filename
                    .as_deref()
                    .unwrap_or("");
                let size = si
                    .size
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let binge_group = si
                    .binge_group
                    .as_deref()
                    .unwrap_or("");
                let service_id = si
                    .service_id
                    .as_deref()
                    .unwrap_or("");
                let addon_id = si
                    .addon_id
                    .map(|id| id.to_string())
                    .unwrap_or_default();

                if filename.is_empty()
                    && size.is_empty()
                    && binge_group.is_empty()
                    && service_id.is_empty()
                    && addon_id.is_empty()
                {
                    // Plain HTTP with no stable metadata — fall back to URL path.
                    let stable = url
                        .split('?')
                        .next()
                        .unwrap_or(url.as_str());
                    return Some(format!("http:{stable}"));
                }

                Some(format!(
                    "http:{}:{}:{}:{}:{}",
                    filename, size, binge_group, service_id, addon_id
                ))
            }
            crate::stream::StreamDescriptor::Local(path) => {
                Some(format!("local:{}", path.display()))
            }
            crate::stream::StreamDescriptor::Rtsp { url } => {
                Some(format!("rtsp:{url}"))
            }
            crate::stream::StreamDescriptor::Opendal { addon_id, path } => {
                Some(format!("opendal:{addon_id}:{path}"))
            }
        }
    }

    fn deduplicate_streams(streams: Vec<db::Media>) -> Vec<db::Media> {
        let mut seen = std::collections::HashSet::new();
        streams
            .into_iter()
            .filter(|stream| match Self::stream_dedup_key(stream) {
                Some(key) => seen.insert(key),
                None => true,
            })
            .collect()
    }
}

fn strip_video_ext(name: &str) -> &str {
    if let Some((stem, ext)) = name.rsplit_once('.') {
        if remux_sdks::remux::VideoContainer::parse_known(ext).is_some() {
            return stem;
        }
    }
    name
}

fn match_probe_version<'a>(
    versions: &'a [remuxdb::MediaInfo],
    stream: &db::Media,
) -> Option<&'a remuxdb::MediaInfo> {
    let si = stream
        .stream_info
        .as_ref()?;

    // Torrent: match by info_hash + file_idx. Covers both raw Torrent descriptors and
    // debrid Http streams where the hash comes from AIOStreams streamData.
    if let Some((hash, hash_file_idx)) = si.torrent_identity() {
        if let Some(v) = versions
            .iter()
            .find(|v| {
                v.sources
                    .iter()
                    .any(|s| {
                        s.torrent_info_hash
                            .as_deref()
                            == Some(hash)
                            && s.torrent_file_idx == hash_file_idx
                    })
            })
        {
            return Some(v);
        }
    }

    // Usenet: match by indexer_guid (+ indexer name when available) against version sources
    if let Some(ref guid) = si.usenet_guid {
        if let Some(v) = versions
            .iter()
            .find(|v| {
                v.sources
                    .iter()
                    .any(|s| {
                        s.indexer_guid
                            .as_deref()
                            == Some(guid.as_str())
                            && (si
                                .usenet_indexer
                                .is_none()
                                || s.indexer == si.usenet_indexer)
                    })
            })
        {
            return Some(v);
        }
    }

    // HTTP: match by exact file size
    if let Some(size) = si.size {
        if let Some(v) = versions
            .iter()
            .find(|v| v.size == Some(size))
        {
            return Some(v);
        }
    }

    // Fallback: match by filename (with and without video extension) against version sources
    let filename = si
        .filename
        .as_deref()?;
    let stem = strip_video_ext(filename);
    versions
        .iter()
        .find(|v| {
            v.sources
                .iter()
                .any(|s| {
                    s.filename
                        .as_deref()
                        .map(|sf| sf == filename || strip_video_ext(sf) == stem)
                        .unwrap_or(false)
                })
        })
}

impl AddonService {
    #[tracing::instrument(skip_all, fields(title = %media.title, kind = %media.kind))]
    pub async fn refresh_streams(
        &self,
        media: &mut db::Media,
        ctx: &AppContext,
        user_id: Option<Uuid>,
    ) -> Result<()> {
        const STREAMS_TTL_SECS: i64 = 60;
        static STREAM_LOCKS: KeyedLock<Uuid> = KeyedLock::new();

        // Fast path: TTL not expired — skip the lock entirely.
        let is_fresh = |refreshed: Option<chrono::NaiveDateTime>| {
            refreshed.is_some_and(|r| {
                (chrono::Utc::now().naive_utc() - r).num_seconds() < STREAMS_TTL_SECS
            })
        };
        if is_fresh(media.streams_refreshed_at) {
            return Ok(());
        }

        // Acquire per-media lock to prevent concurrent refreshes.
        let _guard = STREAM_LOCKS
            .lock(media.id)
            .await;

        // Re-check after acquiring lock — another task may have just refreshed.
        let refreshed_at = sqlx::query_scalar::<_, Option<chrono::NaiveDateTime>>(
            "SELECT streams_refreshed_at FROM media WHERE id = ?",
        )
        .bind(media.id)
        .fetch_optional(&ctx.db)
        .await
        .ok()
        .flatten()
        .flatten();
        if is_fresh(refreshed_at) {
            media.streams_refreshed_at = refreshed_at;
            return Ok(());
        }

        media
            .grandparent(&ctx.db)
            .await
            .ok();

        let instant = Instant::now();
        let probe_versions_fut = async {
            let Some(url) = ctx
                .config
                .remuxdb_url
                .clone()
            else {
                return None;
            };
            // RemuxDB stores episodes under the series id, so an episode's own
            // id would look up the wrong thing: no series id, no lookup.
            let external_id = if media.kind == db::MediaKind::Episode {
                media
                    .grandparent
                    .as_deref()
                    .and_then(|gp| {
                        gp.external_ids
                            .stremio_lookup_id()
                    })
            } else {
                media
                    .external_ids
                    .stremio_lookup_id()
            };
            let Some(external_id) = external_id else {
                return None;
            };
            let cfg = db::Settings::get_config_or_default(&ctx.db).await;
            if !cfg
                .remuxdb_enabled
                .unwrap_or(true)
            {
                return None;
            }
            let (season, episode) = if media.kind == db::MediaKind::Episode {
                let (Some(season), Some(episode)) = (media.parent_idx, media.idx)
                else {
                    // Without both, the lookup would be series-wide.
                    return None;
                };
                (Some(season as i32), Some(episode as i32))
            } else {
                (None, None)
            };
            remuxdb::fetch_probe(
                &url,
                cfg.remuxdb_token
                    .as_deref(),
                Some(crate::common::server_id().as_str()),
                &external_id,
                season,
                episode,
            )
            .await
        };
        let probe_t = std::time::Instant::now();
        let (raw, probe_versions) = tokio::join!(
            self.get_streams(media, ctx, user_id),
            tokio::time::timeout(std::time::Duration::from_secs(5), probe_versions_fut,)
        );
        let raw = raw?;
        let probe_versions = match probe_versions {
            Ok(v) => v,
            Err(_) => {
                debug!(remuxdb_elapsed = ?probe_t.elapsed(), "remuxdb probe timed out");
                None
            }
        };
        debug!(
            raw_count = raw.len(),
            remuxdb_elapsed = ?probe_t.elapsed(),
            "raw streams fetched"
        );

        // Dedup by descriptor content; order preserves addon priority (DB load order)
        // for which duplicate survives. First occurrence wins, so higher-priority
        // addons' streams survive ties.
        let deduped = Self::deduplicate_streams(raw);

        let sources: Vec<&str> = {
            let mut seen = std::collections::HashSet::new();
            deduped
                .iter()
                .filter_map(|m| {
                    m.stream_info
                        .as_ref()?
                        .source
                        .as_deref()
                })
                .filter(|s| seen.insert(*s))
                .collect()
        };
        info!(streams = deduped.len(), ?sources, elapsed = ?instant.elapsed(), "streams synced");
        if deduped.is_empty() {
            return Ok(());
        }

        let now = chrono::Utc::now().naive_utc();
        sqlx::query("UPDATE media SET streams_refreshed_at = ? WHERE id = ?")
            .bind(now)
            .bind(media.id)
            .execute(&ctx.db)
            .await?;
        media.streams_refreshed_at = Some(now);
        let mut sources: Vec<db::Media> = deduped
            .into_iter()
            .enumerate()
            .map(|(idx, mut s)| {
                // Stable ID derived from content so the same stream always maps to
                // the same UUID across refreshes, enabling safe upsert semantics.
                let id_key = Self::stream_dedup_key(&s)
                    .unwrap_or_else(|| format!("source_{idx}"));
                s.id = Uuid::new_v5(&media.id, id_key.as_bytes());
                s.parent_id = Some(media.id);
                s.runtime = media.runtime;
                s.idx = Some(idx as i64);
                s.created_at = now;
                s.updated_at = now;
                s
            })
            .collect();

        if let Some(ref versions) = probe_versions {
            for source in &mut sources {
                if source
                    .probe_data
                    .is_some()
                {
                    continue;
                }
                if let Some(version) = match_probe_version(versions, source) {
                    source.probe_data = Some(version.into());
                    debug!(id = %source.id, "remuxdb: probe data applied from version");
                }
            }
        }

        db::Media::upsert(&ctx.db, &sources).await?;

        // delete stale items
        sqlx::query(
            "DELETE FROM media WHERE kind = 'stream' AND parent_id = ? AND updated_at < datetime('now', '-1 days')",
        )
        .bind(media.id)
        .execute(&ctx.db)
        .await?;
        Ok(())
    }

    pub async fn fetch_segments(
        &self,
        media: &db::Media,
        ctx: &AppContext,
        background: bool,
    ) -> MediaSegments {
        let addons: Vec<(String, Arc<dyn SegmentAddon>)> = self
            .inner
            .load()
            .iter()
            .filter(|r| {
                r.row
                    .resources
                    .contains(&ResourceType::Segment)
            })
            .filter_map(|r| {
                r.segment
                    .as_ref()
                    .and_then(|s| {
                        let supports = s.supports(media);
                        debug!(
                            addon = %r.row.name,
                            media_kind = ?media.kind,
                            supports,
                            "segment addon filter"
                        );
                        if supports {
                            Some((
                                r.row
                                    .name
                                    .clone(),
                                s.clone(),
                            ))
                        } else {
                            None
                        }
                    })
            })
            .collect();

        let addon_count = addons.len();
        let instant = Instant::now();
        let mut merged = MediaSegments::default();
        for (name, addon) in addons {
            match addon
                .segment_fetch(media, ctx)
                .await
            {
                Ok(segs) if !segs.is_empty() => merged.merge_from(segs),
                Ok(_) => {}
                Err(e) => {
                    error!(addon = %name, item = %media.id, error = %e, "segment addon failed")
                }
            }
        }
        let found = [
            &merged.intro,
            &merged.outro,
            &merged.recap,
            &merged.preview,
            &merged.commercial,
        ]
        .iter()
        .filter(|s| s.is_some())
        .count();
        if background {
            debug!(segments = found, addons = addon_count, elapsed = ?instant.elapsed(), "segments fetched");
        } else {
            info!(segments = found, addons = addon_count, elapsed = ?instant.elapsed(), "segments fetched");
        }
        merged
    }

    pub async fn lyric_fetch(
        &self,
        req: &LyricSearchRequest,
    ) -> Result<Option<LyricDto>> {
        let addons: Vec<(String, Arc<dyn LyricAddon>)> = self
            .inner
            .load()
            .iter()
            .filter(|r| {
                r.row
                    .resources
                    .contains(&ResourceType::Lyrics)
            })
            .filter_map(|r| {
                r.lyric
                    .as_ref()
                    .map(|l| {
                        (
                            r.row
                                .name
                                .clone(),
                            l.clone(),
                        )
                    })
            })
            .collect();

        for (name, addon) in addons {
            match addon
                .lyric_fetch(req)
                .await
            {
                Ok(Some(l)) => return Ok(Some(l)),
                Ok(None) => continue,
                Err(e) => {
                    warn!(addon = %name, error = %e, "lyric addon fetch failed")
                }
            }
        }
        Ok(None)
    }

    pub async fn lyric_search(
        &self,
        req: &LyricSearchRequest,
    ) -> Result<Vec<RemoteLyricInfoDto>> {
        let addons: Vec<(String, Arc<dyn LyricAddon>)> = self
            .inner
            .load()
            .iter()
            .filter(|r| {
                r.row
                    .resources
                    .contains(&ResourceType::Lyrics)
            })
            .filter_map(|r| {
                r.lyric
                    .as_ref()
                    .map(|l| {
                        (
                            r.row
                                .name
                                .clone(),
                            l.clone(),
                        )
                    })
            })
            .collect();

        let mut out = Vec::new();
        for (name, addon) in addons {
            match addon
                .lyric_search(req)
                .await
            {
                Ok(items) => out.extend(items),
                Err(e) => {
                    warn!(addon = %name, error = %e, "lyric addon search failed")
                }
            }
        }
        Ok(out)
    }

    pub async fn lyric_get_by_composite_id(
        &self,
        composite_id: &str,
    ) -> Result<Option<LyricDto>> {
        let addons: Vec<Arc<dyn LyricAddon>> = self
            .inner
            .load()
            .iter()
            .filter_map(|r| {
                r.lyric
                    .clone()
            })
            .collect();

        for addon in addons {
            let prefix = format!("{}_", addon.provider_id());
            if let Some(inner) = composite_id.strip_prefix(&prefix) {
                return addon
                    .lyric_get_by_id(inner)
                    .await;
            }
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// AddonCatalogStream
// ---------------------------------------------------------------------------

struct AddonCatalogStream {
    addon: Arc<dyn CatalogAddon>,
    local_id: String,
}

#[async_trait]
impl RemoteMediaStream for AddonCatalogStream {
    async fn stream(
        &self,
        ctx: &AppContext,
    ) -> Result<Pin<Box<dyn Stream<Item = db::Media> + Send>>> {
        self.addon
            .catalog_stream(ctx, &self.local_id)
            .await?
            .ok_or_else(|| {
                anyhow!("catalog addon does not serve catalog '{}'", self.local_id)
            })
    }
}

/// Runs one addon's `meta_fetch` under `timeout`, and tries once more if that
/// first attempt timed out.
///
/// A single flaky addon (observed: AIO/aiometadata hanging up to its own 30s
/// upstream timeout) must not stall an entire item's refresh, hence the cap.
/// But a timeout is final for the item, not just the call:
/// `on_meta_fetch_timeout` blacklists the whole series for the rest of the
/// run, so every episode under it silently skips the addon. During a large
/// library refresh a burst of fetches can stall for a few seconds at once
/// while the runtime is busy (Cinemeta answers in ~0.1s when idle), and one
/// slow moment shouldn't cost a series its metadata. The second attempt gets
/// the same timeout, so a genuinely hung addon costs at most twice the cap.
async fn fetch_meta_with_retry(
    addon: &dyn MetaAddon,
    addon_name: &str,
    media: &db::Media,
    ctx: &AppContext,
    config: &api::ServerConfiguration,
    timeout: Duration,
) -> Result<Option<db::Media>> {
    if let Ok(result) =
        tokio::time::timeout(timeout, addon.meta_fetch(media, ctx, config)).await
    {
        return result;
    }

    // The cooldown can have started while the first attempt was waiting.
    // Same reasoning as the check before the first attempt: a retry that
    // would only sit out the cooldown is a guaranteed timeout, so don't.
    let cooldown = addon
        .rate_limit_cooldown()
        .await;
    if cooldown >= timeout {
        debug!(
            id = %media.id,
            addon = %addon_name,
            cooldown = ?cooldown,
            "not retrying addon meta_fetch: shared rate limit cooldown exceeds timeout"
        );
    } else {
        let started = std::time::Instant::now();
        let retried =
            tokio::time::timeout(timeout, addon.meta_fetch(media, ctx, config)).await;
        info!(
            id = %media.id,
            addon = %addon_name,
            outcome = if retried.is_ok() { "answered" } else { "timed out again" },
            elapsed = ?started.elapsed(),
            "addon meta_fetch timed out, retried once"
        );
        if let Ok(result) = retried {
            return result;
        }
    }
    addon.on_meta_fetch_timeout(media);
    Err(anyhow!("addon meta_fetch timed out after {:?}", timeout))
}

pub fn make_media_id(addon_id: Uuid, local_id: &str) -> String {
    format!("addon:{addon_id}:{local_id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torrent_stream(hash: &str, file_hint: &str, file_idx: usize) -> db::Media {
        db::Media {
            stream_info: Some(crate::stream::StreamInfo {
                descriptor: crate::stream::StreamDescriptor::Torrent {
                    info_hash: hash.to_string(),
                    file_hint: Some(file_hint.to_string()),
                    file_idx: Some(file_idx),
                    trackers: Vec::new(),
                },
                filename: Some(file_hint.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    // Regression test for the duplicate-row race: two independent "new"
    // stubs for the exact same content (same tmdb id, as if discovered via
    // two different addon catalogs at once) must converge on a single row
    // even when their `process_meta_item` calls genuinely race, because
    // neither's pre-upsert dedup check can see the other's not-yet-committed
    // insert. Correctness here depends on the per-kind external-id unique
    // indexes (migrations/202609080002_media_external_id_unique_indexes.sql)
    // plus the post-upsert id-correction check in `process_meta_item_inner`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_new_items_with_same_external_id_do_not_duplicate() {
        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;

        let ext = db::ExternalIds {
            imdb: db::NonEmptyString::try_new("tt9999999").ok(),
            tmdb: Some(999999),
            ..Default::default()
        };

        let config = Arc::new(db::Settings::get_config_or_default(&ctx.db).await);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(4));

        let a = db::Media {
            id: uuid::Uuid::new_v4(),
            kind: db::MediaKind::Movie,
            title: "Race Movie".into(),
            external_ids: ext.clone(),
            ..Default::default()
        };
        let b = db::Media {
            id: uuid::Uuid::new_v4(),
            kind: db::MediaKind::Movie,
            title: "Race Movie".into(),
            external_ids: ext,
            ..Default::default()
        };

        let task_a = tokio::spawn({
            let addons = ctx
                .addons
                .clone();
            let ctx = ctx.clone();
            let config = config.clone();
            let semaphore = semaphore.clone();
            async move {
                addons
                    .process_meta_item(a, ctx, false, config, semaphore)
                    .await
            }
        });
        let task_b = tokio::spawn({
            let addons = ctx
                .addons
                .clone();
            let ctx = ctx.clone();
            let config = config.clone();
            let semaphore = semaphore.clone();
            async move {
                addons
                    .process_meta_item(b, ctx, false, config, semaphore)
                    .await
            }
        });

        let (id_a, id_b) = tokio::join!(task_a, task_b);
        let id_a = id_a.unwrap();
        let id_b = id_b.unwrap();

        assert_eq!(
            id_a, id_b,
            "both concurrent imports of the same content must converge on one row"
        );

        let count: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM media WHERE kind = 'movie' AND json_extract(external_ids, '$.tmdb') = 999999",
        )
        .fetch_one(&ctx.db)
        .await
        .unwrap();
        assert_eq!(count, 1, "exactly one row should survive the race");
    }

    #[test]
    fn torrent_dedup_preserves_distinct_files_in_a_bundle() {
        let first = torrent_stream("abc", "Bundle/Movie.One.mkv", 0);
        let duplicate = torrent_stream("ABC", "Bundle/Movie.One.mkv", 7);
        let second = torrent_stream("abc", "Bundle/Movie.Two.mkv", 1);

        assert_eq!(
            AddonService::stream_dedup_key(&first),
            AddonService::stream_dedup_key(&duplicate)
        );
        assert_ne!(
            AddonService::stream_dedup_key(&first),
            AddonService::stream_dedup_key(&second)
        );
    }

    #[test]
    fn stream_dedup_preserves_addon_load_order() {
        let first = torrent_stream("aaa", "Movie.2026.720p.WEBRip.mkv", 0);
        let duplicate = torrent_stream("AAA", "Movie.2026.720p.WEBRip.mkv", 9);
        let second = torrent_stream("bbb", "Movie.2026.2160p.BluRay.Remux.mkv", 0);
        let expected = vec![
            first
                .stream_info
                .as_ref()
                .unwrap()
                .filename
                .clone(),
            second
                .stream_info
                .as_ref()
                .unwrap()
                .filename
                .clone(),
        ];

        let deduped = AddonService::deduplicate_streams(vec![first, duplicate, second]);
        let filenames: Vec<_> = deduped
            .iter()
            .map(|stream| {
                stream
                    .stream_info
                    .as_ref()
                    .unwrap()
                    .filename
                    .clone()
            })
            .collect();

        assert_eq!(filenames, expected);
    }

    fn make_image(path: &str) -> db::MediaImage {
        db::MediaImage {
            id: uuid::Uuid::new_v4(),
            media_id: uuid::Uuid::nil(),
            image_type: db::ImageKind::Primary.to_string(),
            image_index: 0,
            path: path.into(),
            width: None,
            height: None,
        }
    }

    // Simulates the refresh_meta accumulation: patch multiple addon results
    // into a fresh db::Media with replace=false, then apply once to `media`.
    fn accumulate(patches: Vec<db::Media>, force_refresh: bool) -> db::Media {
        let mut combined: Option<db::Media> = None;
        for patch in patches {
            let acc = combined.get_or_insert_with(db::Media::default);
            apply_meta(acc, patch, false);
        }
        let mut media = db::Media::default();
        if let Some(c) = combined {
            apply_meta(&mut media, c, force_refresh);
        }
        media
    }

    #[test]
    fn highest_priority_addon_wins_description() {
        let high = db::Media {
            description: Some("from high priority".into()),
            ..Default::default()
        };
        let low = db::Media {
            description: Some("from low priority".into()),
            ..Default::default()
        };
        // high priority addon is first (lower priority number, ORDER BY priority ASC)
        let result = accumulate(vec![high, low], true);
        assert_eq!(
            result
                .description
                .as_deref(),
            Some("from high priority")
        );
    }

    #[test]
    fn lower_priority_addon_fills_gaps_left_by_higher() {
        let high = db::Media {
            description: None, // high priority addon has no description
            ..Default::default()
        };
        let low = db::Media {
            description: Some("fallback".into()),
            ..Default::default()
        };
        let result = accumulate(vec![high, low], true);
        assert_eq!(
            result
                .description
                .as_deref(),
            Some("fallback")
        );
    }

    #[test]
    fn highest_priority_addon_wins_primary_image() {
        let mut high = db::Media::default();
        high.images
            .primary = vec![make_image("https://high.example/poster.jpg")];

        let mut low = db::Media::default();
        low.images
            .primary = vec![make_image("https://low.example/poster.jpg")];

        let result = accumulate(vec![high, low], true);
        assert_eq!(
            result
                .images
                .primary[0]
                .path,
            "https://high.example/poster.jpg"
        );
    }

    #[test]
    fn lower_priority_fills_missing_image_type() {
        let mut high = db::Media::default();
        high.images
            .primary = vec![make_image("https://high.example/poster.jpg")];
        // high priority has no backdrop

        let mut low = db::Media::default();
        low.images
            .backdrop = vec![make_image("https://low.example/backdrop.jpg")];

        let result = accumulate(vec![high, low], true);
        assert_eq!(
            result
                .images
                .primary[0]
                .path,
            "https://high.example/poster.jpg"
        );
        assert_eq!(
            result
                .images
                .backdrop[0]
                .path,
            "https://low.example/backdrop.jpg"
        );
    }

    #[test]
    fn force_refresh_replaces_existing_media_values() {
        let patch = db::Media {
            description: Some("new description".into()),
            ..Default::default()
        };
        let mut media = db::Media {
            description: Some("old description".into()),
            ..Default::default()
        };
        apply_meta(&mut media, patch, true);
        assert_eq!(
            media
                .description
                .as_deref(),
            Some("new description")
        );
    }

    #[test]
    fn no_force_refresh_preserves_existing_media_values() {
        let patch = db::Media {
            description: Some("new description".into()),
            ..Default::default()
        };
        let mut media = db::Media {
            description: Some("old description".into()),
            ..Default::default()
        };
        apply_meta(&mut media, patch, false);
        assert_eq!(
            media
                .description
                .as_deref(),
            Some("old description")
        );
    }

    fn days_ago(n: i64) -> chrono::NaiveDateTime {
        (chrono::Utc::now() - chrono::Duration::days(n)).naive_utc()
    }

    fn days_from_now(n: i64) -> chrono::NaiveDateTime {
        (chrono::Utc::now() + chrono::Duration::days(n)).naive_utc()
    }

    fn child_with(
        refreshed_at: Option<chrono::NaiveDateTime>,
        digital_released_at: Option<chrono::NaiveDateTime>,
    ) -> db::Media {
        db::Media {
            refreshed_at,
            digital_released_at,
            ..Default::default()
        }
    }

    // --- child_refresh_force ---

    #[test]
    fn active_window_episode_returns_some_true() {
        let child = child_with(Some(days_ago(1)), None);
        assert_eq!(child_refresh_force(false, true, &child), Some(true));
    }

    #[test]
    fn active_window_episode_no_refreshed_at_returns_some_true() {
        let child = child_with(None, None);
        assert_eq!(child_refresh_force(false, true, &child), Some(true));
    }

    #[test]
    fn inactive_episode_with_refreshed_at_returns_none() {
        let child = child_with(Some(days_ago(1)), None);
        assert_eq!(child_refresh_force(false, false, &child), None);
    }

    #[test]
    fn inactive_episode_no_refreshed_at_returns_some_false() {
        let child = child_with(None, None);
        assert_eq!(child_refresh_force(false, false, &child), Some(false));
    }

    #[test]
    fn force_refresh_overrides_everything() {
        let child = child_with(Some(days_ago(1)), Some(days_ago(300)));
        assert_eq!(child_refresh_force(true, false, &child), Some(true));
    }

    // --- episode_in_active_window ---

    #[test]
    fn no_released_at_is_active() {
        let child = child_with(None, None);
        assert!(episode_in_active_window(&child));
    }

    #[test]
    fn future_released_at_is_active() {
        let child = child_with(None, Some(days_from_now(30)));
        assert!(episode_in_active_window(&child));
    }

    #[test]
    fn recent_released_at_is_active() {
        let child = child_with(None, Some(days_ago(30)));
        assert!(episode_in_active_window(&child));
    }

    #[test]
    fn old_released_at_is_inactive() {
        let child = child_with(None, Some(days_ago(200)));
        assert!(!episode_in_active_window(&child));
    }

    // A freshly fetched TMDB episode carries its air date in `released_at` and
    // never a digital date, so an old episode must read as inactive from the
    // air date alone, not be refreshed every night forever.
    #[test]
    fn old_air_date_without_digital_date_is_inactive() {
        let child = db::Media {
            released_at: Some(days_ago(400)),
            ..Default::default()
        };
        assert!(!episode_in_active_window(&child));
    }

    #[test]
    fn recent_air_date_without_digital_date_is_active() {
        let child = db::Media {
            released_at: Some(days_ago(10)),
            ..Default::default()
        };
        assert!(episode_in_active_window(&child));
    }

    #[test]
    fn upcoming_air_date_without_digital_date_is_active() {
        let child = db::Media {
            released_at: Some(days_from_now(7)),
            ..Default::default()
        };
        assert!(episode_in_active_window(&child));
    }

    // The stored digital date wins over the freshly fetched row, which has none.
    #[test]
    fn stored_old_digital_date_makes_fresh_row_inactive() {
        let child = db::Media::default();
        assert!(!episode_in_active_window_with(&child, Some(days_ago(300))));
    }

    #[test]
    fn stored_recent_digital_date_makes_fresh_row_active() {
        let child = db::Media {
            released_at: Some(days_ago(400)),
            ..Default::default()
        };
        assert!(episode_in_active_window_with(&child, Some(days_ago(5))));
    }

    // --- series_is_active ---

    #[test]
    fn series_active_when_status_none() {
        assert!(series_is_active(&None));
    }

    #[test]
    fn series_active_when_continuing() {
        assert!(series_is_active(&Some(db::MediaStatus::Continuing)));
    }

    #[test]
    fn series_inactive_when_ended() {
        assert!(!series_is_active(&Some(db::MediaStatus::Ended)));
    }

    #[test]
    fn series_inactive_when_unreleased() {
        assert!(!series_is_active(&Some(db::MediaStatus::Unreleased)));
    }

    // --- apply_title_format idempotency ---

    #[test]
    fn apply_title_format_strips_episode_prefix() {
        let mut media = db::Media {
            kind: db::MediaKind::Episode,
            title: "S3E4 - Tumbleton".into(),
            idx: Some(4),
            parent_idx: Some(3),
            ..Default::default()
        };
        apply_title_format(&mut media);
        assert_eq!(media.title, "Tumbleton");
    }

    #[test]
    fn apply_title_format_keeps_clean_episode_title() {
        let mut media = db::Media {
            kind: db::MediaKind::Episode,
            title: "Tumbleton".into(),
            idx: Some(4),
            parent_idx: Some(3),
            ..Default::default()
        };
        apply_title_format(&mut media);
        assert_eq!(media.title, "Tumbleton");
    }

    #[test]
    fn merge_media_respects_locked_name() {
        let mut target = db::Media {
            title: "User Title".to_string(),
            locked_fields: vec![db::MetadataField::Name],
            ..Default::default()
        };
        let source = db::Media {
            title: "Provider Title".to_string(),
            ..Default::default()
        };
        merge_media(&mut target, &source, true);
        assert_eq!(
            target.title, "User Title",
            "locked Name must not be overwritten"
        );
    }

    #[test]
    fn merge_media_respects_is_locked() {
        let mut target = db::Media {
            title: "User Title".to_string(),
            description: Some("User Overview".to_string()),
            is_locked: true,
            ..Default::default()
        };
        let source = db::Media {
            title: "Provider Title".to_string(),
            description: Some("Provider Overview".to_string()),
            ..Default::default()
        };
        merge_media(&mut target, &source, true);
        assert_eq!(target.title, "User Title");
        assert_eq!(
            target
                .description
                .as_deref(),
            Some("User Overview")
        );
    }

    #[test]
    fn merge_media_unlocked_fields_still_update() {
        let mut target = db::Media {
            locked_fields: vec![db::MetadataField::Name],
            ..Default::default()
        };
        let source = db::Media {
            description: Some("Provider Overview".to_string()),
            ..Default::default()
        };
        merge_media(&mut target, &source, true);
        assert_eq!(
            target
                .description
                .as_deref(),
            Some("Provider Overview"),
            "unlocked Overview must still be updated"
        );
    }

    #[tokio::test]
    async fn save_pending_relations_preserves_locked_genres() {
        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;
        let item = db::Media {
            id: Uuid::new_v4(),
            title: "Locked Genres Series".to_string(),
            kind: db::MediaKind::Series,
            locked_fields: vec![db::MetadataField::Genres],
            ..Default::default()
        };
        let genre = db::Media {
            id: Uuid::new_v4(),
            title: "Acoustic".to_string(),
            kind: db::MediaKind::Genre,
            ..Default::default()
        };
        let actor = db::Media {
            id: Uuid::new_v4(),
            title: "Actor".to_string(),
            kind: db::MediaKind::Person,
            external_ids: db::ExternalIds {
                tmdb: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };
        let item_id = item.id;
        let genre_id = genre.id;
        db::Media::upsert(&ctx.db, &[item.clone(), genre.clone(), actor.clone()])
            .await
            .unwrap();
        db::MediaRelation::upsert(
            &ctx.db,
            &[db::MediaRelation {
                left_media_id: item.id,
                right_media_id: genre.id,
                ..Default::default()
            }],
        )
        .await
        .unwrap();

        let refreshed = db::Media {
            relations: Some(vec![(
                db::MediaRelation {
                    left_media_id: item.id,
                    right_media_id: actor.id,
                    role: Some(db::RelationRole::Actor),
                    ..Default::default()
                },
                actor,
            )]),
            ..item
        };
        save_pending_relations(ctx, &[refreshed]).await;

        let relations = db::MediaRelation::get_by_left_ids(&ctx.db, &[item_id])
            .await
            .unwrap();
        assert!(
            relations
                .iter()
                .any(|relation| relation.right_media_id == genre_id)
        );
    }

    #[tokio::test]
    async fn save_pending_relations_preserves_locked_cast() {
        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;
        let item = db::Media {
            id: Uuid::new_v4(),
            title: "Locked Cast Series".to_string(),
            kind: db::MediaKind::Series,
            locked_fields: vec![db::MetadataField::Cast],
            ..Default::default()
        };
        let actor_tmdb_id = 42i64;
        let director_tmdb_id = 43i64;
        let actor = db::Media {
            id: crate::common::stable_media_uuid(
                &db::MediaKind::Person,
                &actor_tmdb_id.to_string(),
            ),
            title: "Actor".to_string(),
            kind: db::MediaKind::Person,
            external_ids: db::ExternalIds {
                tmdb: Some(actor_tmdb_id),
                ..Default::default()
            },
            ..Default::default()
        };
        let director = db::Media {
            id: crate::common::stable_media_uuid(
                &db::MediaKind::Person,
                &director_tmdb_id.to_string(),
            ),
            title: "Director".to_string(),
            kind: db::MediaKind::Person,
            external_ids: db::ExternalIds {
                tmdb: Some(director_tmdb_id),
                ..Default::default()
            },
            ..Default::default()
        };
        let genre = db::Media {
            id: Uuid::new_v4(),
            title: "Drama".to_string(),
            kind: db::MediaKind::Genre,
            ..Default::default()
        };
        let item_id = item.id;
        let actor_id = actor.id;
        let director_id = director.id;
        db::Media::upsert(
            &ctx.db,
            &[item.clone(), actor.clone(), director.clone(), genre.clone()],
        )
        .await
        .unwrap();
        db::MediaRelation::upsert(
            &ctx.db,
            &[
                db::MediaRelation {
                    left_media_id: item.id,
                    right_media_id: actor.id,
                    role: Some(db::RelationRole::Actor),
                    ..Default::default()
                },
                db::MediaRelation {
                    left_media_id: item.id,
                    right_media_id: director.id,
                    role: Some(db::RelationRole::Director),
                    ..Default::default()
                },
            ],
        )
        .await
        .unwrap();

        // Simulate a RefreshLibrary patch that only carries genre relations —
        // `apply_meta` deliberately omits Person relations when Cast is locked
        // (see the merge-time filter above), so the refreshed item's pending
        // relations never include the existing actor/director.
        let refreshed = db::Media {
            relations: Some(vec![(
                db::MediaRelation {
                    left_media_id: item.id,
                    right_media_id: genre.id,
                    ..Default::default()
                },
                genre,
            )]),
            ..item
        };
        save_pending_relations(ctx, &[refreshed]).await;

        let relations = db::MediaRelation::get_by_left_ids(&ctx.db, &[item_id])
            .await
            .unwrap();
        assert!(
            relations
                .iter()
                .any(|relation| relation.right_media_id == actor_id),
            "Cast-locked actor relation must survive a refresh that omits it"
        );
        assert!(
            relations
                .iter()
                .any(|relation| relation.right_media_id == director_id),
            "Cast-locked crew (director) relation must survive a refresh that omits it"
        );
    }

    #[test]
    fn recognized_manifest_media_kind_drops_unrecognized_custom_type() {
        assert_eq!(
            recognized_manifest_media_kind(sdks::stremio::MediaType::Other(
                "anime".to_string()
            )),
            None
        );
        assert_eq!(
            recognized_manifest_media_kind(sdks::stremio::MediaType::Series),
            Some(sdks::remux::MediaKind::Series)
        );
        assert_eq!(
            recognized_manifest_media_kind(sdks::stremio::MediaType::Other(
                "episode".to_string()
            )),
            Some(sdks::remux::MediaKind::Episode)
        );
    }

    // --- get_direct_children resource guard ---

    struct SpyTree {
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl TreeAddon for SpyTree {
        fn supports(&self, root: &db::Media) -> bool {
            matches!(root.kind, db::MediaKind::Series)
        }

        async fn get_children(
            &self,
            _root: &db::Media,
            _ctx: &AppContext,
        ) -> anyhow::Result<Option<Vec<db::Media>>> {
            self.called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(Some(vec![]))
        }
    }

    fn stream_only_runtime_with_tree(
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> AddonRuntime {
        let now = chrono::Utc::now().naive_utc();
        AddonRuntime {
            row: addon::Addon {
                id: uuid::Uuid::new_v4(),
                name: "stream-only".into(),
                preset: AddonPresetRef {
                    kind: "scripted".into(),
                    config: serde_json::Value::Null.into(),
                },
                resources: vec![ResourceType::Stream], // no Meta
                types: vec![],
                enabled: true,
                priority: 0,
                created_at: now,
                updated_at: now,
                system: false,
                is_default: false,
                http_redirect_stream: false,
                subtitle_extraction: false,
                probe_on_scan: false,
                service_filter: vec![],
            },
            caps: AddonCapabilities {
                tree: Some(std::sync::Arc::new(SpyTree { called })),
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn get_direct_children_skips_addons_without_meta_resource() {
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runtime = stream_only_runtime_with_tree(called.clone());

        let service = AddonService {
            inner: std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(vec![runtime])),
        };

        let series = db::Media {
            kind: db::MediaKind::Series,
            ..Default::default()
        };

        // AppContext is required by the trait signature but is never reached when the
        // addon is filtered out before get_children is called. Use new_test_server so
        // port binding is handled correctly (same as all other integration tests).
        let (_, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = guard
            .0
            .clone();

        let children = service
            .get_direct_children(&series, &ctx, &api::ServerConfiguration::default())
            .await;

        assert!(
            children.is_empty(),
            "stream-only addon should not contribute children"
        );
        assert!(
            !called.load(std::sync::atomic::Ordering::SeqCst),
            "stream-only addon's get_children should never be called"
        );
    }

    #[test]
    fn subtitle_query_cache_suffix_keeps_default_keys_and_separates_releases() {
        let release = |filename: &str, size| SubtitleQuery {
            filename: Some(filename.into()),
            video_size: size,
        };
        assert_eq!(SubtitleQuery::default().cache_suffix(), "");
        assert_ne!(
            release("a.mkv", Some(1)).cache_suffix(),
            release("b.mkv", Some(1)).cache_suffix()
        );
        assert_ne!(
            release("a.mkv", Some(1)).cache_suffix(),
            release("a.mkv", Some(2)).cache_suffix()
        );
        assert_ne!(
            release("a.mkv", None).cache_suffix(),
            SubtitleQuery::default().cache_suffix()
        );
    }

    struct CountingSubtitleAddon {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl SubtitleAddon for CountingSubtitleAddon {
        fn supports(&self, _media: &db::Media) -> bool {
            true
        }

        async fn subtitle_fetch(
            &self,
            _media: &db::Media,
            _query: &SubtitleQuery,
            _db: &SqlitePool,
        ) -> Result<Vec<SubtitleInfo>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Give a concurrent second caller a chance to reach the cache
            // check while this one is still "in flight".
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Ok(vec![SubtitleInfo {
                id: "sub".into(),
                url: None,
                lang: Some("eng".into()),
                is_forced: false,
                is_hi: false,
                filename: None,
                from_trusted: None,
                ai_translated: None,
            }])
        }
    }

    /// Two near-simultaneous callers for the same item/user (e.g. an Items
    /// detail fetch racing a PlaybackInfo call) must not each pay the full
    /// addon round-trip — that's real duplicated latency and addon load, not
    /// just a log artifact.
    #[tokio::test]
    async fn fetch_subtitles_coalesces_concurrent_calls_for_the_same_item() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let now = chrono::Utc::now().naive_utc();
        let runtime = AddonRuntime {
            row: addon::Addon {
                id: uuid::Uuid::new_v4(),
                name: "counting-subtitle".into(),
                preset: AddonPresetRef {
                    kind: "scripted".into(),
                    config: serde_json::Value::Null.into(),
                },
                resources: vec![ResourceType::Subtitles],
                types: vec![],
                enabled: true,
                priority: 0,
                created_at: now,
                updated_at: now,
                system: true,
                is_default: false,
                http_redirect_stream: false,
                subtitle_extraction: false,
                probe_on_scan: false,
                service_filter: vec![],
            },
            caps: AddonCapabilities {
                subtitle: Some(std::sync::Arc::new(CountingSubtitleAddon {
                    calls: calls.clone(),
                })),
                ..Default::default()
            },
        };

        let service = AddonService {
            inner: std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(vec![runtime])),
        };

        let (_, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = guard
            .0
            .clone();

        let mut media_a = db::Media {
            id: uuid::Uuid::new_v4(),
            kind: db::MediaKind::Movie,
            title: "Concurrent Fetch Test".into(),
            ..Default::default()
        };
        let mut media_b = media_a.clone();
        let user_id = uuid::Uuid::new_v4();

        let query = SubtitleQuery::default();
        let (subs_a, subs_b) = tokio::join!(
            service.fetch_subtitles(&mut media_a, &query, &ctx, false, Some(user_id),),
            service.fetch_subtitles(&mut media_b, &query, &ctx, false, Some(user_id),),
        );

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "concurrent fetches for the same item/user must coalesce into a single addon call"
        );
        assert_eq!(subs_a.len(), 1);
        assert_eq!(subs_b.len(), 1);
    }

    fn unreachable_runtime(
        resources: Vec<ResourceType>,
        kind: Option<Arc<dyn AddonKind>>,
    ) -> AddonRuntime {
        let now = chrono::Utc::now().naive_utc();
        AddonRuntime {
            row: addon::Addon {
                id: uuid::Uuid::new_v4(),
                name: "unreachable".into(),
                preset: AddonPresetRef {
                    kind: "scripted".into(),
                    config: serde_json::Value::Null.into(),
                },
                resources,
                types: vec![],
                enabled: true,
                priority: 0,
                created_at: now,
                updated_at: now,
                system: true,
                is_default: false,
                http_redirect_stream: false,
                subtitle_extraction: false,
                probe_on_scan: false,
                service_filter: vec![],
            },
            caps: AddonCapabilities {
                manifest_unreachable: true,
                kind,
                ..Default::default()
            },
        }
    }

    struct RecoveredManifestKind;

    #[async_trait]
    impl AddonKind for RecoveredManifestKind {
        fn id(&self) -> &'static str {
            "recovered"
        }

        async fn available_info(
            &self,
        ) -> Result<
            Option<(
                Vec<remux_sdks::stremio::ResourceRef>,
                Vec<remux_sdks::stremio::MediaType>,
            )>,
        > {
            Ok(Some((
                vec![remux_sdks::stremio::ResourceRef {
                    name: ResourceType::Meta,
                    types: vec![],
                    id_prefixes: Some(vec!["tt".into()]),
                }],
                vec![],
            )))
        }
    }

    struct FailingManifestKind;

    #[async_trait]
    impl AddonKind for FailingManifestKind {
        fn id(&self) -> &'static str {
            "failing"
        }

        async fn available_info(&self) -> Result<Option<ManifestInfo>> {
            Err(anyhow!("manifest unreachable"))
        }
    }

    fn media_with_imdb(imdb: &str) -> db::Media {
        db::Media {
            kind: db::MediaKind::Movie,
            external_ids: db::ExternalIds {
                imdb: db::NonEmptyString::try_new(imdb.to_string()).ok(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn unreachable_addon_is_skipped_until_its_manifest_loads() {
        let mut runtime = unreachable_runtime(
            vec![ResourceType::Meta],
            Some(Arc::new(FailingManifestKind)),
        );
        runtime
            .caps
            .meta = Some(Arc::new(AlwaysMeta));
        let media = media_with_imdb("tt0111161");
        assert!(!PickCap::<dyn MetaAddon>::pick(&runtime, &media).await);
    }

    #[tokio::test]
    async fn recovered_addon_applies_its_manifest_id_prefixes() {
        let mut runtime = unreachable_runtime(
            vec![ResourceType::Meta],
            Some(Arc::new(RecoveredManifestKind)),
        );
        runtime
            .caps
            .meta = Some(Arc::new(AlwaysMeta));
        assert!(
            PickCap::<dyn MetaAddon>::pick(&runtime, &media_with_imdb("tt0111161"))
                .await
        );
        assert!(
            !PickCap::<dyn MetaAddon>::pick(&runtime, &media_with_imdb("xx0111161"))
                .await
        );
    }

    struct AlwaysMeta;

    #[async_trait]
    impl MetaAddon for AlwaysMeta {
        async fn supports(&self, _media: &db::Media) -> bool {
            true
        }

        async fn meta_fetch(
            &self,
            _media: &db::Media,
            _ctx: &AppContext,
            _config: &crate::api::ServerConfiguration,
        ) -> Result<Option<db::Media>> {
            Ok(None)
        }
    }

    /// Hangs on the first call, answers on the next ones: a stall that is
    /// over by the time the call is repeated.
    struct StallsOnce {
        calls: std::sync::atomic::AtomicUsize,
        timeouts: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl MetaAddon for StallsOnce {
        async fn supports(&self, _media: &db::Media) -> bool {
            true
        }

        async fn meta_fetch(
            &self,
            media: &db::Media,
            _ctx: &AppContext,
            _config: &crate::api::ServerConfiguration,
        ) -> Result<Option<db::Media>> {
            let n = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                std::future::pending::<()>().await;
            }
            Ok(Some(db::Media {
                title: media
                    .title
                    .clone(),
                ..Default::default()
            }))
        }

        fn on_meta_fetch_timeout(&self, _media: &db::Media) {
            self.timeouts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn a_single_stalled_meta_fetch_is_retried_instead_of_failing_the_item() {
        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;
        let config = db::Settings::get_config_or_default(&ctx.db).await;
        let addon = StallsOnce {
            calls: Default::default(),
            timeouts: Default::default(),
        };
        let media = db::Media {
            kind: db::MediaKind::Episode,
            title: "Pilot".into(),
            ..Default::default()
        };

        let patch = fetch_meta_with_retry(
            &addon,
            "stalls-once",
            &media,
            ctx,
            &config,
            Duration::from_millis(100),
        )
        .await
        .expect("second attempt succeeds")
        .expect("addon returned a patch");

        assert_eq!(patch.title, "Pilot");
        assert_eq!(
            addon
                .timeouts
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a recovered fetch must not blacklist the series"
        );
    }

    #[tokio::test]
    async fn a_meta_fetch_that_keeps_stalling_still_times_out() {
        struct Hangs {
            calls: std::sync::atomic::AtomicUsize,
            timeouts: std::sync::atomic::AtomicUsize,
        }

        #[async_trait]
        impl MetaAddon for Hangs {
            async fn supports(&self, _media: &db::Media) -> bool {
                true
            }
            async fn meta_fetch(
                &self,
                _media: &db::Media,
                _ctx: &AppContext,
                _config: &crate::api::ServerConfiguration,
            ) -> Result<Option<db::Media>> {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::pending().await
            }
            fn on_meta_fetch_timeout(&self, _media: &db::Media) {
                self.timeouts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;
        let config = db::Settings::get_config_or_default(&ctx.db).await;
        let addon = Hangs {
            calls: Default::default(),
            timeouts: Default::default(),
        };
        let err = fetch_meta_with_retry(
            &addon,
            "hangs",
            &db::Media::default(),
            ctx,
            &config,
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("timed out")
        );
        assert_eq!(
            addon
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a first timeout is retried once"
        );
        assert_eq!(
            addon
                .timeouts
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn no_retry_when_the_rate_limit_cooldown_outlasts_the_timeout() {
        struct Cooling {
            calls: std::sync::atomic::AtomicUsize,
            timeouts: std::sync::atomic::AtomicUsize,
        }

        #[async_trait]
        impl MetaAddon for Cooling {
            async fn supports(&self, _media: &db::Media) -> bool {
                true
            }
            async fn meta_fetch(
                &self,
                _media: &db::Media,
                _ctx: &AppContext,
                _config: &crate::api::ServerConfiguration,
            ) -> Result<Option<db::Media>> {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::pending().await
            }
            fn on_meta_fetch_timeout(&self, _media: &db::Media) {
                self.timeouts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            async fn rate_limit_cooldown(&self) -> Duration {
                Duration::from_secs(60)
            }
        }

        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = &guard.0;
        let config = db::Settings::get_config_or_default(&ctx.db).await;
        let addon = Cooling {
            calls: Default::default(),
            timeouts: Default::default(),
        };
        let err = fetch_meta_with_retry(
            &addon,
            "cooling",
            &db::Media::default(),
            ctx,
            &config,
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("timed out")
        );
        assert_eq!(
            addon
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the retry must be skipped during a cooldown"
        );
        assert_eq!(
            addon
                .timeouts
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn manifest_cache_remembers_success_and_throttles_failure() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let failing = ManifestCache::default();
        for _ in 0..3 {
            let res = failing
                .get_or_fetch(|| async {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(anyhow!("down"))
                })
                .await;
            assert!(res.is_err());
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let loaded = ManifestCache::default();
        for _ in 0..3 {
            loaded
                .get_or_fetch(|| async {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok((vec![], vec![]))
                })
                .await
                .unwrap();
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn subtitles_are_not_cached_while_a_subtitle_addon_is_unreachable() {
        let mut runtime = unreachable_runtime(
            vec![ResourceType::Subtitles],
            Some(Arc::new(FailingManifestKind)),
        );
        runtime
            .caps
            .manifest_unreachable = true;
        let service = AddonService {
            inner: Arc::new(ArcSwap::from_pointee(vec![runtime])),
        };
        let (_, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = guard
            .0
            .clone();
        let mut media = db::Media {
            id: uuid::Uuid::new_v4(),
            kind: db::MediaKind::Movie,
            title: "Unreachable Subtitles".into(),
            ..Default::default()
        };

        service
            .fetch_subtitles(&mut media, &SubtitleQuery::default(), &ctx, false, None)
            .await;

        assert!(
            ctx.store
                .get::<Vec<SubtitleInfo>>(&format!("addon-subtitles:{}:anon", media.id))
                .is_none()
        );
    }

    /// Remote search mints a fresh id (and fresh, possibly-drifted data) per
    /// request. A result that already exists locally must be replaced with
    /// the stored row wholesale — not just its id — or a client that keeps
    /// the id (next episode, continue watching) gets 404 on it once the
    /// store entry is gone, and in the meantime sees data that can differ
    /// from what it actually has. Unknown items keep their own id and data.
    #[tokio::test]
    async fn search_results_adopt_existing_rows() {
        use crate::integration_test::{authenticated_server, seed_movie};
        let (_server, guard, _token) = authenticated_server().await;
        let ctx = &guard.0;
        let stored = seed_movie(ctx).await;
        let mut results = vec![
            db::Media {
                id: Uuid::new_v4(),
                // Deliberately different from the stored row's title, to
                // prove the whole item is replaced, not just its id.
                title: "Heat (remote addon's stale title)".into(),
                kind: db::MediaKind::Movie,
                external_ids: db::ExternalIds {
                    imdb: stored
                        .external_ids
                        .imdb
                        .clone(),
                    ..Default::default()
                },
                // Transient, caller-attached bookkeeping unrelated to which
                // row is correct — must survive the swap.
                relations: Some(vec![]),
                ..Default::default()
            },
            db::Media {
                id: Uuid::new_v4(),
                title: "Unknown".into(),
                kind: db::MediaKind::Movie,
                external_ids: db::ExternalIds {
                    imdb: db::NonEmptyString::try_new("tt0000001".to_string()).ok(),
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        let unknown_id = results[1].id;
        results.push(db::Media {
            id: Uuid::new_v4(),
            title: stored
                .title
                .clone(),
            kind: db::MediaKind::Movie,
            external_ids: db::ExternalIds {
                tmdb: stored
                    .external_ids
                    .tmdb,
                ..Default::default()
            },
            ..Default::default()
        });
        db::Media::adopt_existing_rows(&ctx.db, &mut results).await;
        assert_eq!(
            results[0].id, stored.id,
            "known item takes the stored row's id"
        );
        assert_eq!(
            results[0].title, stored.title,
            "known item is replaced wholesale with the stored row, not just its id"
        );
        assert!(
            results[0]
                .relations
                .is_some(),
            "caller-attached relations survive the swap"
        );
        assert_eq!(results[1].id, unknown_id, "unknown item keeps its own id");
        assert_eq!(
            results[1].title, "Unknown",
            "unknown item keeps its own data"
        );
        assert_eq!(
            results[2].id, stored.id,
            "a match on a lower-priority id adopts the stored row's id too"
        );
        assert_eq!(
            results[2].title, stored.title,
            "a match on a lower-priority id is also replaced wholesale"
        );
    }

    // Regression test: `ExternalIds::is_empty()` only looks at
    // imdb/tmdb/tvdb/custom_stremio_id, so gating kind-detection on it (as
    // opposed to the kind-aware `external_id_fields`) silently skipped
    // Artist/Album/Track adoption entirely — their only identity is
    // deezer_*/youtube_id.
    #[tokio::test]
    async fn search_results_adopt_existing_rows_for_music_kinds() {
        use crate::integration_test::authenticated_server;
        let (_server, guard, _token) = authenticated_server().await;
        let ctx = &guard.0;

        let mut stored = db::Media {
            id: Uuid::new_v4(),
            title: "Actual Album".into(),
            kind: db::MediaKind::Album,
            external_ids: db::ExternalIds {
                deezer_album: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };
        stored
            .save(&ctx.db)
            .await
            .unwrap();

        let mut results = vec![db::Media {
            id: Uuid::new_v4(),
            title: "Remote Album (stale)".into(),
            kind: db::MediaKind::Album,
            external_ids: db::ExternalIds {
                deezer_album: Some(42),
                ..Default::default()
            },
            ..Default::default()
        }];
        db::Media::adopt_existing_rows(&ctx.db, &mut results).await;
        assert_eq!(
            results[0].id, stored.id,
            "a deezer-only match still adopts the stored row"
        );
        assert_eq!(
            results[0].title, stored.title,
            "a deezer-only match is replaced wholesale too"
        );
    }

    /// Answers with its hits (or an error if `fail`) after sleeping `delay`
    /// and, if gated, once the test opens the gate.
    #[derive(Default)]
    struct FixedSearch {
        hits: Vec<db::Media>,
        fail: bool,
        delay: Duration,
        gate: Option<Arc<tokio::sync::Notify>>,
    }

    #[async_trait]
    impl SearchAddon for FixedSearch {
        async fn search_supports(&self, _kind: &db::MediaKind) -> bool {
            true
        }

        async fn search(
            &self,
            _kind: &db::MediaKind,
            _query: &str,
            _limit: usize,
            _ctx: &AppContext,
        ) -> Result<Option<Vec<db::Media>>> {
            tokio::time::sleep(self.delay).await;
            if let Some(gate) = &self.gate {
                gate.notified()
                    .await;
            }
            if self.fail {
                return Err(anyhow!("search addon down"));
            }
            Ok(Some(
                self.hits
                    .clone(),
            ))
        }
    }

    fn search_runtime(results: Vec<db::Media>) -> AddonRuntime {
        fixed_search_runtime(Arc::new(FixedSearch {
            hits: results,
            ..Default::default()
        }))
    }

    fn fixed_search_runtime(search: Arc<FixedSearch>) -> AddonRuntime {
        let mut runtime = unreachable_runtime(vec![ResourceType::Search], None);
        runtime
            .caps
            .search = Some(search);
        runtime
    }

    fn search_hit(title: &str, imdb: &str) -> db::Media {
        db::Media {
            id: Uuid::new_v4(),
            title: title.into(),
            ..media_with_imdb(imdb)
        }
    }

    async fn search_titles(
        service: &AddonService,
        ctx: &AppContext,
        query: &str,
        limit: usize,
    ) -> Vec<String> {
        service
            .search(&db::MediaKind::Movie, query, limit, ctx, None)
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.title)
            .collect()
    }

    async fn search_with(runtimes: Vec<AddonRuntime>, limit: usize) -> Vec<String> {
        let service = AddonService {
            inner: Arc::new(ArcSwap::from_pointee(runtimes)),
        };
        let (_, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        search_titles(&service, &guard.0, "query", limit).await
    }

    /// Results from every search addon are appended in priority order; the
    /// same item from two addons (shared imdb id) appears once, as the
    /// higher-priority copy.
    #[tokio::test]
    async fn search_appends_addons_in_priority_order_and_dedupes() {
        let titles = search_with(
            vec![
                search_runtime(vec![]),
                search_runtime(vec![
                    search_hit("Shared (high)", "tt9900001"),
                    search_hit("High 2", "tt9900002"),
                    search_hit("High 3", "tt9900004"),
                ]),
                search_runtime(vec![
                    search_hit("Shared (low)", "tt9900001"),
                    search_hit("Low 2", "tt9900003"),
                ]),
            ],
            20,
        )
        .await;
        assert_eq!(titles, ["Shared (high)", "High 2", "High 3", "Low 2"]);
    }

    /// A higher-priority addon having answered doesn't cut off a slower
    /// lower-priority one: every addon is awaited up to the timeout.
    #[tokio::test]
    async fn search_waits_for_slower_lower_priority_addon() {
        let titles = search_with(
            vec![
                search_runtime(vec![search_hit("Fast", "tt9900001")]),
                fixed_search_runtime(Arc::new(FixedSearch {
                    hits: vec![search_hit("Slow", "tt9900002")],
                    delay: Duration::from_millis(300),
                    ..Default::default()
                })),
            ],
            20,
        )
        .await;
        assert_eq!(titles, ["Fast", "Slow"]);
    }

    /// An addon that doesn't answer within the configured
    /// `addon_fetch_timeout_secs` is left out of the results.
    #[tokio::test]
    async fn search_leaves_out_addon_exceeding_timeout() {
        let service = AddonService {
            inner: Arc::new(ArcSwap::from_pointee(vec![
                fixed_search_runtime(Arc::new(FixedSearch {
                    hits: vec![search_hit("Never", "tt9900002")],
                    gate: Some(Arc::new(tokio::sync::Notify::new())),
                    ..Default::default()
                })),
                search_runtime(vec![search_hit("Fast", "tt9900001")]),
            ])),
        };
        let (_, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        db::Settings::set_config(
            &guard
                .0
                .db,
            &crate::api::ServerConfiguration {
                addon_fetch_timeout_secs: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let started = tokio::time::Instant::now();
        assert_eq!(
            search_titles(&service, &guard.0, "query", 20).await,
            ["Fast"]
        );
        assert!(started.elapsed() >= Duration::from_secs(1));
    }

    /// Results that share an external id only through a chain are one item:
    /// `{tmdb}`, `{tmdb, imdb}`, `{imdb}` collapses to the first, also when
    /// the linking result comes last and joins two earlier groups. An
    /// unrelated result stays separate. The imdb-only result is asked about
    /// on `/find`, which TMDB answers consistently with the chain.
    #[tokio::test]
    async fn search_dedupes_results_linked_through_a_chain_of_ids() {
        let tmdb = httpmock::MockServer::start();
        mock_find(&tmdb, "tt9900101", &[9900101], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let hit = |title: &str, tmdb: Option<i64>, imdb: Option<&str>| db::Media {
            id: Uuid::new_v4(),
            title: title.into(),
            kind: db::MediaKind::Movie,
            external_ids: db::ExternalIds {
                tmdb,
                imdb: imdb
                    .and_then(|i| db::NonEmptyString::try_new(i.to_string()).ok()),
                ..Default::default()
            },
            ..Default::default()
        };
        let tmdb_only = hit("First", Some(9900101), None);
        let both = hit("Both", Some(9900101), Some("tt9900101"));
        let imdb_only = hit("Imdb", None, Some("tt9900101"));
        let other = hit("Other", Some(9900102), Some("tt9900102"));
        for chain in [
            [tmdb_only.clone(), both.clone(), imdb_only.clone()],
            [tmdb_only.clone(), imdb_only.clone(), both.clone()],
        ] {
            let mut runtimes: Vec<AddonRuntime> = chain
                .into_iter()
                .map(|m| search_runtime(vec![m]))
                .collect();
            runtimes.push(search_runtime(vec![other.clone()]));
            let service = AddonService {
                inner: Arc::new(ArcSwap::from_pointee(runtimes)),
            };
            assert_eq!(
                search_titles(&service, &guard.0, "query", 20).await,
                ["First", "Other"]
            );
        }
    }

    /// Results linked through one wrong id (a Stremio copy of Inception
    /// carrying Interstellar's imdb id) stay apart when the groups they would
    /// join carry different tmdb ids; whichever comes last.
    #[tokio::test]
    async fn search_keeps_apart_results_linked_by_a_conflicting_id() {
        let hit = |title: &str, tmdb: i64, imdb: Option<&str>| db::Media {
            id: Uuid::new_v4(),
            title: title.into(),
            kind: db::MediaKind::Movie,
            external_ids: db::ExternalIds {
                tmdb: Some(tmdb),
                imdb: imdb
                    .and_then(|i| db::NonEmptyString::try_new(i.to_string()).ok()),
                ..Default::default()
            },
            ..Default::default()
        };
        let inception = hit("Inception", 27205, None);
        let wrong = hit("Inception (wrong imdb)", 27205, Some("tt0816692"));
        let interstellar = hit("Interstellar", 157336, Some("tt0816692"));
        assert_eq!(
            search_with(
                vec![
                    search_runtime(vec![inception.clone()]),
                    search_runtime(vec![wrong.clone(), interstellar.clone()]),
                ],
                20,
            )
            .await,
            ["Inception", "Interstellar"]
        );
        assert_eq!(
            search_with(
                vec![
                    search_runtime(vec![inception]),
                    search_runtime(vec![interstellar]),
                    search_runtime(vec![wrong]),
                ],
                20,
            )
            .await,
            ["Inception", "Interstellar"]
        );
    }

    /// An addon whose search failed doesn't hide the others' results.
    #[tokio::test]
    async fn failed_search_addon_does_not_hide_other_results() {
        let titles = search_with(
            vec![
                fixed_search_runtime(Arc::new(FixedSearch {
                    fail: true,
                    ..Default::default()
                })),
                search_runtime(vec![search_hit("Found", "tt9900001")]),
            ],
            20,
        )
        .await;
        assert_eq!(titles, ["Found"]);
    }

    /// A TMDB result and a Stremio result for the same title (a `tmdb:` id, as
    /// AIOStreams' search sends) collapse to the first occurrence.
    #[tokio::test]
    async fn search_dedupes_tmdb_and_stremio_copies() {
        let tmdb = db::Media {
            id: Uuid::new_v4(),
            title: "Inception (TMDB)".into(),
            kind: db::MediaKind::Movie,
            external_ids: db::ExternalIds {
                tmdb: Some(27205),
                ..Default::default()
            },
            ..Default::default()
        };
        let meta: remux_sdks::stremio::Meta =
            serde_json::from_value(serde_json::json!({
                "id": "tmdb:27205",
                "type": "movie",
                "name": "Inception (Stremio)",
            }))
            .unwrap();
        let stremio = db::Media::try_from(meta).unwrap();
        let titles = search_with(
            vec![search_runtime(vec![tmdb]), search_runtime(vec![stremio])],
            20,
        )
        .await;
        assert_eq!(titles, ["Inception (TMDB)"]);
    }

    /// `/find/{imdb}` placing it as these movie and tv ids. Ids of their own
    /// per test: the response cache is keyed on the url and httpmock reuses
    /// servers.
    fn mock_find<'a>(
        server: &'a httpmock::MockServer,
        imdb: &str,
        movie: &[i64],
        tv: &[i64],
    ) -> httpmock::Mock<'a> {
        let movie_results: Vec<_> = movie
            .iter()
            .map(|id| {
                serde_json::json!({
                    "id": id, "title": "Film", "adult": false, "original_language": "en"
                })
            })
            .collect();
        let tv_results: Vec<_> = tv
            .iter()
            .map(|id| serde_json::json!({"id": id, "name": "Show"}))
            .collect();
        server.mock(|when, then| {
            when.path(format!("/find/{imdb}"))
                .query_param("external_source", "imdb_id");
            then.status(200)
                .json_body(serde_json::json!({
                    "movie_results": movie_results,
                    "tv_results": tv_results
                }));
        })
    }

    /// TMDB-shaped: a tmdb id alone.
    fn tmdb_only(kind: db::MediaKind, title: &str, tmdb: i64) -> db::Media {
        db::Media {
            id: Uuid::new_v4(),
            title: title.into(),
            kind,
            external_ids: db::ExternalIds {
                tmdb: Some(tmdb),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Cinemeta-shaped: an imdb id alone.
    fn imdb_only(kind: db::MediaKind, title: &str, imdb: &str) -> db::Media {
        db::Media {
            kind,
            ..search_hit(title, imdb)
        }
    }

    async fn search_kind(
        ctx: &AppContext,
        kind: db::MediaKind,
        runtimes: Vec<AddonRuntime>,
    ) -> Vec<db::Media> {
        search_kind_limit(ctx, kind, runtimes, 20).await
    }

    async fn search_kind_limit(
        ctx: &AppContext,
        kind: db::MediaKind,
        runtimes: Vec<AddonRuntime>,
        limit: usize,
    ) -> Vec<db::Media> {
        AddonService {
            inner: Arc::new(ArcSwap::from_pointee(runtimes)),
        }
        .search(&kind, "query", limit, ctx, None)
        .await
        .unwrap()
    }

    fn titles_of(results: &[db::Media]) -> Vec<&str> {
        results
            .iter()
            .map(|m| {
                m.title
                    .as_str()
            })
            .collect()
    }

    /// A TMDB copy and a Cinemeta copy of one series merge through TMDB's
    /// `/find` into the higher-priority copy, in either order.
    #[tokio::test]
    async fn search_merges_an_imdb_only_copy_through_tmdb_find() {
        use db::MediaKind::Series;
        let tmdb = httpmock::MockServer::start();
        mock_find(&tmdb, "tt9870001", &[], &[9870001]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let tmdb_copy =
            || search_runtime(vec![tmdb_only(Series, "BoB (TMDB)", 9870001)]);
        let cinemeta =
            || search_runtime(vec![imdb_only(Series, "BoB (Cinemeta)", "tt9870001")]);

        let results =
            search_kind(&guard.0, Series, vec![tmdb_copy(), cinemeta()]).await;
        assert_eq!(titles_of(&results), ["BoB (TMDB)"]);

        let results =
            search_kind(&guard.0, Series, vec![cinemeta(), tmdb_copy()]).await;
        assert_eq!(titles_of(&results), ["BoB (Cinemeta)"]);
        assert_eq!(
            results[0]
                .external_ids
                .tmdb,
            Some(9870001)
        );
        assert!(
            guard
                .0
                .store
                .get::<WeakSearchFind>(search_find_running_key(&Series, "tt9870001"))
                .is_none(),
            "a finished lookup clears its running marker"
        );
    }

    /// An imdb id TMDB places only as another kind, or nowhere, merges
    /// nothing.
    #[tokio::test]
    async fn search_keeps_imdb_only_copies_tmdb_does_not_place() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        mock_find(&tmdb, "tt9870002", &[], &[9870002]);
        mock_find(&tmdb, "tt9870003", &[], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = search_kind(
            &guard.0,
            Movie,
            vec![
                search_runtime(vec![tmdb_only(Movie, "Film (TMDB)", 9870002)]),
                search_runtime(vec![
                    imdb_only(Movie, "Film (Cinemeta)", "tt9870002"),
                    imdb_only(Movie, "Other (Cinemeta)", "tt9870003"),
                ]),
            ],
        )
        .await;
        assert_eq!(
            titles_of(&results),
            ["Film (TMDB)", "Film (Cinemeta)", "Other (Cinemeta)"]
        );
    }

    /// A single answering addon has no other addon's copy to merge with, so
    /// TMDB isn't asked. Across addons every imdb-only result is, more than
    /// `SEARCH_FIND_CONCURRENCY` of them, so a twin listed last and titled
    /// otherwise still merges. A repeat is answered from remembered answers.
    #[tokio::test]
    async fn search_asks_tmdb_find_about_every_result_across_addons() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let others = tmdb.mock(|when, then| {
            when.path_contains("/find/tt98701");
            then.status(200)
                .json_body(serde_json::json!({"movie_results": [], "tv_results": []}));
        });
        let twin = tmdb.mock(|when, then| {
            when.path("/find/tt9870299");
            then.status(200)
                .json_body(serde_json::json!({
                    "movie_results": [{
                        "id": 9870100, "title": "Film", "adult": false,
                        "original_language": "en"
                    }],
                    "tv_results": []
                }));
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let tmdb_copy = || tmdb_only(Movie, "Twin (TMDB)", 9870100);
        let cinemeta = |n: usize| {
            let mut hits: Vec<db::Media> = (0..n)
                .map(|i| {
                    imdb_only(
                        Movie,
                        &format!("Cinemeta {i}"),
                        &format!("tt98701{i:02}"),
                    )
                })
                .collect();
            hits.push(imdb_only(Movie, "Twin (Cinemeta)", "tt9870299"));
            hits
        };

        let single = vec![
            search_runtime([vec![tmdb_copy()], cinemeta(2)].concat()),
            search_runtime(vec![]),
        ];
        assert_eq!(
            search_kind(&guard.0, Movie, single)
                .await
                .len(),
            4
        );
        others.assert_hits(0);
        twin.assert_hits(0);

        let n = SEARCH_FIND_CONCURRENCY + 5;
        for _ in 0..2 {
            let both = vec![
                search_runtime(vec![tmdb_copy()]),
                search_runtime(cinemeta(n)),
            ];
            let results = search_kind(&guard.0, Movie, both).await;
            assert_eq!(results.len(), n + 1);
            assert!(!titles_of(&results).contains(&"Twin (Cinemeta)"));
        }
        others.assert_hits(n);
        twin.assert_hits(1);
    }

    /// Merges happen before `limit` cuts the results, and only imdb-only
    /// results that can still make `limit` are asked about.
    #[tokio::test]
    async fn search_asks_tmdb_find_only_about_results_that_can_make_the_limit() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let twin = tmdb.mock(|when, then| {
            when.path("/find/tt9870500");
            then.status(200)
                .json_body(serde_json::json!({
                    "movie_results": [{
                        "id": 9870500, "title": "Film", "adult": false,
                        "original_language": "en"
                    }],
                    "tv_results": []
                }));
        });
        let other = tmdb.mock(|when, then| {
            when.path("/find/tt9870501");
            then.status(200)
                .json_body(serde_json::json!({"movie_results": [], "tv_results": []}));
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let runtimes = || {
            vec![
                search_runtime(vec![
                    tmdb_only(Movie, "A (TMDB)", 9870500),
                    tmdb_only(Movie, "B (TMDB)", 9870599),
                ]),
                search_runtime(vec![
                    imdb_only(Movie, "A (Cinemeta)", "tt9870500"),
                    imdb_only(Movie, "C (Cinemeta)", "tt9870501"),
                ]),
            ]
        };

        // The two TMDB results fill a limit of two whatever TMDB answers.
        let results = search_kind_limit(&guard.0, Movie, runtimes(), 2).await;
        assert_eq!(titles_of(&results), ["A (TMDB)", "B (TMDB)"]);
        twin.assert_hits(0);
        other.assert_hits(0);

        // With room for one more, "A (Cinemeta)" merges and "C (Cinemeta)"
        // takes its place.
        let results = search_kind_limit(&guard.0, Movie, runtimes(), 3).await;
        assert_eq!(
            titles_of(&results),
            ["A (TMDB)", "B (TMDB)", "C (Cinemeta)"]
        );
        twin.assert_hits(1);
        other.assert_hits(1);
    }

    /// Imdb-only results titled like a tmdb-only one are asked about first,
    /// then the rest by position; without a tmdb-only result, none are.
    #[test]
    fn search_find_candidates_put_title_matches_first() {
        use db::MediaKind::Movie;
        let results = [
            imdb_only(Movie, "Other", "tt1"),
            tmdb_only(Movie, "The Target!", 1),
            imdb_only(Movie, "Another", "tt2"),
            imdb_only(Movie, "the target", "tt3"),
        ];
        let ids = |results: &[db::Media], limit: usize| -> Vec<String> {
            search_find_candidates(&Movie, results, limit)
                .into_iter()
                .map(ToString::to_string)
                .collect()
        };
        assert_eq!(ids(&results, 20), ["tt3", "tt1", "tt2"]);
        // "The Target!" alone fills a limit of one for the results after it.
        assert_eq!(ids(&results, 1), ["tt1"]);
        assert!(ids(&[results[0].clone(), results[2].clone()], 20).is_empty());
    }

    /// Only results of the searched kind with a well-formed imdb id are
    /// asked about; the rest stay, so they count as other results.
    #[test]
    fn search_find_candidates_skip_other_kinds_and_malformed_imdb_ids() {
        use db::MediaKind::{Movie, Series};
        let results = [
            tmdb_only(Movie, "Target", 1),
            imdb_only(Series, "Series-typed", "tt10"),
            imdb_only(Movie, "Traversal", "tt1/../../movie/550"),
            imdb_only(Movie, "Query", "tt1375666?x=1"),
            imdb_only(Movie, "Fragment", "tt1375666#f"),
            imdb_only(Movie, "Prefix only", "tt"),
            imdb_only(Movie, "Well-formed", "tt11"),
        ];
        let ids = |kind: db::MediaKind, limit: usize| -> Vec<String> {
            search_find_candidates(&kind, &results, limit)
                .into_iter()
                .map(ToString::to_string)
                .collect()
        };
        assert_eq!(ids(Movie, 20), ["tt11"]);
        assert!(ids(Movie, 6).is_empty());
        // No tmdb-only Series result to match.
        assert!(ids(Series, 20).is_empty());
    }

    fn find_running(ctx: &AppContext, kind: &db::MediaKind, imdb: &str) -> bool {
        ctx.store
            .get::<WeakSearchFind>(search_find_running_key(kind, imdb))
            .is_some()
    }

    /// A lookup outliving the budget doesn't cost the answers that landed.
    #[tokio::test]
    async fn tmdb_ids_via_find_keeps_answers_that_beat_a_slow_one() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        mock_find(&tmdb, "tt9870004", &[9870004], &[]);
        tmdb.mock(|when, then| {
            when.path("/find/tt9870005");
            then.status(200)
                .delay(Duration::from_secs(2))
                .json_body(serde_json::json!({"movie_results": [], "tv_results": []}));
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = [
            tmdb_only(Movie, "Fast (TMDB)", 9870004),
            imdb_only(Movie, "Slow (Cinemeta)", "tt9870005"),
            imdb_only(Movie, "Fast (Cinemeta)", "tt9870004"),
        ];
        let budget = Duration::from_millis(250);
        let limits =
            SearchFindLimits::new(SEARCH_FIND_CONCURRENCY, SEARCH_FIND_QUEUE_MAX);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        let fast = db::NonEmptyString::try_new("tt9870004".to_string()).unwrap();
        assert_eq!(found, [(fast, 9870004)]);
    }

    /// Cached answers, a cached miss included, are used without a permit:
    /// with every permit held, as by slow lookups, only the uncached id
    /// waits.
    #[tokio::test]
    async fn tmdb_ids_via_find_answers_from_the_cache_while_permits_are_taken() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let hit = mock_find(&tmdb, "tt9870700", &[9870700], &[]);
        let miss = mock_find(&tmdb, "tt9870701", &[], &[]);
        let uncached = mock_find(&tmdb, "tt9870702", &[9870702], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let client = crate::common::tmdb_search_find_client(
            &guard
                .0
                .db,
            &tmdb.base_url(),
        )
        .await
        .unwrap();
        for imdb in ["tt9870700", "tt9870701"] {
            MediaResolveService::find_tmdb_id_by(
                imdb.into(),
                "imdb_id",
                false,
                &client,
            )
            .await
            .unwrap();
        }
        let results = [
            tmdb_only(Movie, "Hit (TMDB)", 9870700),
            tmdb_only(Movie, "Uncached (TMDB)", 9870702),
            imdb_only(Movie, "Uncached (Cinemeta)", "tt9870702"),
            imdb_only(Movie, "Miss (Cinemeta)", "tt9870701"),
            imdb_only(Movie, "Hit (Cinemeta)", "tt9870700"),
        ];
        // Limits of its own: holding the shared permits would hold up other
        // tests' lookups.
        let limits = SearchFindLimits::new(0, SEARCH_FIND_QUEUE_MAX);
        let budget = Duration::from_millis(250);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        uncached.assert_hits(0);
        let hit_id = db::NonEmptyString::try_new("tt9870700".to_string()).unwrap();
        assert_eq!(found, [(hit_id, 9870700)]);
        hit.assert_hits(1);
        miss.assert_hits(1);
    }

    /// A lookup another search (keystroke) already started is joined, not
    /// sent again: its answer merges this search's copies too.
    #[tokio::test]
    async fn search_joins_a_find_already_running() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = tmdb.mock(|when, then| {
            when.path("/find/tt9870400");
            then.status(200)
                .json_body(serde_json::json!({"movie_results": [], "tv_results": []}));
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        // The running lookup's own handle; the store holds a weak one.
        let running: SearchFind = futures::FutureExt::shared(rx);
        guard
            .0
            .store
            .save(
                search_find_running_key(&Movie, "tt9870400"),
                running
                    .downgrade()
                    .unwrap(),
                SEARCH_FIND_RUNNING_TTL,
            );
        let search = search_kind(
            &guard.0,
            Movie,
            vec![
                search_runtime(vec![tmdb_only(Movie, "Film (TMDB)", 9870400)]),
                search_runtime(vec![imdb_only(Movie, "Film (Cinemeta)", "tt9870400")]),
            ],
        );
        let (results, _) = tokio::join!(search, async move {
            tokio::task::yield_now().await;
            tx.send(Some(9870400))
        });
        assert_eq!(titles_of(&results), ["Film (TMDB)"]);
        find.assert_hits(0);
    }

    /// A 429 on a lookup cools down only the lookups, not the TMDB search
    /// they serve; the copies just stay apart.
    #[tokio::test]
    async fn search_find_rate_limit_leaves_the_tmdb_search_unthrottled() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = tmdb.mock(|when, then| {
            when.path("/find/tt9870600");
            then.status(429);
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = search_kind(
            &guard.0,
            Movie,
            vec![
                search_runtime(vec![tmdb_only(Movie, "Film (TMDB)", 9870600)]),
                search_runtime(vec![imdb_only(Movie, "Film (Cinemeta)", "tt9870600")]),
            ],
        )
        .await;
        assert_eq!(titles_of(&results), ["Film (TMDB)", "Film (Cinemeta)"]);
        // Not retried, and the shared TMDB cooldown stays clear.
        find.assert_hits(1);
        assert_eq!(
            crate::common::tmdb_rate_limit()
                .remaining_cooldown()
                .await,
            Duration::ZERO
        );
    }

    /// A result whose own kind isn't the searched one is never looked up nor
    /// given the searched kind's ids: a movie-typed meta in a Series search
    /// took the tv id and opened as the movie with that number. A tmdb-only
    /// result of another kind sharing the number gets no imdb id either.
    #[tokio::test]
    async fn search_find_leaves_results_of_another_kind_alone() {
        use db::MediaKind::{Movie, Series};
        let tmdb = httpmock::MockServer::start();
        let wire = mock_find(&tmdb, "tt9871001", &[], &[9871001]);
        let interstellar = mock_find(&tmdb, "tt9871002", &[9871002], &[]);
        let show = mock_find(&tmdb, "tt9871003", &[], &[9871003]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;

        let results = search_kind(
            &guard.0,
            Series,
            vec![
                search_runtime(vec![
                    tmdb_only(Series, "The Wire (TMDB)", 9871001),
                    tmdb_only(Series, "Show (TMDB)", 9871003),
                    tmdb_only(Movie, "Film numbered like Show", 9871003),
                ]),
                search_runtime(vec![
                    imdb_only(Movie, "The Wire (movie-typed)", "tt9871001"),
                    imdb_only(Series, "Show (Cinemeta)", "tt9871003"),
                ]),
            ],
        )
        .await;
        assert_eq!(
            titles_of(&results),
            [
                "The Wire (TMDB)",
                "Show (TMDB)",
                "Film numbered like Show",
                "The Wire (movie-typed)"
            ]
        );
        wire.assert_hits(0);
        show.assert_hits(1);
        assert_eq!(
            results[2]
                .external_ids
                .imdb,
            None
        );
        assert_eq!(
            results[3]
                .external_ids
                .tmdb,
            None
        );

        let results = search_kind(
            &guard.0,
            Movie,
            vec![
                search_runtime(vec![tmdb_only(Movie, "Interstellar (TMDB)", 9871002)]),
                search_runtime(vec![imdb_only(
                    Series,
                    "Interstellar (series-typed)",
                    "tt9871002",
                )]),
            ],
        )
        .await;
        assert_eq!(
            titles_of(&results),
            ["Interstellar (TMDB)", "Interstellar (series-typed)"]
        );
        interstellar.assert_hits(0);
        assert_eq!(
            results[1]
                .external_ids
                .tmdb,
            None
        );
    }

    /// A stored imdb-only row adopted by the Cinemeta copy is listed, not
    /// the TMDB copy listed before it that `/find` links it to.
    #[tokio::test]
    async fn search_lists_the_stored_row_a_tmdb_find_links_to() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        mock_find(&tmdb, "tt9871100", &[9871100], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let mut stored = imdb_only(Movie, "Heat (stored)", "tt9871100");
        stored
            .save(
                &guard
                    .0
                    .db,
            )
            .await
            .unwrap();
        let results = search_kind(
            &guard.0,
            Movie,
            vec![
                search_runtime(vec![tmdb_only(Movie, "Heat (TMDB)", 9871100)]),
                search_runtime(vec![imdb_only(Movie, "Heat (Cinemeta)", "tt9871100")]),
            ],
        )
        .await;
        assert_eq!(titles_of(&results), ["Heat (stored)"]);
        assert_eq!(results[0].id, stored.id);
    }

    /// An imdb-only result sharing its imdb id with a copy carrying another
    /// title's tmdb id isn't merged into it before `/find`: it is asked about
    /// and reaches the final dedupe with its own tmdb id. Merged first, an
    /// imdb-only Interstellar was never asked about and stayed hidden behind
    /// an Inception copy carrying Interstellar's imdb id. The wrong copy
    /// itself still collapses into the TMDB result it shares a tmdb id with.
    #[tokio::test]
    async fn link_via_find_asks_about_an_imdb_only_result_a_wrong_copy_shares() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = mock_find(&tmdb, "tt9871800", &[9871802], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let mut wrong_copy =
            imdb_only(Movie, "Inception (Interstellar's imdb id)", "tt9871800");
        wrong_copy
            .external_ids
            .tmdb = Some(9871801);
        let results = vec![
            tmdb_only(Movie, "Inception (TMDB)", 9871801),
            wrong_copy,
            imdb_only(Movie, "Interstellar (Cinemeta)", "tt9871800"),
        ];
        let results = link_via_find(&Movie, results, 20, &guard.0).await;
        assert_eq!(
            titles_of(&results),
            ["Inception (TMDB)", "Interstellar (Cinemeta)"]
        );
        assert_eq!(
            results[1]
                .external_ids
                .tmdb,
            Some(9871802)
        );
        find.assert_hits(1);
    }

    /// Before `/find`, candidates merge with each other but not into a
    /// result that has a tmdb id; everything else dedupes as usual, and kept
    /// results stay in input order.
    #[test]
    fn dedupe_holding_find_candidates_apart_keeps_candidates_out_of_tmdb_groups() {
        use db::MediaKind::Movie;
        let mut linked = imdb_only(Movie, "Linked", "tt1");
        linked
            .external_ids
            .tmdb = Some(1);
        let results = vec![
            tmdb_only(Movie, "Tmdb", 1),
            imdb_only(Movie, "Imdb", "tt1"),
            linked,
            imdb_only(Movie, "Imdb again", "tt1"),
            tmdb_only(Movie, "Tmdb again", 1),
        ];
        let kept = dedupe_holding_find_candidates_apart(&Movie, results);
        assert_eq!(titles_of(&kept), ["Tmdb", "Imdb"]);
    }

    /// With no tmdb-only result, only imdb-only results sharing their imdb id
    /// with a result that has a tmdb id are asked about.
    #[test]
    fn search_find_candidates_without_a_tmdb_only_result_ask_about_shared_imdb_ids() {
        use db::MediaKind::Movie;
        let mut linked = imdb_only(Movie, "Wrong copy", "tt1");
        linked
            .external_ids
            .tmdb = Some(1);
        let results = [
            imdb_only(Movie, "Unrelated", "tt2"),
            linked,
            imdb_only(Movie, "Victim", "tt1"),
        ];
        let ids: Vec<String> = search_find_candidates(&Movie, &results, 20)
            .into_iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(ids, ["tt1"]);
    }

    /// Round 5's F4O/F4X and absorbed-base shapes: an Inception copy
    /// carrying Interstellar's imdb id, an imdb-only Interstellar, and no
    /// tmdb-only result left after the first dedupe (none listed, or the TMDB
    /// Inception listed after the wrong copy and absorbed into it). In every
    /// order Interstellar is asked about and gets its own tmdb id, which the
    /// final dedupe's conflict check keeps apart from the wrong copy's.
    #[tokio::test]
    async fn link_via_find_asks_about_an_imdb_only_result_with_no_tmdb_only_one_left() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = mock_find(&tmdb, "tt9871900", &[9871902], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let mut wrong_copy = imdb_only(Movie, "Inception (wrong imdb)", "tt9871900");
        wrong_copy
            .external_ids
            .tmdb = Some(9871901);
        let victim = || imdb_only(Movie, "Interstellar (imdb-only)", "tt9871900");
        let base = || tmdb_only(Movie, "Inception (TMDB)", 9871901);
        for results in [
            vec![wrong_copy.clone(), victim()],
            vec![victim(), wrong_copy.clone()],
            vec![wrong_copy.clone(), base(), victim()],
        ] {
            let results = link_via_find(&Movie, results, 20, &guard.0).await;
            let victim = results
                .iter()
                .find(|m| m.title == "Interstellar (imdb-only)")
                .unwrap();
            assert_eq!(
                victim
                    .external_ids
                    .tmdb,
                Some(9871902)
            );
            assert!(
                results
                    .iter()
                    .any(|m| m
                        .external_ids
                        .tmdb
                        == Some(9871901))
            );
        }
        // Asked once, then remembered.
        find.assert_hits(1);
    }

    /// `/find` answers are kept apart from the response cache, which a
    /// library refresh churns: a lookup's answer is remembered, and one
    /// remembered is used without asking TMDB.
    #[tokio::test]
    async fn search_find_answers_are_kept_apart_from_the_response_cache() {
        use db::MediaKind::Series;
        let tmdb = httpmock::MockServer::start();
        let asked = mock_find(&tmdb, "tt9871200", &[], &[9871200]);
        let remembered = mock_find(&tmdb, "tt9871201", &[], &[9871201]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let runtimes = |imdb: &str, tmdb: i64| {
            vec![
                search_runtime(vec![tmdb_only(Series, "Show (TMDB)", tmdb)]),
                search_runtime(vec![imdb_only(Series, "Show (Cinemeta)", imdb)]),
            ]
        };

        let results =
            search_kind(&guard.0, Series, runtimes("tt9871200", 9871200)).await;
        assert_eq!(titles_of(&results), ["Show (TMDB)"]);
        asked.assert_hits(1);
        let answer = SEARCH_FIND_ANSWERS
            .get::<SearchFindAnswer>("tt9871200")
            .unwrap();
        assert_eq!((answer.movie, answer.tv), (None, Some(9871200)));

        // Not in the response cache, so only `SEARCH_FIND_ANSWERS` knows it.
        SEARCH_FIND_ANSWERS.save(
            "tt9871201",
            SearchFindAnswer {
                movie: None,
                tv: Some(9871201),
            },
            Duration::from_secs(60),
        );
        let results =
            search_kind(&guard.0, Series, runtimes("tt9871201", 9871201)).await;
        assert_eq!(titles_of(&results), ["Show (TMDB)"]);
        remembered.assert_hits(0);
    }

    /// A lookup still queued when its search stopped waiting is dropped
    /// unsent and its running marker cleared; the next search asks again.
    #[tokio::test]
    async fn tmdb_ids_via_find_drops_a_queued_lookup_nobody_waits_for() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = mock_find(&tmdb, "tt9871300", &[9871300], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = [
            tmdb_only(Movie, "Film (TMDB)", 9871300),
            imdb_only(Movie, "Film (Cinemeta)", "tt9871300"),
        ];
        let limits = SearchFindLimits::new(0, SEARCH_FIND_QUEUE_MAX);
        let budget = Duration::from_millis(100);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        assert!(found.is_empty());
        assert!(find_running(&guard.0, &Movie, "tt9871300"));

        limits
            .permits
            .add_permits(1);
        tokio::time::timeout(Duration::from_secs(5), async {
            while find_running(&guard.0, &Movie, "tt9871300") {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        find.assert_hits(0);

        let budget = Duration::from_secs(5);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        let imdb = db::NonEmptyString::try_new("tt9871300".to_string()).unwrap();
        assert_eq!(found, [(imdb, 9871300)]);
        find.assert_hits(1);
    }

    /// Past the queue cap a search asks about no more.
    #[tokio::test]
    async fn tmdb_ids_via_find_queues_no_lookups_past_the_cap() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let first = mock_find(&tmdb, "tt9871400", &[9871400], &[]);
        let second = mock_find(&tmdb, "tt9871401", &[9871401], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = [
            tmdb_only(Movie, "A (TMDB)", 9871400),
            tmdb_only(Movie, "B (TMDB)", 9871401),
            imdb_only(Movie, "A (Cinemeta)", "tt9871400"),
            imdb_only(Movie, "B (Cinemeta)", "tt9871401"),
        ];
        let limits = SearchFindLimits::new(0, 1);
        let budget = Duration::from_millis(100);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        assert!(found.is_empty());
        assert!(find_running(&guard.0, &Movie, "tt9871400"));
        assert!(
            !find_running(&guard.0, &Movie, "tt9871401"),
            "queued past the cap"
        );
        first.assert_hits(0);
        second.assert_hits(0);
    }

    /// A stalled lookup gives its permit back after the request timeout, so
    /// the lookup queued behind it still answers within the budget, and the
    /// timeout isn't remembered as a miss.
    #[tokio::test]
    async fn tmdb_ids_via_find_releases_the_permit_of_a_stalled_lookup() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        tmdb.mock(|when, then| {
            when.path("/find/tt9871700");
            then.status(200)
                .delay(Duration::from_secs(10))
                .json_body(serde_json::json!({"movie_results": [], "tv_results": []}));
        });
        mock_find(&tmdb, "tt9871701", &[9871701], &[]);
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        let results = [
            tmdb_only(Movie, "A (TMDB)", 9871700),
            tmdb_only(Movie, "B (TMDB)", 9871701),
            imdb_only(Movie, "Stalled (Cinemeta)", "tt9871700"),
            imdb_only(Movie, "Queued (Cinemeta)", "tt9871701"),
        ];
        // One permit, so the second lookup waits for the stalled one.
        let limits = SearchFindLimits {
            request_timeout: Duration::from_millis(200),
            ..SearchFindLimits::new(1, SEARCH_FIND_QUEUE_MAX)
        };
        let budget = Duration::from_secs(3);
        let found =
            tmdb_ids_via_find(&Movie, &results, 20, budget, &limits, &guard.0).await;
        let queued = db::NonEmptyString::try_new("tt9871701".to_string()).unwrap();
        assert_eq!(found, [(queued, 9871701)]);
        assert!(
            SEARCH_FIND_ANSWERS
                .get::<SearchFindAnswer>("tt9871700")
                .is_none()
        );
        assert!(!find_running(&guard.0, &Movie, "tt9871700"));
    }

    /// A 200 `/find` body without result keys, as a proxy wrapping an error
    /// sends, is no answer: it isn't remembered as a miss, so the next
    /// search asks again.
    #[tokio::test]
    async fn search_asks_again_after_a_find_body_without_result_keys() {
        use db::MediaKind::Movie;
        let tmdb = httpmock::MockServer::start();
        let find = tmdb.mock(|when, then| {
            when.path("/find/tt9871500");
            then.status(200)
                .json_body(serde_json::json!({"success": false, "status_code": 34}));
        });
        let guard = crate::integration_test::ctx_with_tmdb(&tmdb).await;
        for _ in 0..2 {
            let results = search_kind(
                &guard.0,
                Movie,
                vec![
                    search_runtime(vec![tmdb_only(Movie, "Film (TMDB)", 9871500)]),
                    search_runtime(vec![imdb_only(
                        Movie,
                        "Film (Cinemeta)",
                        "tt9871500",
                    )]),
                ],
            )
            .await;
            assert_eq!(titles_of(&results), ["Film (TMDB)", "Film (Cinemeta)"]);
        }
        find.assert_hits(2);
        assert!(
            SEARCH_FIND_ANSWERS
                .get::<SearchFindAnswer>("tt9871500")
                .is_none()
        );
    }
}

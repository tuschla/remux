use crate::{
    sdks,
    sdks::{CachedEndpoint, ClientError, Endpoint, WithExtraQuery},
};
use anyhow::{Result, anyhow};
use futures::{
    future,
    stream::{self, Stream, StreamExt},
};
use std::{
    pin::Pin,
    time::{Duration, Instant},
};
use tracing::debug;

/// A 404 on a paginated catalog page is the addon's normal "no more pages"
/// signal, not an error — some addons (e.g. fankai) 404 past the last page
/// instead of returning an empty array. Any other error (5xx, rate limit,
/// network blip) is a real problem and must not be conflated with reaching
/// the end of the catalog.
fn is_404(e: &ClientError) -> bool {
    matches!(
        e,
        ClientError::Http { status: 404, .. } | ClientError::Json { status: 404, .. }
    )
}

#[derive(Clone)]
pub struct StremioService {
    pub client: sdks::RestClient,
    extra_query: Vec<(String, String)>,
}

impl StremioService {
    pub fn from_url(url: &str) -> Result<Self> {
        let parsed = url::Url::parse(url).map_err(|e| anyhow::anyhow!("{e}"))?;
        let extra_query: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let mut clean = parsed.clone();
        clean.set_query(None);
        let base = clean
            .as_str()
            .trim_end_matches('/')
            .to_string()
            + "/";
        Ok(Self {
            client: sdks::stremio::client(&base)?,
            extra_query,
        })
    }

    /// Shares a 429 cooldown across every client built for the same addon.
    /// `from_url` builds a fresh `RestClient` on every call (addons are not
    /// cached), so without this each concurrent or sequential call starts
    /// with no memory of a prior 429 from the same addon.
    pub fn with_shared_rate_limit(mut self, limit: sdks::SharedRateLimit) -> Self {
        self.client = self
            .client
            .with_shared_rate_limit(limit);
        self
    }

    fn ep<EP: Endpoint + Clone>(&self, endpoint: EP) -> WithExtraQuery<EP> {
        WithExtraQuery {
            endpoint,
            extra: self
                .extra_query
                .clone(),
        }
    }

    pub async fn get_manifest(&self) -> Result<sdks::stremio::Manifest> {
        Ok(self
            .client
            .execute(
                self.ep(sdks::stremio::ManifestEndpoint)
                    .with_cache(Duration::from_secs(3600)),
            )
            .await?)
    }

    pub async fn get_meta(
        &self,
        media_type: sdks::stremio::MediaType,
        id: impl Into<String>,
    ) -> Result<Option<sdks::stremio::Meta>> {
        Ok(self
            .client
            .execute(
                self.ep(sdks::stremio::MetaEndpoint {
                    media_type,
                    id: id.into(),
                    season: None,
                    episode: None,
                })
                .with_cache(Duration::from_secs(3600))
                // Addons report upstream failures as a 200 OK with an
                // error-shaped body, not via HTTP status — caching one of
                // those would replay the same stale failure from the HTTP
                // cache for up to an hour, regardless of the addon-local
                // `medias_cache` skip in `fetch_and_cache_meta`.
                .should_cache(|response: &sdks::stremio::MetaResponse| {
                    (!response
                        .meta
                        .as_ref()
                        .is_some_and(|m| m.is_error()))
                    .then_some(Duration::from_secs(3600))
                }),
            )
            .await?
            .meta)
    }

    pub async fn search(
        &self,
        media_type: sdks::stremio::MediaType,
        q: String,
    ) -> Result<Vec<sdks::stremio::Meta>> {
        let catalog = self
            .get_manifest()
            .await?
            .get_search_catalog(&media_type.to_string())
            .ok_or_else(|| anyhow!("no search catalog for type {}", media_type))?;
        Ok(self
            .client
            .execute(
                self.ep(sdks::stremio::CatalogEndpoint {
                    kind: catalog
                        .kind
                        .clone(),
                    id: catalog
                        .id
                        .clone(),
                    search: Some(q),
                    genre: None,
                    skip: None,
                })
                .with_cache(Duration::from_secs(60)),
            )
            .await?
            .metas)
    }

    pub async fn get_streams(
        &self,
        media_type: sdks::stremio::MediaType,
        id: impl Into<String>,
    ) -> Result<Vec<sdks::stremio::Stream>> {
        Ok(self
            .client
            .execute(
                self.ep(sdks::stremio::StreamEndpoint {
                    kind: media_type,
                    id: id.into(),
                })
                .with_cache(Duration::from_secs(300)),
            )
            .await?
            .streams)
    }

    pub async fn get_subtitles(
        &self,
        media_type: sdks::stremio::MediaType,
        imdb_id: &str,
        season: Option<i64>,
        episode: Option<i64>,
        filename: Option<String>,
        video_size: Option<i64>,
    ) -> Result<Vec<sdks::stremio::Subtitle>> {
        Ok(self
            .client
            .execute(
                self.ep(sdks::stremio::SubtitlesEndpoint {
                    media_type,
                    imdb_id: imdb_id.to_string(),
                    season,
                    episode,
                    filename,
                    video_size,
                })
                .with_cache(Duration::from_secs(86_400)),
            )
            .await?
            .subtitles)
    }

    pub async fn get_catalog_stream(
        &self,
        kind: String,
        id: String,
        supports_skip: bool,
        page_concurrency: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<sdks::stremio::Meta>> + Send>>> {
        let client = self
            .client
            .clone();
        let extra_query = self
            .extra_query
            .clone();

        let t0 = Instant::now();
        let first_page = client
            .execute(WithExtraQuery {
                endpoint: sdks::stremio::CatalogEndpoint {
                    kind: kind.clone(),
                    id: id.clone(),
                    search: None,
                    genre: None,
                    skip: None,
                },
                extra: extra_query.clone(),
            })
            .await?;

        let page_size = first_page
            .metas
            .len() as u32;
        debug!(
            kind = %kind,
            id = %id,
            page_size,
            page_concurrency = page_concurrency.max(1),
            elapsed = ?t0.elapsed(),
            "catalog first page"
        );
        if page_size == 0 || !supports_skip {
            return Ok(Box::pin(stream::iter(
                first_page
                    .metas
                    .into_iter()
                    .map(Ok::<_, anyhow::Error>),
            )));
        }

        let first = stream::once(future::ready(Ok(first_page)));

        let rest = stream::iter(1..999u32)
            .map(move |page| {
                let client = client.clone();
                let kind = kind.clone();
                let id = id.clone();
                let extra_query = extra_query.clone();
                async move {
                    let skip = page * page_size;
                    let mut raw_response = None;
                    let result = client
                        .execute_observed(
                            WithExtraQuery {
                                endpoint: sdks::stremio::CatalogEndpoint {
                                    kind: kind.clone(),
                                    id: id.clone(),
                                    search: None,
                                    genre: None,
                                    skip: Some(skip),
                                },
                                extra: extra_query,
                            },
                            |status, body| {
                                // Bounded: this fires for every page regardless of
                                // outcome, including large successful pages whose
                                // body is never read below — and an addon's raw
                                // response is untrusted, unbounded-size input that
                                // must not be logged in full (see the `Secret`
                                // wrapper `ClientError` itself uses for bodies).
                                let snippet: String = body
                                    .chars()
                                    .take(300)
                                    .collect();
                                raw_response = Some((status, snippet))
                            },
                        )
                        .await;
                    match &result {
                        Ok(response) => tracing::debug!(
                            kind = %kind,
                            id = %id,
                            page,
                            skip,
                            metas = response.metas.len(),
                            "catalog page fetched"
                        ),
                        // A 404 here is the addon's normal "no more pages" signal
                        // (see `is_404`), not a real failure — every completed
                        // catalog import ends on one. Logging it as a failure
                        // would flag completely normal pagination on every run.
                        Err(error) if is_404(error) => {
                            tracing::debug!(
                                kind = %kind,
                                id = %id,
                                page,
                                skip,
                                "catalog pagination reached end of catalog (404)"
                            );
                        }
                        Err(error) => {
                            let status = raw_response
                                .as_ref()
                                .map(|(status, _)| *status)
                                .unwrap_or_default();
                            let body = raw_response
                                .as_ref()
                                .map(|(_, body)| body.as_str())
                                .unwrap_or_default();
                            tracing::info!(
                                kind = %kind,
                                id = %id,
                                page,
                                skip,
                                status,
                                body = %body,
                                error = %error,
                                "catalog page fetch failed"
                            )
                        }
                    }
                    if let Ok(response) = &result
                        && response
                            .metas
                            .is_empty()
                    {
                        let status = raw_response
                            .as_ref()
                            .map(|(status, _)| *status)
                            .unwrap_or_default();
                        let body = raw_response
                            .as_ref()
                            .map(|(_, body)| body.as_str())
                            .unwrap_or_default();
                        tracing::debug!(
                            kind = %kind,
                            id = %id,
                            page,
                            skip,
                            status,
                            body = %body,
                            "catalog page returned no metas"
                        );
                    }
                    result
                }
            })
            .buffered(page_concurrency.max(1));

        let pages = first
            .chain(rest)
            // `scan` ends the stream at the first page it maps to `None`, so the
            // 404-vs-error distinction has to happen here. A 404 or an empty page
            // is the end of the catalog. Any other error is passed on as the last
            // item: the caller has to know the catalog was cut short, not that it
            // ended, or it would drop every item after the failed page.
            .scan(false, |failed, result| {
                if *failed {
                    return future::ready(None);
                }
                future::ready(match result {
                    Ok(response) => {
                        let metas = response.metas;
                        if metas.is_empty() {
                            None
                        } else {
                            Some(
                                stream::iter(
                                    metas
                                        .into_iter()
                                        .map(Ok::<_, anyhow::Error>),
                                )
                                .left_stream(),
                            )
                        }
                    }
                    Err(e) if is_404(&e) => {
                        debug!(
                            "stopping catalog pagination: reached end of catalog (404)"
                        );
                        None
                    }
                    Err(e) => {
                        *failed = true;
                        Some(
                            stream::once(future::ready(Err(anyhow::Error::from(e))))
                                .right_stream(),
                        )
                    }
                })
            })
            .flatten();

        Ok(Box::pin(pages))
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;

    use super::*;

    #[test]
    fn is_404_matches_only_404_status() {
        assert!(is_404(&ClientError::Http {
            status: 404,
            message: "not found".to_string(),
            endpoint: None,
            body: None,
        }));
        assert!(!is_404(&ClientError::Http {
            status: 500,
            message: "server error".to_string(),
            endpoint: None,
            body: None,
        }));
        assert!(!is_404(&ClientError::RateLimited {
            retry_after_secs: 30
        }));
    }

    fn mock_page(server: &httpmock::MockServer, path: &str, names: &[&str]) {
        let metas: Vec<_> = names
            .iter()
            .map(|n| serde_json::json!({"id": n, "type": "movie", "name": n}))
            .collect();
        server.mock(|when, then| {
            when.path(path);
            then.status(200)
                .json_body(serde_json::json!({"metas": metas}));
        });
    }

    /// Regression test for the take_while/filter_map bug: take_while drops the
    /// first item it rejects instead of passing it downstream, so a 404 on page
    /// 2 must still yield page 1's items and stop cleanly there.
    #[tokio::test]
    async fn get_catalog_stream_stops_cleanly_on_404() {
        let server = httpmock::MockServer::start();
        mock_page(&server, "/catalog/movie/test.json", &["a", "b"]);
        server.mock(|when, then| {
            when.path("/catalog/movie/test/skip=2.json");
            then.status(404);
        });

        let svc = StremioService::from_url(&server.base_url()).unwrap();
        let stream = svc
            .get_catalog_stream("movie".to_string(), "test".to_string(), true, 3)
            .await
            .unwrap();
        let names: Vec<String> = stream
            .map(|m| m.map(|m| m.id))
            .try_collect()
            .await
            .unwrap();

        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }

    /// A non-404 error (5xx) is not the end of the catalog: the stream must
    /// still yield page 1's items, then end on that error so the caller can
    /// tell a cut-short catalog from a complete one.
    #[tokio::test]
    async fn get_catalog_stream_ends_on_non_404_error() {
        let server = httpmock::MockServer::start();
        mock_page(&server, "/catalog/movie/test.json", &["a", "b"]);
        server.mock(|when, then| {
            when.path("/catalog/movie/test/skip=2.json");
            then.status(500);
        });

        let svc = StremioService::from_url(&server.base_url()).unwrap();
        let stream = svc
            .get_catalog_stream("movie".to_string(), "test".to_string(), true, 3)
            .await
            .unwrap();
        let mut items: Vec<Result<String>> = stream
            .map(|m| m.map(|m| m.id))
            .collect()
            .await;

        let last = items.pop();
        assert!(
            matches!(last, Some(Err(_))),
            "a failed page must end the stream on an error"
        );
        let names: Vec<String> = items
            .into_iter()
            .map(|name| name.unwrap())
            .collect();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }
}

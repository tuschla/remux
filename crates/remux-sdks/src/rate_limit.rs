use http::{HeaderMap, header};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use tokio::{sync::Mutex, time::Instant};

pub(crate) const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(60);

const MAX_RETRY_AFTER: Duration = Duration::from_secs(48 * 60 * 60);

/// Longest a request inside [`interactive`] waits on a 429, either for its
/// own Retry-After or for a shared cooldown before sending. A longer
/// requested wait still feeds the shared cooldown in full, but this caller
/// gets `RateLimited` in seconds instead of being held for the whole window
/// (60 s by default, up to `MAX_RETRY_AFTER`), and later interactive callers
/// get it at once without sending.
#[cfg(not(target_arch = "wasm32"))]
const MAX_IN_REQUEST_WAIT: Duration = Duration::from_secs(5);

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    static INTERACTIVE: ();
}

/// Runs `fut` as an interactive request, such as a user's search.
///
/// Inside it, a 429 is waited on for at most [`MAX_IN_REQUEST_WAIT`] and
/// not retried, and a request is refused without sending while more than
/// that is left of the shared cooldown: someone is waiting for the answer,
/// and a fallback beats a minute-long stall. Outside it, background work
/// (catalog import, metadata refresh) waits out every cooldown in full and
/// retries, because giving up there loses data, e.g. a catalog page that
/// was never fetched. Tasks spawned from `fut` do not inherit the scope.
#[cfg(not(target_arch = "wasm32"))]
pub async fn interactive<F: Future>(fut: F) -> F::Output {
    INTERACTIVE
        .scope((), fut)
        .await
}

#[cfg(not(target_arch = "wasm32"))]
fn is_interactive() -> bool {
    INTERACTIVE
        .try_with(|_| ())
        .is_ok()
}

/// Response extension set by [`RetryAfterMiddleware`] on a 429 whose wait it
/// cut short at [`MAX_IN_REQUEST_WAIT`]. The retry middleware returns such a
/// response as-is: each retry would hit the same limit and wait again,
/// turning one request into several back-to-back capped waits. A 429 whose
/// whole wait fit under the cap stays untagged and is retried as usual.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
pub(crate) struct RateLimitedResponse;

/// Returned by [`RetryAfterMiddleware`] inside [`interactive`], without
/// sending, when the shared cooldown has more than [`MAX_IN_REQUEST_WAIT`] left. Mapped to
/// `ClientError::RateLimited`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub(crate) struct CooldownActive {
    pub(crate) remaining: Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl CooldownActive {
    /// Rounded up, so a sub-second remainder never reads as "retry now".
    pub(crate) fn retry_after_secs(&self) -> u64 {
        self.remaining
            .as_secs()
            + u64::from(
                self.remaining
                    .subsec_nanos()
                    > 0,
            )
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Display for CooldownActive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shared rate-limit cooldown active, retry after {}s",
            self.retry_after_secs()
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::error::Error for CooldownActive {}

/// Shared 429 cooldown for one provider.
///
/// Give clones of one value to every [`RestClient`](crate::RestClient) that
/// talks to the same upstream. After a 429, later requests wait out the
/// cooldown; inside [`interactive`] they fail fast, without sending, when
/// more than [`MAX_IN_REQUEST_WAIT`] of it is left. It
/// intentionally does not limit request concurrency; callers control that
/// themselves.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct SharedRateLimit {
    blocked_until: Arc<Mutex<Instant>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SharedRateLimit {
    pub fn new() -> Self {
        Self {
            blocked_until: Arc::new(Mutex::new(Instant::now())),
        }
    }

    async fn block_for(&self, delay: Duration) {
        let candidate = Instant::now() + delay;
        let mut blocked_until = self
            .blocked_until
            .lock()
            .await;
        *blocked_until = (*blocked_until).max(candidate);
    }

    /// Waits for the cooldown to clear, unless that would take longer than
    /// `max_wait`; then it returns at once. Giving up never touches the
    /// cooldown, so callers refused here don't extend it.
    async fn wait_for_cooldown(
        &self,
        max_wait: Duration,
    ) -> Result<(), CooldownActive> {
        // `None` (overflow) means no limit.
        let give_up_after = Instant::now().checked_add(max_wait);
        loop {
            let blocked_until = *self
                .blocked_until
                .lock()
                .await;
            let now = Instant::now();
            if blocked_until <= now {
                return Ok(());
            }
            if give_up_after.is_some_and(|limit| blocked_until > limit) {
                return Err(CooldownActive {
                    remaining: blocked_until - now,
                });
            }
            tokio::time::sleep_until(blocked_until).await;
        }
    }

    /// Time left before the cooldown clears — `Duration::ZERO` if not
    /// currently blocked. Unlike `wait_for_cooldown`, this never sleeps: a
    /// caller can use it to decide whether it's even worth starting a
    /// request under its own bounded timeout, rather than beginning the
    /// attempt and having the timeout fire mid-wait, indistinguishable from
    /// a genuinely hung request.
    pub async fn remaining_cooldown(&self) -> Duration {
        let blocked_until = *self
            .blocked_until
            .lock()
            .await;
        blocked_until.saturating_duration_since(Instant::now())
    }
}

pub(crate) fn retry_after(
    headers: &HeaderMap,
    now: SystemTime,
    default: Duration,
) -> Duration {
    headers
        .get(header::RETRY_AFTER)
        .and_then(|value| {
            value
                .to_str()
                .ok()
        })
        .and_then(|value| parse_retry_after(value, now))
        .unwrap_or(default)
        .min(MAX_RETRY_AFTER)
}

fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = chrono::DateTime::parse_from_rfc2822(value)
        .map(|date| date.timestamp())
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .map(|date| {
                    date.and_utc()
                        .timestamp()
                })
        })
        .ok()?;
    let retry_at = u64::try_from(retry_at).unwrap_or_default();
    let now = now
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(Duration::from_secs(retry_at.saturating_sub(now)))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct RetryAfterMiddleware {
    pub(crate) default_retry_after: Duration,
    pub(crate) shared_rate_limit: Option<SharedRateLimit>,
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl reqwest_middleware::Middleware for RetryAfterMiddleware {
    async fn handle(
        &self,
        req: reqwest::Request,
        extensions: &mut http::Extensions,
        next: reqwest_middleware::Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        let interactive = is_interactive();
        // Interactive: waiting out a long cooldown here would hold this
        // request for the rest of a Retry-After window (60 s by default) and
        // then send it into the same limit, re-arming the cooldown. Fail fast
        // instead; short cooldowns (TMDB's 2 s) are still waited out.
        let max_wait = if interactive {
            MAX_IN_REQUEST_WAIT
        } else {
            Duration::MAX
        };
        if let Some(limit) = &self.shared_rate_limit {
            limit
                .wait_for_cooldown(max_wait)
                .await
                .map_err(reqwest_middleware::Error::middleware)?;
        }
        let mut response = next
            .run(req, extensions)
            .await?;
        if response.status() != http::StatusCode::TOO_MANY_REQUESTS {
            return Ok(response);
        }

        let delay = retry_after(
            response.headers(),
            SystemTime::now(),
            self.default_retry_after,
        );
        if delay.is_zero() {
            return Ok(response);
        }
        let wait = delay.min(max_wait);
        tracing::warn!(
            host = log_host(response.url()),
            retry_after_secs = delay.as_secs(),
            wait_secs = wait.as_secs(),
            shared = self
                .shared_rate_limit
                .is_some(),
            "upstream returned 429; backing off before this request returns"
        );
        if let Some(limit) = &self.shared_rate_limit {
            limit
                .block_for(delay)
                .await;
        }
        if interactive {
            tokio::time::sleep(wait).await;
            if wait < delay {
                response
                    .extensions_mut()
                    .insert(RateLimitedResponse);
            }
        } else if let Some(limit) = &self.shared_rate_limit {
            // Also waits out a longer cooldown another request recorded.
            limit
                .wait_for_cooldown(Duration::MAX)
                .await
                .map_err(reqwest_middleware::Error::middleware)?;
        } else {
            tokio::time::sleep(delay).await;
        }
        Ok(response)
    }
}

/// What the 429 warning names the upstream by: the host only. The userinfo,
/// path and query of addon URLs often carry credentials.
#[cfg(not(target_arch = "wasm32"))]
fn log_host(url: &url::Url) -> &str {
    url.host_str()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_429_warning_names_only_the_host() {
        let url = url::Url::parse(
            "http://user:pass@addon.example:8080/SECRET-TOKEN/catalog/movie/search=x.json?apikey=KEY#frag",
        )
        .unwrap();
        assert_eq!(log_host(&url), "addon.example");
    }

    #[test]
    fn parses_delay_seconds() {
        let now = UNIX_EPOCH + Duration::from_secs(10);
        assert_eq!(parse_retry_after("42", now), Some(Duration::from_secs(42)));
    }

    #[test]
    fn parses_http_date() {
        let now = UNIX_EPOCH + Duration::from_secs(1_445_412_450);
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn a_date_in_the_past_means_no_wait() {
        let now = UNIX_EPOCH + Duration::from_secs(1_600_000_000);
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn defaults_to_the_configured_default_when_absent_or_unparseable() {
        assert_eq!(
            retry_after(&HeaderMap::new(), UNIX_EPOCH, DEFAULT_RETRY_AFTER),
            DEFAULT_RETRY_AFTER
        );

        let mut headers = HeaderMap::new();
        headers.insert(
            header::RETRY_AFTER,
            "not-a-delay"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            retry_after(&headers, UNIX_EPOCH, Duration::from_secs(2)),
            Duration::from_secs(2)
        );
    }

    #[test]
    fn clamps_an_absurd_retry_after() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::RETRY_AFTER,
            "999999999"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            retry_after(&headers, UNIX_EPOCH, DEFAULT_RETRY_AFTER),
            MAX_RETRY_AFTER
        );
    }

    #[test]
    fn clamps_an_absurd_future_date() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::RETRY_AFTER,
            "Fri, 01 Jan 2100 00:00:00 GMT"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            retry_after(&headers, UNIX_EPOCH, DEFAULT_RETRY_AFTER),
            MAX_RETRY_AFTER
        );
    }

    #[test]
    fn honours_a_multi_hour_backoff() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::RETRY_AFTER,
            "21600"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            retry_after(&headers, UNIX_EPOCH, DEFAULT_RETRY_AFTER),
            Duration::from_secs(21600)
        );
    }

    #[test]
    fn honours_a_delay_under_the_cap() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::RETRY_AFTER,
            "5".parse()
                .unwrap(),
        );
        assert_eq!(
            retry_after(&headers, UNIX_EPOCH, DEFAULT_RETRY_AFTER),
            Duration::from_secs(5)
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod shared_rate_limit {
        use super::super::SharedRateLimit;
        use std::time::Duration;
        use tokio::time::Instant;

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_fresh_limit_never_blocks() {
            let limit = SharedRateLimit::new();
            let started = Instant::now();
            limit
                .wait_for_cooldown(Duration::MAX)
                .await
                .unwrap();
            assert_eq!(
                started.elapsed(),
                Duration::ZERO,
                "a limit that was never tripped must not delay callers"
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn remaining_cooldown_is_zero_when_not_blocked() {
            let limit = SharedRateLimit::new();
            assert_eq!(
                limit
                    .remaining_cooldown()
                    .await,
                Duration::ZERO
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn remaining_cooldown_reports_time_left_without_waiting() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(5))
                .await;

            let started = Instant::now();
            let remaining = limit
                .remaining_cooldown()
                .await;
            assert_eq!(
                started.elapsed(),
                Duration::ZERO,
                "remaining_cooldown must never sleep, unlike wait_for_cooldown"
            );
            assert_eq!(remaining, Duration::from_secs(5));
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn tripping_the_limit_parks_a_later_waiter_for_the_delay() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(5))
                .await;

            let started = Instant::now();
            limit
                .wait_for_cooldown(Duration::MAX)
                .await
                .unwrap();
            assert_eq!(started.elapsed(), Duration::from_secs(5));
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_shorter_block_never_shrinks_an_existing_longer_cooldown() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(10))
                .await;
            // A second, shorter delay (e.g. from a request that raced the first
            // 429 and got a smaller Retry-After) must not pull the deadline in —
            // whoever asked for the longest cooldown wins.
            limit
                .block_for(Duration::from_secs(1))
                .await;

            let started = Instant::now();
            limit
                .wait_for_cooldown(Duration::MAX)
                .await
                .unwrap();
            assert_eq!(
                started.elapsed(),
                Duration::from_secs(10),
                "a shorter subsequent block must not shrink the cooldown"
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_longer_block_extends_an_existing_shorter_cooldown() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(1))
                .await;
            limit
                .block_for(Duration::from_secs(10))
                .await;

            let started = Instant::now();
            limit
                .wait_for_cooldown(Duration::MAX)
                .await
                .unwrap();
            assert_eq!(started.elapsed(), Duration::from_secs(10));
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn concurrent_waiters_all_release_at_the_same_deadline() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(3))
                .await;

            let started = Instant::now();
            let (a, b) = tokio::join!(
                limit.wait_for_cooldown(Duration::MAX),
                limit.wait_for_cooldown(Duration::MAX)
            );
            a.unwrap();
            b.unwrap();
            assert_eq!(started.elapsed(), Duration::from_secs(3));
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_cooldown_within_max_wait_is_waited_out() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(2))
                .await;

            let started = Instant::now();
            limit
                .wait_for_cooldown(Duration::from_secs(5))
                .await
                .unwrap();
            assert_eq!(started.elapsed(), Duration::from_secs(2));
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_cooldown_beyond_max_wait_is_refused_at_once_without_extending_it() {
            let limit = SharedRateLimit::new();
            limit
                .block_for(Duration::from_secs(60))
                .await;

            let started = Instant::now();
            let refused = limit
                .wait_for_cooldown(Duration::from_secs(5))
                .await
                .unwrap_err();
            assert_eq!(started.elapsed(), Duration::ZERO);
            assert_eq!(refused.remaining, Duration::from_secs(60));
            assert_eq!(
                limit
                    .remaining_cooldown()
                    .await,
                Duration::from_secs(60)
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod middleware {
        use crate::{ClientError, Endpoint, RestClient};
        use std::time::{Duration, Instant};

        #[derive(Clone)]
        struct Probe(&'static str);

        impl Endpoint for Probe {
            type Output = Vec<String>;

            fn path(&self) -> String {
                self.0
                    .to_string()
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn a_429_parks_the_caller_for_the_requested_delay() {
            let server = httpmock::MockServer::start();
            let limited = server.mock(|when, then| {
                when.path("/limited");
                then.status(429)
                    .header("Retry-After", "1");
            });
            let client = RestClient::new(&server.base_url()).unwrap();

            let started = Instant::now();
            let error = client
                .execute(Probe("/limited"))
                .await
                .unwrap_err();
            let elapsed = started.elapsed();

            match error {
                ClientError::RateLimited { retry_after_secs } => {
                    assert_eq!(retry_after_secs, 1)
                }
                other => panic!("expected rate-limit error, got {other}"),
            }
            assert!(
                elapsed >= Duration::from_millis(750),
                "caller should have been parked for the requested delay, took {elapsed:?}"
            );
            assert_eq!(limited.hits(), 1);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn a_successful_response_is_never_delayed() {
            let server = httpmock::MockServer::start();
            let ok = server.mock(|when, then| {
                when.path("/ok");
                then.status(200)
                    .json_body(serde_json::json!([]));
            });
            let client = RestClient::new(&server.base_url()).unwrap();

            let started = Instant::now();
            client
                .execute(Probe("/ok"))
                .await
                .unwrap();

            assert!(started.elapsed() < Duration::from_millis(500));
            assert_eq!(ok.hits(), 1);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn clients_for_different_origins_do_not_block_each_other() {
            let limited_server = httpmock::MockServer::start();
            limited_server.mock(|when, then| {
                when.path("/limited");
                then.status(429)
                    .header("Retry-After", "30");
            });
            let other_server = httpmock::MockServer::start();
            let other = other_server.mock(|when, then| {
                when.path("/ok");
                then.status(200)
                    .json_body(serde_json::json!([]));
            });

            let limited_client = RestClient::new(&limited_server.base_url()).unwrap();
            let other_client = RestClient::new(&other_server.base_url()).unwrap();

            let limited = tokio::spawn(async move {
                let _ = limited_client
                    .execute(Probe("/limited"))
                    .await;
            });

            let started = Instant::now();
            other_client
                .execute(Probe("/ok"))
                .await
                .unwrap();
            assert!(
                started.elapsed() < Duration::from_millis(500),
                "unrelated origin was blocked by another host's 429"
            );
            assert_eq!(other.hits(), 1);
            limited.abort();
        }

        #[tokio::test(flavor = "current_thread")]
        async fn clients_sharing_a_rate_limit_block_each_other_after_a_429() {
            use crate::rate_limit::SharedRateLimit;

            let limited_server = httpmock::MockServer::start();
            limited_server.mock(|when, then| {
                when.path("/limited");
                then.status(429)
                    .header("Retry-After", "1");
            });
            let other_server = httpmock::MockServer::start();
            let other = other_server.mock(|when, then| {
                when.path("/ok");
                then.status(200)
                    .json_body(serde_json::json!([]));
            });

            let shared = SharedRateLimit::new();
            let limited_client = RestClient::new(&limited_server.base_url())
                .unwrap()
                .with_shared_rate_limit(shared.clone());
            // Same shared limit, entirely different underlying host — mirrors
            // two independently-constructed clients for one logical provider
            // (see `tmdb_rate_limit`), not two requests to the same server.
            let other_client = RestClient::new(&other_server.base_url())
                .unwrap()
                .with_shared_rate_limit(shared.clone());

            // Spawned so it races the second request instead of fully waiting
            // out its own cooldown before this task even starts the other
            // call — otherwise the shared deadline would already be in the
            // past by the time `other_client` checked it.
            let limited = tokio::spawn(async move {
                let _ = limited_client
                    .execute(Probe("/limited"))
                    .await;
            });
            // Wait for the cooldown to actually be recorded rather than
            // guessing how long that takes — a fixed sleep here would be a
            // timing race against however long the spawned task takes to
            // reach that point under CI/executor load.
            while shared
                .remaining_cooldown()
                .await
                == Duration::ZERO
            {
                tokio::task::yield_now().await;
            }

            let started = Instant::now();
            other_client
                .execute(Probe("/ok"))
                .await
                .unwrap();
            assert!(
                started.elapsed() >= Duration::from_millis(700),
                "a client sharing the tripped limit should have been parked too, took {:?}",
                started.elapsed()
            );
            assert_eq!(other.hits(), 1);
            limited
                .await
                .unwrap();
        }
    }

    /// The real retry and rate-limit middlewares, stacked the way `build_mw`
    /// stacks them, in front of a canned 429 instead of a socket, so paused
    /// time makes every wait exact.
    #[cfg(not(target_arch = "wasm32"))]
    mod in_request_wait {
        use super::super::{
            DEFAULT_RETRY_AFTER, MAX_IN_REQUEST_WAIT, RetryAfterMiddleware,
            SharedRateLimit, interactive,
        };
        use reqwest::ResponseBuilderExt;
        use reqwest_middleware::ClientWithMiddleware;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use tokio::time::Instant;

        const CREDENTIALED_URL: &str =
            "http://addon.example/SECRET-TOKEN/catalog/movie/search=x.json";

        struct Canned429 {
            retry_after: Option<&'static str>,
            hits: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl reqwest_middleware::Middleware for Canned429 {
            async fn handle(
                &self,
                req: reqwest::Request,
                _: &mut http::Extensions,
                _: reqwest_middleware::Next<'_>,
            ) -> reqwest_middleware::Result<reqwest::Response> {
                self.hits
                    .fetch_add(1, Ordering::SeqCst);
                let mut response = http::Response::builder()
                    .status(http::StatusCode::TOO_MANY_REQUESTS)
                    .url(
                        req.url()
                            .clone(),
                    );
                if let Some(value) = self.retry_after {
                    response = response.header(http::header::RETRY_AFTER, value);
                }
                Ok(response
                    .body("")
                    .unwrap()
                    .into())
            }
        }

        fn client(
            retry_after: Option<&'static str>,
            shared: &SharedRateLimit,
        ) -> (ClientWithMiddleware, Arc<AtomicUsize>) {
            let hits = Arc::new(AtomicUsize::new(0));
            let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
                .with(
                    reqwest_retry::RetryTransientMiddleware::new_with_policy_and_strategy(
                        crate::ExponentialBackoff::builder().build_with_max_retries(3),
                        crate::NoTimeoutRetryStrategy,
                    ),
                )
                .with(RetryAfterMiddleware {
                    default_retry_after: DEFAULT_RETRY_AFTER,
                    shared_rate_limit: Some(shared.clone()),
                })
                .with(Canned429 {
                    retry_after,
                    hits: hits.clone(),
                })
                .build();
            (client, hits)
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_429_without_retry_after_returns_after_the_capped_wait_without_a_retry()
         {
            let shared = SharedRateLimit::new();
            let (client, hits) = client(None, &shared);

            let started = Instant::now();
            let response = interactive(
                client
                    .get(CREDENTIALED_URL)
                    .send(),
            )
            .await
            .unwrap();

            assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(
                hits.load(Ordering::SeqCst),
                1,
                "each retry would hit the same limit and wait again"
            );
            assert_eq!(started.elapsed(), MAX_IN_REQUEST_WAIT);
            // The rest of the default window still gates later requests.
            assert_eq!(
                shared
                    .remaining_cooldown()
                    .await,
                DEFAULT_RETRY_AFTER - MAX_IN_REQUEST_WAIT
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_request_during_a_long_cooldown_is_refused_without_sending() {
            let shared = SharedRateLimit::new();
            let (client, hits) = client(None, &shared);
            interactive(
                client
                    .get(CREDENTIALED_URL)
                    .send(),
            )
            .await
            .unwrap();
            assert_eq!(hits.load(Ordering::SeqCst), 1);
            let cooldown = shared
                .remaining_cooldown()
                .await;

            let started = Instant::now();
            let error = interactive(
                client
                    .get(CREDENTIALED_URL)
                    .send(),
            )
            .await
            .unwrap_err();

            assert_eq!(started.elapsed(), std::time::Duration::ZERO);
            assert_eq!(hits.load(Ordering::SeqCst), 1, "nothing may be sent");
            let error = crate::middleware_error(error);
            assert!(
                matches!(
                    error,
                    crate::ClientError::RateLimited {
                        retry_after_secs: 55
                    }
                ),
                "{error:?}"
            );
            assert_eq!(
                shared
                    .remaining_cooldown()
                    .await,
                cooldown,
                "a refused request must not re-arm the cooldown"
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn a_429_whose_wait_fits_under_the_cap_is_waited_in_full_and_retried() {
            let shared = SharedRateLimit::new();
            let (client, hits) = client(Some("2"), &shared);

            let started = Instant::now();
            let response = interactive(
                client
                    .get(CREDENTIALED_URL)
                    .send(),
            )
            .await
            .unwrap();

            assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(
                hits.load(Ordering::SeqCst),
                4,
                "a wait that was not cut short leaves the retry policy in charge"
            );
            assert!(
                started.elapsed() >= 4 * std::time::Duration::from_secs(2),
                "every attempt should have waited its full Retry-After, took {:?}",
                started.elapsed()
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn outside_interactive_a_429_waits_the_full_default_and_is_retried() {
            let shared = SharedRateLimit::new();
            let (client, hits) = client(None, &shared);

            let started = Instant::now();
            let response = client
                .get(CREDENTIALED_URL)
                .send()
                .await
                .unwrap();

            assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(
                hits.load(Ordering::SeqCst),
                4,
                "background work keeps retrying, e.g. a catalog page"
            );
            assert!(
                started.elapsed() >= 4 * DEFAULT_RETRY_AFTER,
                "every attempt should have waited the full default, took {:?}",
                started.elapsed()
            );
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn outside_interactive_a_long_cooldown_is_waited_not_refused() {
            let shared = SharedRateLimit::new();
            shared
                .block_for(DEFAULT_RETRY_AFTER)
                .await;
            // `0`: the upstream asks for no wait, so only the existing
            // cooldown delays the request.
            let (client, hits) = client(Some("0"), &shared);

            let started = Instant::now();
            client
                .get(CREDENTIALED_URL)
                .send()
                .await
                .unwrap();

            assert!(started.elapsed() >= DEFAULT_RETRY_AFTER);
            assert_eq!(hits.load(Ordering::SeqCst), 4);
        }
    }
}

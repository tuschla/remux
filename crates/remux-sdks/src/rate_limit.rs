use http::{HeaderMap, header};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use tokio::{sync::Mutex, time::Instant};

pub(crate) const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(60);

const MAX_RETRY_AFTER: Duration = Duration::from_secs(48 * 60 * 60);

/// Longest a request inside [`interactive`] waits on a 429 or shared cooldown.
#[cfg(not(target_arch = "wasm32"))]
const MAX_IN_REQUEST_WAIT: Duration = Duration::from_secs(5);

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    static INTERACTIVE: ();
}

/// Runs `fut` as a request someone is waiting on, such as a search: inside
/// it, a 429 or shared cooldown longer than [`MAX_IN_REQUEST_WAIT`] fails the
/// request at once instead of being waited out. Tasks spawned from `fut` do
/// not inherit the scope.
#[cfg(not(target_arch = "wasm32"))]
pub async fn interactive<F: Future>(fut: F) -> F::Output {
    INTERACTIVE
        .scope((), fut)
        .await
}

/// Marks a 429 returned without waiting out its cooldown, so the retry
/// middleware doesn't retry it into the same limit.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
pub(crate) struct RateLimitedResponse;

/// Shared 429 cooldown for one provider.
///
/// Give clones of one value to every [`RestClient`](crate::RestClient) that
/// talks to the same upstream. After a 429, it parks all later requests until
/// the shared cooldown has elapsed. It intentionally does not limit request
/// concurrency; callers control that themselves.
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

    /// Waits for the cooldown to clear; returns `false` at once instead if it
    /// lasts longer than `max_wait`.
    async fn wait_for_cooldown(&self, max_wait: Option<Duration>) -> bool {
        let deadline = max_wait.map(|max_wait| Instant::now() + max_wait);
        loop {
            let blocked_until = *self
                .blocked_until
                .lock()
                .await;
            let now = Instant::now();
            if blocked_until <= now {
                return true;
            }
            if deadline.is_some_and(|deadline| blocked_until > deadline) {
                return false;
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
        let max_wait = INTERACTIVE
            .try_with(|_| MAX_IN_REQUEST_WAIT)
            .ok();
        if let Some(limit) = &self.shared_rate_limit
            && !limit
                .wait_for_cooldown(max_wait)
                .await
        {
            // Refused unsent, as a final 429 so callers still get `RateLimited`.
            return Ok(http::Response::builder()
                .status(http::StatusCode::TOO_MANY_REQUESTS)
                .extension(RateLimitedResponse)
                .body("")
                .expect("a bare 429 is a valid response")
                .into());
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
        // Host only: addon URLs carry credentials in the path and query.
        tracing::warn!(
            host = response
                .url()
                .host_str()
                .unwrap_or_default(),
            retry_after_secs = delay.as_secs(),
            shared = self
                .shared_rate_limit
                .is_some(),
            "upstream returned 429"
        );
        let waited = match &self.shared_rate_limit {
            Some(limit) => {
                limit
                    .block_for(delay)
                    .await;
                // Also waits out a longer cooldown another request recorded.
                limit
                    .wait_for_cooldown(max_wait)
                    .await
            }
            None if max_wait.is_some_and(|max_wait| delay > max_wait) => false,
            None => {
                tokio::time::sleep(delay).await;
                true
            }
        };
        if !waited {
            response
                .extensions_mut()
                .insert(RateLimitedResponse);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                .wait_for_cooldown(None)
                .await;
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
                .wait_for_cooldown(None)
                .await;
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
                .wait_for_cooldown(None)
                .await;
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
                .wait_for_cooldown(None)
                .await;
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
                limit.wait_for_cooldown(None),
                limit.wait_for_cooldown(None)
            );
            assert!(a && b);
            assert_eq!(started.elapsed(), Duration::from_secs(3));
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

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn an_interactive_request_during_a_long_cooldown_is_refused_unsent() {
            use crate::{SharedRateLimit, interactive};

            let server = httpmock::MockServer::start();
            let upstream = server.mock(|when, then| {
                when.path("/search");
                then.status(200)
                    .json_body(serde_json::json!([]));
            });
            let shared = SharedRateLimit::new();
            shared
                .block_for(Duration::from_secs(60))
                .await;
            let client = RestClient::new(&server.base_url())
                .unwrap()
                .with_shared_rate_limit(shared.clone());

            let started = tokio::time::Instant::now();
            let error = interactive(client.execute(Probe("/search")))
                .await
                .unwrap_err();

            assert!(matches!(error, ClientError::RateLimited { .. }), "{error}");
            assert_eq!(started.elapsed(), Duration::ZERO);
            assert_eq!(upstream.hits(), 0);
            // A refusal must not re-arm the cooldown.
            assert_eq!(
                shared
                    .remaining_cooldown()
                    .await,
                Duration::from_secs(60)
            );
        }

        #[tokio::test(flavor = "current_thread")]
        async fn an_interactive_request_gets_a_long_429_at_once_and_unretried() {
            use crate::{ExponentialBackoff, SharedRateLimit, interactive};

            let server = httpmock::MockServer::start();
            let limited = server.mock(|when, then| {
                when.path("/search");
                then.status(429);
            });
            let shared = SharedRateLimit::new();
            let client = RestClient::new(&server.base_url())
                .unwrap()
                .with_retry(
                    ExponentialBackoff::builder()
                        .jitter(reqwest_retry::Jitter::None)
                        .build_with_max_retries(1),
                )
                .with_shared_rate_limit(shared.clone());

            let started = Instant::now();
            let error = tokio::time::timeout(
                Duration::from_secs(5),
                interactive(client.execute(Probe("/search"))),
            )
            .await
            .expect("must not wait out the 60 s cooldown")
            .unwrap_err();

            assert!(matches!(error, ClientError::RateLimited { .. }), "{error}");
            // A retry would first back off for 1 s.
            assert!(started.elapsed() < Duration::from_millis(500));
            assert_eq!(limited.hits(), 1);
            // Later requests still wait out the full cooldown.
            assert!(
                shared
                    .remaining_cooldown()
                    .await
                    > Duration::from_secs(55)
            );
        }
    }
}

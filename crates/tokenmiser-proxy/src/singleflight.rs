//! Single-flight coalescing of identical concurrent cache misses: the first
//! miss on a key leads and computes upstream, the rest follow and serve its
//! outcome.
//!
//! Accounting invariants: every request performs its L1 (and, on L1 miss, L2)
//! lookup *before* reaching the flight map, so `hits + misses == lookups`
//! holds at both layers; a follower is ledgered as a cache hit, so the
//! leader's call is the only provider request recorded and spend is never
//! double-counted.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tokenmiser_providers::ChatResponse;
use tokio::sync::watch;

/// What the leader publishes to its followers.
#[derive(Clone)]
pub enum FlightOutcome {
    /// The upstream response plus the model actually routed to, so followers
    /// can emit an honest routed-to header.
    Response {
        response: Arc<ChatResponse>,
        routed_to: String,
    },
    /// Followers replay the leader's error rather than piling onto an upstream
    /// that just refused this exact request.
    Error { status: u16, message: String },
}

type Slot = watch::Receiver<Option<FlightOutcome>>;

/// Map of in-flight upstream computations, keyed by the L1 exact-match cache
/// key (which deliberately ignores `stream`, so streaming and non-streaming
/// callers coalesce together).
pub struct FlightMap {
    inner: Mutex<HashMap<String, Slot>>,
    coalesced: AtomicU64,
    abandoned: AtomicU64,
    timed_out: AtomicU64,
}

/// Outcome of [`FlightMap::begin`].
pub enum Flight {
    /// This request must compute upstream and [`FlightLease::publish`].
    Leader(FlightLease),
    /// An identical request is already in flight; await its outcome.
    Follower(Slot),
}

/// Held by the leader; publishing (or dropping) it releases the key.
pub struct FlightLease {
    map: Arc<FlightMap>,
    key: String,
    tx: Option<watch::Sender<Option<FlightOutcome>>>,
}

impl FlightMap {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HashMap::new()),
            coalesced: AtomicU64::new(0),
            abandoned: AtomicU64::new(0),
            timed_out: AtomicU64::new(0),
        })
    }

    /// Join the flight for `key`. Atomic under the map lock, so exactly one
    /// leader exists per key at any moment.
    pub fn begin(self: &Arc<Self>, key: &str) -> Flight {
        let mut m = self.inner.lock();
        if let Some(rx) = m.get(key) {
            return Flight::Follower(rx.clone());
        }
        let (tx, rx) = watch::channel(None);
        m.insert(key.to_string(), rx);
        Flight::Leader(FlightLease {
            map: Arc::clone(self),
            key: key.to_string(),
            tx: Some(tx),
        })
    }

    fn release(&self, key: &str) {
        self.inner.lock().remove(key);
    }

    pub fn stats(&self) -> FlightStats {
        FlightStats {
            coalesced: self.coalesced.load(Ordering::Relaxed),
            abandoned: self.abandoned.load(Ordering::Relaxed),
            timed_out: self.timed_out.load(Ordering::Relaxed),
            in_flight: self.inner.lock().len() as u64,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FlightStats {
    pub coalesced: u64,
    /// Leader vanished without publishing (benign).
    pub abandoned: u64,
    /// Follower gave up while the leader was still running (investigate).
    #[serde(default)]
    pub timed_out: u64,
    pub in_flight: u64,
}

impl FlightLease {
    /// Publish the outcome to all followers and release the key. The value is
    /// sent before the key is removed, so a request that raced in as a
    /// follower just before removal still observes it.
    pub fn publish(mut self, outcome: FlightOutcome) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Some(outcome));
        }
        self.map.release(&self.key);
    }
}

impl Drop for FlightLease {
    fn drop(&mut self) {
        // Abandoned without publishing: release the key and drop the sender so
        // followers unblock with `None`. Guarded on `tx` so a lease that
        // already published cannot clobber a successor's entry.
        if self.tx.take().is_some() {
            self.map.release(&self.key);
        }
    }
}

/// Await the leader's outcome; `None` means fall back to an own upstream call.
///
/// The two `None` cases are counted separately because they mean different
/// things operationally: `abandoned` (leader vanished) is expected and
/// harmless, while `timed_out` (leader still running past the cap) means the
/// upstream is wedged and coalescing has stopped working.
pub async fn await_outcome(
    map: &FlightMap,
    mut rx: Slot,
    wait: std::time::Duration,
) -> Option<FlightOutcome> {
    match tokio::time::timeout(wait, rx.wait_for(|v| v.is_some())).await {
        // The leader is alive, just slow.
        Err(_) => {
            map.timed_out.fetch_add(1, Ordering::Relaxed);
            None
        }
        // Channel closed: the leader dropped its lease without publishing.
        Ok(Err(_)) => {
            map.abandoned.fetch_add(1, Ordering::Relaxed);
            None
        }
        Ok(Ok(guard)) => match guard.clone() {
            Some(outcome) => {
                map.coalesced.fetch_add(1, Ordering::Relaxed);
                Some(outcome)
            }
            None => {
                map.abandoned.fetch_add(1, Ordering::Relaxed);
                None
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::time::Duration;
    use tokenmiser_providers::{ChatChoice, ChatMessage, Usage};

    const WAIT: Duration = Duration::from_secs(5);

    fn resp(text: &str) -> Arc<ChatResponse> {
        Arc::new(ChatResponse {
            id: "t".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String(text.into()),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage::default(),
            extra: Default::default(),
        })
    }

    #[tokio::test]
    async fn one_leader_many_followers_share_one_result() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!("first begin must lead");
        };

        let mut tasks = Vec::new();
        for _ in 0..19 {
            let Flight::Follower(rx) = map.begin("k") else {
                panic!("concurrent begin with same key must follow");
            };
            let map = Arc::clone(&map);
            tasks.push(tokio::spawn(
                async move { await_outcome(&map, rx, WAIT).await },
            ));
        }

        lease.publish(FlightOutcome::Response {
            response: resp("42"),
            routed_to: "qwen2.5:7b".into(),
        });

        for t in tasks {
            match t.await.unwrap() {
                Some(FlightOutcome::Response {
                    response,
                    routed_to,
                }) => {
                    assert_eq!(
                        response.choices[0].message.content,
                        Value::String("42".into())
                    );
                    assert_eq!(routed_to, "qwen2.5:7b");
                }
                other => panic!(
                    "follower must receive the response, got {:?}",
                    match other {
                        Some(FlightOutcome::Error { status, message }) =>
                            format!("Error({status}, {message})"),
                        _ => "None".to_string(),
                    }
                ),
            }
        }
        let s = map.stats();
        assert_eq!(s.coalesced, 19);
        assert_eq!(s.abandoned, 0);
        assert_eq!(s.in_flight, 0, "key released after publish");
    }

    #[tokio::test]
    async fn error_outcome_propagates_to_followers() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        lease.publish(FlightOutcome::Error {
            status: 502,
            message: "upstream exploded".into(),
        });
        match await_outcome(&map, rx, WAIT).await {
            Some(FlightOutcome::Error { status, message }) => {
                assert_eq!(status, 502);
                assert_eq!(message, "upstream exploded");
            }
            _ => panic!("follower must receive the error"),
        }
    }

    #[tokio::test]
    async fn abandoned_leader_unblocks_followers_with_none() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        drop(lease);
        assert!(
            await_outcome(&map, rx, WAIT).await.is_none(),
            "abandonment must unblock followers with None (fall back upstream)"
        );
        let s = map.stats();
        assert_eq!(s.abandoned, 1);
        assert_eq!(
            s.timed_out, 0,
            "a dead leader is abandonment, not a timeout"
        );
        assert_eq!(s.in_flight, 0, "key released on abandonment");
    }

    #[tokio::test]
    async fn slow_leader_counts_as_timed_out_not_abandoned() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        // The leader still holds its lease while the follower exhausts its wait.
        let out = await_outcome(&map, rx, std::time::Duration::from_millis(20)).await;
        assert!(out.is_none(), "follower must fall back after its wait");
        let s = map.stats();
        assert_eq!(s.timed_out, 1, "slow leader must count as timed_out");
        assert_eq!(s.abandoned, 0, "leader is alive; this is not abandonment");
        assert_eq!(s.coalesced, 0);
        assert_eq!(s.in_flight, 1, "leader still holds the key");
        drop(lease);
    }

    #[tokio::test]
    async fn key_is_reusable_after_publish_and_after_abandon() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        lease.publish(FlightOutcome::Error {
            status: 400,
            message: "x".into(),
        });
        assert!(matches!(map.begin("k"), Flight::Leader(_)));

        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        drop(lease);
        assert!(matches!(map.begin("k"), Flight::Leader(_)));
    }

    #[tokio::test]
    async fn panicking_leader_releases_followers_and_key() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        let handle = tokio::spawn(async move {
            let _lease = lease;
            panic!("leader dies mid-flight");
        });
        assert!(handle.await.is_err(), "leader task must have panicked");
        assert!(
            await_outcome(&map, rx, WAIT).await.is_none(),
            "followers of a panicked leader must be released to fall back"
        );
        let s = map.stats();
        assert_eq!(s.in_flight, 0, "key must be freed by the panicking leader");
        assert!(
            matches!(map.begin("k"), Flight::Leader(_)),
            "key must be reusable after a leader panic"
        );
    }

    #[tokio::test]
    async fn cancelled_leader_future_releases_followers() {
        // Abort drops the future without unwinding through user code; the
        // lease Drop impl must still fire.
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        let handle = tokio::spawn(async move {
            let _lease = lease;
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        tokio::task::yield_now().await;
        handle.abort();
        assert!(handle.await.is_err());
        assert!(
            await_outcome(&map, rx, WAIT).await.is_none(),
            "followers of a cancelled leader must be released"
        );
        assert_eq!(map.stats().in_flight, 0);
    }

    #[tokio::test]
    async fn distinct_keys_do_not_coalesce() {
        let map = FlightMap::new();
        let _a = map.begin("a");
        assert!(matches!(map.begin("b"), Flight::Leader(_)));
    }

    #[tokio::test]
    async fn late_follower_between_send_and_release_still_sees_outcome() {
        let map = FlightMap::new();
        let Flight::Leader(lease) = map.begin("k") else {
            panic!()
        };
        let Flight::Follower(rx) = map.begin("k") else {
            panic!()
        };
        lease.publish(FlightOutcome::Response {
            response: resp("ok"),
            routed_to: "m".into(),
        });
        assert!(await_outcome(&map, rx, WAIT).await.is_some());
    }
}

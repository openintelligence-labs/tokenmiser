//! Per-connection outbound backpressure (architecture §14.4).
//!
//! A slow client streaming a frontier-model response can balloon outbound
//! buffers and crash the proxy. Pingora handles inbound flow control; the
//! outbound path needs explicit accounting because we're proxying.
//!
//! Token-bucket: each connection starts with `burst_bytes` and refills at
//! `refill_bytes_per_sec`. `take()` returns the number of bytes the caller
//! may write right now; if the bucket is empty we sleep until enough has
//! accrued. Cheap and correct.

use std::time::{Duration, Instant};

pub struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    available: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Default per-architecture-§14.4: 1MB burst, 256KB/s sustained.
    pub fn default_streaming() -> Self {
        Self::new(1024.0 * 1024.0, 256.0 * 1024.0)
    }

    pub fn new(burst_bytes: f64, refill_per_sec: f64) -> Self {
        Self {
            capacity: burst_bytes,
            refill_per_sec,
            available: burst_bytes,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.available = (self.available + elapsed * self.refill_per_sec).min(self.capacity);
            self.last_refill = now;
        }
    }

    /// Async: wait until at least `n` bytes are available, then debit and
    /// return. Yields to the runtime in small steps so a stalled bucket
    /// doesn't busy-wait.
    pub async fn acquire(&mut self, n: usize) {
        let needed = n as f64;
        loop {
            self.refill();
            if self.available >= needed {
                self.available -= needed;
                return;
            }
            let deficit = needed - self.available;
            let wait_secs = deficit / self.refill_per_sec;
            // Clamp so we yield often enough to notice client disconnects.
            let dur = Duration::from_secs_f64(wait_secs.min(0.05));
            tokio::time::sleep(dur).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn initial_burst_is_immediate() {
        let mut tb = TokenBucket::new(1000.0, 100.0);
        let start = Instant::now();
        tb.acquire(500).await;
        assert!(start.elapsed() < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn refill_throttles_subsequent_writes() {
        let mut tb = TokenBucket::new(100.0, 100.0);
        tb.acquire(100).await; // drain
        let start = Instant::now();
        tb.acquire(50).await; // need 0.5s to refill
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(400),
            "expected ~500ms refill wait, got {:?}",
            elapsed
        );
    }
}

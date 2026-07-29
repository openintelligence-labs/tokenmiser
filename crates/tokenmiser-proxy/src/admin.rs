//! Admin endpoints (`/stats`, `/healthz`, future dashboard). v0.1 serves
//! these through the same Pingora ingress; v1.0 will fan them out to a
//! separate Axum service on the admin port so we can route dashboard
//! traffic differently from LLM traffic.

use std::sync::Arc;

use crate::AppState;

/// Lightweight wrapper around `AppState` for the admin server in v1.0.
#[derive(Clone)]
pub struct AdminState {
    pub app: Arc<AppState>,
}

impl AdminState {
    pub fn new(app: Arc<AppState>) -> Self {
        Self { app }
    }
}

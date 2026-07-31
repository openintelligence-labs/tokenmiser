//! Admin endpoints (`/stats`, `/healthz`, dashboard), served through the same
//! Pingora ingress as LLM traffic.

use std::sync::Arc;

use crate::AppState;

/// Wrapper around `AppState` for the admin server.
#[derive(Clone)]
pub struct AdminState {
    pub app: Arc<AppState>,
}

impl AdminState {
    pub fn new(app: Arc<AppState>) -> Self {
        Self { app }
    }
}

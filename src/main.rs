use axum::{routing::get, Router};
use std::net::SocketAddr;
use tokenmiser::CostLedger;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/healthz", get(health))
        .route("/stats", get(stats));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8443));
    println!("tokenmiser listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn stats() -> String {
    let ledger = CostLedger::default();
    serde_json::to_string_pretty(&ledger).unwrap()
}

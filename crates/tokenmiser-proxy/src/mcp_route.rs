//! HTTP routes for the MCP budget gateway (architecture §11.7).
//!
//! Endpoints:
//! - `POST /v1/mcp/tools/call` — JSON-RPC-shaped tool invocation. We
//!   enforce the per-(agent, tool) budget cap before forwarding (v0.95.1
//!   will add actual upstream forwarding once we ship an MCP client; v0.95
//!   exposes the budget-check + post-call cost reporting surface so agent
//!   frameworks can integrate today).
//! - `POST /v1/mcp/budgets` — set per-(agent, tool) caps at runtime.
//! - `GET  /v1/mcp/budgets` — snapshot per-(agent, tool) spend.
//!
//! Wire shape is intentionally JSON-RPC 2.0 (`{"jsonrpc":"2.0","method":"tools/call",...}`)
//! so MCP clients can point at this endpoint directly. We ignore the
//! `params.arguments` field for budget purposes — the cap is on the
//! identity (agent, tool), not the argument content.

use serde::{Deserialize, Serialize};
use tokenmiser_mcp::ToolBudget;

/// JSON-RPC envelope for inbound tool-call requests.
#[derive(Debug, Deserialize)]
pub struct McpToolsCallRequest {
    #[serde(default)]
    pub jsonrpc: Option<String>,
    pub id: Option<serde_json::Value>,
    pub method: Option<String>,
    pub params: McpToolsCallParams,
}

#[derive(Debug, Deserialize)]
pub struct McpToolsCallParams {
    /// Tool name, per MCP spec.
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// `tokenmiser`-specific extension: which agent is calling. Defaults
    /// to `"default"` for clients that don't set it.
    #[serde(default)]
    pub agent: Option<String>,
    /// Optional: estimated cost in USD this call will incur. If supplied,
    /// we pre-check against the cap. v0.95.2 will integrate token-count
    /// estimation here.
    #[serde(default)]
    pub estimated_cost_usd: Option<f64>,
    /// Optional: actual cost (filled in on the second call by clients that
    /// want post-hoc reporting).
    #[serde(default)]
    pub actual_cost_usd: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct McpToolsCallResponse {
    pub jsonrpc: &'static str,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<McpResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

#[derive(Debug, Serialize)]
pub struct McpResult {
    pub allowed: bool,
    pub agent: String,
    pub tool: String,
    pub spent_usd: f64,
    pub calls: u64,
}

#[derive(Debug, Serialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

/// JSON-RPC error codes we use. The MCP spec reserves -32000..-32099 for
/// server-implementation-defined errors.
pub const ERR_BUDGET_EXCEEDED: i32 = -32001;
pub const ERR_BAD_REQUEST: i32 = -32600;

#[derive(Debug, Deserialize)]
pub struct SetBudgetRequest {
    pub agent: String,
    pub tool: String,
    pub cap_usd: f64,
    #[serde(default = "default_window")]
    pub window_secs: u64,
}

fn default_window() -> u64 {
    3600
}

impl SetBudgetRequest {
    pub fn into_budget(self) -> (String, String, ToolBudget) {
        (
            self.agent,
            self.tool,
            ToolBudget {
                cap_usd: self.cap_usd,
                window_secs: self.window_secs,
            },
        )
    }
}

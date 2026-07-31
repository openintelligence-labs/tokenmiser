//! HTTP routes for the MCP budget gateway.
//!
//! The wire shape is JSON-RPC 2.0 so MCP clients can point at these endpoints
//! directly. `params.arguments` is ignored: the cap is on the (agent, tool)
//! identity, not the argument content.

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
    /// TokenMiser extension naming the calling agent.
    #[serde(default)]
    pub agent: Option<String>,
    /// Estimated USD cost, pre-checked against the cap when supplied.
    #[serde(default)]
    pub estimated_cost_usd: Option<f64>,
    /// Actual USD cost, for clients doing post-hoc reporting.
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

/// The MCP spec reserves -32000..-32099 for server-defined errors.
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

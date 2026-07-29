//! MCP gateway with per-tool budget caps (architecture §11.7).
//!
//! What's shipping in v0.95:
//! - **Budget tracker**: per-(agent, tool) spend ledger with hard caps.
//! - **HTTP endpoint** (wired into the proxy admin server in v1.0):
//!   `POST /v1/mcp/tools/call` accepts a JSON-RPC `tools/call` payload,
//!   checks the per-tool budget, forwards to the wrapped MCP server, and
//!   records the actual cost from the response.
//!
//! What's NOT shipping in v0.95 (deferred to v0.95.1):
//! - Full MCP server spec compliance (initialize / capabilities / etc).
//!   The gateway speaks the tool-call subset only.
//! - Bidirectional stdio MCP — HTTP only.
//!
//! The architectural moat is the **per-tool budget caps**. Every agent
//! framework wants this; nobody else ships it for OSS MCP yet.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::warn;

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error(
        "budget exceeded for agent={agent} tool={tool}: spent ${spent_usd:.4} of ${cap_usd:.4}"
    )]
    Exceeded {
        agent: String,
        tool: String,
        spent_usd: f64,
        cap_usd: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolBudget {
    /// Maximum USD this (agent, tool) tuple may spend in this window.
    pub cap_usd: f64,
    /// Window length in seconds (0 = lifetime).
    pub window_secs: u64,
}

impl Default for ToolBudget {
    fn default() -> Self {
        Self {
            cap_usd: 1.0,
            window_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSpend {
    pub spent_usd: f64,
    pub calls: u64,
    pub denied: u64,
    pub window_started_at_unix: i64,
}

pub struct McpBudgetGateway {
    /// (agent, tool) → budget config.
    budgets: Mutex<HashMap<(String, String), ToolBudget>>,
    /// (agent, tool) → current spend.
    spend: Mutex<HashMap<(String, String), ToolSpend>>,
    /// Default budget for any (agent, tool) without an explicit config.
    default_budget: ToolBudget,
}

impl McpBudgetGateway {
    pub fn new(default_budget: ToolBudget) -> Arc<Self> {
        Arc::new(Self {
            budgets: Mutex::new(HashMap::new()),
            spend: Mutex::new(HashMap::new()),
            default_budget,
        })
    }

    pub fn set_budget(&self, agent: &str, tool: &str, b: ToolBudget) {
        self.budgets
            .lock()
            .insert((agent.to_string(), tool.to_string()), b);
    }

    /// Check if a call is allowed. Returns the current spend snapshot on
    /// the allowed path or a `BudgetError` if the cap is breached.
    pub fn check(&self, agent: &str, tool: &str) -> Result<ToolSpend, BudgetError> {
        let key = (agent.to_string(), tool.to_string());
        let budget = self
            .budgets
            .lock()
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default_budget.clone());

        let mut spend_map = self.spend.lock();
        let now = now_unix();
        let spend = spend_map.entry(key.clone()).or_default();

        // Window rollover.
        if budget.window_secs > 0
            && spend.window_started_at_unix > 0
            && (now - spend.window_started_at_unix) as u64 >= budget.window_secs
        {
            *spend = ToolSpend {
                window_started_at_unix: now,
                ..Default::default()
            };
        }
        if spend.window_started_at_unix == 0 {
            spend.window_started_at_unix = now;
        }

        if spend.spent_usd >= budget.cap_usd {
            spend.denied += 1;
            let denied = spend.clone();
            warn!(
                agent = %agent,
                tool = %tool,
                spent = spend.spent_usd,
                cap = budget.cap_usd,
                "MCP budget denied"
            );
            return Err(BudgetError::Exceeded {
                agent: agent.into(),
                tool: tool.into(),
                spent_usd: denied.spent_usd,
                cap_usd: budget.cap_usd,
            });
        }
        Ok(spend.clone())
    }

    /// Record the cost of a completed tool call. Call this *after* the
    /// upstream returns, with the actual spend.
    pub fn record(&self, agent: &str, tool: &str, cost_usd: f64) {
        let key = (agent.to_string(), tool.to_string());
        let mut spend_map = self.spend.lock();
        let spend = spend_map.entry(key).or_default();
        if spend.window_started_at_unix == 0 {
            spend.window_started_at_unix = now_unix();
        }
        spend.spent_usd += cost_usd;
        spend.calls += 1;
    }

    pub fn snapshot(&self) -> HashMap<String, ToolSpend> {
        self.spend
            .lock()
            .iter()
            .map(|((agent, tool), s)| (format!("{agent}::{tool}"), s.clone()))
            .collect()
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_allows_until_cap() {
        let g = McpBudgetGateway::new(ToolBudget {
            cap_usd: 0.10,
            window_secs: 3600,
        });
        for _ in 0..5 {
            g.check("agent-a", "search").unwrap();
            g.record("agent-a", "search", 0.02);
        }
        // Now spent = 0.10, next check should deny.
        assert!(g.check("agent-a", "search").is_err());
    }

    #[test]
    fn explicit_budget_overrides_default() {
        let g = McpBudgetGateway::new(ToolBudget {
            cap_usd: 100.0,
            window_secs: 3600,
        });
        g.set_budget(
            "agent-a",
            "expensive_tool",
            ToolBudget {
                cap_usd: 0.05,
                window_secs: 3600,
            },
        );
        g.check("agent-a", "expensive_tool").unwrap();
        g.record("agent-a", "expensive_tool", 0.06);
        assert!(g.check("agent-a", "expensive_tool").is_err());
        // Different tool still allowed via default.
        g.check("agent-a", "cheap_tool").unwrap();
    }

    #[test]
    fn snapshot_includes_calls_and_denials() {
        let g = McpBudgetGateway::new(ToolBudget {
            cap_usd: 0.01,
            window_secs: 3600,
        });
        g.check("a", "t").unwrap();
        g.record("a", "t", 0.02);
        let _ = g.check("a", "t"); // denied
        let snap = g.snapshot();
        let entry = snap.get("a::t").unwrap();
        assert_eq!(entry.calls, 1);
        assert_eq!(entry.denied, 1);
        assert!((entry.spent_usd - 0.02).abs() < 1e-9);
    }
}

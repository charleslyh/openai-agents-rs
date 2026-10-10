//! Built-in retry policies (Python: `retry_policies`).

use super::{ReplaySafety, RetryDecision, RetryPolicy, RetryPolicyContext};

/// Built-in policies (Python: `retry_policies`).
pub mod retry_policies {
    use super::*;

    /// Never retry.
    pub fn never() -> RetryPolicy {
        RetryPolicy::new(|_| async { false })
    }

    /// Follow the provider's advice (Python: `provider_suggested`).
    pub fn provider_suggested() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            let advice = ctx.provider_advice.as_ref();
            let reason = advice.and_then(|a| a.reason.clone());
            let retry_after = advice.and_then(|a| a.retry_after);
            match advice.and_then(|a| a.suggested) {
                None => RetryDecision::no(),
                Some(false) => {
                    let mut decision = RetryDecision::no();
                    decision.reason = reason;
                    if ctx.replay_safety() == Some(ReplaySafety::Unsafe) {
                        decision.delegable_replay_veto()
                    } else {
                        decision.hard_veto()
                    }
                }
                Some(true) => {
                    let mut decision = RetryDecision::yes();
                    decision.delay = retry_after;
                    decision.reason = reason;
                    if ctx.replay_safety() == Some(ReplaySafety::Safe) {
                        decision.replay_safe_approval()
                    } else {
                        decision
                    }
                }
            }
        })
    }

    /// Retry network failures and timeouts.
    pub fn network_error() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            ctx.normalized.is_network_error || ctx.normalized.is_timeout
        })
    }

    /// Retry when the provider sent a `Retry-After`, waiting that long.
    pub fn retry_after() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            let delay = ctx
                .normalized
                .retry_after
                .or_else(|| ctx.provider_advice.as_ref().and_then(|a| a.retry_after));
            match delay {
                Some(delay) => RetryDecision::yes().with_delay(delay),
                None => RetryDecision::no(),
            }
        })
    }

    /// Retry the given HTTP statuses.
    pub fn http_status(statuses: impl IntoIterator<Item = u16>) -> RetryPolicy {
        let allowed: Vec<u16> = statuses.into_iter().collect();
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let retry = ctx
                .normalized
                .status_code
                .is_some_and(|code| allowed.contains(&code));
            async move { retry }
        })
    }

    /// Retry only when every policy agrees (Python: `all`).
    pub fn all(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let policies = policies.clone();
            async move {
                let mut merged = RetryDecision::yes();
                let mut delegable: Option<RetryDecision> = None;
                for policy in &policies {
                    let decision = policy.evaluate(ctx.clone()).await;
                    if decision.hard_veto {
                        if decision.delegable_replay_veto {
                            delegable.get_or_insert(decision);
                            continue;
                        }
                        return decision;
                    }
                    if !decision.retry {
                        return decision;
                    }
                    if decision.delay.is_some() {
                        merged.delay = decision.delay;
                    }
                    if decision.reason.is_some() {
                        merged.reason = decision.reason.clone();
                    }
                    if decision.approve_unsafe_replay {
                        merged.approve_unsafe_replay = true;
                    }
                    if decision.approves_replay {
                        merged = merged.replay_safe_approval();
                    }
                }
                match delegable {
                    Some(veto) => RetryDecision::resolve_delegable_veto(veto, merged),
                    None => merged,
                }
            }
        })
    }

    /// Retry when any policy agrees (Python: `any`).
    pub fn any(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let policies = policies.clone();
            async move {
                let mut first_positive: Option<RetryDecision> = None;
                let mut last_negative: Option<RetryDecision> = None;
                let mut delegable: Option<RetryDecision> = None;
                for policy in &policies {
                    let decision = policy.evaluate(ctx.clone()).await;
                    if decision.hard_veto {
                        if decision.delegable_replay_veto {
                            delegable.get_or_insert(decision);
                            continue;
                        }
                        return decision;
                    }
                    if decision.retry {
                        first_positive = Some(match first_positive {
                            None => decision,
                            Some(existing) => RetryDecision::merge_positive(existing, decision),
                        });
                        continue;
                    }
                    last_negative = Some(decision);
                }
                if let Some(veto) = delegable {
                    return match first_positive {
                        None => veto,
                        Some(positive) => RetryDecision::resolve_delegable_veto(veto, positive),
                    };
                }
                first_positive
                    .or(last_negative)
                    .unwrap_or_else(RetryDecision::no)
            }
        })
    }
}

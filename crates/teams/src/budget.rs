//! Limits that keep a team from running away with cost.
//!
//! A team multiplies model calls, so every limit has a default and the defaults are small.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::error::TeamError;

/// Limits of one team.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Budget {
    /// Teammates alive at the same time (default 3).
    pub max_members: usize,
    /// Model turns of one teammate run (default 10).
    pub max_turns_per_run: usize,
    /// Teammate runs in total, across the team (default 50).
    pub max_total_runs: u64,
    /// Tokens in total, across the team. `None` means unlimited (default).
    pub max_total_tokens: Option<u64>,
    /// Largest message body in bytes (default 16 KiB).
    pub max_message_bytes: usize,
    /// Largest task description in bytes (default 8 KiB).
    pub max_task_bytes: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_members: 3,
            max_turns_per_run: 10,
            max_total_runs: 50,
            max_total_tokens: None,
            max_message_bytes: 16 * 1024,
            max_task_bytes: 8 * 1024,
        }
    }
}

impl Budget {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Teammates alive at once.
    #[must_use]
    pub fn max_members(mut self, n: usize) -> Self {
        self.max_members = n;
        self
    }

    /// Model turns of one run.
    #[must_use]
    pub fn max_turns_per_run(mut self, n: usize) -> Self {
        self.max_turns_per_run = n;
        self
    }

    /// Runs in total.
    #[must_use]
    pub fn max_total_runs(mut self, n: u64) -> Self {
        self.max_total_runs = n;
        self
    }

    /// Tokens in total.
    #[must_use]
    pub fn max_total_tokens(mut self, n: u64) -> Self {
        self.max_total_tokens = Some(n);
        self
    }

    /// Largest message body in bytes.
    #[must_use]
    pub fn max_message_bytes(mut self, n: usize) -> Self {
        self.max_message_bytes = n;
        self
    }

    /// Largest task description in bytes.
    #[must_use]
    pub fn max_task_bytes(mut self, n: usize) -> Self {
        self.max_task_bytes = n;
        self
    }
}

/// Running totals checked against a [`Budget`].
pub(crate) struct Meter {
    budget: Budget,
    runs: AtomicU64,
    tokens: AtomicU64,
    reported: AtomicBool,
}

impl Meter {
    pub(crate) fn new(budget: Budget) -> Self {
        Self {
            budget,
            runs: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            reported: AtomicBool::new(false),
        }
    }

    pub(crate) fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Count one more run, or refuse when a limit is reached.
    pub(crate) fn start_run(&self) -> Result<(), TeamError> {
        if let Some(max) = self.budget.max_total_tokens {
            if self.tokens.load(Ordering::SeqCst) >= max {
                return Err(TeamError::BudgetExceeded(format!(
                    "the team used its {max} token budget"
                )));
            }
        }
        // A compare-exchange loop, so a refused run is never counted.
        let mut runs = self.runs.load(Ordering::SeqCst);
        loop {
            if runs >= self.budget.max_total_runs {
                return Err(TeamError::BudgetExceeded(format!(
                    "the team reached its limit of {} runs",
                    self.budget.max_total_runs
                )));
            }
            match self.runs.compare_exchange_weak(
                runs,
                runs + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => runs = current,
            }
        }
    }

    pub(crate) fn add_tokens(&self, tokens: u64) {
        self.tokens.fetch_add(tokens, Ordering::SeqCst);
    }

    pub(crate) fn runs(&self) -> u64 {
        self.runs.load(Ordering::SeqCst)
    }

    pub(crate) fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::SeqCst)
    }

    /// `true` only the first time, so exhaustion is reported once, not on every refused call.
    pub(crate) fn first_report(&self) -> bool {
        !self.reported.swap(true, Ordering::SeqCst)
    }
}

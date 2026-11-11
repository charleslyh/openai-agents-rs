//! Reducers that drop whole exchanges, built from one question: which exchanges stay?
//!
//! An [`ExchangeSelector`] answers only that. [`select_exchanges`] and [`DropExchanges`] do the
//! rest and guarantee the rules every reducer must follow, so a selector cannot break them:
//! exchanges are the smallest unit (a tool call never leaves its output behind), pinned items
//! always stay, the latest exchange always stays, and the order is preserved.

use async_trait::async_trait;
use serde_json::Value;

use crate::context::{ExchangeView, TokenCounter};
use crate::error::AgentsError;
use crate::reducer::{ContextReducer, ReduceInput, ReduceOutcome};

/// What a selector gets to look at.
pub struct SelectionView<'a> {
    /// The exchanges, oldest first. Never empty.
    pub exchanges: &'a [Vec<Value>],
    /// Estimated tokens of each exchange.
    pub costs: &'a [usize],
    /// Tokens that stay whatever is chosen: pinned items plus the reserved tokens.
    pub fixed_tokens: usize,
}

impl SelectionView<'_> {
    /// Index of the latest exchange, which is always kept.
    pub fn latest(&self) -> usize {
        self.exchanges.len() - 1
    }
}

/// Chooses which exchanges to keep.
pub trait ExchangeSelector: Send + Sync {
    /// Short name, used in traces.
    fn name(&self) -> &str;

    /// Indices into `view.exchanges` to keep. Order and duplicates do not matter, indices out of
    /// range are ignored, and the latest exchange is kept even when left out.
    fn select(&self, view: &SelectionView<'_>) -> Vec<usize>;
}

/// Keep the exchanges `selector` chooses. Returns the new items and how many exchanges were
/// dropped. A history without exchanges, or one where everything is kept, is returned as is.
pub fn select_exchanges(
    selector: &dyn ExchangeSelector,
    items: &[Value],
    reserved_tokens: usize,
    counter: &TokenCounter,
) -> (Vec<Value>, usize) {
    let history = ExchangeView::split(items);
    let total = history.exchanges.len();
    if total == 0 {
        return (items.to_vec(), 0);
    }
    let costs: Vec<usize> = history
        .exchanges
        .iter()
        .map(|exchange| exchange.iter().map(|item| counter(item)).sum())
        .collect();
    let fixed_tokens = reserved_tokens
        + history
            .pinned
            .iter()
            .map(|item| counter(item))
            .sum::<usize>();
    let view = SelectionView {
        exchanges: &history.exchanges,
        costs: &costs,
        fixed_tokens,
    };

    let mut keep = vec![false; total];
    keep[total - 1] = true;
    for index in selector.select(&view) {
        if let Some(slot) = keep.get_mut(index) {
            *slot = true;
        }
    }
    let kept = keep.iter().filter(|&&k| k).count();
    if kept == total {
        return (items.to_vec(), 0);
    }

    let mut result = history.pinned;
    result.extend(
        history
            .exchanges
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .flat_map(|(exchange, _)| exchange),
    );
    (result, total - kept)
}

/// A [`ContextReducer`] that drops exchanges as an [`ExchangeSelector`] chooses.
pub struct DropExchanges<S> {
    selector: S,
}

impl<S: ExchangeSelector> DropExchanges<S> {
    /// Reduce with `selector`.
    pub fn new(selector: S) -> Self {
        Self { selector }
    }
}

#[async_trait]
impl<S: ExchangeSelector> ContextReducer for DropExchanges<S> {
    fn name(&self) -> &str {
        self.selector.name()
    }

    async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
        let (items, dropped) = select_exchanges(
            &self.selector,
            input.items,
            input.reserved_tokens,
            input.counter,
        );
        Ok(ReduceOutcome::new(&input, items).dropped_exchanges(dropped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::default_token_counter;
    use serde_json::json;

    struct Choose(Vec<usize>);

    impl ExchangeSelector for Choose {
        fn name(&self) -> &str {
            "choose"
        }

        fn select(&self, _: &SelectionView<'_>) -> Vec<usize> {
            self.0.clone()
        }
    }

    fn history() -> Vec<Value> {
        let mut items = vec![json!({"role": "system", "content": "s"})];
        for i in 0..4 {
            items.push(json!({"role": "user", "content": format!("q{i}")}));
            items.push(json!({"type": "function_call", "call_id": format!("c{i}"),
                              "name": "t", "arguments": "{}"}));
            items.push(
                json!({"type": "function_call_output", "call_id": format!("c{i}"),
                              "output": "ok"}),
            );
        }
        items
    }

    #[test]
    fn a_selector_that_keeps_nothing_still_leaves_pinned_and_the_latest_exchange() {
        let items = history();
        let (kept, dropped) =
            select_exchanges(&Choose(vec![]), &items, 0, &default_token_counter());
        assert_eq!(dropped, 3);
        assert_eq!(kept.len(), 1 + 3);
        assert_eq!(kept[0]["role"], "system");
        assert_eq!(kept[1]["content"], "q3");
    }

    #[test]
    fn indices_are_deduplicated_ordered_and_bounds_checked() {
        let items = history();
        let (kept, dropped) = select_exchanges(
            &Choose(vec![2, 0, 0, 99]),
            &items,
            0,
            &default_token_counter(),
        );
        let users: Vec<&str> = kept
            .iter()
            .filter_map(|i| i["content"].as_str())
            .filter(|t| t.starts_with('q'))
            .collect();
        assert_eq!(users, ["q0", "q2", "q3"]);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn keeping_everything_returns_the_input_unchanged() {
        let items = history();
        let (kept, dropped) = select_exchanges(
            &Choose(vec![0, 1, 2, 3]),
            &items,
            0,
            &default_token_counter(),
        );
        assert_eq!(kept, items);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn a_history_without_exchanges_is_returned_as_is() {
        let items = vec![json!({"role": "system", "content": "s"})];
        let (kept, dropped) =
            select_exchanges(&Choose(vec![]), &items, 0, &default_token_counter());
        assert_eq!((kept, dropped), (items, 0));
    }
}

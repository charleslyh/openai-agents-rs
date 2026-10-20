//! Arithmetic the model can call, without handing it an interpreter (Rust-only, X-019).
//!
//! "Let the model evaluate this" is an easy thing to ask for and a dangerous thing to give: the
//! usual answer is an `eval` of whatever the model wrote. This one is a small parser instead —
//! numbers, the four operators, parentheses and a unary minus, and nothing else — so there is no
//! way to reach a variable, a function, a loop or the file system, however the expression is
//! written.
//!
//! What it refuses is part of the tool: an expression longer than [`MAX_EXPRESSION`], a number the
//! parser cannot read, a division by zero and an empty expression are all errors the model sees,
//! because a tool that silently answers `0` teaches the model nothing.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use openai_agents_core::error::UserError;
use openai_agents_core::{AgentsError, FunctionTool};

/// The longest expression accepted. Long enough for arithmetic, short enough that nothing long and
/// strange is worth parsing.
pub const MAX_EXPRESSION: usize = 200;

/// Arguments of the `calculator` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CalculatorArgs {
    /// The expression to evaluate, e.g. `(12 + 3) * 4`.
    pub expression: String,
}

/// The `calculator` tool: `expression` in, `result` out.
pub fn calculator() -> FunctionTool {
    FunctionTool::typed(
        "calculator",
        "Evaluate an arithmetic expression: numbers, + - * /, parentheses and a unary minus. \
         Nothing else is accepted; there are no variables or functions.",
        |_ctx, args: CalculatorArgs| async move {
            let value = evaluate(&args.expression)?;
            Ok(json!({"result": value}))
        },
    )
}

/// Parse and evaluate `expression`, or say why it cannot be done.
pub fn evaluate(expression: &str) -> Result<f64, AgentsError> {
    let trimmed = expression.trim();
    if trimmed.is_empty() {
        return Err(err("the expression is empty"));
    }
    if trimmed.len() > MAX_EXPRESSION {
        return Err(err(format!(
            "the expression is longer than {MAX_EXPRESSION} characters"
        )));
    }
    let mut parser = Parser::new(trimmed);
    let value = parser.parse()?;
    parser.skip_spaces();
    if !parser.is_at_end() {
        return Err(err(format!(
            "unexpected {:?} at position {}",
            parser.peek().unwrap_or(' '),
            parser.position
        )));
    }
    if !value.is_finite() {
        return Err(err("the result is not a finite number"));
    }
    Ok(value)
}

fn err(message: impl Into<String>) -> AgentsError {
    AgentsError::from(UserError::new(format!("calculator: {}", message.into())))
}

/// Recursive descent over the tokens of one expression: no state to escape from, and nothing to
/// evaluate twice.
struct Parser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }

    fn parse(&mut self) -> Result<f64, AgentsError> {
        self.sum()
    }

    fn sum(&mut self) -> Result<f64, AgentsError> {
        let mut left = self.product()?;
        loop {
            self.skip_spaces();
            match self.peek() {
                Some('+') => {
                    self.position += 1;
                    left += self.product()?;
                }
                Some('-') => {
                    self.position += 1;
                    left -= self.product()?;
                }
                _ => return Ok(left),
            }
        }
    }

    fn product(&mut self) -> Result<f64, AgentsError> {
        let mut left = self.unary()?;
        loop {
            self.skip_spaces();
            match self.peek() {
                Some('*') => {
                    self.position += 1;
                    left *= self.unary()?;
                }
                Some('/') => {
                    self.position += 1;
                    let right = self.unary()?;
                    if right == 0.0 {
                        return Err(err("division by zero"));
                    }
                    left /= right;
                }
                _ => return Ok(left),
            }
        }
    }

    fn unary(&mut self) -> Result<f64, AgentsError> {
        self.skip_spaces();
        match self.peek() {
            Some('-') => {
                self.position += 1;
                Ok(-self.unary()?)
            }
            Some('+') => {
                self.position += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> Result<f64, AgentsError> {
        self.skip_spaces();
        match self.peek() {
            Some('(') => {
                self.position += 1;
                let value = self.sum()?;
                self.skip_spaces();
                if self.peek() != Some(')') {
                    return Err(err("a '(' is never closed"));
                }
                self.position += 1;
                Ok(value)
            }
            _ => self.number(),
        }
    }

    fn number(&mut self) -> Result<f64, AgentsError> {
        let start = self.position;
        let mut saw_digit = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                saw_digit = true;
                self.position += 1;
            } else if c == '.' && !self.input[start..self.position].contains('.') {
                self.position += 1;
            } else {
                break;
            }
        }
        if !saw_digit {
            return Err(err(format!(
                "expected a number at position {}",
                self.position
            )));
        }
        self.input[start..self.position]
            .parse::<f64>()
            .map_err(|_| {
                err(format!(
                    "{} is not a number",
                    &self.input[start..self.position]
                ))
            })
    }

    fn skip_spaces(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_works() {
        assert_eq!(evaluate("1 + 2").unwrap(), 3.0);
        assert_eq!(evaluate("(12 + 3) * 4").unwrap(), 60.0);
        assert_eq!(evaluate("-3 + 1").unwrap(), -2.0);
        assert_eq!(evaluate("10 / 4").unwrap(), 2.5);
        assert_eq!(evaluate("  2*3  ").unwrap(), 6.0);
    }

    /// Scientific notation is not part of the grammar: it reads as a number followed by junk.
    #[test]
    fn notation_outside_the_grammar_is_an_error() {
        assert!(evaluate("1.5e2").is_err());
        assert!(evaluate("0x10").is_err());
    }

    #[test]
    fn precedence_is_the_usual_one() {
        assert_eq!(evaluate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(evaluate("(2 + 3) * 4").unwrap(), 20.0);
    }

    /// The point of a parser instead of an `eval`: nothing that is not arithmetic gets through.
    #[test]
    fn anything_that_is_not_arithmetic_is_an_error() {
        for expression in [
            "",
            "   ",
            "1 +",
            "(1 + 2",
            "1 + 2)",
            "1 / 0",
            "abc",
            "1; drop table",
            "1 + 2 * (3",
            "__import__('os').system('ls')",
            "1 2",
            &"1".repeat(MAX_EXPRESSION + 1),
        ] {
            assert!(
                evaluate(expression).is_err(),
                "`{expression}` must be refused"
            );
        }
    }

    /// Nesting is bounded by the length limit, but a parser that recursed without a limit of its
    /// own would still be one deep expression from a stack overflow.
    #[test]
    fn deeply_nested_input_is_an_error_not_a_crash() {
        let open = "(".repeat(MAX_EXPRESSION - 2);
        assert!(evaluate(&format!("{open}1")).is_err());
        let minus = "-".repeat(MAX_EXPRESSION - 1);
        // An even number of unary minuses is positive, and must not overflow the stack either.
        let _ = evaluate(&format!("{minus}1"));
        let sums = "1+".repeat(MAX_EXPRESSION / 2 - 1) + "1";
        assert_eq!(evaluate(&sums).unwrap(), (MAX_EXPRESSION / 2) as f64);
    }

    /// Whatever the text, the answer is a number or an error: never a panic.
    #[test]
    fn arbitrary_text_never_panics() {
        let alphabet: Vec<char> = "0123456789.+-*/() \tx\u{e9}\u{200b}\u{1F600}"
            .chars()
            .collect();
        // A small deterministic generator, so a failure reproduces.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let length = (next() % 40) as usize;
            let text: String = (0..length)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let _ = evaluate(&text);
        }
    }

    #[test]
    fn overflow_and_oddities_are_errors() {
        assert!(evaluate(&format!(
            "{0}*{0}*{0}*{0}*{0}*{0}*{0}*{0}*{0}*{0}",
            "9".repeat(30)
        ))
        .is_err());
        assert!(evaluate("0.1.2").is_err());
        assert!(evaluate(".").is_err());
        assert!(evaluate("1..2").is_err());
    }
}

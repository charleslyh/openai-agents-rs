//! `cargo opai new` — create a project that uses the OpenAI Agents Rust SDK.
//!
//! The templates are the same shape as the repository's examples, trimmed to one file each:
//! `chat` (one run), `tools` (a `#[function_tool]` and a `FunctionTool::typed`) and `hitl` (a call
//! that waits for a human). They are compiled into this binary, so it works from anywhere.
//!
//! ```bash
//! cargo install opai-agents-scaffold   # installs `cargo-opai`
//! cargo opai new my-agent --template tools
//! cd my-agent && cargo run
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Version the generated `Cargo.toml` asks for. A path is used instead when `--path` is given.
const SDK_VERSION: &str = "0.1.0";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Template {
    Chat,
    Tools,
    Hitl,
}

impl Template {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "chat" | "hello" => Some(Self::Chat),
            "tools" | "tool" => Some(Self::Tools),
            "hitl" | "approval" => Some(Self::Hitl),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Tools => "tools",
            Self::Hitl => "hitl",
        }
    }

    fn main_rs(self) -> &'static str {
        match self {
            Self::Chat => include_str!("../templates/chat/src/main.rs"),
            Self::Tools => include_str!("../templates/tools/src/main.rs"),
            Self::Hitl => include_str!("../templates/hitl/src/main.rs"),
        }
    }
}

const TEMPLATE_CARGO_TOML: &str = include_str!("../templates/Cargo.toml.tpl");
const TEMPLATE_README: &str = include_str!("../templates/README.md");

fn usage() -> &'static str {
    "\
Create a project that uses the OpenAI Agents Rust SDK.

Usage: cargo opai new <name> [--template <chat|tools|hitl>] [--path <dir>] [--force]

Arguments:
  <name>                 Directory to create; also the package name. Letters, digits, `-` and `_`,
                         starting with a letter.

Options:
  -t, --template <name>  chat (default): one run.
                         tools: a `#[function_tool]` and a `FunctionTool::typed`.
                         hitl: a tool call that waits for a human.
  -p, --path <dir>       Depend on the SDK at this path instead of the published version. Use it
                         inside a checkout of the repository, or before the SDK is published.
  -f, --force            Write into a directory that already has files.
  -h, --help             Show this message."
}

struct Request {
    name: String,
    template: Template,
    sdk_path: Option<PathBuf>,
    force: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Request>, String> {
    let mut name = None;
    let mut template = Template::Chat;
    let mut sdk_path = None;
    let mut force = false;

    let mut iter = args
        .into_iter()
        // The program name and `new`, which `main` has already checked.
        .skip(2)
        .peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" | "help" => return Ok(None),
            "-f" | "--force" => force = true,
            "-t" | "--template" => {
                let value = iter.next().ok_or("`--template` needs a value")?;
                template = Template::parse(&value).ok_or_else(|| {
                    format!("unknown template `{value}`; expected chat, tools or hitl")
                })?;
            }
            "-p" | "--path" => {
                let value = iter.next().ok_or("`--path` needs a value")?;
                sdk_path = Some(PathBuf::from(value));
            }
            other if other.starts_with('-') => return Err(format!("unknown option `{other}`")),
            other => {
                if name.replace(other.to_string()).is_some() {
                    return Err("one project name at a time".to_string());
                }
            }
        }
    }
    let name = name.ok_or("a project name is required: cargo opai new <name>")?;
    if !is_package_name(&name) {
        return Err(format!(
            "`{name}` is not a usable package name: letters, digits, `-` and `_`, starting with a \
             letter"
        ));
    }
    Ok(Some(Request {
        name,
        template,
        sdk_path,
        force,
    }))
}

/// A package name Cargo accepts and a directory name that cannot escape the working directory.
fn is_package_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && !name.eq_ignore_ascii_case("test")
}

fn render(template: &str, request: &Request) -> String {
    let dependency = match &request.sdk_path {
        Some(path) => {
            let path = path.canonicalize().unwrap_or_else(|_| path.clone());
            format!(
                "path = \"{}\"",
                path.display().to_string().replace('\\', "/")
            )
        }
        None => format!("version = \"{SDK_VERSION}\""),
    };
    template
        .replace("{{name}}", &request.name)
        .replace("{{dependency}}", &dependency)
}

fn create(request: &Request) -> Result<PathBuf, String> {
    let root = Path::new(&request.name);
    if root.exists() {
        let empty = fs::read_dir(root)
            .map_err(|e| format!("cannot read {}: {e}", root.display()))?
            .next()
            .is_none();
        if !empty && !request.force {
            return Err(format!(
                "{} already exists and is not empty; pass --force to write into it",
                root.display()
            ));
        }
    }
    let src = root.join("src");
    fs::create_dir_all(&src).map_err(|e| format!("cannot create {}: {e}", src.display()))?;
    let files = [
        ("Cargo.toml", render(TEMPLATE_CARGO_TOML, request)),
        ("README.md", render(TEMPLATE_README, request)),
        ("src/main.rs", request.template.main_rs().to_string()),
        (".gitignore", "/target\n.env\n".to_string()),
    ];
    for (relative, contents) in files {
        let path = root.join(relative);
        fs::write(&path, contents).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(root.canonicalize().unwrap_or_else(|_| root.to_path_buf()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    // `cargo opai ...` passes `opai` as the first argument; drop it so the same binary works
    // when run directly.
    let args: Vec<String> = match args.get(1).map(String::as_str) {
        Some("opai") => std::iter::once(args[0].clone())
            .chain(args.into_iter().skip(2))
            .collect(),
        _ => args,
    };
    match args.get(1).map(String::as_str) {
        Some("new") => {}
        Some(other) => {
            eprintln!(
                "unknown subcommand `{other}`; the only one is `new`.\n\n{}",
                usage()
            );
            return ExitCode::from(2);
        }
        None => {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
    }
    let request = match parse_args(args) {
        Ok(Some(request)) => request,
        Ok(None) => {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("{message}\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    match create(&request) {
        Ok(root) => {
            println!(
                "Created {} ({} template) in {}",
                request.name,
                request.template.name(),
                root.display()
            );
            println!("\nNext:");
            println!("  cd {}", request.name);
            println!("  AGENTS_OFFLINE=1 cargo run -- \"Hello\"   # no key needed");
            println!("  OPENAI_API_KEY=sk-... cargo run -- \"Hello\"");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("cargo-opai".to_string())
            .chain(list.iter().map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn the_default_is_a_chat_project() {
        let request = parse_args(args(&["new", "my-agent"]))
            .expect("parse")
            .expect("some");
        assert_eq!(request.name, "my-agent");
        assert!(request.template == Template::Chat);
        assert!(!request.force);
    }

    #[test]
    fn a_template_and_a_path_are_read() {
        let request = parse_args(args(&["new", "a", "-t", "hitl", "--path", "."]))
            .expect("parse")
            .expect("some");
        assert!(request.template == Template::Hitl);
        assert_eq!(request.sdk_path.as_deref(), Some(Path::new(".")));
        assert!(parse_args(args(&["new", "a", "--template", "nope"])).is_err());
    }

    #[test]
    fn a_name_must_not_escape_the_directory() {
        for bad in ["", "../x", "a/b", "-x", "1a", "a b", "x;rm"] {
            assert!(
                parse_args(args(&["new", bad])).is_err(),
                "`{bad}` must be refused"
            );
        }
        assert!(parse_args(args(&["new"])).is_err());
        assert!(parse_args(args(&["new", "a", "b"])).is_err());
        assert!(parse_args(args(&["new", "a", "--nope"])).is_err());
    }

    #[test]
    fn help_prints_without_a_name() {
        assert!(parse_args(args(&["new", "--help"])).expect("ok").is_none());
    }

    #[test]
    fn every_template_renders_without_a_placeholder_and_is_rustcpp_free() {
        for template in [Template::Chat, Template::Tools, Template::Hitl] {
            let request = Request {
                name: "example-agent".to_string(),
                template,
                sdk_path: None,
                force: false,
            };
            for text in [TEMPLATE_CARGO_TOML, TEMPLATE_README, template.main_rs()] {
                let rendered = render(text, &request);
                assert!(!rendered.contains("{{"), "{}: {rendered}", template.name());
                assert!(!rendered.contains("}}"));
            }
            let rendered = render(TEMPLATE_CARGO_TOML, &request);
            assert!(rendered.contains("name = \"example-agent\""));
            assert!(rendered.contains("version = \"0.1.0\""));
        }
    }

    #[test]
    fn a_path_dependency_replaces_the_version() {
        let request = Request {
            name: "a".to_string(),
            template: Template::Chat,
            sdk_path: Some(PathBuf::from(".")),
            force: false,
        };
        let rendered = render(TEMPLATE_CARGO_TOML, &request);
        assert!(rendered.contains("path = \""), "{rendered}");
        assert!(!rendered.contains("opai-agents = { version"), "{rendered}");
    }

    #[test]
    fn a_project_is_written_and_an_occupied_directory_needs_force() {
        let dir = tempfile::tempdir().expect("tempdir");
        let name = format!("{}/agent-{}", dir.path().display(), std::process::id());
        let request = Request {
            name,
            template: Template::Tools,
            sdk_path: None,
            force: false,
        };
        let root = create(&request).expect("create");
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("src/main.rs").is_file());
        assert!(fs::read_to_string(root.join("src/main.rs"))
            .expect("read")
            .contains("#[function_tool"));

        assert!(create(&request).is_err(), "not empty without --force");
        let forced = Request {
            force: true,
            ..request
        };
        assert!(create(&forced).is_ok());
    }
}

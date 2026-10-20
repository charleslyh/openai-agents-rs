//! Fetching a URL, with the private network off limits (Rust-only, X-019).
//!
//! A tool that fetches what the model names is the easiest way to turn an agent into a client for
//! the network it runs on: `http://169.254.169.254/latest/meta-data/` is one prompt away, and a
//! metadata service answers with credentials. So the addresses this tool will not reach are the
//! point of it, not a setting:
//!
//! * **Every resolved address is checked, at connect time.** The check is a DNS resolver of its
//!   own, so it runs on the addresses the connection is about to use — not on a name checked once
//!   and resolved again afterwards.
//! * **Loopback, link-local, private, multicast, reserved and unspecified** addresses are refused,
//!   in IPv6 as well, including IPv4-mapped IPv6 forms. The `9.`, `11.`, `21.` and `30.` /8 blocks
//!   (public-looking reserved space) and the `100.64.0.0/10` carrier range are refused with them.
//! * **Redirects are checked hop by hop**, and a redirect to another origin is refused outright
//!   unless it is asked for: that is where credentials would otherwise be sent somewhere else.
//! * `https` is allowed anywhere; `http` only to loopback — which is itself refused unless private
//!   addresses are allowed — or when [`FetchOptions::allow_insecure_http`] is switched on.
//! * An address written into the URL (`http://2130706433/`, `http://0x7f.1/`) never reaches the
//!   resolver, so it is checked as the URL is read, on the first request and on every redirect.
//! * No system proxy is used: a proxy would be the address the connection goes to, and the check
//!   would be on the wrong one.
//! * A response is read in pieces and stops at the limit, so a server that never stops sending
//!   cannot fill memory.
//!
//! [`FetchOptions::allow_private`] opens it up — for a test against a local server, or an agent
//! whose whole job is a service next to it. It is named "allow" on purpose: saying it is what turns
//! a default off.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use ipnet::Ipv4Net;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::{Attempt, Policy};
use reqwest::{Client, Url};

use openai_agents_core::error::UserError;
use openai_agents_core::{AgentsError, FunctionTool};

/// The largest response returned.
pub const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

/// How long a connection may take to open.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the whole fetch may take.
pub const TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

/// How many redirects may be followed.
pub const MAX_REDIRECTS: usize = 5;

/// Arguments of the `web_fetch` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchArgs {
    /// The URL to fetch. `https` (or `http` on loopback, or with `allow_insecure_http`).
    pub url: String,
    /// Return only the text of the page, without tags (default: false).
    #[serde(default)]
    pub as_text: Option<bool>,
}

/// How [`web_fetch`] behaves. The defaults are the safe ones.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Reach loopback, link-local, private and reserved addresses too (default: false).
    pub allow_private: bool,
    /// Allow `http://` to a host that is not loopback (default: false).
    pub allow_insecure_http: bool,
    /// Follow a redirect that changes origin (default: false).
    pub allow_cross_origin_redirects: bool,
    /// Largest response returned (default: [`MAX_RESPONSE_BYTES`]).
    pub max_bytes: usize,
    /// Timeout for the whole fetch (default: [`TOTAL_TIMEOUT`]).
    pub timeout: Duration,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            allow_private: false,
            allow_insecure_http: false,
            allow_cross_origin_redirects: false,
            max_bytes: MAX_RESPONSE_BYTES,
            timeout: TOTAL_TIMEOUT,
        }
    }
}

/// The `web_fetch` tool with the defaults: private networks and cross-origin redirects refused.
pub fn web_fetch() -> FunctionTool {
    web_fetch_with(FetchOptions::default())
}

/// The `web_fetch` tool with `options`.
pub fn web_fetch_with(options: FetchOptions) -> FunctionTool {
    FunctionTool::typed(
        "web_fetch",
        "Fetch a URL and return its body as text. https (or http on loopback). Private, \
         loopback and link-local addresses are refused, and so is a redirect to another origin.",
        move |_ctx, args: FetchArgs| {
            let options = options.clone();
            async move { fetch(&args.url, args.as_text.unwrap_or(false), &options).await }
        },
    )
}

/// Fetch `raw_url` under `options`: what the tool does, callable without a model.
pub async fn fetch(
    raw_url: &str,
    as_text: bool,
    options: &FetchOptions,
) -> Result<serde_json::Value, AgentsError> {
    let url = parse_url(raw_url, options)?;
    let client = client(options)?;
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| tool_err(format!("cannot fetch {raw_url}: {}", describe(&e))))?;
    let status = response.status().as_u16();
    if !response.status().is_success() {
        return Ok(json!({
            "url": raw_url,
            "status": status,
            "text": "",
            "error": format!("the server answered {status}"),
        }));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let too_big = || {
        tool_err(format!(
            "{raw_url} answers more than {} bytes, the most this tool reads",
            options.max_bytes
        ))
    };
    if response
        .content_length()
        .is_some_and(|length| length > options.max_bytes as u64)
    {
        return Err(too_big());
    }
    // Read in pieces and stop at the limit: a server that never stops sending must not be able to
    // fill the process's memory, which reading the whole body first would let it do. (The client
    // is built without decompression, so the limit is on what crossed the wire as well.)
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| tool_err(format!("cannot read {raw_url}: {}", describe(&e))))?
    {
        if bytes.len() + chunk.len() > options.max_bytes {
            return Err(too_big());
        }
        bytes.extend_from_slice(&chunk);
    }
    let body = String::from_utf8_lossy(&bytes).to_string();
    Ok(json!({
        "url": raw_url,
        "status": status,
        "content_type": content_type,
        "text": if as_text { strip_tags(&body) } else { body },
    }))
}

/// An error with the reason behind it: reqwest's own message stops at "error following redirect".
fn describe(error: &reqwest::Error) -> String {
    use std::error::Error;
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

fn client(options: &FetchOptions) -> Result<Client, AgentsError> {
    let policy = options.clone();
    Client::builder()
        .dns_resolver(Arc::new(GuardedResolver {
            allow_private: options.allow_private,
        }))
        .redirect(Policy::custom(
            move |attempt: Attempt| match check_redirect(attempt.previous(), attempt.url(), &policy)
            {
                Ok(()) => attempt.follow(),
                Err(reason) => attempt.error(reason),
            },
        ))
        // A proxy would make the connection go to the proxy, and the resolver would be checking
        // the proxy's address instead of the one the model named.
        .no_proxy()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(options.timeout)
        .user_agent(concat!("openai-agents-rs/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| tool_err(format!("cannot build an HTTP client: {e}")))
}

/// Whether the redirect from `previous` (the chain so far) to `next` may be followed.
///
/// Every hop is checked as the first request is: the scheme, an address written into the URL, and
/// whether the origin changed. A redirect is the usual way to turn a harmless URL into a request
/// for something else.
pub fn check_redirect(previous: &[Url], next: &Url, options: &FetchOptions) -> Result<(), String> {
    if previous.len() >= MAX_REDIRECTS {
        return Err(format!("more than {MAX_REDIRECTS} redirects"));
    }
    check_url(next, options)?;
    if !options.allow_cross_origin_redirects {
        let from = previous.last().and_then(origin);
        if from.is_none() || from != origin(next) {
            return Err("a redirect to another origin is refused".to_string());
        }
    }
    Ok(())
}

/// A resolver that answers only with addresses the tool may connect to.
///
/// Checking here rather than before the request is what makes the check real: these are the
/// addresses the connection is about to use.
struct GuardedResolver {
    allow_private: bool,
}

impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let allow_private = self.allow_private;
        let host = name.as_str().to_string();
        Box::pin(async move {
            let resolved = tokio::net::lookup_host((host.as_str(), 0)).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("cannot resolve {host}: {e}").into()
                },
            )?;
            let allowed: Vec<SocketAddr> = resolved
                .filter(|addr| allow_private || is_reachable(addr.ip()))
                .collect();
            if allowed.is_empty() {
                return Err(format!(
                    "{host} resolves to addresses this tool does not fetch from (private, \
                     loopback, link-local or reserved)"
                )
                .into());
            }
            Ok(Box::new(allowed.into_iter()) as Addrs)
        })
    }
}

fn origin(url: &Url) -> Option<(String, Option<u16>)> {
    Some((
        url.host_str()?.to_ascii_lowercase(),
        url.port_or_known_default(),
    ))
}

fn parse_url(raw: &str, options: &FetchOptions) -> Result<Url, AgentsError> {
    let url = Url::parse(raw.trim()).map_err(|e| tool_err(format!("{raw} is not a URL: {e}")))?;
    check_url(&url, options).map_err(tool_err)?;
    Ok(url)
}

/// The checks every URL passes, first request or redirect.
///
/// An address written into the URL (`http://169.254.169.254/`, and the forms the URL parser turns
/// into one: `http://2852039166/`, `http://0xa9.0xfe.0xa9.0xfe/`, `http://[::ffff:10.0.0.1]/`) never
/// goes through the resolver, so it has to be checked here.
fn check_url(url: &Url, options: &FetchOptions) -> Result<(), String> {
    let host = url.host().ok_or("the URL has no host")?;
    match url.scheme() {
        "https" => {}
        "http" => {
            let loopback = match &host {
                url::Host::Domain(name) => name.eq_ignore_ascii_case("localhost"),
                url::Host::Ipv4(ip) => ip.is_loopback(),
                url::Host::Ipv6(ip) => ip.is_loopback(),
            };
            if !loopback && !options.allow_insecure_http {
                return Err("only https is fetched, except on loopback".to_string());
            }
        }
        other => {
            return Err(format!(
                "{other} is not a scheme this tool fetches: use http or https"
            ))
        }
    }
    let literal = match host {
        url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
        url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
        url::Host::Domain(_) => None,
    };
    match literal {
        Some(ip) if !options.allow_private && !is_reachable(ip) => {
            Err("that address is private, loopback, link-local or reserved".to_string())
        }
        _ => Ok(()),
    }
}

/// Whether the tool may connect to `ip`: the SSRF check itself.
///
/// Refused: loopback, unspecified, link-local, private (10/8, 172.16/12, 192.168/16, 127/8),
/// multicast, reserved — including 9/8, 11/8, 21/8, 30/8, which look public but are not — the
/// carrier-grade 100.64/10, documentation ranges, benchmarking 198.18/15, and every IPv6 form of
/// the same, including IPv4-mapped and IPv4-compatible ones.
pub fn is_reachable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_reachable_v4(ip),
        IpAddr::V6(ip) => match ip.to_ipv4_mapped() {
            Some(mapped) => is_reachable_v4(mapped),
            None => {
                if let Some(compatible) = to_ipv4_compatible(ip) {
                    return is_reachable_v4(compatible);
                }
                !ip.is_loopback()
                    && !ip.is_unspecified()
                    && !ip.is_multicast()
                    && !ip.is_unique_local()
                    && !ip.is_unicast_link_local()
                    && !is_ipv6_reserved(ip)
            }
        },
    }
}

fn to_ipv4_compatible(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let segments = ip.segments();
    if segments[0..6] == [0, 0, 0, 0, 0, 0] && !(segments[6] == 0 && segments[7] <= 1) {
        Some(Ipv4Addr::new(
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        ))
    } else {
        None
    }
}

fn is_ipv6_reserved(ip: Ipv6Addr) -> bool {
    // 2001:db8::/32 (documentation) and 100::/64 (discard).
    let segments = ip.segments();
    matches!(segments[0], 0x2001) && segments[1] == 0x0db8
        || segments[0..6] == [0x0100, 0, 0, 0, 0, 0]
}

fn is_reachable_v4(ip: Ipv4Addr) -> bool {
    if ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_link_local()
        || ip.is_private()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_documentation()
    {
        return false;
    }
    // Reserved but not caught by the standard predicates: 9/8, 11/8, 21/8, 30/8 are allocated to
    // the DoD and others and are not routable; 100.64/10 is carrier-grade NAT; 192.0.0/24 and
    // 198.18/15 are protocol and benchmark ranges.
    const RESERVED: &[&str] = &[
        "0.0.0.0/8",
        "9.0.0.0/8",
        "11.0.0.0/8",
        "21.0.0.0/8",
        "30.0.0.0/8",
        "100.64.0.0/10",
        "169.254.0.0/16",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "240.0.0.0/4",
    ];
    !RESERVED
        .iter()
        .filter_map(|range| range.parse::<Ipv4Net>().ok())
        .any(|net| net.contains(&ip))
}

/// Remove tags and collapse whitespace, so a page is something a model can read.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn tool_err(message: impl Into<String>) -> AgentsError {
    AgentsError::from(UserError::new(format!("web_fetch: {}", message.into())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    /// The addresses an agent must not be able to reach.
    #[test]
    fn private_and_reserved_addresses_are_refused() {
        for address in [
            "127.0.0.1",
            "127.1.2.3",
            "0.0.0.0",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "169.254.1.1",
            "100.64.0.1",
            "9.1.1.1",
            "11.1.1.1",
            "21.1.1.1",
            "30.1.1.1",
            "224.0.0.1",
            "255.255.255.255",
            "192.0.0.1",
            "198.18.0.1",
            "240.0.0.1",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "fd00::abcd",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "::ffff:10.0.0.1",
            "::ffff:9.1.1.1",
        ] {
            assert!(!is_reachable(ip(address)), "`{address}` must be refused");
        }
    }

    #[test]
    fn a_public_address_is_allowed() {
        for address in [
            "1.1.1.1",
            "8.8.8.8",
            "93.184.216.34",
            "2606:2800:220:1:248:1893:25c8:1946",
        ] {
            assert!(is_reachable(ip(address)), "`{address}` must be allowed");
        }
    }

    #[test]
    fn only_http_and_https_are_fetched() {
        let options = FetchOptions::default();
        assert!(parse_url("https://example.com", &options).is_ok());
        // Loopback is http-able, but it is still loopback: refused unless private is allowed.
        assert!(parse_url("http://127.0.0.1:8080/x", &options).is_err());
        let local = FetchOptions {
            allow_private: true,
            ..FetchOptions::default()
        };
        assert!(parse_url("http://127.0.0.1:8080/x", &local).is_ok());
        assert!(parse_url("http://localhost:8080/x", &local).is_ok());
        assert!(parse_url("http://example.com", &options).is_err());
        assert!(parse_url("file:///etc/passwd", &options).is_err());
        assert!(parse_url("gopher://example.com", &options).is_err());
        assert!(parse_url("not a url", &options).is_err());
    }

    #[test]
    fn tags_come_out() {
        assert_eq!(strip_tags("<p>a <b>b</b></p>"), "a b");
    }
}

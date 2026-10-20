//! `web_fetch` against a real socket (X-019).
//!
//! The unit tests check the address table. These check the thing that matters: that a request the
//! tool refuses never reaches the server. Every test starts a local server that counts the
//! connections it accepts, so "refused" means "zero connections", not "an error came back".

#![cfg(feature = "webfetch")]

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openai_agents_tools::web_fetch::{check_redirect, fetch, FetchOptions};
use reqwest::Url;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// How a test server answers.
#[derive(Clone)]
enum Reply {
    /// 200 with this body.
    Body(Vec<u8>),
    /// 302 to this location.
    Redirect(String),
    /// 200 that keeps sending until the client goes away.
    Endless,
}

struct Server {
    addr: SocketAddr,
    connections: Arc<AtomicUsize>,
}

async fn serve(reply: Reply) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let connections = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&connections);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            counted.fetch_add(1, Ordering::SeqCst);
            let reply = reply.clone();
            tokio::spawn(async move {
                let mut buffer = [0u8; 4096];
                let mut seen = Vec::new();
                while !seen.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buffer).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => seen.extend_from_slice(&buffer[..n]),
                    }
                }
                match reply {
                    Reply::Body(body) => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                        let _ = socket.write_all(&body).await;
                    }
                    Reply::Redirect(to) => {
                        let head = format!(
                            "HTTP/1.1 302 Found\r\nlocation: {to}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                    }
                    Reply::Endless => {
                        // No content-length: the body goes on until the client gives up.
                        let _ = socket
                            .write_all(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n")
                            .await;
                        let chunk = vec![b'x'; 64 * 1024];
                        while socket.write_all(&chunk).await.is_ok() {}
                    }
                }
            });
        }
    });
    Server { addr, connections }
}

fn defaults() -> FetchOptions {
    FetchOptions::default()
}

fn local() -> FetchOptions {
    FetchOptions {
        allow_private: true,
        ..FetchOptions::default()
    }
}

fn contacted(server: &Server) -> usize {
    server.connections.load(Ordering::SeqCst)
}

#[tokio::test]
async fn a_loopback_address_is_never_connected_to() {
    let server = serve(Reply::Body(b"secret".to_vec())).await;
    let url = format!("http://{}/", server.addr);
    let err = fetch(&url, false, &defaults())
        .await
        .expect_err("loopback is refused");
    assert!(err.to_string().contains("private, loopback"), "{err}");
    assert_eq!(
        contacted(&server),
        0,
        "the server must not have been reached"
    );
}

/// The URL parser turns these into the same loopback address; none may get through.
#[tokio::test]
async fn other_spellings_of_the_same_address_are_refused_too() {
    let server = serve(Reply::Body(b"secret".to_vec())).await;
    let port = server.addr.port();
    for host in [
        "2130706433".to_string(),         // decimal
        "0x7f000001".to_string(),         // hexadecimal
        "0177.0.0.1".to_string(),         // octal
        "127.1".to_string(),              // short form
        "[::1]".to_string(),              // IPv6 loopback
        "[::ffff:127.0.0.1]".to_string(), // IPv4-mapped
        "0".to_string(),                  // 0.0.0.0
    ] {
        let url = format!("http://{host}:{port}/");
        // Refused for one reason or another; what matters is the connection count below.
        assert!(fetch(&url, false, &defaults()).await.is_err(), "{url}");
    }
    assert_eq!(contacted(&server), 0, "none of them reached the server");
}

#[tokio::test]
async fn a_name_that_resolves_to_loopback_is_refused_by_the_resolver() {
    let server = serve(Reply::Body(b"secret".to_vec())).await;
    let url = format!("http://localhost:{}/", server.addr.port());
    let err = fetch(&url, false, &defaults())
        .await
        .expect_err("localhost resolves to loopback");
    assert!(err.to_string().contains("does not fetch from"), "{err}");
    assert_eq!(
        contacted(&server),
        0,
        "the server must not have been reached"
    );
}

#[tokio::test]
async fn the_metadata_address_is_refused() {
    for url in [
        "http://169.254.169.254/latest/meta-data/",
        "https://169.254.169.254/",
        "http://[fd00:ec2::254]/",
        "http://metadata.google.internal.invalid/",
    ] {
        let started = std::time::Instant::now();
        assert!(fetch(url, false, &defaults()).await.is_err(), "{url}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{url} must be refused, not timed out"
        );
    }
}

#[tokio::test]
async fn allowing_private_addresses_is_a_deliberate_switch_and_then_it_works() {
    let server = serve(Reply::Body(b"<p>hello <b>there</b></p>".to_vec())).await;
    let url = format!("http://{}/", server.addr);
    let page = fetch(&url, false, &local()).await.expect("allowed");
    assert_eq!(page["status"], 200);
    assert_eq!(page["text"], "<p>hello <b>there</b></p>");
    assert_eq!(page["content_type"], "text/html");
    let text = fetch(&url, true, &local()).await.expect("allowed");
    assert_eq!(text["text"], "hello there");
    assert_eq!(contacted(&server), 2);
}

#[tokio::test]
async fn a_redirect_to_another_origin_is_not_followed() {
    let target = serve(Reply::Body(b"target".to_vec())).await;
    let redirector = serve(Reply::Redirect(format!("http://{}/", target.addr))).await;
    let url = format!("http://{}/", redirector.addr);
    let err = fetch(&url, false, &local())
        .await
        .expect_err("different port, different origin");
    assert!(err.to_string().contains("another origin"), "{err}");
    assert_eq!(contacted(&redirector), 1);
    assert_eq!(
        contacted(&target),
        0,
        "the redirect target was never contacted"
    );
}

#[tokio::test]
async fn a_redirect_to_a_private_address_is_refused_even_when_origins_may_change() {
    // The first hop is allowed to be local; the redirect to the metadata address is not allowed
    // to be reached unless private addresses are, so check the hop itself.
    let options = FetchOptions {
        allow_cross_origin_redirects: true,
        ..defaults()
    };
    let from = Url::parse("https://example.com/").unwrap();
    for target in [
        "https://169.254.169.254/latest",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://10.0.0.5/",
        "https://2130706433/",
    ] {
        let next = Url::parse(target).unwrap();
        assert!(
            check_redirect(std::slice::from_ref(&from), &next, &options).is_err(),
            "{target}"
        );
    }
    // A public address is fine when cross-origin redirects are allowed.
    let next = Url::parse("https://93.184.216.34/").unwrap();
    assert!(check_redirect(&[from], &next, &options).is_ok());
}

#[test]
fn a_redirect_that_downgrades_to_http_is_refused() {
    let options = FetchOptions {
        allow_cross_origin_redirects: true,
        ..defaults()
    };
    let from = Url::parse("https://example.com/").unwrap();
    let next = Url::parse("http://example.org/").unwrap();
    assert!(check_redirect(std::slice::from_ref(&from), &next, &options).is_err());
    let allowed = FetchOptions {
        allow_insecure_http: true,
        ..options
    };
    assert!(check_redirect(&[from], &next, &allowed).is_ok());
}

#[test]
fn a_redirect_chain_is_bounded() {
    let hop = Url::parse("https://example.com/").unwrap();
    let chain = vec![hop.clone(); 5];
    assert!(check_redirect(&chain, &hop, &defaults()).is_err());
    assert!(check_redirect(&chain[..4], &hop, &defaults()).is_ok());
}

#[tokio::test]
async fn a_large_response_is_refused() {
    let server = serve(Reply::Body(vec![b'x'; 10_000])).await;
    let url = format!("http://{}/", server.addr);
    let options = FetchOptions {
        max_bytes: 1_000,
        ..local()
    };
    let err = fetch(&url, false, &options).await.expect_err("too big");
    assert!(err.to_string().contains("1000 bytes"), "{err}");
}

/// A server that never stops: the read ends at the limit instead of filling memory.
#[tokio::test]
async fn an_endless_response_is_cut_off_at_the_limit() {
    let server = serve(Reply::Endless).await;
    let url = format!("http://{}/", server.addr);
    let options = FetchOptions {
        max_bytes: 100_000,
        timeout: Duration::from_secs(20),
        ..local()
    };
    let started = std::time::Instant::now();
    let err = fetch(&url, false, &options).await.expect_err("endless");
    assert!(err.to_string().contains("100000 bytes"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "it stopped at the limit, not at the timeout"
    );
}

#[tokio::test]
async fn an_error_status_is_reported_not_raised() {
    // Nothing listens here: a refused connection is an error, and says so.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let err = fetch(&format!("http://{addr}/"), false, &local())
        .await
        .expect_err("nothing is listening");
    assert!(err.to_string().contains("cannot fetch"), "{err}");
}

#[tokio::test]
async fn other_schemes_are_refused() {
    for url in [
        "file:///etc/passwd",
        "ftp://example.com/",
        "gopher://example.com/",
        "data:text/plain,x",
    ] {
        assert!(fetch(url, false, &local()).await.is_err(), "{url}");
    }
}

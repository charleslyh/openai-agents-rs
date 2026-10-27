#!/usr/bin/env python3
"""Publish the workspace crates to crates.io one at a time, in dependency order, pacing the uploads.

Why not `cargo publish --workspace`: crates.io rate-limits *new crates* with a leaky bucket (a burst
of 5, then about one every 10 minutes). Cargo's upload step does not retry a 429 and honours
`Retry-After` for at most 10 seconds, and one invocation cannot pause between two uploads. A
re-publish of a version that already exists still spends a token, so a re-run must not upload what
is already there.

What this does:
  * orders crates with `cargo metadata` (normal and build dependencies that carry a version);
    a cycle is an error
  * asks the sparse index first and skips versions that are already published
  * paces new-crate uploads with a local model of the bucket, and sleeps until the time the server
    names when it answers 429
  * publishes with `cargo publish -p <crate>`; the token is read by cargo from the environment
    variable CARGO_REGISTRY_TOKEN and is never put on a command line or printed

Usage:
  python3 scripts/publish.py --plan          # print the order and what would be skipped
  python3 scripts/publish.py --dry-run       # `cargo publish --workspace --dry-run`
  python3 scripts/publish.py                 # publish for real (needs CARGO_REGISTRY_TOKEN)
  python3 scripts/publish.py --from NAME     # start at NAME in the order (crates before it skipped)

Standard library only.
"""
from __future__ import annotations

import argparse
import email.utils
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
INDEX = "https://index.crates.io"

# Documented in docs/RELEASING.md. The 2026-10-21 upload of this workspace is the evidence:
# five crates in six seconds, then one roughly every ten minutes.
NEW_CRATE_BURST = 5
NEW_CRATE_INTERVAL_S = 600.0
# New versions of crates that already exist have a looser limit. The exact figure is not verified;
# this is a conservative pause, not a promise. A 429 is still handled whatever it is.
EXISTING_CRATE_INTERVAL_S = 60.0
MAX_429_RETRIES = 6


class PublishError(Exception):
    pass


@dataclass
class Crate:
    name: str
    version: str
    deps: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- order


def load_crates() -> list[Crate]:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return crates_from_metadata(json.loads(out))


def crates_from_metadata(meta: dict) -> list[Crate]:
    members = {p["name"]: p for p in meta["packages"] if p.get("publish") != []}
    crates = {n: Crate(n, p["version"]) for n, p in members.items()}
    for name, pkg in members.items():
        for dep in pkg["dependencies"]:
            if dep["name"] not in members or dep.get("kind") == "dev":
                continue
            # A path dependency with no version requirement is stripped on publish; only a
            # versioned one is a real edge.
            if dep["req"] in ("*", ""):
                continue
            crates[name].deps.add(dep["name"])
    return list(crates.values())


def publish_order(crates: list[Crate]) -> list[Crate]:
    """Dependencies first; ties by name so the order is stable. A cycle raises."""
    by_name = {c.name: c for c in crates}
    done: list[Crate] = []
    placed: set[str] = set()
    remaining = set(by_name)
    while remaining:
        ready = sorted(n for n in remaining if by_name[n].deps <= placed)
        if not ready:
            raise PublishError(f"dependency cycle among: {', '.join(sorted(remaining))}")
        for n in ready:
            done.append(by_name[n])
            placed.add(n)
            remaining.discard(n)
    return done


# --------------------------------------------------------------------------- index


def index_path(name: str) -> str:
    n = name.lower()
    if len(n) == 1:
        return f"1/{n}"
    if len(n) == 2:
        return f"2/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[0:2]}/{n[2:4]}/{n}"


def parse_index(body: str) -> dict[str, bool]:
    """version -> yanked, from the sparse index's one-JSON-object-per-line body."""
    versions: dict[str, bool] = {}
    for line in body.splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            versions[row["vers"]] = bool(row.get("yanked"))
    return versions


def fetch_index(name: str) -> dict[str, bool] | None:
    """None when the crate does not exist yet."""
    req = urllib.request.Request(
        f"{INDEX}/{index_path(name)}", headers={"User-Agent": "opai-agents-publish-script"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return parse_index(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            return None
        raise PublishError(f"index lookup for {name} failed: HTTP {e.code}") from e


# --------------------------------------------------------------------------- pacing


class Bucket:
    """A model of the new-crate leaky bucket. `now` is injected so tests need no real clock."""

    def __init__(self, burst: int, interval_s: float, tokens: float | None = None, now: float = 0.0):
        self.burst = float(burst)
        self.interval = interval_s
        self.tokens = self.burst if tokens is None else min(float(tokens), self.burst)
        self.at = now

    def _refill(self, now: float) -> None:
        self.tokens = min(self.burst, self.tokens + (now - self.at) / self.interval)
        self.at = max(self.at, now)

    def wait_for_token(self, now: float) -> float:
        """Seconds to wait before one upload may go."""
        self._refill(now)
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) * self.interval

    def take(self, now: float) -> None:
        self._refill(now)
        self.tokens = max(0.0, self.tokens - 1.0)


def estimate_seconds(new_crates: int, burst: int = NEW_CRATE_BURST,
                     interval: float = NEW_CRATE_INTERVAL_S, tokens: float | None = None) -> float:
    """Total waiting for `new_crates` first-time uploads, ignoring upload time."""
    bucket = Bucket(burst, interval, tokens)
    t = 0.0
    for _ in range(new_crates):
        t += bucket.wait_for_token(t)
        bucket.take(t)
    return t


def retry_after_seconds(text: str, now: datetime | None = None) -> float | None:
    """How long a 429 asks us to wait.

    The crates.io message reads like: "You have published too many new crates in a short period of
    time. Please try again after Tue, 21 Oct 2026 02:48:00 GMT or email help@crates.io ...".
    A plain `Retry-After: 120` header, if it shows up in the output, is read too.
    """
    now = now or datetime.now(timezone.utc)
    m = re.search(r"try again after ([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2} GMT)", text)
    if m:
        when = email.utils.parsedate_to_datetime(m.group(1))
        return max(0.0, (when - now).total_seconds())
    m = re.search(r"retry-after:\s*(\d+)", text, re.I)
    if m:
        return float(m.group(1))
    return None


def is_rate_limited(text: str) -> bool:
    return "429" in text or "too many requests" in text.lower() or "published too many" in text.lower()


def is_already_published(text: str) -> bool:
    return "already exists" in text.lower() or "already uploaded" in text.lower()


# --------------------------------------------------------------------------- run


def redact(text: str) -> str:
    token = os.environ.get("CARGO_REGISTRY_TOKEN")
    return text.replace(token, "***") if token else text


def say(msg: str) -> None:
    print(msg, flush=True)


def cargo_publish(name: str) -> subprocess.CompletedProcess:
    # No --token: cargo reads CARGO_REGISTRY_TOKEN itself, so it never appears in `ps`.
    return subprocess.run(
        ["cargo", "publish", "-p", name], cwd=ROOT, capture_output=True, text=True
    )


def wait_until_visible(name: str, version: str, sleep: Callable[[float], None] = time.sleep) -> None:
    for _ in range(24):
        found = fetch_index(name)
        if found is not None and version in found:
            return
        sleep(5)
    say(f"  note: {name} {version} is not in the index yet; continuing, cargo will wait for it")


def run(args: argparse.Namespace) -> int:
    order = publish_order(load_crates())
    if args.from_crate:
        names = [c.name for c in order]
        if args.from_crate not in names:
            raise PublishError(f"--from {args.from_crate}: not a publishable crate ({', '.join(names)})")
        skipped_before = set(names[: names.index(args.from_crate)])
    else:
        skipped_before = set()

    status: dict[str, str] = {}
    for c in order:
        if c.name in skipped_before:
            status[c.name] = "skip (before --from)"
            continue
        found = fetch_index(c.name)
        if found is None:
            status[c.name] = "new"
        elif c.version in found:
            status[c.name] = "published"
        else:
            status[c.name] = "update"

    say("publish order:")
    for i, c in enumerate(order, 1):
        say(f"  {i:2}. {c.name} {c.version}  [{status[c.name]}]")
    new = sum(1 for s in status.values() if s == "new")
    est = estimate_seconds(new, tokens=args.bucket_tokens)
    say(f"new crates: {new}; estimated pacing wait: {est / 60:.0f} min (burst {NEW_CRATE_BURST}, "
        f"then one per {NEW_CRATE_INTERVAL_S / 60:.0f} min)")

    if args.plan:
        return 0
    if args.dry_run:
        # Only a single resolution of the whole set can rehearse a first release.
        return subprocess.run(
            ["cargo", "publish", "--workspace", "--dry-run", "--allow-dirty"], cwd=ROOT
        ).returncode
    if not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise PublishError("CARGO_REGISTRY_TOKEN is not set")

    start = time.monotonic()
    bucket = Bucket(NEW_CRATE_BURST, NEW_CRATE_INTERVAL_S, args.bucket_tokens, now=0.0)
    last_existing_at: float | None = None
    for c in order:
        st = status[c.name]
        if st in ("published", "skip (before --from)"):
            continue
        now = time.monotonic() - start
        if st == "new":
            pause = bucket.wait_for_token(now)
        else:
            pause = 0.0
            if last_existing_at is not None:
                pause = max(0.0, EXISTING_CRATE_INTERVAL_S - (now - last_existing_at))
        if pause > 0:
            say(f"pacing: waiting {pause:.0f}s before {c.name}")
            time.sleep(pause)
        for attempt in range(MAX_429_RETRIES + 1):
            say(f"publishing {c.name} {c.version} ...")
            proc = cargo_publish(c.name)
            text = redact(proc.stdout + proc.stderr)
            if proc.returncode == 0:
                break
            if is_already_published(text):
                say(f"  {c.name} {c.version} is already there; moving on")
                break
            if is_rate_limited(text) and attempt < MAX_429_RETRIES:
                wait = retry_after_seconds(text)
                wait = (wait if wait is not None else NEW_CRATE_INTERVAL_S) + 1.0
                say(f"  rate limited; sleeping {wait:.0f}s (attempt {attempt + 1}/{MAX_429_RETRIES})")
                time.sleep(wait)
                continue
            print(text, file=sys.stderr)
            raise PublishError(f"publishing {c.name} failed; fix it and re-run (use --from {c.name})")
        done_at = time.monotonic() - start
        if st == "new":
            bucket.take(done_at)
        else:
            last_existing_at = done_at
        wait_until_visible(c.name, c.version)
    say("done")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the order and the status, publish nothing")
    ap.add_argument("--dry-run", action="store_true", help="cargo publish --workspace --dry-run")
    ap.add_argument("--from", dest="from_crate", metavar="NAME", help="start at this crate")
    ap.add_argument("--bucket-tokens", type=float, default=None,
                    help="new-crate tokens assumed available now (default: a full burst); "
                         "use a smaller number if you published new crates a moment ago")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except PublishError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

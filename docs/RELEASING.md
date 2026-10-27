# Releasing

How to publish the crates to crates.io without running into its rate limit.

## What the limit is

crates.io limits how fast **new crates** can be created, with a leaky bucket: a burst of **5**, then
about **one every 10 minutes**. A `429` response says so and tells you to email
`help@crates.io` for a higher limit.

This is a rate, not a cap on how many crates you may own and not a documented "maximum simultaneous
uploads". No separate limit on concurrent uploads has been found in the crates.io documentation.
Two facts decide the design:

- Cargo does not retry a failed upload, and it honours `Retry-After` for at most 10 seconds.
  One `cargo publish --workspace` call cannot pause between two uploads, so it fails as soon as the
  bucket is empty, and part of the workspace is already published by then.
- crates.io checks the rate limit before it checks whether the version exists. Uploading something
  already published still spends a token. A blind re-run makes the wall closer, not farther.

Not verified: the exact limit for publishing a **new version of an existing crate**. The script
pauses 60 seconds between those, which is a guess; if a `429` arrives it sleeps as the server says.

### Evidence from this repository

The first release, 2026-10-21 (UTC), as read from the sparse index:

| Time | Crate | Note |
| --- | --- | --- |
| 02:36:56 to 02:37:02 | `core`, `macros`, `scaffold`, `guardrails`, `mcp` | 5 crates in 6 seconds: the burst |
| 02:48 | `observability` | then one about every 10 minutes |
| 02:57 | `openai` | |
| 03:08 | `providers` | |
| 03:17 | `sessions` | |
| 03:27 | `sqlite` | |
| 03:37 | `teams` | |
| 03:47 | `testing` | |
| 03:57 | `tools` | |
| 04:07 | `opai-agents` | about 1.5 hours in all |

## Why this workspace hits it

The table below is the 0.1.0 layout of fourteen crates. Since 0.2.0 `sqlite` lives in `sessions` and
`scaffold` is not published, so the set is twelve crates: batch 1 has two, batch 2 has eight.

Publish order follows dependencies, and most crates depend only on `core`:

| Batch | Crates | Count |
| --- | --- | --- |
| 1 | `core`, `macros` | 2 |
| 2 | `testing`, `openai`, `mcp`, `sessions`, `observability`, `tools`, `guardrails`, `teams` | 8 |
| 3 | `providers` | 1 |
| 4 | `opai-agents` (facade) | 1 |

Batch 1 leaves 2 tokens of the burst. Batch 2 unlocks eight uploads at once; the third one fails.

## The process

Never use `cargo publish --workspace` for the real upload. Use the script:

```sh
python3 scripts/publish.py --plan        # order, what is already on crates.io, estimated wait
python3 scripts/publish.py --dry-run     # cargo publish --workspace --dry-run: rehearses the whole set
export CARGO_REGISTRY_TOKEN=...          # environment only; never on a command line, never in a file
python3 scripts/publish.py               # one crate at a time, paced
```

What the script does:

1. Orders crates by the dependencies that carry a version requirement. A cycle is an error.
2. Asks the sparse index (`index.crates.io`) for each crate and **skips versions already
   published**. A re-run after an interruption spends no tokens on finished crates.
3. Models the bucket (burst 5, one per 10 minutes) and waits before an upload that would be
   refused.
4. If crates.io answers `429` anyway, it reads the time in the message (`try again after ...`) and
   sleeps until then, then tries again, up to six times.
5. Waits until a crate shows in the index before publishing its dependents.

If something fails, fix it and run again; `--from NAME` starts at a given crate. The GitHub
workflow `.github/workflows/release.yml` does the same: run it by hand (it defaults to a dry run) or
push a `v*` tag. It allows only one run at a time. Put required reviewers on the `release`
environment to gate real uploads.

If you published new crates a moment ago, tell the script the bucket is not full:
`--bucket-tokens 0`.

### How long a first release takes

For \(N\) new crates the wait is about \(\max(0, N - 5) \times 10\) minutes. With 12 new
crates that is 5 at once and 7 more at 10 minutes each: **about 70 minutes** (the 14 crates of
0.1.0 took about 90). A release of new
versions of existing crates does not use the new-crate bucket.

### Asking for a higher limit

The `429` text names `help@crates.io`. Send it before a release that creates several new crates:
the crate names, why they must go out together, and that you have a script that paces them. A
raised limit shortens the first release; it is not needed for ordinary releases.

## Versions

Every crate shares `[workspace.package] version` in the root `Cargo.toml`, and the `opai-agents*`
entries in `[workspace.dependencies]` carry the same number (that is the requirement a published
crate states for its siblings). To release `X.Y.Z`: change those thirteen numbers, update the
install snippets, run `python3 scripts/check_docs_drift.py`, which names anything left behind, then
follow the process above.

### Release tools, evaluated

`release-plz` and `cargo-release` both automate the bump, the changelog and the publish. Neither
is adopted, for these reasons:

- Both publish with plain `cargo publish`, one crate per call, with no pacing between uploads and
  no reading of the `429` time. That pacing is the part of releasing this workspace cannot do
  without, and `scripts/publish.py` already does it, and skips what the index already has.
- Every crate moves with one version, so the "only release crates that changed" feature that
  `release-plz` is chiefly for has nothing to decide here.
- Revisit if the crates start to version independently: then use one of them for the bump and
  changelog and keep `scripts/publish.py` for the upload.

## Fewer new crates

Every new crate name costs a token the first time. When something new is an optional extension, make
it a **feature of an existing crate** rather than another crate. Merging existing crates into one
saves tokens only if they merge into a name that already exists; a new merged name costs one.

## Retiring a crate

A published crate cannot be deleted from crates.io. When one is folded into another (for example
`opai-agents-sqlite` into `opai-agents-sessions`):

1. Publish the last version with a README and `description` that say where the code moved.
2. Yank versions that should no longer be picked up (`cargo yank --version X crate`). Yanking stops
   new resolutions; existing lockfiles still work.
3. Note the move in `CHANGELOG.md`.

## Open items

- **Trusted Publishing.** crates.io may support exchanging a CI identity for a short-lived token,
  which would remove the long-lived `CARGO_REGISTRY_TOKEN` secret. It was seen in one third-party
  document only and has not been checked against crates.io's own documentation. Evaluate before
  relying on it.
- The original `429` text from the failed attempt, to tell "new crate" from "new version" limits.

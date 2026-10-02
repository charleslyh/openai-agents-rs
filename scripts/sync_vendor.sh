#!/usr/bin/env bash
# Sync / verify the vendored openai-agents-python tree at the pinned version.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_FILE="$ROOT/vendor/PINNED_VERSION"
VENDOR_DIR="$ROOT/vendor/openai-agents-python"
EXPECTED="${1:-}"

if [[ -z "$EXPECTED" ]]; then
  if [[ ! -f "$PIN_FILE" ]]; then
    echo "missing $PIN_FILE" >&2
    exit 1
  fi
  EXPECTED="$(tr -d '[:space:]' < "$PIN_FILE")"
fi

check_only=0
if [[ "${2:-}" == "--check" ]] || [[ "${1:-}" == "--check" ]]; then
  check_only=1
  if [[ "${1:-}" == "--check" ]]; then
    EXPECTED="$(tr -d '[:space:]' < "$PIN_FILE")"
  fi
fi

if [[ ! -d "$VENDOR_DIR" ]]; then
  if [[ "$check_only" -eq 1 ]]; then
    echo "vendor missing: $VENDOR_DIR" >&2
    exit 1
  fi
  mkdir -p "$ROOT/vendor"
  TMP="$(mktemp -d)"
  ARCHIVE="$TMP/oap.tar.gz"
  URL="https://codeload.github.com/openai/openai-agents-python/tar.gz/refs/tags/${EXPECTED}"
  echo "Downloading $URL ..."
  curl -fsSL -o "$ARCHIVE" "$URL"
  tar -xzf "$ARCHIVE" -C "$TMP"
  mv "$TMP"/openai-agents-python-* "$VENDOR_DIR"
  echo "$EXPECTED" > "$PIN_FILE"
  rm -rf "$TMP"
fi

PYPROJECT="$VENDOR_DIR/pyproject.toml"
if [[ ! -f "$PYPROJECT" ]]; then
  echo "missing $PYPROJECT" >&2
  exit 1
fi

ACTUAL="$(python3 - <<'PY' "$PYPROJECT"
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
if not m:
    raise SystemExit("version not found in pyproject.toml")
print(m.group(1))
PY
)"

PIN_TAG="${EXPECTED#v}"
if [[ "$ACTUAL" != "$PIN_TAG" ]]; then
  echo "vendor version mismatch: pyproject=$ACTUAL pinned=$EXPECTED" >&2
  exit 1
fi

echo "vendor ok: openai-agents-python $ACTUAL (pin $EXPECTED)"

#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR
###############################################################################
# Test script for downloading files using requests
# Usage:
#   scripts/test_download_requests.sh [BASE_DIR]
# If BASE_DIR is omitted, defaults to $PWD/.test_data
###############################################################################
BASE_DIR="${1:-$PWD/.test_data}"
mkdir -p "$BASE_DIR"

python3 - <<'PY' "$BASE_DIR"
import sys, requests, pathlib
url = "https://raw.githubusercontent.com/vinta/awesome-python/master/README.md"
r = requests.get(url, timeout=10)
# write into the BASE_DIR
base = pathlib.Path(sys.argv[1])
path = base / "README.md"
path.write_bytes(r.content)
print(f"✓ downloaded {len(r.content):,} bytes → {path.resolve()}")
PY
# Clean up the downloaded file
rm -f "$BASE_DIR/README.md"

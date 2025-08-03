#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR

###############################################################################
# Test script for downloading the Cora dataset using fsspec
#
# Usage:
#   source scripts/setup_web_proxy.sh
#   scripts/test_download_fsspec.sh [BASE_DIR]
#
# If BASE_DIR is omitted, defaults to $PWD/.test_data
###############################################################################

CACHE=${1:-"$PWD/.test_cora"}
mkdir -p "$CACHE"
export CACHE

python3 - <<'PY'
import os, warnings
from pathlib import Path
from torch_geometric.datasets import Planetoid

# patch Planetoid.url to a host reachable through ACES proxy
warnings.filterwarnings("ignore", category=UserWarning)
root = Path(os.getenv("CACHE")) / "Planetoid"
ds = Planetoid(root=str(root), name="Cora")
print(f"✓ Cora OK  – {len(ds)} graph(s) downloaded to {root}")
PY

rm -r $CACHE

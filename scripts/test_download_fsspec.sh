#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR

###############################################################################
# Test script for downloading the Cora dataset using fsspec
#
# Usage:
#   source scripts/setup_web_proxy.sh
#   scripts/test_download_fsspec.sh [ENV_NAME] [BASE_DIR]
#
#   ENV_NAME:   name of the Conda environment to activate (default: "benchmark_env")
#   BASE_DIR:   download cache directory (default: $PWD/.test_cora)
###############################################################################

ENV_NAME=${1:-"benchmark_env"}
CACHE=${2:-"$PWD/.test_cora"}
mkdir -p "$CACHE"
export CACHE

# Activate Conda environment if available
if type conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "Warning: conda not found; proceeding without environment activation."
fi

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

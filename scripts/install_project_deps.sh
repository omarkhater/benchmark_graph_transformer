#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR

###############################################################################
# Install project dependencies with Poetry
# Usage:
#   scripts/install_project_deps.sh [PROJECT_ROOT]
# This script:
#   1) Ensures Poetry is available
#   2) Runs `poetry install` in the project root (defaults to repo root)
###############################################################################

# Resolve script and defaults
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
DEFAULT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
# First arg: environment name; second arg: project root
ENV_NAME=${1:-"benchmark_env"}
PROJECT_ROOT=${2:-"$DEFAULT_ROOT"}
ENV_PATH="${PROJECT_ROOT%/}/$ENV_NAME"

# ── 1) Initialize conda ───────────────────────────────────────────────────────
if ! type conda >/dev/null 2>&1; then
  echo "Error: conda not found" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

# ── 2) Activate target environment ────────────────────────────────────────────
if [[ -d "$ENV_PATH" ]]; then
  echo "Activating conda env at '$ENV_PATH'..."
  conda activate "$ENV_PATH"
else
  echo "Error: conda env not found at '$ENV_PATH'" >&2
  exit 1
fi


# ── 3) Configure cache directories ──────────────────────────────────────────
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export POETRY_CACHE_DIR="$PROJECT_ROOT/.cache/poetry"
mkdir -p "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"
# ── 3) Install dependencies via Poetry ──────────────────────────────────────
cd "$PROJECT_ROOT"

echo "Installing project dependencies with Poetry at '$PROJECT_ROOT'..."
poetry install -v

echo
# Success message
echo "✓ ${SCRIPT_NAME} completed successfully"

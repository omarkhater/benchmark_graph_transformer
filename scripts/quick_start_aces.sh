#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR

###############################################################################
# Setup script for configuring a Python 3.12 environment with Poetry
# Usage:
#   scripts/setup_python_aces.sh [ENV_NAME] [ENV_LOCATION]
# This script:
#   1) Ensures conda is initialized
#   2) Creates and activates a Python 3.12 conda environment at the specified location
#   3) Upgrades pip and installs Poetry
#   4) Performs sanity checks to confirm setup (Python and Poetry versions)

# ─── 1) Ensure conda is initialized ──────────────────────────────────────────
if ! type conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
# Ensure named envs are created in the project directory
ENV_LOCATION=${2:-"$(pwd)"}
export CONDA_ENVS_PATH="$ENV_LOCATION"
export _CONDA_ENVS_PATH_FOR_QUICKSTART="$CONDA_ENVS_PATH"

# ─── 2) Create & activate a Python 3.12 env ──────────────────────────────────
# Get the environment name from the first argument or default to "benchmark_env"
ENV_NAME=${1:-"benchmark_env"}
if [[ -z "$ENV_NAME" ]]; then
  echo "Error: Environment name cannot be empty" >&2
  exit 1
fi

# Create & activate a named conda environment
if conda env list | grep -qE "^$ENV_NAME[[:space:]]"; then
  echo "Environment '$ENV_NAME' already exists. Activating..."
else
  echo "Creating named conda environment '$ENV_NAME' with Python 3.12..."
  conda create --name "$ENV_NAME" python=3.12 -y
fi

echo "Activating named conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# ─── 3) Upgrade pip & install Poetry ─────────────────────────────────────────
echo "Upgrading pip and installing Poetry..."
pip install --upgrade pip
pip install poetry

# ─── 4) Sanity checks ───────────────────────────────────────────────────────
echo "✅ Python: $(python --version)"
echo "✅ Poetry: $(poetry --version)"

echo
echo "Environment '$ENV_NAME' has Python 3.12 and Poetry installed."
echo "✓ ${SCRIPT_NAME} completed successfully"

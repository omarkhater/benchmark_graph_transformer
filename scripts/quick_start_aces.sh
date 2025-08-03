#!/usr/bin/env bash

###############################################################################
# Quick start script for configuring a Python 3.12 environment with Poetry
# and setting up the WebProxy module for ACES cluster.
# Usage:
#   scripts/quick_start_aces.sh [ENV_NAME]
# This script:
#   1) Sets up a Python 3.12 environment with Poetry
#   2) Configures the WebProxy module and environment variables
#   3) Installs project dependencies with Poetry
#   4) Test downloading via requests
#   5) Test downloading via fsspec
# ─── 1) Setup Python environment with Poetry ────────────────────────────────

PROJECT_DIR=$PWD
env_name="${1:-benchmark_env}"

# Setup Python environment with Poetry
if [[ -f $PROJECT_DIR/scripts/setup_python_aces.sh ]]; then
  echo "######################"
  echo "Setting up Python environment with Poetry..."
  chmod +x "$PROJECT_DIR/scripts/setup_python_aces.sh"
  bash \
    "$PROJECT_DIR/scripts/setup_python_aces.sh" \
    "$env_name" \
    || echo "‼ Error running setup_python_aces.sh"
  echo "######################"
else
  echo "setup_python_aces.sh not found in $PROJECT_DIR/scripts/"
fi

# Setup WebProxy module and environment variables
if [[ -f $PROJECT_DIR/scripts/setup_web_proxy.sh ]]; then
  echo "######################"
  echo "Setting up WebProxy module and environment variables..."
  chmod +x "$PROJECT_DIR/scripts/setup_web_proxy.sh"
  bash \
    "$PROJECT_DIR/scripts/setup_web_proxy.sh" \
    || echo "‼ Error running setup_web_proxy.sh"
  echo "######################"
else
  echo "setup_web_proxy.sh not found in $PROJECT_DIR/scripts/"
fi

# Install project dependencies with Poetry in specified environment
if [[ -f $PROJECT_DIR/scripts/install_project_deps.sh ]]; then
  echo "######################"
  echo "Installing project dependencies in '$env_name'..."
  chmod +x "$PROJECT_DIR/scripts/install_project_deps.sh"
  bash \
    "$PROJECT_DIR/scripts/install_project_deps.sh" \
    "$env_name" \
    "$PROJECT_DIR" \
    || echo "‼ Error running install_project_deps.sh"
  echo "######################"
else
  echo "install_project_deps.sh not found in $PROJECT_DIR/scripts/"
fi

# Test downloading via requests
if [[ -f $PROJECT_DIR/scripts/test_download_requests.sh ]]; then
  echo "######################"
  echo "Testing download using requests..."
  chmod +x "$PROJECT_DIR/scripts/test_download_requests.sh"
  bash \
    "$PROJECT_DIR/scripts/test_download_requests.sh" \
    "$PWD/.test_data" \
    || echo "‼ Error running test_download_requests.sh"
  echo "######################"
else
  echo "test_download_requests.sh not found in $PROJECT_DIR/scripts/"
fi

# Test downloading via fsspec
if [[ -f $PROJECT_DIR/scripts/test_download_fsspec.sh ]]; then
  echo "######################"
  echo "Testing download using fsspec..."
  chmod +x "$PROJECT_DIR/scripts/test_download_fsspec.sh"
  bash \
    "$PROJECT_DIR/scripts/test_download_fsspec.sh" \
    "$PROJECT_DIR/$env_name" \
    "$PWD/.test_data" \
    || echo "‼ Error running test_download_fsspec.sh"
  echo "######################"
else
  echo "test_download_fsspec.sh not found in $PROJECT_DIR/scripts/"
fi

echo "Quick start setup complete!"

cd "$PROJECT_DIR"

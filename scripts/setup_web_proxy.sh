#!/usr/bin/env bash
set -e
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
trap 'echo "${SCRIPT_NAME} failed at line ${LINENO}" >&2; exit 1' ERR
###############################################################################
# Setup script for configuring the WebProxy module and environment variables
# Usage:
#   source scripts/setup_web_proxy.sh
#
# This script:
#   1) Loads the WebProxy module if available
#   2) Exports every standard proxy var that HTTP libraries check
###############################################################################

# ── 1) your cluster’s proxy ───────────────────────────────────────────────────
PROXY=http://10.71.8.1:8080

# ── 2) enable outbound proxy ─────────────────────────────────────────────────
module load WebProxy >/dev/null 2>&1 || true     # no-op if missing

export http_proxy=$PROXY   https_proxy=$PROXY
export HTTP_PROXY=$PROXY   HTTPS_PROXY=$PROXY

# uppercase “ALL_PROXY” is widely supported; lowercase “all_proxy” too
export ALL_PROXY=$PROXY    all_proxy=$PROXY

# same for NO_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p ~/.config/fsspec
cat > ~/.config/fsspec/config.json <<EOF
{
  "http": {
    "client_kwargs": { "trust_env": true },
    "proxy": "$PROXY"
  },
  "https": {
    "client_kwargs": { "trust_env": true },
    "proxy": "$PROXY"
  }
}
EOF
echo "✓ proxy set to $PROXY"
echo "Ready – Python (and any HTTP client) will now go through the proxy."
echo "✓ ${SCRIPT_NAME} completed successfully"

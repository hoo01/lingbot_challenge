#!/usr/bin/env bash
# Reuse a validated Linux/RTX 5090 runtime in a separate new directory.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/scripts/install_runtime.py" "$@"

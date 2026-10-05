#!/usr/bin/env bash
# Development reproduction entry; organizer scene/CLI integration is pending.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_eval.sh" "$@"

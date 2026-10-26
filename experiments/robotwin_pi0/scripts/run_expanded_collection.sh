#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
exec 9>"$PROJECT_DIR/.expanded-collection.lock"
flock -n 9
printf '%s\n' "$$" > "$PROJECT_DIR/report/expanded-collection.pid"
cd "$ROBOTWIN_WORKSPACE"
python -u "$PROJECT_DIR/scripts/run_expanded_collection.py"

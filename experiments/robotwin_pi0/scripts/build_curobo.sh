#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$(python - <<'PY'
from pathlib import Path
import sys
print(':'.join(str(p) for p in (Path(sys.prefix)/'lib/python3.11/site-packages/nvidia').glob('*/include')))
PY
)${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
uv pip install --no-build-isolation --no-deps -e "$ROBOTWIN_WORKSPACE/projects/curobo-v0.7.8"

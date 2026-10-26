# Source from bash. Uses the project conda environment plus isolated Zoo additions.
FETCH_OFFICIAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FETCH_OFFICIAL_ROOT/../env.sh"
source "$FETCH_OFFICIAL_ROOT/.venv/bin/activate"
export PYTHONPATH="$FETCH_OFFICIAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$FETCH_OFFICIAL_ROOT/upstream/rl-baselines3-zoo"
unset FETCH_OFFICIAL_ROOT

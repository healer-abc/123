#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
cd "$ROBOTWIN_WORKSPACE"
case "${1:-status}" in
  status)
    for log in env_install_finish environment-check task-check-clean task-check-randomized; do
      if [ -f "$PROJECT_DIR/logs/$log.log" ]; then
        echo "$log"
        tail -25 "$PROJECT_DIR/logs/$log.log"
      fi
    done
    ls -lh "$ROBOTWIN_PATH"/assets/*.zip
    ps -u "$(id -un)" -o pid,ppid,etime,stat,wchan:20,args | rg 'install_env.sh|check_env.py|check_nccl.py|uv pip|importlib.util|nvidia-smi|download_assets'
    ;;
  check)
    if python -u "$PROJECT_DIR/scripts/check_env.py" > "$PROJECT_DIR/logs/environment-check.log" 2>&1; then
      cat "$PROJECT_DIR/logs/environment-check.log"
    else
      cat "$PROJECT_DIR/logs/environment-check.log"
      exit 1
    fi
    ;;
  task)
    for variant in clean randomized; do
      task_args=()
      [ "$variant" = randomized ] && task_args+=(--randomized)
      if timeout --kill-after=10 240 python -u "$PROJECT_DIR/scripts/check_task.py" "${task_args[@]}" > "$PROJECT_DIR/logs/task-check-$variant.log" 2>&1; then
        cat "$PROJECT_DIR/logs/task-check-$variant.log"
      else
        cat "$PROJECT_DIR/logs/task-check-$variant.log"
        exit 1
      fi
    done
    ;;
  ray)
    if timeout --kill-after=10 180 python -u "$PROJECT_DIR/scripts/check_ray.py" > "$PROJECT_DIR/logs/ray-check.log" 2>&1; then
      cat "$PROJECT_DIR/logs/ray-check.log"
    else
      cat "$PROJECT_DIR/logs/ray-check.log"
      exit 1
    fi
    ;;
  finalize)
    python -m pip check > "$PROJECT_DIR/report/dependency-check.txt"
    uv pip freeze > "$PROJECT_DIR/report/requirements-lock.txt"
    "$ROBOTWIN_WORKSPACE/miniforge3/bin/conda" list --explicit -p "$PROJECT_DIR/toolchain" > "$PROJECT_DIR/report/toolchain-explicit.txt"
    bash -n "$PROJECT_DIR/env.sh" "$PROJECT_DIR/scripts/install_env.sh" "$PROJECT_DIR/scripts/manage_env.sh"
    python - <<'PY'
from pathlib import Path
import os
import yaml
root = Path(os.environ['ROBOTWIN_EXPERIMENT_ROOT'])
robotwin = Path(os.environ['ROBOTWIN_PATH'])
for variant in ['clean', 'randomized']:
    config = yaml.safe_load((robotwin / 'task_config' / f'demo_{variant}.yml').read_text())
    config.update(planner_backend='mplib', episode_num=5, save_path=str(Path(os.environ['ROBOTWIN_WORKSPACE']) / 'data/robotwin_pi0/pilot' / variant))
    target = root / 'configs' / f'demo_{variant}_pi0.yml'
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    link = robotwin / 'task_config' / target.name
    if not link.exists():
        link.symlink_to(target)
    print(f'Pilot config: {target}')
assert '"torch==2.11.0"' in (Path(os.environ['RLINF_ROOT']) / 'pyproject.toml').read_text(), 'Official pyproject was not restored'
print((root / 'report/dependency-check.txt').read_text())
PY
    df -h "$ROBOTWIN_WORKSPACE"
    ;;
  nccl)
    python - <<'PY'
import os
import signal
import subprocess
for line in subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,args='], text=True).splitlines():
    pid, command = line.strip().split(None, 1)
    if command.startswith('/home/zhsu/jyluo/experiments/robotwin_pi0/.venv/bin/python ') and '/torchrun ' in command and 'robotwin_pi0/scripts/check_nccl.py' in command:
        os.kill(int(pid), signal.SIGTERM)
PY
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_CUMEM_HOST_ENABLE=0
    export NCCL_CUMEM_ENABLE=0
    timeout --kill-after=10 100 torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29619 "$PROJECT_DIR/scripts/check_nccl.py" > "$PROJECT_DIR/logs/nccl-check-cumem.log" 2>&1
    cat "$PROJECT_DIR/logs/nccl-check-cumem.log"
    ;;
  unpack)
    python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import zipfile
assets = Path(os.environ['ROBOTWIN_PATH']) / 'assets'
expected = json.loads((Path(os.environ['ROBOTWIN_EXPERIMENT_ROOT']) / 'source_materials/assets-tree.json').read_text())
for item in expected:
    if item['path'] not in {'embodiments.zip', 'objects.zip', 'background_texture.zip'}:
        continue
    path = assets / item['path']
    assert path.stat().st_size == item['size'], path
    checksum = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert checksum == item['lfs']['oid'], path
    print(f'Checksum verified: {path.name}', flush=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(assets)
    print(f'Extracted: {path.name}', flush=True)
PY
    cd "$ROBOTWIN_PATH"
    python script/update_embodiment_config_path.py
    ;;
  inspect)
    python -m pip check
    ;;
  compatibility)
    uv pip install 'transformers==4.53.2' 'peft==0.17.1' trimesh rtree
    uv pip install --no-deps --reinstall-package rlinf-transformer-openpi 'rlinf-transformer-openpi==4.53.2'
    uv pip freeze > "$PROJECT_DIR/report/requirements-lock.txt"
    python -m pip check
    mkdir -p "$OPENPI_DATA_HOME/big_vision"
    cp "$PROJECT_DIR/.cache/openpi/paligemma_tokenizer.model" "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
    ;;
  ffmpeg)
    export CONDA_PKGS_DIRS="$PROJECT_DIR/.cache/conda"
    export CONDA_REMOTE_READ_TIMEOUT_SECS=120
    export NO_PROXY="${NO_PROXY},mirrors.tuna.tsinghua.edu.cn"
    export no_proxy="$NO_PROXY"
    "$ROBOTWIN_WORKSPACE/miniforge3/bin/conda" install -y -p "$PROJECT_DIR/toolchain" 'ffmpeg>=6,<8' --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge > "$PROJECT_DIR/logs/ffmpeg_install.log" 2>&1
    tail -20 "$PROJECT_DIR/logs/ffmpeg_install.log"
    ;;
  finish)
    python - <<'PY'
import os
from pathlib import Path
import signal
import subprocess
import time
processes = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,args='], text=True).splitlines()
for line in processes:
    pid, command = line.strip().split(None, 1)
    if command == 'bash /home/zhsu/jyluo/experiments/robotwin_pi0/scripts/install_env.sh' or command == 'bash /home/zhsu/jyluo/projects/RLinf/requirements/embodied/download_assets.sh --assets openpi' or ('/.venv/bin/hf download RLinf/openpi_tokenizer' in command and command.startswith('/home/zhsu/jyluo/experiments/robotwin_pi0/.venv/bin/python ')):
        os.kill(int(pid), signal.SIGTERM)
time.sleep(2)
root = Path(os.environ['ROBOTWIN_EXPERIMENT_ROOT'])
tokenizer = Path(os.environ['ROBOTWIN_WORKSPACE']) / 'myopenpi/dataset/ckpt/big_vision/paligemma_tokenizer.model'
assert tokenizer.is_file(), tokenizer
destination = root / '.cache/openpi/paligemma_tokenizer.model'
destination.parent.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy2(tokenizer, destination)
print(f'Tokenizer copied into {destination}', flush=True)
PY
    bash "$PROJECT_DIR/scripts/install_env.sh" > "$PROJECT_DIR/logs/env_install_finish.log" 2>&1
    tail -35 "$PROJECT_DIR/logs/env_install_finish.log"
    ;;
  diagnose)
    rg -n 'ray_tracing_denoiser|OIDN|denoiser|render_config|shader' "$ROBOTWIN_PATH/envs/_base_task.py" "$ROBOTWIN_PATH/envs/camera/camera.py"
    rg -n 'TOKENIZER|tokenizer.model|OPENPI_DATA_HOME|\.cache/openpi' "$RLINF_ROOT/rlinf/models/embodiment/openpi" "$VIRTUAL_ENV/lib/python3.11/site-packages/openpi/models/tokenizer.py"
    ps -u "$(id -un)" -o pid,ppid,etime,args | rg 'hf download|download_assets|install_env|check_nccl|curl --noproxy'
    python - <<'PY'
import importlib.metadata as m
for name in ['rlinf-openpi', 'rlinf-transformer-openpi', 'transformers', 'peft', 'tokenizers']:
    print(name, m.version(name))
    print([r for r in m.requires(name) if 'extra ==' not in r])
PY
    ;;
  *)
    echo "Usage: $0 {status|check|task|ray|nccl|unpack|inspect|compatibility|ffmpeg|finish|diagnose|finalize}" >&2
    exit 2
    ;;
esac

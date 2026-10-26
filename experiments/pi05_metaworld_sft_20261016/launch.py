"""Set CUDA shared-library search path before importing the RLinf PyTorch build."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DEPENDENCIES=Path('/home/zhsu/jyluo/experiments/pi0_metaworld_sft_hf_20261016/deps/nvidia')
env=os.environ.copy()
libs=[str(path) for path in DEPENDENCIES.glob('*/lib')]
libs.append('/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib')
env['LD_LIBRARY_PATH']=':'.join(libs+[env.get('LD_LIBRARY_PATH','')]).rstrip(':')
env.setdefault('CUDA_VISIBLE_DEVICES','1')
env.setdefault('MUJOCO_EGL_DEVICE_ID','1')
env.setdefault('TORCH_COMPILE_DISABLE','1')
env.setdefault('JAX_PLATFORMS','cpu')
env.setdefault('OMP_NUM_THREADS','4')
raise SystemExit(subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),*sys.argv[1:]],env=env).returncode)

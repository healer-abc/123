"""Set CUDA libraries before importing the PyTorch wheel used by OpenPI."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT.parent / "pi0_metaworld_sft_hf_20261016/deps/nvidia"
environment = os.environ.copy()
libraries = [str(path) for path in DEPENDENCIES.glob("*/lib")]
libraries.append("/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib")
environment["LD_LIBRARY_PATH"] = ":".join(libraries + [environment.get("LD_LIBRARY_PATH", "")]).rstrip(":")
environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
environment.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
environment.setdefault("MUJOCO_GL", "egl")
environment.setdefault("PYOPENGL_PLATFORM", "egl")
environment.setdefault("JAX_PLATFORMS", "cpu")
environment.setdefault("TORCH_COMPILE_DISABLE", "1")
environment.setdefault("OMP_NUM_THREADS", "4")
raise SystemExit(subprocess.run([sys.executable, "-u", str(ROOT / "rollout.py"), *sys.argv[1:]],
                                env=environment).returncode)

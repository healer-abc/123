"""Probe Mesa software rendering without loading the NVIDIA EGL vendor."""
import os
from pathlib import Path
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LP_NUM_THREADS"] = "2"

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
from OpenGL import GL

root = Path(__file__).resolve().parents[1]
gym.register_envs(gymnasium_robotics)
env = gym.make("FetchPush-v4", render_mode="rgb_array")
try:
    env.reset(seed=99)
    frame = env.render()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    assert "llvmpipe" in renderer.lower() or "softpipe" in renderer.lower(), renderer
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
    assert frame.std() > 0
    imageio.imwrite(root / "checks/cpu_render.png", frame)
    report = {"status": "passed", "renderer": renderer, "frame_shape": list(frame.shape),
              "egl_vendor": os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"],
              "software_rendering": True}
    (root / "checks/cpu_rendering.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
finally:
    env.close()

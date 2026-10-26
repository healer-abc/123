"""Isolated baseline runtime. Import before torch or MuJoCo."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED = Path('/home/zhsu/dp3_ablation_v52')
PYTHON = SHARED / 'env/bin/python'
DP3 = ROOT / 'sources/dp3/3D-Diffusion-Policy-master'
DP = ROOT / 'sources/dp/diffusion_policy-main'
ACT = ROOT / 'runtime_act'
TASKS = ['shelf-place', 'pick-place-wall', 'pick-place', 'stick-push', 'door-open']
METHODS = ['DP', 'ACT', 'SimpleDP3']
PREVIOUS = ROOT.parent / 'metaworld_official_baselines_20261019'
DEMONSTRATIONS = 50
TRAIN_EPISODES = 40
ADDITIONAL_SEED = 43


def shared_split(n_episodes):
    import json
    import numpy as np
    split = json.loads((ROOT / 'manifests/dataset_split.json').read_text())
    assert n_episodes == split['episodes'] == DEMONSTRATIONS
    train, val = split['train_episode_ids'], split['validation_episode_ids']
    assert len(train) == TRAIN_EPISODES and len(val) == DEMONSTRATIONS-TRAIN_EPISODES
    assert set(train).isdisjoint(val) and set(train+val) == set(range(n_episodes))
    mask = np.zeros(n_episodes, dtype=bool)
    mask[train] = True
    return mask


def environment(gpu):
    e = os.environ.copy()
    repo = SHARED / 'project/3D-Diffusion-Policy'
    paths = [ROOT, ROOT / 'deps', DP3 / '3D-Diffusion-Policy', DP, ACT, ACT / 'detr',
             DP3 / 'third_party/Metaworld']
    paths += [repo / 'third_party' / p for p in
              ['mujoco-py-2.1.2.14', 'gym-0.21.0', 'pytorch3d_simplified',
               'dexart-release', 'rrl-dependencies/mj_envs', 'rrl-dependencies/mjrl']]
    e.update(PYTHONPATH=os.pathsep.join(map(str, paths)),
             PATH=str(SHARED / 'env/bin') + ':' + e.get('PATH', ''),
             CONDA_PREFIX=str(SHARED / 'env'), CUDA_HOME=str(SHARED / 'env'),
             CUDA_VISIBLE_DEVICES=str(gpu),
             MUJOCO_PY_MUJOCO_PATH=str(SHARED / 'runtime/mujoco210'),
             LD_LIBRARY_PATH=':'.join(map(str, [SHARED / 'runtime/mujoco210/bin',
                 SHARED / 'env/lib', SHARED / 'env/lib/python3.10/site-packages/torch/lib']))
                 + ':' + e.get('LD_LIBRARY_PATH', ''),
             MUJOCO_GL='egl', MUJOCO_PY_FORCE_CPU='1', LIBGL_ALWAYS_SOFTWARE='1',
             GALLIUM_DRIVER='llvmpipe',
             __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
             MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
             WANDB_MODE='offline', WANDB_SILENT='true',
             TORCH_HOME=str(ROOT / 'torch_cache'))
    return e


def bootstrap():
    for p in environment(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))['PYTHONPATH'].split(':')[::-1]:
        if p not in sys.path:
            sys.path.insert(0, p)


def software_device():
    import ctypes
    import ctypes.util
    egl = ctypes.CDLL(ctypes.util.find_library('EGL'))
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(
            egl.eglGetProcAddress(b'eglQueryDevicesEXT'))
    string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress(b'eglQueryDeviceStringEXT'))
    devices, n = (ctypes.c_void_p * 8)(), ctypes.c_int()
    assert query(8, devices, ctypes.byref(n))
    indexes = [i for i in range(n.value) if b'EGL_MESA_device_software' in
               (string(devices[i], 0x3055) or b'').split()]
    assert len(indexes) == 1, indexes
    return indexes[0]


def install_render_fix():
    from diffusion_policy_3d.env import MetaWorldEnv
    if getattr(MetaWorldEnv, '_baseline_render_fix', False):
        return
    original = MetaWorldEnv.__init__
    def init(self, *a, **kw):
        original(self, *a, **kw)
        self.device_id = software_device()
    MetaWorldEnv.__init__ = init
    MetaWorldEnv._baseline_render_fix = True


def atomic_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

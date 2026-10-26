#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
cd "$RLINF_ROOT"

# Keep the official installer intact. Use its install flow and dependency patches,
# selecting the MPLib backend already used by the official adjust_bottle config.
INSTALL_FUNCTIONS="$PROJECT_DIR/source_materials/install.functions.sh"
sed '/^main "\$@"$/d' requirements/install.sh > "$INSTALL_FUNCTIONS"
source "$INSTALL_FUNCTIONS"
SCRIPT_PATH="$RLINF_ROOT/requirements/install.sh"
SCRIPT_DIR="$RLINF_ROOT/requirements"
PYPROJECT_FILE="$RLINF_ROOT/pyproject.toml"
configure_nvidia() {
    export UV_TORCH_BACKEND=cu128
    PLATFORM_TORCH_STR="+cu128"
    PLATFORM_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    PLATFORM_TORCH_PACKAGES=(torch torchvision torchaudio)
}
install_robotwin_env() {
    uv pip install mplib==0.2.1 gymnasium==0.29.1 av open3d 'zarr<3' openai "$SAPIEN_SPEC" toppra transforms3d h5py trimesh rtree
    local sapien_dir mplib_dir
    sapien_dir="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("sapien").submodule_search_locations)))')"
    mplib_dir="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("mplib").submodule_search_locations)))')"
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$sapien_dir/wrapper/urdf_loader.py"
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$mplib_dir/planner.py"
}
export UV_TORCH_BACKEND=cu128
main embodied --model openpi --env robotwin --venv "$PROJECT_DIR/.venv" --python 3.11 --torch 2.7.1 --no-root --no-flash-attn
uv pip install --no-deps -e "$RLINF_ROOT"
uv pip install 'transformers==4.53.2' 'peft==0.17.1'
# The OpenPI fork owns patched files in the shared transformers namespace.
uv pip install --no-deps --reinstall-package rlinf-transformer-openpi 'rlinf-transformer-openpi==4.53.2'
# Separate target: upgrading the venv package would violate Torch's exact pin.
uv pip install --target "$PROJECT_DIR/runtime/nccl-2.31.2" --no-deps 'nvidia-nccl-cu12==2.31.2'
uv pip install 'cuda-toolkit[nvcc]==12.8.0' 'warp-lang==1.11.1' 'numpy==1.26.4' 'trimesh==5.2.0' setuptools-scm yourdfpy numpy-quaternion scikit-image importlib-resources
bash "$PROJECT_DIR/scripts/build_curobo.sh"
uv pip freeze > "$PROJECT_DIR/report/requirements-lock.txt"

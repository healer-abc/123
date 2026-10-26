"""Check the isolated runtime, both CUDA devices, and headless SAPIEN rendering."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
    assert Path(sys.prefix).resolve() == (root / ".venv").resolve(), sys.prefix
    result = {"python": sys.version, "prefix": sys.prefix, "packages": {}, "gpus": []}
    for package in ["torch", "torchvision", "ray", "rlinf", "rlinf-openpi", "sapien", "mplib", "gymnasium", "peft", "datasets", "lerobot", "torchcodec"]:
        result["packages"][package] = importlib.metadata.version(package)
    for module in ["torchvision", "ray", "rlinf", "openpi", "sapien", "mplib", "peft", "lerobot", "torchcodec", "envs.adjust_bottle", "robotwin.envs.vector_env", "rlinf.models.embodiment.openpi", "rlinf.data.datasets.openpi.official_sft_data_loader"]:
        importlib.import_module(module)
    video_path = root / "report" / "codec-probe.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5", "-t", "1", "-pix_fmt", "yuv420p", str(video_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path))
    assert decoder[0].shape == (3, 64, 64), decoder[0].shape
    result["video_decode"] = {"frames": len(decoder), "frame_shape": list(decoder[0].shape)}
    assert torch.cuda.device_count() == 2, torch.cuda.device_count()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            layer = torch.nn.Linear(64, 32, device=f"cuda:{index}", dtype=torch.bfloat16)
            optimizer = torch.optim.AdamW(layer.parameters())
            before = layer.weight.detach().clone()
            loss = layer(torch.randn(4, 64, device=f"cuda:{index}", dtype=torch.bfloat16)).float().square().mean()
            loss.backward()
            optimizer.step()
            assert not torch.equal(before, layer.weight)
            torch.cuda.synchronize()
            result["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index), "capability": torch.cuda.get_device_capability(index), "bf16_backward": True})
    import sapien
    scene = sapien.Scene()
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1])
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial()
    material.base_color = [1, 0, 0, 1]
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=material)
    box = builder.build_kinematic(name="probe")
    box.set_pose(sapien.Pose([0, 0, 0.1]))
    camera = scene.add_camera("probe", 64, 64, 1.0, 0.01, 10)
    camera.set_local_pose(sapien.Pose([-1, 0, 0.5], [0.973249, 0, 0.229753, 0]))
    scene.step()
    scene.update_render()
    camera.take_picture()
    rgb = camera.get_picture("Color")
    assert rgb.shape == (64, 64, 4) and np.isfinite(rgb).all(), rgb.shape
    assert float(rgb[..., :3].std()) > 1e-4, "Renderer returned a constant image"
    result["sapien_render"] = {"shape": list(rgb.shape), "rgb_std": float(rgb[..., :3].std())}
    result["status"] = "passed"
    output = root / "report" / "environment-check.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

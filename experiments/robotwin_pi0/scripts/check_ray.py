"""Verify Ray can reserve the two GPUs in separate workers of this environment."""
import json
import os
from pathlib import Path

import ray


@ray.remote(num_gpus=1, num_cpus=1)
class GPUProbe:
    def check(self):
        import sys
        import torch

        value = torch.ones(4, device="cuda")
        return {"gpu_ids": ray.get_gpu_ids(), "prefix": sys.prefix, "name": torch.cuda.get_device_name(0), "sum": value.sum().item()}


ray.init(num_cpus=4, num_gpus=2, include_dashboard=False, object_store_memory=256 * 1024**2, _temp_dir=os.environ["RAY_TMPDIR"])
try:
    workers = [GPUProbe.remote() for _ in range(2)]
    results = ray.get([worker.check.remote() for worker in workers], timeout=90)
    assert len({r["gpu_ids"][0] for r in results}) == 2, results
    assert all(r["prefix"] == os.environ["VIRTUAL_ENV"] and r["sum"] == 4 for r in results), results
    report = {"status": "passed", "workers": results}
    (Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"]) / "report/ray-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
finally:
    ray.shutdown()

"""Reproduce FSDP's initial parameter broadcast without a model or simulator."""
import argparse
import ctypes
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["mixed", "fp32", "bf16"], default="mixed")
args = parser.parse_args()
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
runtime_version = ctypes.c_int()
# Torch's version() reports its build version even when LD_PRELOAD overrides it.
try:
    get_version = ctypes.CDLL(None).ncclGetVersion
except AttributeError:
    bundled = Path(torch.__file__).resolve().parents[1] / "nvidia/nccl/lib/libnccl.so.2"
    get_version = ctypes.CDLL(str(bundled)).ncclGetVersion
assert get_version(ctypes.byref(runtime_version)) == 0
print(f"rank={rank} start, torch={torch.__version__}, nccl_build={torch.cuda.nccl.version()}, nccl_runtime={runtime_version.value}", flush=True)
for count in [1024, 1024 * 1024, 32 * 1024 * 1024]:
    dtypes = [torch.float32, torch.bfloat16] if args.dtype == "mixed" else [torch.float32 if args.dtype == "fp32" else torch.bfloat16]
    tensors = [torch.full((count,), rank + 1.0, dtype=dtype, device=f"cuda:{rank}") for dtype in dtypes]
    print(f"rank={rank} broadcast elements={count}, dtypes={dtypes}", flush=True)
    dist._broadcast_coalesced(dist.group.WORLD, tensors, 250 * 1024 * 1024, 0)
    torch.cuda.synchronize()
    assert all(torch.all(tensor == 1).item() for tensor in tensors)
    print(f"rank={rank} broadcast passed elements={count}", flush=True)
    del tensors
dist.destroy_process_group()

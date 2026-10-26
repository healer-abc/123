"""Run with torchrun --standalone --nproc-per-node=2 to verify GPU collectives."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=90))
value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
assert value.item() == 3, value
print(f"rank={rank} NCCL all_reduce passed, sum={value.item()}", flush=True)
model = FSDP(torch.nn.Linear(64, 32).to(rank), device_id=rank)
optimizer = torch.optim.AdamW(model.parameters())
loss = model(torch.randn(4, 64, device=f"cuda:{rank}")).square().mean()
loss.backward()
optimizer.step()
dist.barrier()
print(f"rank={rank} FSDP backward and optimizer step passed", flush=True)
dist.destroy_process_group()

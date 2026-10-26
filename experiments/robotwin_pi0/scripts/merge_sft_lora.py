"""Merge RLinf's injected linear LoRA checkpoint into a bare Pi0 checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--norm-stats", required=True, type=Path)
parser.add_argument("--dcp", action="store_true", help="Read model tensors from a native DCP training checkpoint")
parser.add_argument("--base-model", type=Path, help="Check frozen tensors against this native Pi0 safetensors file")
args = parser.parse_args()
assert not args.output.exists(), args.output
if args.dcp:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    reader = dcp.FileSystemReader(args.checkpoint)
    prefix = "fsdp_checkpoint.model."
    state = {key[len(prefix):]: torch.empty(meta.size, dtype=meta.properties.dtype)
             for key, meta in reader.read_metadata().state_dict_metadata.items()
             if key.startswith(prefix) and isinstance(meta, TensorStorageMetadata)}
    assert state, f"No Pi0 model tensors in {args.checkpoint}"
    dcp.load({"fsdp_checkpoint": {"model": state}}, storage_reader=reader)
else:
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
frozen_check = None
if args.base_model:
    from safetensors.torch import load_file
    base = load_file(str(args.base_model))
    checked = set()
    for key, value in state.items():
        assert torch.isfinite(value).all(), f"Nonfinite trained tensor: {key}"
        if ".lora_" in key:
            continue
        clean = key.replace(".base_layer.", ".")
        assert clean in base and torch.equal(value, base[clean]), f"Frozen base changed: {clean}"
        checked.add(clean)
    assert checked == base.keys(), "Frozen tensor set differs from the initialization"
    frozen_check = {"base_model": str(args.base_model.resolve()), "unchanged_tensors": len(checked)}
    del base
merged = {}
checks = []
for key, value in state.items():
    if ".lora_" in key:
        continue
    clean = key.replace(".base_layer.", ".")
    if key.endswith(".base_layer.weight"):
        prefix = key.removesuffix("base_layer.weight")
        a = state[prefix + "lora_A.default.weight"].float()
        b = state[prefix + "lora_B.default.weight"].float()
        # RLinf's builder sets lora_alpha == lora_rank and dropout == 0.
        delta = b @ a
        updated = value.float() + delta
        merged[clean] = updated.to(value.dtype)
        if len(checks) < 3:
            generator = torch.Generator().manual_seed(0)
            x = torch.randn(2, a.shape[1], generator=generator)
            expected = F.linear(x, value.float()) + F.linear(F.linear(x, a), b)
            actual = F.linear(x, updated)
            error = float((expected - actual).abs().max())
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            checks.append({"weight": clean, "merge_float32_error": error, "adapter_max_abs": float(delta.abs().max()), "changed_after_cast": int((merged[clean] != value).sum())})
    else:
        if clean in merged:
            raise ValueError(f"Duplicate merged key {clean}")
        merged[clean] = value
assert checks and any(c["adapter_max_abs"] > 0 for c in checks), "No trained LoRA updates found"
assert not any("lora_" in key or "base_layer" in key for key in merged)
args.output.mkdir(parents=True)
torch.save(merged, args.output / "full_weights.pt")
torch.save({key: value for key, value in state.items() if ".lora_" in key}, args.output / "adapter_weights.pt")
shutil.copy2(args.norm_stats, args.output / "norm_stats.json")
record = {"source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_format": "dcp" if args.dcp else "torch", "scaling": 1.0, "checks": checks, "tensor_count": len(merged), "norm_stats_source": str(args.norm_stats.resolve()), "frozen_base_check": frozen_check}
(args.output / "merge-check.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))

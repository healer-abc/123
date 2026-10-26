#!/usr/bin/env python3
"""Export parameter-level trainability for one or more OpenPI train configs."""

import argparse
import csv
from pathlib import Path

import flax.nnx as nnx
import jax

from openpi.training import config as train_config


def classify(path: str) -> tuple[str, bool, bool]:
    is_lora = "lora" in path.lower()
    is_cond_norm = "ada" in path.lower() or "cond" in path.lower()
    if path.startswith("PaliGemma/img/"):
        module = "vision_encoder"
    elif path.startswith("PaliGemma/llm/"):
        module = "action_branch" if "_1" in path else "language_branch"
    elif any(name in path for name in ("action_in_proj", "action_out_proj", "time_mlp")):
        module = "action_time_projection"
    else:
        module = "other"
    return module, is_lora, is_cond_norm


def export_one(name: str, output: Path) -> None:
    config = train_config.get_config(name)
    model = nnx.eval_shape(config.model.create, jax.random.key(config.seed))
    params = nnx.state(model, nnx.Param)
    trainable_paths = set(params.filter(config.trainable_filter).flat_state().keys())

    rows = []
    for path_parts, variable in params.flat_state().items():
        path = "/".join(str(part) for part in path_parts)
        shape = tuple(int(dim) for dim in variable.value.shape)
        count = 1
        for dim in shape:
            count *= dim
        module, is_lora, is_cond_norm = classify(path)
        rows.append(
            {
                "config": name,
                "path": path,
                "shape": "x".join(map(str, shape)),
                "dtype": str(variable.value.dtype),
                "parameter_count": count,
                "module": module,
                "is_lora": str(is_lora).lower(),
                "is_conditional_norm": str(is_cond_norm).lower(),
                "trainable": str(path_parts in trainable_paths).lower(),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = sum(row["parameter_count"] for row in rows)
    trainable = sum(row["parameter_count"] for row in rows if row["trainable"] == "true")
    print(f"{name}: total={total:,} trainable={trainable:,} ({trainable / total:.4%}) -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Train config names")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in args.configs:
        export_one(name, args.output_dir / f"parameter_audit_{name}.csv")


if __name__ == "__main__":
    main()

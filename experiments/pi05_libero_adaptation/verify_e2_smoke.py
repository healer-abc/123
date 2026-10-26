#!/usr/bin/env python3
"""Verify E2's 10-step checkpoint against its exact seeded initialization."""

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
from flax import traverse_util
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.training import config as train_config


def flatten(tree):
    return traverse_util.flatten_dict(tree, sep="/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = train_config.get_config("pi05_libero_lora_frozen_vision")

    # Reproduce scripts/train.py's RNG splitting exactly.
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    initial = nnx.state(model).to_pure_dict()
    initial = config.weight_loader.load(initial)
    initial_flat = flatten(initial)

    trained = model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray)
    trained_again = model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray)
    trained_flat = flatten(trained)
    trained_again_flat = flatten(trained_again)

    vision = []
    frozen_llm = []
    trainable_lora = []
    trainable_projection = []
    reload_equal = []

    audit_model = nnx.eval_shape(config.model.create, jax.random.key(config.seed))
    params = nnx.state(audit_model, nnx.Param)
    trainable_paths = {"/".join(map(str, p)) for p in params.filter(config.trainable_filter).flat_state().keys()}

    for path, after in trained_flat.items():
        before = initial_flat[path]
        before = np.asarray(before, dtype=after.dtype)
        equal = np.array_equal(before, after)
        record = {"path": path, "equal": bool(equal), "max_abs_delta": float(np.max(np.abs(after.astype(np.float32) - before.astype(np.float32))))}
        if path.startswith("PaliGemma/img/"):
            vision.append(record)
        elif path not in trainable_paths:
            frozen_llm.append(record)
        elif "lora" in path.lower():
            trainable_lora.append(record)
        else:
            trainable_projection.append(record)
        reload_equal.append(np.array_equal(after, trained_again_flat[path]))

    result = {
        "checkpoint": str(args.checkpoint),
        "seed": config.seed,
        "vision_leaf_count": len(vision),
        "vision_all_unchanged": all(item["equal"] for item in vision),
        "other_frozen_leaf_count": len(frozen_llm),
        "other_frozen_all_unchanged": all(item["equal"] for item in frozen_llm),
        "trainable_lora_leaf_count": len(trainable_lora),
        "trainable_lora_changed_count": sum(not item["equal"] for item in trainable_lora),
        "trainable_projection_leaf_count": len(trainable_projection),
        "trainable_projection_changed_count": sum(not item["equal"] for item in trainable_projection),
        "double_restore_all_equal": all(reload_equal),
        "unexpected_changed_frozen": [item for item in vision + frozen_llm if not item["equal"]],
        "unchanged_trainable": [item for item in trainable_lora + trainable_projection if item["equal"]],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert result["vision_all_unchanged"]
    assert result["other_frozen_all_unchanged"]
    assert result["trainable_lora_changed_count"] == result["trainable_lora_leaf_count"]
    assert result["trainable_projection_changed_count"] == result["trainable_projection_leaf_count"]
    assert result["double_restore_all_equal"]


if __name__ == "__main__":
    main()

"""Compare E3 smoke weights with pi05_base and verify deterministic reload inference."""
import argparse
import gc
import json
from pathlib import Path

import flax.nnx as nnx
from flax import traverse_util
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.training import config as train_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = train_config.get_config("pi05_libero_action_only")
    shape = nnx.eval_shape(config.model.create, jax.random.key(config.seed))
    params = nnx.state(shape, nnx.Param)
    trainable = {"/".join(map(str, p)) for p in params.filter(config.trainable_filter).flat_state()}
    expected = traverse_util.flatten_dict(config.weight_loader.load(params.to_pure_dict()), sep="/")
    trained = traverse_util.flatten_dict(model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray), sep="/")
    assert expected.keys() == trained.keys()
    frozen, updated = [], []
    for path, after in trained.items():
        before = np.asarray(expected[path], dtype=after.dtype)
        assert np.isfinite(after).all(), path
        equal = bool(np.array_equal(before, after))
        record = {"path": path, "equal": equal,
                  "max_abs_delta": float(np.max(np.abs(after.astype(np.float32) - before.astype(np.float32))))}
        (updated if path in trainable else frozen).append(record)
    result = {"config": config.name, "checkpoint": str(args.checkpoint),
              "base": config.weight_loader.params_path, "frozen_leaves": len(frozen),
              "frozen_all_unchanged": all(r["equal"] for r in frozen),
              "trainable_leaves": len(updated), "changed_trainable_leaves": sum(not r["equal"] for r in updated),
              "leaves": frozen + updated}
    assert result["frozen_all_unchanged"], [r for r in frozen if not r["equal"]]
    assert result["changed_trainable_leaves"] == len(updated), [r for r in updated if r["equal"]]
    del expected, trained, params, shape
    gc.collect()
    observation = {"observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                   "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                   "observation/state": np.zeros(8, dtype=np.float32),
                   "prompt": "pick up the black bowl and place it on the plate"}
    noise = np.random.default_rng(42).standard_normal((10, 32), dtype=np.float32)
    outputs = []
    for _ in range(2):
        policy = policy_config.create_trained_policy(config, args.checkpoint, sample_kwargs={"num_steps": 10})
        outputs.append(np.asarray(policy.infer(observation, noise=noise)["actions"]))
        del policy
        gc.collect()
        jax.clear_caches()
    assert all(a.shape == (10, 7) and np.isfinite(a).all() for a in outputs)
    result["reload_inference_max_abs_delta"] = float(np.max(np.abs(outputs[0] - outputs[1])))
    result["reload_inference_atol"] = 1e-5
    assert np.allclose(outputs[0], outputs[1], atol=1e-5, rtol=0)
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "leaves"}, indent=2))


if __name__ == "__main__":
    main()

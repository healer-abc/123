"""Verify the original kit's saved artifacts and action-gain pairing on CPU."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.logger import configure
import run

torch.set_num_threads(2)


def verify_bundle(path, expected_steps):
    model, config, env = run.load_bundle(path, "cpu", training=True)
    try:
        assert model.num_timesteps == expected_steps
        assert json.loads((path / "complete.json").read_text())["steps"] == expected_steps
        assert str(model.device) == "cpu"
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        initial_updates = model._n_updates
        assert initial_updates > 0
        if config["group"] != "A":
            assert model.replay_buffer.size() == expected_steps
            if config["group"] in ("D", "E"):
                assert torch.isfinite(model.replay_buffer.sample(256).rewards).all()
            model.set_logger(configure(format_strings=[]))
            model.train(gradient_steps=1, batch_size=256)
            assert model._n_updates == initial_updates + 1
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        return {"steps": expected_steps, "updates_before_verification": initial_updates,
                "finite_parameters": True, "load_passed": True,
                "restored_update_passed": config["group"] != "A"}
    finally:
        env.close()


def main():
    provenance = json.loads((ROOT / "source_materials/archive_manifest.json").read_text())
    changes = json.loads((ROOT / "source_materials/local_changes.json").read_text())
    for name, digest in provenance["files_sha256"].items():
        digest = changes.get(name, {}).get("modified_sha256", digest)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    report = {"status": "passed", "source_audit_passed": True,
              "original_kit_modified_files": list(changes), "bundles": {}}
    specs = {"reach_smoke": 3000, "reach_resume": 5000,
             **{f"smoke_{g}_push": 2048 if g == "A" else 300 for g in "ABCDE"}}
    for name, expected in specs.items():
        directory = ROOT / "runs" / name
        report["bundles"][name] = verify_bundle(directory / "final", expected)
        assert json.loads((directory / "summary.json").read_text())["timesteps"] == expected
        with (directory / "progress.csv").open() as file:
            for row in csv.DictReader(file):
                for value in row.values():
                    if value:
                        assert np.isfinite(float(value)), (name, row)
    resumed = json.loads((ROOT / "runs/reach_resume/summary.json").read_text())
    assert resumed["session_steps"] == 2000
    for phase, steps in (("reach_smoke", (1000, 2000, 3000)), ("reach_resume", (4000, 5000))):
        for step in steps:
            directory = ROOT / "runs" / phase / "checkpoints" / f"step_{step}"
            model, _, env = run.load_bundle(directory, "cpu", training=True)
            assert model.num_timesteps == step
            assert torch.isfinite(model.replay_buffer.sample(256).rewards).all()
            env.close()
    report["periodic_checkpoints_loadable"] = True
    report["resume_added_steps"] = resumed["session_steps"]
    report["paired_initial_states_across_gains"] = []
    config = {"group": "E", "task": "Push"}
    reference = run.make_env(config, gain=1.0)
    initial, _ = reference.reset(seed=20000)
    reference.close()
    for gain in (0.6, 0.8, 1.0, 1.2, 1.4):
        env = run.make_env(config, gain=gain)
        obs, _ = env.reset(seed=20000)
        for key in obs:
            np.testing.assert_array_equal(obs[key], initial[key])
        assert env.gain == gain
        env.close()
        report["paired_initial_states_across_gains"].append(gain)
    env = run.make_env(config, training=True)
    env.reset(seed=99)
    sampled_gain = env.gain
    assert 0.8 <= sampled_gain <= 1.2
    for _ in range(5):
        env.step(np.zeros(4, dtype=np.float32))
        assert env.gain == sampled_gain
    env.reset(seed=99)
    assert env.gain == sampled_gain
    env.close()
    report["randomized_gain_seeded_and_fixed_within_episode"] = True
    report["training_quality_validated"] = False
    (ROOT / "checks/acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

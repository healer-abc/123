"""Run an isolated 20-warmup + 200-measurement training profile with real data."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
OPENPI = ROOT.parents[1] / "projects/openpi"
sys.path.insert(0, str(OPENPI / "scripts"))
import train
from openpi.training import config as train_config
from profile_hooks import install_metric_observer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = dataclasses.replace(train_config.get_config(args.config), exp_name=args.exp_name,
                                 num_train_steps=220, batch_size=32, log_interval=1, save_interval=220)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    save_original = train._checkpoints.save_state

    def record(value):
        with args.output.open("a") as file:
            file.write(json.dumps({"timestamp": time.time(), **value}, allow_nan=False) + "\n")

    def observe(data, step):
        nonlocal previous
        if "loss" in data:
            instant = time.monotonic()
            record({"event": "step", "step": step,
                    "elapsed_since_previous_step": None if previous is None else instant - previous,
                    "measured": step >= 20, "loss": float(data["loss"]),
                    "grad_norm": float(data["grad_norm"])})
            previous = instant

    def save(*values, **kwargs):
        record({"event": "save_begin"})
        result = save_original(*values, **kwargs)
        record({"event": "save_blocking_end"})
        return result

    install_metric_observer(train, observe)
    train._checkpoints.save_state = save
    record({"event": "profile_start", "config": config.name, "checkpoint": str(config.checkpoint_dir),
            "warmup_steps": 20, "measured_steps": 200, "allocator_preallocate": False,
            "note": "per-step timing synchronizes through stock training metric device_get"})
    train.main(config)
    record({"event": "profile_complete", "checkpoint_finalized": str(config.checkpoint_dir / "219")})


if __name__ == "__main__":
    main()

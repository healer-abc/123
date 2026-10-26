"""Freeze the development and confirmation LIBERO episode identities."""
import csv
import hashlib
import pathlib

import numpy as np
from libero.libero import benchmark


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "manifests/eval_manifest.csv"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def phase_indices(suite: str):
    if suite in {"libero_spatial", "libero_10"}:
        return (("development", range(0, 20)), ("confirmation", range(20, 50)))
    return (("confirmation", range(0, 50)),)


def main():
    rows = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_index, suite in enumerate(SUITES):
        task_suite = benchmark_dict[suite]()
        assert task_suite.n_tasks == 10
        max_steps = {"libero_spatial": 220, "libero_object": 280,
                     "libero_goal": 300, "libero_10": 520}[suite]
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            states = np.asarray(task_suite.get_task_init_states(task_id))
            if len(states) < 50:
                raise ValueError(f"{suite} task {task_id} has only {len(states)} states")
            for phase, indices in phase_indices(suite):
                for state_index in indices:
                    identity = suite_index * 1000 + task_id * 50 + state_index
                    rows.append({
                        "phase": phase,
                        "suite": suite,
                        "task_id": task_id,
                        "task_description": task.language,
                        "init_state_index": state_index,
                        "init_state_sha256": hashlib.sha256(
                            np.ascontiguousarray(states[state_index]).tobytes()).hexdigest(),
                        "env_seed": 100000 + identity,
                        "policy_seed": 200000 + identity,
                        "max_policy_steps": max_steps,
                        "settling_steps": 10,
                        "replan_steps": 5,
                        "flow_steps": 10,
                    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"development={sum(r['phase'] == 'development' for r in rows)} confirmation={sum(r['phase'] == 'confirmation' for r in rows)}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()

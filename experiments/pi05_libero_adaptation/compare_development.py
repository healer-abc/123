"""Paired task-stratified comparison of E2 against E1 on the frozen dev manifest."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def records(run_id):
    paths = sorted((ROOT / "evaluations" / run_id / "development/episodes").glob("*/*/*.json"))
    values = [json.loads(path.read_text()) for path in paths]
    if len(values) != 400 or not all(x["status"] == "valid" for x in values):
        raise ValueError(f"{run_id} is not a complete valid 400-episode development run")
    return {(x["suite"], x["task_id"], x["init_state_index"]): x for x in values}


def main():
    e1, e2 = records("E1-s42"), records("E2-s42")
    assert e1.keys() == e2.keys()
    tasks = sorted({(suite, task) for suite, task, _ in e1})
    rows = []
    for suite, task in tasks:
        keys = [key for key in e1 if key[:2] == (suite, task)]
        a = np.asarray([e1[key]["success"] for key in keys], dtype=float)
        b = np.asarray([e2[key]["success"] for key in keys], dtype=float)
        rows.append({"suite": suite, "task_id": task, "episodes": len(keys),
                     "e1_successes": int(a.sum()), "e2_successes": int(b.sum()),
                     "e1_rate": float(a.mean()), "e2_rate": float(b.mean()),
                     "delta_e2_minus_e1": float((b - a).mean())})
    e1_point = float(np.mean([row["e1_rate"] for row in rows]))
    e2_point = float(np.mean([row["e2_rate"] for row in rows]))
    generator = np.random.default_rng(20261016)
    deltas = np.empty(10000)
    paired = []
    for suite, task in tasks:
        keys = [key for key in e1 if key[:2] == (suite, task)]
        paired.append(np.asarray([float(e2[key]["success"]) - float(e1[key]["success"]) for key in keys]))
    for iteration in range(len(deltas)):
        deltas[iteration] = np.mean([
            difference[generator.integers(0, len(difference), len(difference))].mean()
            for difference in paired
        ])
    delta = e2_point - e1_point
    interval = np.quantile(deltas, [0.025, 0.975]).tolist()
    output = {"complete": True, "episodes_per_run": 400,
              "e1_task_equal_rate": e1_point, "e2_task_equal_rate": e2_point,
              "delta_e2_minus_e1": delta, "paired_task_stratified_bootstrap_95": interval,
              "engineering_margin": -0.05,
              "development_screen_pass": delta >= -0.05,
              "noninferiority_claim_allowed": interval[0] > -0.05,
              "tasks": rows,
              "next_step": ("E2 retained E1 within the preregistered development point-estimate margin; "
                            "review task-level regressions, then register the action-only E3 branch if E3 remains useful."
                            if delta >= -0.05 else
                            "E2 failed the development screen; review failures and register exactly one of the "
                            "adaRMSNorm or E1-learning-rate diagnostic branches before any E3 training.")}
    path = ROOT / "evaluations/development_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

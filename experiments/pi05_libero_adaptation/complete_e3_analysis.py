"""Wait for E3 development results, then compare with E1/E2 without exposing confirmation."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import development_watchdog as helpers

ROOT = Path(__file__).resolve().parent


def records(run):
    values = [json.loads(p.read_text()) for p in (ROOT / "evaluations" / run / "development/episodes").glob("*/*/*.json")]
    if len(values) != 400 or any(v["status"] != "valid" for v in values):
        raise ValueError(f"{run}: incomplete valid development evaluation")
    if any(v["manifest_sha256"] != helpers.EXPECTED_MANIFEST for v in values):
        raise ValueError(f"{run}: manifest identity mismatch")
    return {(v["suite"], v["task_id"], v["init_state_index"]): v for v in values}


def compare(candidate, reference, seed=20261018):
    assert candidate.keys() == reference.keys() and len(candidate) == 400
    task_ids = sorted({key[:2] for key in candidate})
    tasks, paired = [], []
    for suite, task in task_ids:
        keys = sorted(key for key in candidate if key[:2] == (suite, task))
        assert len(keys) == 20
        for key in keys:
            for column in ("env_seed", "policy_seed", "init_state_sha256", "max_policy_steps", "replan_steps", "flow_steps"):
                assert candidate[key][column] == reference[key][column], (key, column)
        a = np.array([reference[k]["success"] for k in keys], dtype=float)
        b = np.array([candidate[k]["success"] for k in keys], dtype=float)
        paired.append(b - a)
        tasks.append({"suite": suite, "task_id": task, "episodes": 20,
                      "reference_successes": int(a.sum()), "candidate_successes": int(b.sum()),
                      "delta": float((b-a).mean())})
    generator = np.random.default_rng(seed)
    replicates = np.mean([d[generator.integers(0, 20, (10000, 20))].mean(axis=1) for d in paired], axis=0)
    delta = float(np.mean([t["delta"] for t in tasks]))
    return {"environment_delta": delta, "paired_task_stratified_bootstrap_95": np.quantile(replicates, [.025, .975]).tolist(),
            "development_screen_pass": delta >= -.05, "tasks": tasks,
            "note": "development screening only; not an independent confirmation or training-seed uncertainty interval"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    lock = (ROOT / "logs/e3_analysis.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / "logs/e3_analysis.pid").write_text(str(__import__("os").getpid()) + "\n")
    summary_path = ROOT / "evaluations/E3-s42/development/summary.json"
    while not summary_path.exists() or not json.loads(summary_path.read_text()).get("complete"):
        status_path = ROOT / "logs/e3_training_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") in {"blocked", "stopped_by_signal"}:
                raise SystemExit("E3 pipeline blocked/stopped; see e3_training_status.json and alerts.jsonl")
        if not args.wait:
            raise SystemExit("E3 development is not complete")
        time.sleep(30)
    candidate = records("E3-s42")
    output = {"complete": True, "candidate": "E3-s42", "episodes": 400,
              "environment_successes": sum(v["success"] for v in candidate.values()),
              "raw_bound_strict_successes": sum(v["success"] and not v["action_out_of_nominal_bounds"] for v in candidate.values()),
              "against_E1": compare(candidate, records("E1-s42")),
              "against_E2": compare(candidate, records("E2-s42")),
              "confirmation_exposed": False,
              "next_step": "review E3 failures and uniform training profiles, freeze candidate before confirmation"}
    helpers.atomic_json(ROOT / "evaluations/e3_development_comparison.json", output)
    helpers.event("e3_analysis_complete", output=str(ROOT / "evaluations/e3_development_comparison.json"))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

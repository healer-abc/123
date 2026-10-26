"""Validate all 4800 records and compute suite-equal paired confirmation comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import development_watchdog as helpers
from confirmation_gate import confirmation_rows, validate_freeze, sha256

ROOT = helpers.ROOT


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [center - radius, center + radius]


def score(record, strict=False):
    return float(record["success"] and (not strict or not record["action_out_of_nominal_bounds"]))


def summarize(records):
    tasks = []
    for suite, task in sorted({key[:2] for key in records}):
        subset = [v for key, v in records.items() if key[:2] == (suite, task)]
        successes = int(sum(score(v) for v in subset))
        strict = int(sum(score(v, True) for v in subset))
        tasks.append({"suite": suite, "task_id": task, "episodes": len(subset), "successes": successes,
                      "environment_success_rate": successes/len(subset), "wilson_95": wilson(successes, len(subset)),
                      "strict_successes": strict, "strict_success_rate": strict/len(subset),
                      "strict_wilson_95": wilson(strict, len(subset))})
    suites = {suite: {"environment_success_rate": float(np.mean([t["environment_success_rate"] for t in tasks if t["suite"] == suite])),
                      "strict_success_rate": float(np.mean([t["strict_success_rate"] for t in tasks if t["suite"] == suite]))}
              for suite in sorted({t["suite"] for t in tasks})}
    return {"episodes": len(records), "successes": int(sum(score(v) for v in records.values())), "tasks": tasks,
            "suites": suites, "suite_equal_environment_success_rate": float(np.mean([s["environment_success_rate"] for s in suites.values()])),
            "suite_equal_strict_success_rate": float(np.mean([s["strict_success_rate"] for s in suites.values()]))}


def compare(candidate, reference, seed=20261021, replicates=10000, strict=False):
    if candidate.keys() != reference.keys():
        raise ValueError("candidate/reference identities differ")
    rng = np.random.default_rng(seed)
    suite_replicates, suite_deltas = {}, {}
    for suite, task in sorted({key[:2] for key in candidate}):
        keys = sorted(k for k in candidate if k[:2] == (suite, task))
        for key in keys:
            for column in ("env_seed", "policy_seed", "init_state_sha256", "max_policy_steps", "replan_steps", "flow_steps", "settling_steps"):
                if str(candidate[key][column]) != str(reference[key][column]):
                    raise ValueError(f"pairing mismatch: {key} {column}")
        differences = np.array([score(candidate[k], strict) - score(reference[k], strict) for k in keys])
        suite_deltas.setdefault(suite, []).append(float(differences.mean()))
        suite_replicates.setdefault(suite, []).append(differences[rng.integers(0, len(keys), (replicates, len(keys)))].mean(axis=1))
    deltas = {s: float(np.mean(v)) for s, v in suite_deltas.items()}
    distribution = np.mean([np.mean(v, axis=0) for v in suite_replicates.values()], axis=0)
    low, high = map(float, np.quantile(distribution, [.025, .975]))
    # Floor effects under the raw-action-bound criterion cannot establish a useful success-preserving claim.
    all_zero = not any(score(v, strict) for v in candidate.values()) and not any(score(v, strict) for v in reference.values())
    return {"delta_suite_equal": float(np.mean(list(deltas.values()))), "suite_deltas": deltas,
            "paired_task_stratified_bootstrap_95": [low, high], "engineering_margin": -0.05,
            "meets_environment_noninferiority_rule": low > -.05 if not strict else None,
            "strict_claim_allowed": False if strict else None, "both_metrics_at_zero_floor": all_zero,
            "suite_drop_over_10pp": [s for s, d in deltas.items() if d < -.10]}


def load_records(run, rows):
    output = ROOT / "evaluations" / run["run_id"] / "confirmation"
    paths = list((output / "episodes").glob("*/*/*.json"))
    if len(paths) != 1600:
        raise ValueError(f"{run['run_id']}: expected 1600 records, got {len(paths)}")
    result = {}
    for row in rows:
        key = row["suite"], int(row["task_id"]), int(row["init_state_index"])
        path = output / "episodes" / key[0] / f"task_{key[1]:02d}" / f"state_{key[2]:02d}.json"
        value = json.loads(path.read_text())
        if (value["status"] != "valid" or value["config"] != run["config"] or value["checkpoint"] != run["checkpoint"]
                or value["manifest_sha256"] != helpers.EXPECTED_MANIFEST
                or any(str(value.get(k)) != str(v) for k, v in row.items())):
            raise ValueError(f"invalid confirmation record: {path}")
        if sha256(value["trajectory"]) != value["trajectory_sha256"]:
            raise ValueError(f"trajectory hash mismatch: {path}")
        with np.load(value["trajectory"]) as data:
            actions = data["actions"]
            if actions.shape != (value["steps"], 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid saved actions: {path}")
            if bool((np.abs(actions) > 1.000001).any()) != value["action_out_of_nominal_bounds"]:
                raise ValueError(f"saved action-bound flag mismatch: {path}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "manifests/candidate_freeze.json")
    args = parser.parse_args()
    freeze = validate_freeze(args.freeze, ["E0-s42", "E1-s42", "E2-s42"])
    rows = confirmation_rows(helpers.MANIFEST)
    records = {run["run_id"]: load_records(run, rows) for run in freeze["runs"]}
    summaries = {run: summarize(values) for run, values in records.items()}
    comparisons = {}
    for candidate, reference in (("E2-s42", "E1-s42"), ("E1-s42", "E0-s42"), ("E2-s42", "E0-s42")):
        comparisons[f"{candidate}_minus_{reference}"] = {
            "environment": compare(records[candidate], records[reference]),
            "strict_raw_bound": compare(records[candidate], records[reference], strict=True)}
    result = {"complete": True, "created_at": helpers.now(), "freeze_sha256": sha256(args.freeze),
              "primary_comparison": "E2-s42_minus_E1-s42", "runs": summaries, "comparisons": comparisons,
              "limitations": freeze["limitations"], "development_included": False}
    full_benchmark = {}
    for run in freeze["runs"]:
        dev_paths = list((ROOT / "evaluations" / run["run_id"] / "development/episodes").glob("*/*/*.json"))
        if len(dev_paths) == 400:
            combined = dict(records[run["run_id"]])
            for path in dev_paths:
                value = json.loads(path.read_text())
                if value["status"] != "valid" or value["checkpoint"] != run["checkpoint"] or value["manifest_sha256"] != helpers.EXPECTED_MANIFEST:
                    raise ValueError(f"incompatible development supplement: {path}")
                key = value["suite"], value["task_id"], value["init_state_index"]
                if key in combined:
                    raise ValueError("development and confirmation overlap")
                combined[key] = value
            full_benchmark[run["run_id"]] = summarize(combined)
    result["separate_full_benchmark_descriptive_only"] = full_benchmark
    helpers.atomic_json(ROOT / "evaluations/confirmation_comparison.json", result)
    report = ["# E0/E1/E2 固定清单确认评测", "", f"完成时间：{helpers.now()}。每组 1600 次，训练 seed=42。", "",
              "| 模型 | Spatial | Object | Goal | Long | 四套等权环境成功率 | 严格边界成功率 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for run, summary in summaries.items():
        rates = [summary["suites"][s]["environment_success_rate"] for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
        report.append("| " + run + " | " + " | ".join(f"{r:.2%}" for r in rates + [summary["suite_equal_environment_success_rate"], summary["suite_equal_strict_success_rate"]]) + " |")
    primary = comparisons["E2-s42_minus_E1-s42"]["environment"]
    low, high = primary["paired_task_stratified_bootstrap_95"]
    report += ["", f"主要比较 E2−E1：{primary['delta_suite_equal'] * 100:.2f} 个百分点，配对分层 bootstrap 95% CI [{low * 100:.2f}, {high * 100:.2f}]。",
               "环境成功口径预登记非劣规则：" + ("满足。" if primary["meets_environment_noninferiority_rule"] else "未满足；不能声称非劣。"),
               "", "原始动作边界口径独立保留；环境成功不能替代原严格指标。", "",
               "每任务成功数和 Wilson 区间、各比较及两种计分见 confirmation_comparison.json。", "", "限制："]
    report += ["- " + text for text in freeze["limitations"]]
    if full_benchmark:
        report += ["", "完整 2000 次开发+确认的描述性汇总（不替代上方独立列出的确认估计）：", ""]
        report += [f"- {run}: {s['successes']}/2000，四套等权 {s['suite_equal_environment_success_rate']:.2%}" for run, s in full_benchmark.items()]
    (ROOT / "report/confirmation_report.md").write_text("\n".join(report) + "\n")
    for run in records:
        with (ROOT / "manifests/exposure_log.csv").open("a", newline="") as file:
            csv.writer(file).writerow([helpers.now(), run, "frozen confirmation 1600", "confirmation_complete", "post-freeze assessment", "candidate not changed after exposure"])
    helpers.event("confirmation_analysis_complete", output=str(ROOT / "report/confirmation_report.md"))
    print(json.dumps({"complete": True, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()

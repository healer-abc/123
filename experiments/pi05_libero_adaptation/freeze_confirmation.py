"""Freeze E2 based only on reviewed development evidence before any new confirmation run."""
import json
from pathlib import Path
import subprocess

import development_watchdog as helpers
from confirmation_gate import inventory, sha256, confirmation_rows

ROOT, OPENPI = helpers.ROOT, helpers.OPENPI


def main():
    target = ROOT / "manifests/candidate_freeze.json"
    if target.exists():
        raise SystemExit("candidate freeze exists; refusing replacement")
    for run in ("E0-s42", "E1-s42", "E2-s42"):
        if any((ROOT / "evaluations" / run / "confirmation").glob("episodes/*/*/*.json")):
            raise SystemExit("confirmation records already exist; freeze must precede exposure")
    audit = json.loads((ROOT / "report/final_development_review/artifact_audit.json").read_text())
    if not audit["passed"]:
        raise SystemExit("development audit failed")
    confirmation_rows(helpers.MANIFEST)
    notes = ROOT / "report/final_development_review/review_notes.md"
    if not notes.is_file() or not (ROOT / "report/final_development_review/failure_labels.csv").is_file():
        raise SystemExit("review must be recorded before freeze")
    registered = json.loads((ROOT / "manifests/runs.json").read_text())["runs"]
    runs = []
    for run_id in ("E0-s42", "E1-s42", "E2-s42"):
        run = next(r for r in registered if r["run_id"] == run_id)
        print(f"Hashing inference checkpoint: {run_id}", flush=True)
        runs.append({"run_id": run_id, "config": run["config"], "checkpoint": run["final_checkpoint"],
                     "weight_type": "EMA" if run_id == "E0-s42" else "current non-EMA",
                     "training_seed": 42, "inference_artifacts": inventory(run["final_checkpoint"])})
    source_paths = [ROOT / name for name in ("eval_manifest.py", "seeded_policy_server.py", "development_watchdog.py",
        "confirmation_gate.py", "analyze_confirmation.py", "freeze_confirmation.py", "run_confirmation_pipeline.py",
        "report/final_development_review/artifact_audit.json", "report/final_development_review/failure_labels.csv",
        "report/final_development_review/failure_sample.json", "report/final_development_review/review_notes.md",
        "evaluations/development_comparison.json", "evaluations/e3_development_comparison.json", "metrics/profiles/summary.json")]
    source_paths += [OPENPI / name for name in ("src/openpi/training/config.py", "src/openpi/training/checkpoints.py",
        "src/openpi/models/pi0_config.py", "src/openpi/models/pi0.py", "src/openpi/models/gemma.py",
        "src/openpi/models/lora.py", "src/openpi/policies/policy_config.py", "src/openpi/policies/libero_policy.py",
        "src/openpi/policies/policy.py", "src/openpi/transforms.py")]
    source_paths += sorted((OPENPI / "third_party/libero/libero/libero").rglob("*.py"))
    version_code = "import importlib.metadata as m, json, sys; names=['jax','jaxlib','numpy','torch','mujoco','robosuite','libero','openpi-client']; versions={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps({'python':sys.version,'packages':{n:versions.get(n,'not distributed in this environment') for n in names}}))"
    environments = {name: json.loads(subprocess.check_output([str(python), "-c", version_code], text=True))
                    for name, python in (("server", OPENPI / ".venv/bin/python"), ("client", OPENPI / "examples/libero/.venv/bin/python"))}
    freeze = {"status": "frozen", "created_at": helpers.now(), "candidate": "E2-s42", "reference": "E1-s42",
        "anchor": "E0-s42", "excluded": {"E3-s42": "development success 68.25%, 26pp below E1; failed screening"},
        "selection": "E2 development equals E1 overall; Spatial +3.5pp, Long -3.5pp; sampled peak memory -24.5%, step time -12.9% with shared-device limitations",
        "manifest_sha256": sha256(helpers.MANIFEST), "runs": runs,
        "metrics": {"primary_for_prospective_confirmation": "environment_success_suite_equal",
                    "co_reported_original_metric": "raw_bound_strict_success_suite_equal",
                    "primary_comparison": "E2 minus E1", "engineering_margin": -0.05,
                    "bootstrap": "10000 paired resamples within each fixed task, then task and suite equal means",
                    "bootstrap_seed": 20261021, "noninferiority": "95% interval lower endpoint strictly greater than -0.05",
                    "strict_claim": "retain original raw-bound failures; no noninferiority claim for the original strict criterion based on environment success"},
        "execution": {"episodes_per_run": 1600, "planned_total": 4800, "e0_development_supplement": 400, "flow_steps": 10, "replan_steps": 5,
                      "settling_steps": 10, "initial_states": "Spatial/Long 20..49, Object/Goal 0..49, all 40 tasks",
                      "infrastructure_max_attempts_per_episode": 2, "final_checkpoints_only": True,
                      "failure_rule": "environment timeout/NaN is policy failure or explicitly reported policy fault; infrastructure faults retained separately"},
        "limitations": ["one training seed (42), no training-seed uncertainty claim",
                       "E0 historical all-suite results already exposed; this is fixed-benchmark confirmation, not a pristine independent test",
                       "development is used for selection and excluded from confirmation estimates",
                       "E0 EMA and E1/E2 non-EMA recipe comparison is not a single-variable ablation",
                       "no candidate or execution changes after viewing confirmation outcomes"],
        "environment_versions": environments,
        "source_and_evidence_hashes": {str(p): sha256(p) for p in source_paths}}
    helpers.atomic_json(target, freeze)
    sidecar = target.with_suffix(".json.sha256")
    with sidecar.open("x") as file:
        file.write(sha256(target) + "  " + target.name + "\n")
    helpers.event("candidate_frozen", candidate="E2-s42", path=str(target), sha256=sha256(target))


if __name__ == "__main__":
    main()

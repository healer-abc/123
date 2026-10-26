"""Immutable candidate and artifact identity checks used before confirmation launch."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPENPI = ROOT.parents[1] / "projects/openpi"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(checkpoint):
    checkpoint = Path(checkpoint)
    if not (checkpoint / "_CHECKPOINT_METADATA").is_file() or not (checkpoint / "params/_METADATA").is_file():
        raise ValueError(f"checkpoint not finalized: {checkpoint}")
    paths = [checkpoint / "_CHECKPOINT_METADATA"]
    paths += sorted(p for folder in ("params", "assets") for p in (checkpoint / folder).rglob("*") if p.is_file())
    return {str(p.relative_to(checkpoint)): {"size": p.stat().st_size, "sha256": sha256(p)} for p in paths}


def confirmation_rows(manifest):
    with Path(manifest).open() as file:
        all_rows = list(csv.DictReader(file))
    rows = [r for r in all_rows if r["phase"] == "confirmation"]
    expected = {(suite, task, state) for suite in ("libero_spatial", "libero_10", "libero_object", "libero_goal")
                for task in range(10) for state in (range(20, 50) if suite in ("libero_spatial", "libero_10") else range(50))}
    keys = {(r["suite"], int(r["task_id"]), int(r["init_state_index"])) for r in rows}
    if len(rows) != 1600 or keys != expected:
        raise ValueError("confirmation manifest does not contain exact frozen 1600 identities")
    return rows


def validate_freeze(path, run_ids, verify_artifacts=True):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sha256(path) != sidecar.read_text().split()[0]:
        raise ValueError("candidate freeze hash mismatch")
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "frozen" or freeze.get("candidate") != "E2-s42":
        raise ValueError("candidate is not frozen E2")
    if set(run_ids) != {r["run_id"] for r in freeze["runs"]} or len(run_ids) != 3:
        raise ValueError("confirmation run ids differ from frozen E0/E1/E2")
    manifest = ROOT / "manifests/eval_manifest.csv"
    if sha256(manifest) != freeze["manifest_sha256"]:
        raise ValueError("confirmation manifest hash mismatch")
    confirmation_rows(manifest)
    for file, digest in freeze["source_and_evidence_hashes"].items():
        if sha256(file) != digest:
            raise ValueError(f"frozen code/evidence changed: {file}")
    runs = {r["run_id"]: r for r in json.loads((ROOT / "manifests/runs.json").read_text())["runs"]}
    for run in freeze["runs"]:
        registered = runs[run["run_id"]]
        if registered["config"] != run["config"] or registered["final_checkpoint"] != run["checkpoint"]:
            raise ValueError("registered configuration differs from freeze")
        if verify_artifacts and inventory(run["checkpoint"]) != run["inference_artifacts"]:
            raise ValueError(f"checkpoint content changed: {run['run_id']}")
    return freeze

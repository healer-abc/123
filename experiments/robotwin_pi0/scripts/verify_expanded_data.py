"""Cross-check completed collection artifacts and retain their provenance hashes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
report = root / "report"


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


state = read(report / "expanded-collection-state.json")
assert state["status"] == "complete" and not state["errors"]
assert read(report / "data-interface-check.json")["status"] == "passed"
variants = {}
for split in ["clean", "randomized"]:
    repo_id = f"local/robotwin_pi0_{split}200"
    dataset = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
    raw = workspace / f"data/robotwin_pi0/raw/{split}/adjust_bottle/demo_{split}_pi0_train200"
    manifest_path = dataset / "meta/source_episodes.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    info = read(dataset / "meta/info.json")
    converted = read(report / f"{split}200-converted-check.json")
    loader = read(report / f"{split}200-loader-check.json")
    stats_dir = report / f"{split}200_norm_stats"
    stats_source = read(stats_dir / "source.json")
    stats = read(stats_dir / "norm_stats.json")["norm_stats"]
    protocol = read(raw / "collection_protocol.json")["protocol"]
    config_hash = sha256(raw / "collection_config.yml")
    assert state["variants"][split]["phase"] == "complete"
    assert info["total_episodes"] == len(manifest) == len(converted["episodes"]) == 200
    assert len(list((raw / "data").glob("episode*.hdf5"))) == 200
    assert len(list((raw / "instructions").glob("episode*.json"))) == 200
    assert len(list((dataset / "data").rglob("*.parquet"))) == 200
    seeds = [int(value) for value in (raw / "seed.txt").read_text().split()]
    assert seeds == [row["seed"] for row in manifest] and len(set(seeds)) == 200
    assert converted["status"] == loader["status"] == "passed"
    assert converted["repo_id"] == loader["dataset"] == stats_source["repo_id"] == repo_id
    assert config_hash == protocol["config_sha256"]
    assert set(stats) == {"state", "actions"}
    for key_stats in stats.values():
        for values in key_stats.values():
            assert np.asarray(values).shape == (14,) and np.isfinite(values).all()
    for index, (row, checked) in enumerate(zip(manifest, converted["episodes"])):
        assert row["episode_index"] == checked["episode_index"] == index
        assert row["frames"] == checked["frames"] == row["source_frames"] - 1
        assert row["seed"] == checked["seed"] and row["variant"] == split
        assert Path(row["source"]).resolve() == raw / "data" / f"episode{index}.hdf5"
        assert sha256(Path(row["source"])) == row["source_sha256"]
        assert row["config_sha256"] == config_hash
        assert row["denoiser"] == protocol["denoiser"] == "none"
        assert row["planner"] == "curobo"
        assert row["instruction"] in read(raw / "instructions" / f"episode{index}.json")["seen"]
    frames = [row["frames"] for row in manifest]
    assert sum(frames) == info["total_frames"] == converted["total_frames"]
    artifacts = [manifest_path, dataset / "meta/info.json", raw / "collection_config.yml",
                 raw / "collection_protocol.json", stats_dir / "norm_stats.json",
                 stats_dir / "source.json", report / f"{split}200-converted-check.json",
                 report / f"{split}200-loader-check.json"]
    variants[split] = {
        "episodes": 200, "training_frames": sum(frames), "raw_frames": sum(frames) + 200,
        "frames_per_episode": {"min": min(frames), "max": max(frames), "mean": sum(frames) / 200},
        "seed_range": [min(seeds), max(seeds)], "repo_id": repo_id,
        "raw_dataset": str(raw), "lerobot_dataset": str(dataset),
        "norm_stats": str(stats_dir / "norm_stats.json"),
        "raw_source_hashes_verified": 200,
        "artifact_sha256": {str(path.relative_to(workspace)): sha256(path) for path in artifacts},
    }
result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
          "total_episodes": 400, "total_training_frames": sum(v["training_frames"] for v in variants.values()),
          "variants": variants,
          "validation": "Raw hashes, seeds, instructions, configuration and report consistency checked here; all state/action pairs and sampled camera pixels checked by converted-check reports."}
(report / "expanded-data-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "variants"}, indent=2))

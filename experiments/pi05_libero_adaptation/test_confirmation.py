"""Checks for confirmation isolation, unequal-count suite weighting and frozen evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_confirmation as analysis
import confirmation_gate as gate
import development_watchdog as watchdog


def record(success):
    return {"success": success, "action_out_of_nominal_bounds": True, "env_seed": 1, "policy_seed": 2,
            "init_state_sha256": "fixed", "max_policy_steps": "220", "replan_steps": "5", "flow_steps": "10", "settling_steps": "10"}


class ConfirmationTest(unittest.TestCase):
    def test_exact_confirmation_identities_and_disjoint_development(self):
        rows = gate.confirmation_rows(watchdog.MANIFEST)
        with watchdog.MANIFEST.open() as file:
            dev = {(r["suite"], r["task_id"], r["init_state_index"]) for r in csv.DictReader(file) if r["phase"] == "development"}
        confirm = {(r["suite"], r["task_id"], r["init_state_index"]) for r in rows}
        self.assertEqual(len(confirm), 1600)
        self.assertFalse(dev & confirm)

    def test_suite_equal_weighting_does_not_pool_unequal_counts(self):
        records = {(suite, 0, state): record(suite == "libero_spatial")
                   for suite, count in (("libero_spatial", 30), ("libero_object", 50)) for state in range(count)}
        summary = analysis.summarize(records)
        self.assertEqual(summary["suite_equal_environment_success_rate"], .5)
        self.assertNotEqual(summary["suite_equal_environment_success_rate"], 30/80)
        self.assertEqual(summary["suite_equal_strict_success_rate"], 0)

    def test_pairing_rejects_changed_seed(self):
        a = {("libero_spatial", 0, 20): record(True)}
        b = {k: {**v, "policy_seed": 7} for k, v in a.items()}
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            analysis.compare(a, b)

    def test_deterministic_paired_difference_and_strict_floor(self):
        reference = {("libero_spatial", 0, i): record(False) for i in range(20, 50)}
        candidate = {k: record(True) for k in reference}
        result = analysis.compare(candidate, reference, replicates=100)
        self.assertEqual(result["paired_task_stratified_bootstrap_95"], [1, 1])
        strict = analysis.compare(candidate, reference, replicates=100, strict=True)
        self.assertTrue(strict["both_metrics_at_zero_floor"])
        self.assertFalse(strict["strict_claim_allowed"])
        self.assertIsNone(strict["meets_environment_noninferiority_rule"])

    def test_confirmation_summary_cannot_read_development_outputs(self):
        row = {"phase": "confirmation", "suite": "libero_spatial", "task_id": "0", "init_state_index": "20"}
        job = {"run_id": "E2-s42", "checkpoint": "/fixed", "config": "test", "phase": "confirmation"}
        with tempfile.TemporaryDirectory() as temp, patch.object(watchdog, "ROOT", Path(temp)):
            output = Path(temp) / "evaluations/E2-s42/development/episodes/libero_spatial/task_00/state_20.json"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({**row, "status": "valid", "success": True}))
            summary = watchdog.summarize(job, [row])
            self.assertEqual(summary["completed"], 0)
            self.assertTrue((Path(temp) / "evaluations/E2-s42/confirmation/summary.json").exists())

    def test_inventory_changes_if_checkpoint_payload_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "params").mkdir()
            (root / "_CHECKPOINT_METADATA").write_text("final")
            (root / "params/_METADATA").write_text("metadata")
            (root / "params/weights").write_bytes(b"first")
            first = gate.inventory(root)
            (root / "params/weights").write_bytes(b"other")
            self.assertNotEqual(first, gate.inventory(root))

    def test_changed_freeze_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "freeze.json"
            path.write_text('{}')
            path.with_suffix('.json.sha256').write_text(gate.sha256(path))
            path.write_text('{"candidate":"changed"}')
            with self.assertRaisesRegex(ValueError, "freeze hash mismatch"):
                gate.validate_freeze(path, ["E0-s42", "E1-s42", "E2-s42"])


if __name__ == "__main__":
    unittest.main()

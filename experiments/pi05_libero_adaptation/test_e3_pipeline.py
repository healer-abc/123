"""E3 trainability, recipe and supervisor checks without GPU allocation."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
import training_watchdog as watchdog
import complete_e3_analysis as analysis


class E3PipelineTest(unittest.TestCase):
    def test_parameter_audit_is_exact_action_whitelist(self):
        path = DIRECTORY / "audits/parameter_audit_pi05_libero_action_only.csv"
        with path.open() as file:
            rows = list(csv.DictReader(file))
        selected = [r for r in rows if r["trainable"] == "true"]
        self.assertEqual(len(selected), 19)
        self.assertEqual(sum(int(r["parameter_count"]) for r in selected), 430098464)
        self.assertEqual(sum(int(r["parameter_count"]) for r in rows), 3353433872)
        self.assertTrue(all(r["module"] in {"action_branch", "action_time_projection"} for r in selected))
        self.assertFalse(any(r["is_lora"] == "true" for r in rows))

    def test_checkpoint_gate_requires_training_state_and_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "params").mkdir()
            self.assertFalse(watchdog.complete_checkpoint(path))
            (path / "train_state").mkdir()
            self.assertFalse(watchdog.complete_checkpoint(path))
            (path / "_CHECKPOINT_METADATA").write_text("{}")
            self.assertTrue(watchdog.complete_checkpoint(path))

    def test_memory_sampling_counts_only_owned_process(self):
        with patch.object(watchdog.subprocess, "check_output", side_effect=[
            "21000, 64000, 75\n", "123, 10000\n456, 10682\n"]):
            value = watchdog.sample(1, 123)
        self.assertEqual(value["process_mib"], 10000)
        self.assertEqual(value["gpu_used_mib"], 21000)

    def test_training_environment_uses_local_data_and_no_preallocation(self):
        env = watchdog.environment(0)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertNotIn("XLA_PYTHON_CLIENT_MEM_FRACTION", env)

    def test_pairing_identical_results_has_zero_delta_and_interval(self):
        reference = analysis.records("E1-s42")
        result = analysis.compare(reference, reference)
        self.assertEqual(result["environment_delta"], 0.0)
        self.assertEqual(result["paired_task_stratified_bootstrap_95"], [0.0, 0.0])
        self.assertEqual(len(result["tasks"]), 20)

    def test_pairing_rejects_changed_policy_noise_seed(self):
        reference = analysis.records("E1-s42")
        candidate = {key: dict(value) for key, value in reference.items()}
        first = next(iter(candidate))
        candidate[first]["policy_seed"] += 1
        with self.assertRaises(AssertionError):
            analysis.compare(candidate, reference)


if __name__ == "__main__":
    unittest.main()

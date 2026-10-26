import unittest
from types import SimpleNamespace

from cost_profiles_watchdog import validate_history
from profile_hooks import install_metric_observer


class ProfileRecoveryTest(unittest.TestCase):
    def test_observer_survives_log_rebinding_during_initialization(self):
        native, observed = [], []

        def obsolete(*_args, **_kwargs):
            raise AssertionError("pre-init W&B log should not be retained")

        module = SimpleNamespace(wandb=SimpleNamespace(log=obsolete))

        def initialize():
            module.wandb.log = lambda data, step=None: native.append((data, step))
            return "initialized"

        module.init_wandb = initialize
        install_metric_observer(module, lambda data, step: observed.append((data, step)))
        self.assertEqual(module.init_wandb(), "initialized")
        module.wandb.log({"loss": .1}, step=20)
        self.assertEqual(native, [({"loss": .1}, 20)])
        self.assertEqual(observed, native)

    def rows(self):
        return [{"_step": i, "_timestamp": 1000 + i * 4.0, "loss": .1, "grad_norm": .2} for i in range(220)]

    def test_history_has_exactly_200_post_warmup_intervals(self):
        intervals = validate_history(self.rows())
        self.assertEqual(len(intervals), 200)
        self.assertEqual(set(intervals), {4.0})

    def test_recovery_rejects_missing_steps_nonfinite_or_clock_reversal(self):
        with self.assertRaises(ValueError):
            validate_history(self.rows()[:-1])
        rows = self.rows()
        rows[30]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            validate_history(rows)
        rows = self.rows()
        rows[30]["_timestamp"] = rows[29]["_timestamp"] - 1
        with self.assertRaises(ValueError):
            validate_history(rows)


if __name__ == "__main__":
    unittest.main()

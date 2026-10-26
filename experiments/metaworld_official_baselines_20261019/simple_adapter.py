from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from rollout import RunnerMixin


class Runner(RunnerMixin, BaseRunner):
    def __init__(self, output_dir, task_name, **kw):
        self.configure(output_dir, task_name, 'SimpleDP3', **kw)

from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from rollout import RunnerMixin
from runtime import shared_split
from diffusion_policy_3d.dataset.metaworld_dataset import MetaworldDataset
from diffusion_policy_3d.common.sampler import SequenceSampler


class Dataset(MetaworldDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = shared_split(self.replay_buffer.n_episodes)
        self.sampler = SequenceSampler(self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=self.train_mask)


class Runner(RunnerMixin, BaseRunner):
    def __init__(self, output_dir, task_name, **kw):
        self.configure(output_dir, task_name, 'SimpleDP3', **kw)

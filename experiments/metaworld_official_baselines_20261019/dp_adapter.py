"""MetaWorld I/O for unmodified official Diffusion Policy image workspace."""
import copy
import numpy as np
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from rollout import RunnerMixin


class Dataset(BaseImageDataset):
    def __init__(self, zarr_path, horizon, pad_before, pad_after, seed=42, val_ratio=0.02):
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['img', 'state', 'action'])
        self.train_mask = ~get_val_mask(self.replay_buffer.n_episodes, val_ratio, seed)
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.sampler = self.make_sampler(self.train_mask)

    def make_sampler(self, mask):
        return SequenceSampler(self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=mask)

    def get_validation_dataset(self):
        val = copy.copy(self)
        val.sampler = val.make_sampler(~self.train_mask)
        return val

    def get_normalizer(self, mode='limits', **kw):
        n = LinearNormalizer()
        n.fit({'action': self.replay_buffer['action'], 'agent_pos': self.replay_buffer['state']},
              last_n_dims=1, mode=mode, **kw)
        n['image'] = get_image_range_normalizer()
        return n

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, i):
        s = self.sampler.sample_sequence(i)
        # The official generator converts observations to HWC on disk.
        assert s['img'].shape[-1] == 3
        return dict_apply({'obs': {'image': np.moveaxis(s['img'], -1, 1).astype(np.float32) / 255.,
            'agent_pos': s['state'].astype(np.float32)},
            'action': s['action'].astype(np.float32)}, torch.from_numpy)


class Runner(RunnerMixin, BaseImageRunner):
    def __init__(self, output_dir, task_name, **kw):
        self.configure(output_dir, task_name, 'DP', **kw)

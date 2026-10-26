"""Explicit, untrained bridge from MetaWorld Sawyer observations to base pi05."""
import sys
from pathlib import Path

WORKSPACE = Path('/home/zhsu/jyluo')
sys.path.insert(0, str(WORKSPACE / 'projects/openpi/src'))
# Append rather than prepend: avoid the old RLinf OpenPI and Transformers copies.
sys.path.append(str(WORKSPACE / 'experiments/pi0_metaworld_baseline_20261015/deps'))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils
from openpi_client import image_tools


class MetaWorldAdapter:
    def __init__(self, protocol):
        self.protocol = protocol

    def make_env(self, name):
        import metaworld
        benchmark = metaworld.MT1(name, seed=self.protocol['benchmark_seed'])
        env = benchmark.train_classes[name](render_mode='rgb_array')
        env.model.cam_pos[self.protocol['camera_id']] = self.protocol['camera_position']
        indices = np.random.default_rng(self.protocol['task_selection_seed']).choice(
            len(benchmark.train_tasks), self.protocol['trials_per_task'], replace=False).tolist()
        return env, benchmark.train_tasks, indices

    def render(self, env):
        image = env.mujoco_renderer.render('rgb_array', camera_id=self.protocol['camera_id'])
        image = np.ascontiguousarray(image[::-1, ::-1])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'Invalid image: {image.shape}, {image.dtype}')
        return image

    def encode_state(self, env, obs):
        state = np.zeros(32, dtype=np.float32)
        state[:3] = 2 * (np.asarray(obs[:3]) - env.hand_low) / (env.hand_high - env.hand_low) - 1
        state[3] = 2 * float(obs[3]) - 1
        return np.clip(state, -1, 1)

    def decode_actions(self, env, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (self.protocol['action_horizon'], 32) or not np.isfinite(raw).all():
            raise ValueError(f'Invalid policy actions: {raw.shape}, finite={np.isfinite(raw).all()}')
        selected = raw[:, :4]
        clipped = np.clip(selected, env.action_space.low, env.action_space.high)
        return clipped, int(np.count_nonzero(clipped != selected))


class FrozenBasePolicy:
    def __init__(self, protocol):
        self.protocol = protocol
        self.config = Pi0Config(pi05=True, action_horizon=protocol['action_horizon'], discrete_state_input=True)
        params = model_lib.restore_params(Path(protocol['checkpoint']) / 'params', dtype=jnp.bfloat16)
        self.model = self.config.load(params)
        self.model.eval()
        self.sample = nnx_utils.module_jit(self.model.sample_actions)
        self.tokenizer = PaligemmaTokenizer(self.config.max_token_len)
        self.rng = jax.random.key(0)

    def reset(self, seed):
        self.rng = jax.random.key(seed)

    def infer(self, image, state, prompt):
        image = image_tools.resize_with_pad(image, self.protocol['image_size'], self.protocol['image_size'])
        image = image_tools.convert_to_uint8(image)
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        image = image.astype(np.float32) / 127.5 - 1
        observation = model_lib.Observation(
            images={'base_0_rgb': jnp.asarray(image[None]),
                    'left_wrist_0_rgb': jnp.zeros((1, 224, 224, 3)),
                    'right_wrist_0_rgb': jnp.zeros((1, 224, 224, 3))},
            image_masks={'base_0_rgb': jnp.array([True]),
                         'left_wrist_0_rgb': jnp.array([False]),
                         'right_wrist_0_rgb': jnp.array([False])},
            state=jnp.asarray(state[None]), tokenized_prompt=jnp.asarray(tokens[None]),
            tokenized_prompt_mask=jnp.asarray(mask[None]))
        self.rng, sample_rng = jax.random.split(self.rng)
        actions = self.sample(sample_rng, observation, num_steps=self.protocol['flow_steps'])
        return np.asarray(actions[0])  # Device synchronization is included in caller timing.

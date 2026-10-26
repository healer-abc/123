"""Diagnostic-only observer around the frozen ActionGain behavior; not used for training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from run import ActionGain


class ObservedActionGain(ActionGain):
    def __init__(self, env, low=1., high=1., *, training_seed):
        super().__init__(env, low, high)
        self.training_seed = training_seed
        self.clip_records = []
        self.episode_index = -1

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.episode_index += 1
        self.episode_seed = seed
        self.episode_steps = 0
        return result

    def step(self, action):
        # Match the frozen wrapper's dtype and product exactly; consume no RNG.
        scaled = np.asarray(action, dtype=np.float32).copy()
        scaled[:3] *= self.gain
        mask = np.abs(scaled[:3]) > 1.
        # Pass ORIGINAL action to the original implementation, exactly once.
        transition = super().step(action)
        self.episode_steps += 1
        self.clip_records.append({
            'training_seed': self.training_seed, 'episode_index': self.episode_index,
            'episode_seed': self.episode_seed, 'step': self.episode_steps, 'gain': self.gain,
            'clipped_components': int(mask.sum()), 'translation_components': 3,
            'any_translation_clipped': int(mask.any()),
        })
        return transition


def summarize_clip_records(records):
    """Aggregate fixed evaluation gains; do not pool arbitrary random gains together."""
    totals = {}
    for row in records:
        key = (row['training_seed'], row['gain'])
        counts = totals.setdefault(key, [0, 0, 0, 0])
        counts[0] += row['clipped_components']
        counts[1] += row['translation_components']
        counts[2] += row['any_translation_clipped']
        counts[3] += 1
    return [{'training_seed': seed, 'gain': gain,
             'component_clip_rate': counts[0] / counts[1],
             'step_clip_rate': counts[2] / counts[3], 'steps': counts[3]}
            for (seed, gain), counts in sorted(totals.items())]

"""Physically specified joint-action bridge; does not use goals or object state."""
import importlib.util
from pathlib import Path

LEGACY_ROOT = Path('/home/zhsu/jyluo/experiments/pi05_metaworld_base_20261015')
spec = importlib.util.spec_from_file_location('legacy_base_adapter', LEGACY_ROOT/'adapter.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import normalize
from openpi_client import image_tools


class JointBridge(legacy.MetaWorldAdapter):
    def __init__(self, protocol):
        super().__init__(protocol)
        self.stats = normalize.load(Path(protocol['checkpoint'])/'assets'/protocol['normalization_asset'])
        self.normalizer = transforms.Normalize({'state': self.stats['state']}, use_quantiles=True)
        self.unnormalizer = transforms.Unnormalize({'actions': self.stats['actions']}, use_quantiles=True)

    @staticmethod
    def joint_indices(env):
        joints = [env.model.joint(f'right_j{i}') for i in range(7)]
        return np.array([int(j.qposadr[0]) for j in joints]), np.array([int(j.dofadr[0]) for j in joints])

    @staticmethod
    def gripper_closure(env):
        # r_close range [0,.04], l_close [-.03,0]; use actual joint travel, not
        # obs[3] which has a nonzero fully-closed floor of approximately .30.
        right = float(env.data.joint('r_close').qpos[0])
        left = float(env.data.joint('l_close').qpos[0])
        return float(np.clip((right-left)/.07, 0, 1))

    def physical_state(self, env):
        qpos, _ = self.joint_indices(env)
        return np.r_[env.data.qpos[qpos], self.gripper_closure(env)].astype(np.float32)

    def encode_state(self, env, obs=None):
        return self.normalizer({'state': self.physical_state(env)})['state'].astype(np.float32)

    def unnormalize(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != 32 or not np.isfinite(raw).all():
            raise ValueError(f'Invalid raw policy output: {raw.shape}')
        return self.unnormalizer({'actions': raw})['actions'][..., :8]

    @staticmethod
    def jacobians(env):
        _, dofs = JointBridge.joint_indices(env)
        position = np.zeros((3,env.model.nv))
        rotation = np.zeros_like(position)
        left, right = np.zeros_like(position), np.zeros_like(position)
        mujoco.mj_jacSite(env.model,env.data,left,None,env.model.site('leftEndEffector').id)
        mujoco.mj_jacSite(env.model,env.data,right,None,env.model.site('rightEndEffector').id)
        position[:] = (left+right)/2
        mujoco.mj_jacBody(env.model,env.data,None,rotation,env.model.body('hand').id)
        return position[:,dofs], rotation[:,dofs]

    def decode_action(self, env, physical):
        physical = np.asarray(physical,dtype=np.float64)
        if physical.shape != (8,) or not np.isfinite(physical).all():
            raise ValueError('Expected finite 7 DROID joint commands plus gripper closure')
        # DROID RobotIKSolver.joint_velocity_to_delta: commands are dimensionless,
        # limited together by their infinity norm, then scaled by .2 rad/tick.
        velocity = physical[:7]/max(1.,float(np.max(np.abs(physical[:7]))))
        velocity = velocity*self.protocol['droid_max_joint_delta_rad']*self.protocol['policy_hz']
        jp, jr = self.jacobians(env)
        # MetaWorld fixes the tool orientation; remove the rotational component
        # rather than silently commanding an unsupported rotational motion.
        null = np.eye(7)-jr.T@np.linalg.solve(jr@jr.T+1e-6*np.eye(3),jr)
        projected = null@velocity
        displacement = jp@projected*env.dt
        unbounded = displacement/env.action_scale
        closure = float(np.clip(physical[7],0,1))
        # These are position actuators with r_close=[0,.04], l_close=[-.03,0].
        # At static control u: closure=(clip(u,0,.04)-clip(-u,-.03,0))/.07.
        # Invert it, then add position-error feedback to compensate the very
        # overdamped fingers. Endpoint sign checks alone do not test this.
        equilibrium = .035*closure if closure<=6/7 else .07*closure-.03
        gripper = equilibrium+self.protocol['gripper_servo_gain']*(closure-self.gripper_closure(env))
        preclip = np.r_[unbounded,gripper]
        action = np.clip(preclip,env.action_space.low,env.action_space.high).astype(np.float32)
        return action, dict(joint_velocity=velocity, projected_velocity=projected,
                            requested_displacement=displacement, preclip_action=preclip,
                            clipped_components=int(np.count_nonzero(np.abs(preclip[:3])>1)),
                            removed_rotation_speed=float(np.linalg.norm(jr@velocity)),
                            residual_rotation_speed=float(np.linalg.norm(jr@projected)))


class FrozenBasePolicy(legacy.FrozenBasePolicy):
    def infer(self,image,state,prompt):
        if np.shape(state)!=(8,):
            raise ValueError('Tokenize the eight real state dimensions before padding')
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224))
        tokens,mask = self.tokenizer.tokenize(prompt,state)
        image = image.astype(np.float32)/127.5-1
        observation = model_lib.Observation(
            images={'base_0_rgb':jnp.asarray(image[None]),
                    'left_wrist_0_rgb':jnp.zeros((1,224,224,3)),
                    'right_wrist_0_rgb':jnp.zeros((1,224,224,3))},
            image_masks={'base_0_rgb':jnp.array([True]),'left_wrist_0_rgb':jnp.array([False]),
                         'right_wrist_0_rgb':jnp.array([False])},
            state=jnp.asarray(np.pad(state,(0,24))[None]),
            tokenized_prompt=jnp.asarray(tokens[None]),tokenized_prompt_mask=jnp.asarray(mask[None]))
        self.rng,key = jax.random.split(self.rng)
        return np.asarray(self.sample(key,observation,num_steps=self.protocol['flow_steps'])[0])

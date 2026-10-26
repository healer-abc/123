"""Validate Jacobian, normalization, gripper sign and observations in real MuJoCo."""
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from adapter import JointBridge
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())


def tcp(model,data):
    return (data.site('leftEndEffector').xpos+data.site('rightEndEffector').xpos)/2


def main():
    bridge=JointBridge(P)
    records=[]
    for name in P['tasks']:
        env,tasks,indices=bridge.make_env(name)
        try:
            env.set_task(tasks[indices[0]]);env.reset(seed=42)
            qpos,dofs=bridge.joint_indices(env)
            jp,jr=bridge.jacobians(env)
            numeric=np.zeros_like(jp)
            for i,index in enumerate(qpos):
                points=[]
                for sign in (-1,1):
                    scratch=mujoco.MjData(env.model)
                    scratch.qpos[:]=env.data.qpos
                    scratch.qpos[index]+=sign*1e-6
                    mujoco.mj_kinematics(env.model,scratch)
                    points.append(tcp(env.model,scratch).copy())
                numeric[:,i]=(points[1]-points[0])/2e-6
            jac_error=float(np.max(np.abs(jp-numeric)))
            assert jac_error<1e-5,(name,jac_error)
            state=bridge.physical_state(env)
            normalized=bridge.encode_state(env)
            restored=bridge.unnormalizer({'actions':np.zeros((1,32))})['actions'][0,:8]
            expected=(bridge.stats['actions'].q01+bridge.stats['actions'].q99)/2
            assert np.allclose(restored,expected,atol=1e-6)
            inverse=bridge.stats['state'].q01+(normalized+1)/2*(bridge.stats['state'].q99-bridge.stats['state'].q01+1e-6)
            assert np.allclose(inverse,state,atol=2e-6)
            action,detail=bridge.decode_action(env,np.zeros(8))
            assert np.array_equal(action[:3],np.zeros(3,dtype=np.float32))
            command=np.array([.1,-.2,.15,.1,-.1,.2,.1,.8])
            action,detail=bridge.decode_action(env,command)
            assert np.allclose(detail['joint_velocity'],command[:7]*3)
            assert np.allclose(action[:3]*env.action_scale,detail['requested_displacement'],atol=1e-8)
            assert detail['residual_rotation_speed']<1e-5
            modified=command.copy();modified[3]=-10
            assert bridge.decode_action(env,modified)[0][-1]==action[-1]
            limited=bridge.decode_action(env,modified)[1]['joint_velocity']
            assert np.allclose(limited,modified[:7]/10*3)
            assert np.isclose(np.max(np.abs(limited)),3)
            modified=command.copy();modified[7]=0
            assert bridge.decode_action(env,modified)[0][-1]<=0
            for _ in range(50):env.step(np.array([0,0,0,1.]))
            closed=bridge.gripper_closure(env)
            for _ in range(50):env.step(np.array([0,0,0,-1.]))
            opened=bridge.gripper_closure(env)
            assert closed>.9 and opened<.1,(name,closed,opened)
            tracking=[]
            for target in (0.,.25,.5,.75,1.):
                env.reset(seed=42)
                command=np.zeros(8);command[7]=target
                for _ in range(100):
                    action,_=bridge.decode_action(env,command)
                    env.step(action)
                measured=bridge.gripper_closure(env)
                tracking.append(dict(target=target,measured=measured))
                assert abs(measured-target)<.08,(name,'gripper position tracking',target,measured)
            records.append(dict(task=name,jacobian_finite_difference_max_error=jac_error,
                                normalization_roundtrip=True,gripper_closed=closed,gripper_open=opened,
                                droid_command_to_rad_per_second=True,uniform_infinity_norm_limiting=True,
                                zero_joint_velocity_is_zero_translation=True,
                                fourth_dimension_not_gripper=True,eighth_dimension_is_gripper=True,
                                physical_state=state.tolist(),normalized_state=normalized.tolist(),
                                out_of_quantile_state_dimensions=int(np.count_nonzero(np.abs(normalized)>1)),
                                gripper_position_tracking=tracking,env_dt=env.dt,action_scale=env.action_scale))
        finally:env.close()
    (ROOT/'diagnostics/interface_validation.json').write_text(json.dumps(records,indent=2)+'\n')
    stamp=dict(protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),tasks=len(records),passed=True)
    (ROOT/'diagnostics/interface_validation_stamp.json').write_text(json.dumps(stamp,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()

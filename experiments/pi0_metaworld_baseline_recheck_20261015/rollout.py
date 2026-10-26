"""Frozen reference pi0 policy evaluated in the five requested MetaWorld v2 tasks."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))
sys.path.insert(0, str(ROOT / "sources"))
cuda_libs = [str(path) for path in (ROOT / "deps/nvidia").glob("*/lib")]
cuda_libs.append("/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_libs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("OPENPI_DATA_HOME", "/home/zhsu/jyluo/myopenpi/dataset/ckpt")
os.environ.setdefault("OMP_NUM_THREADS", "4")

PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
PROTOCOL_HASH = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
CHECKPOINT = ROOT / "checkpoints/RLinf-Pi0-MetaWorld-SFT"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_simulation(env):
    # MetaWorld can swallow simulator errors and return the last stable observation.
    # Such episodes are infrastructure errors, not legitimate policy failures.
    if env._did_see_sim_exception:
        raise RuntimeError("MetaWorld reported an internal simulation exception")
    for name in ("qpos", "qvel", "qacc"):
        import numpy as np
        if not np.isfinite(getattr(env.data, name)).all():
            raise RuntimeError(f"Non-finite MuJoCo {name}")


def reset_episode(env, selected_task, seed):
    env.set_task(selected_task)
    # This native v2 revision explicitly ignores the reset(seed=...) argument.
    env.seed(seed)
    observation, info = env.reset()
    check_simulation(env)
    return observation, info


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def heartbeat(task, phase, **extra):
    atomic_json(ROOT / "logs" / f"heartbeat_{task}.json",
                {"pid": os.getpid(), "time": time.time(), "task": task, "phase": phase, **extra})


def make_env(task):
    import metaworld
    import numpy as np
    benchmark = metaworld.MT1(task, seed=PROTOCOL["benchmark_seed"])
    env = benchmark.train_classes[task](render_mode="rgb_array")
    env.model.cam_pos[PROTOCOL["camera_id"]] = PROTOCOL["camera_position"]
    indices = np.random.default_rng(PROTOCOL["task_selection_seed"]).choice(
        len(benchmark.train_tasks), PROTOCOL["trials_per_task"], replace=False).tolist()
    return env, benchmark.train_tasks, indices


def render_image(env):
    import numpy as np
    image = env.mujoco_renderer.render("rgb_array", camera_id=PROTOCOL["camera_id"])
    image = np.ascontiguousarray(image[::-1, ::-1])
    if list(image.shape) != PROTOCOL["image_shape"] or image.dtype != np.uint8:
        raise ValueError(f"Unexpected rendered observation: {image.shape}, {image.dtype}")
    return image


def build_policy():
    import safetensors.torch
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from openpi.training.config import ModelTransformFactory
    from openpi import transforms
    from metaworld_policy import MetaworldInputs, MetaworldOutputs

    verified = json.loads((CHECKPOINT / "verified.json").read_text())
    if verified["sha256"] != PROTOCOL["checkpoint_sha256"]:
        raise ValueError("Wrong checkpoint hash")
    config = Pi0Config(action_horizon=PROTOCOL["action_horizon"])
    model = PI0Pytorch(config)
    missing, unexpected = safetensors.torch.load_model(model, str(CHECKPOINT / "model.safetensors"), strict=False)
    atomic_json(ROOT / "logs" / f"weight_load_{os.getpid()}.json",
                {"missing": sorted(missing), "unexpected": sorted(unexpected), "config": str(config)})
    if missing or unexpected:
        raise ValueError(f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.requires_grad_(False)
    model.eval()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    norm_stats = normalize.load(CHECKPOINT / "lerobot/metaworld_mt50")
    model_transforms = ModelTransformFactory()(config)
    return Policy(
        model,
        transforms=[MetaworldInputs(config.model_type), transforms.Normalize(norm_stats, use_quantiles=False),
                    *model_transforms.inputs],
        output_transforms=[*model_transforms.outputs,
                           transforms.Unnormalize(norm_stats, use_quantiles=False), MetaworldOutputs()],
        sample_kwargs={"num_steps": PROTOCOL["denoising_steps"]},
        is_pytorch=True, pytorch_device="cuda:0")


def smoke_env():
    import imageio.v2 as imageio
    import numpy as np
    records = []
    for task in PROTOCOL["tasks"]:
        env, tasks, indices = make_env(task)
        try:
            observation, info = reset_episode(env, tasks[indices[0]], PROTOCOL["episode_seeds"][0])
            observation, reward, terminated, truncated, info = env.step(np.zeros(4))
            image = render_image(env)
            imageio.imwrite(ROOT / "videos" / f"{task}_interface_smoke.png", image)
            record = {"task": task, "observation_shape": list(observation.shape),
                      "action_shape": list(env.action_space.shape), "image_shape": list(image.shape),
                      "info_keys": list(info), "random_action_baseline": False,
                      "note": "one zero action for interface validation; excluded from experiment"}
            records.append(record)
            print(json.dumps(record), flush=True)
        finally:
            env.close()
    atomic_json(ROOT / "logs/env_smoke.json", records)


def evaluate(task, limit=None, output_group="results"):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    torch.set_num_threads(4)
    heartbeat(task, "loading_policy")
    policy = build_policy()
    heartbeat(task, "constructing_environment")
    env, tasks, task_indices = make_env(task)
    prompts = json.loads((ROOT / "sources/rlinf_metaworld_config.json").read_text())["TASK_DESCRIPTIONS"]
    prompt = prompts[task.replace("-v2", "-v3")]
    count = PROTOCOL["trials_per_task"] if limit is None else limit
    try:
        for episode in range(count):
            seed = PROTOCOL["episode_seeds"][episode]
            result_file = ROOT / output_group / task / f"episode_{episode:02d}.json"
            if result_file.exists():
                existing = json.loads(result_file.read_text())
                if existing["protocol_sha256"] != PROTOCOL_HASH:
                    raise ValueError(f"Protocol changed for {result_file}")
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            chosen_task = tasks[task_indices[episode]]
            observation, info = reset_episode(env, chosen_task, seed)
            for _ in range(PROTOCOL["settling_steps"]):
                observation, _, terminated, truncated, info = env.step(np.zeros(4))
                check_simulation(env)
                if terminated or truncated:
                    raise RuntimeError("Environment terminated during settling")
            initial_qpos, initial_qvel = env.data.qpos.copy(), env.data.qvel.copy()
            initial_mocap_pos, initial_mocap_quat = env.data.mocap_pos.copy(), env.data.mocap_quat.copy()
            heartbeat(task, "rollout", episode=episode, seed=seed, step=0)
            action_plan = deque()
            states, actions, rewards, successes, inference_ms = [], [], [], [], []
            full_observations, next_observations, info_records, image_hashes = [], [], [], []
            start = time.monotonic()
            video_file = ROOT / "videos" / output_group / task / f"episode_{episode:02d}.mp4"
            video_file.parent.mkdir(parents=True, exist_ok=True)
            trajectory = result_file.with_suffix(".npz")
            result_file.parent.mkdir(parents=True, exist_ok=True)
            first_success_step = None
            terminated = truncated = False
            with imageio.get_writer(video_file, fps=5, codec="libx264", quality=7) as writer:
                for step in range(PROTOCOL["max_policy_steps"]):
                    image = render_image(env)
                    if step % 4 == 0:
                        writer.append_data(image)
                    states.append(np.asarray(observation[:4], dtype=np.float32).copy())
                    full_observations.append(observation.copy())
                    if not action_plan:
                        image_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                        batch = {"observation/image": image, "observation/state": states[-1], "prompt": prompt}
                        with torch.inference_mode():
                            result = policy.infer(batch)
                        plan = np.asarray(result["actions"], dtype=np.float32)
                        if plan.shape != (PROTOCOL["action_horizon"], 4) or not np.isfinite(plan).all():
                            raise ValueError(f"Invalid model action plan: {plan.shape}")
                        action_plan.extend(plan[:PROTOCOL["action_chunk"]])
                        inference_ms.append(result["policy_timing"]["infer_ms"])
                        heartbeat(task, "rollout", episode=episode, seed=seed, step=step,
                                  memory_allocated_mb=torch.cuda.memory_allocated() / 1048576)
                    action = action_plan.popleft()
                    observation, reward, terminated, truncated, info = env.step(action)
                    check_simulation(env)
                    if not np.isfinite(observation).all() or not np.isfinite(reward):
                        raise ValueError("Non-finite environment observation/reward")
                    success = bool(info["success"])
                    if success and first_success_step is None:
                        first_success_step = step + 1
                    actions.append(action)
                    rewards.append(float(reward))
                    successes.append(success)
                    next_observations.append(observation.copy())
                    info_records.append({key: float(value) for key, value in info.items()
                                         if np.isscalar(value)})
                    if terminated or truncated:
                        break
            np.savez_compressed(trajectory, states=np.asarray(states), actions=np.asarray(actions),
                                rewards=np.asarray(rewards), successes=np.asarray(successes),
                                inference_ms=np.asarray(inference_ms),
                                full_observations=np.asarray(full_observations),
                                next_observations=np.asarray(next_observations),
                                initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                                initial_mocap_pos=initial_mocap_pos, initial_mocap_quat=initial_mocap_quat,
                                image_sha256=np.asarray(image_hashes),
                                info_json=np.asarray(json.dumps(info_records)))
            task_parameters = pickle.loads(chosen_task.data)
            record = {"task": task, "episode": episode, "seed": seed, "task_index": task_indices[episode],
                      "task_data_sha256": hashlib.sha256(chosen_task.data).hexdigest(),
                      "reset_rand_vec": np.asarray(task_parameters["rand_vec"]).tolist(),
                      "success": bool(any(successes)), "success_at_last_step": bool(successes[-1]),
                      "first_success_step": first_success_step, "steps": len(actions),
                      "return": float(sum(rewards)), "terminated": bool(terminated), "truncated": bool(truncated),
                      "elapsed_seconds": time.monotonic() - start, "prompt": prompt,
                      "mean_inference_ms": float(np.mean(inference_ms)),
                      "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576,
                      "video": str(video_file), "trajectory": str(trajectory),
                      "video_sha256": file_sha256(video_file), "trajectory_sha256": file_sha256(trajectory),
                      "environment_seed_applied": seed, "simulation_valid": True,
                      "rollout_source_sha256": file_sha256(__file__),
                      "checkpoint_sha256": PROTOCOL["checkpoint_sha256"], "protocol_sha256": PROTOCOL_HASH}
            atomic_json(result_file, record)
            heartbeat(task, "episode_complete", episode=episode, seed=seed, success=record["success"])
            print(json.dumps(record), flush=True)
        heartbeat(task, "complete", episodes=count)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=PROTOCOL["tasks"])
    parser.add_argument("--env-smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-group", default="results")
    args = parser.parse_args()
    try:
        if args.env_smoke:
            smoke_env()
        else:
            evaluate(args.task, args.limit, args.output_group)
    except Exception:
        error = traceback.format_exc()
        heartbeat(args.task or "env_smoke", "error", error=error)
        print(error, file=sys.stderr, flush=True)
        raise

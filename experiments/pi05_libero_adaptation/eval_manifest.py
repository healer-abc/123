"""Evaluate one checkpoint on frozen LIBERO manifest rows with paired policy noise."""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests/eval_manifest.csv"
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def axis_angle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def make_env(task, seed):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env


def load_rows(phase, suites):
    with MANIFEST.open() as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row["phase"] == phase and row["suite"] in suites]


def prepare_image(image):
    image = np.ascontiguousarray(image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))


def run_episode(client, row, task_suite, env, output, checkpoint, config, heartbeat=lambda **kwargs: None):
    task_id = int(row["task_id"])
    state_index = int(row["init_state_index"])
    initial_states = np.asarray(task_suite.get_task_init_states(task_id))
    initial_state = initial_states[state_index]
    assert hashlib.sha256(np.ascontiguousarray(initial_state).tobytes()).hexdigest() == row["init_state_sha256"]
    env_seed = int(row["env_seed"])
    policy_seed = int(row["policy_seed"])
    np.random.seed(env_seed)
    env.seed(env_seed)
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(int(row["settling_steps"])):
        observation, _, done, _ = env.step(DUMMY_ACTION)
        if done:
            raise RuntimeError("environment succeeded/terminated during settling")

    rng = np.random.default_rng(policy_seed)
    queue = collections.deque()
    actions, inference_ms, frames = [], [], []
    success = False
    start = time.monotonic()
    for _step in range(int(row["max_policy_steps"])):
        if _step % 25 == 0:
            heartbeat(stage="rollout", row=row, step=_step)
        image = prepare_image(observation["agentview_image"])
        wrist = prepare_image(observation["robot0_eye_in_hand_image"])
        frames.append(image)
        if not queue:
            state = np.concatenate((observation["robot0_eef_pos"],
                                    axis_angle(observation["robot0_eef_quat"]),
                                    observation["robot0_gripper_qpos"]))
            noise = rng.standard_normal((10, 32), dtype=np.float32)
            result = client.infer({"observation/image": image,
                                   "observation/wrist_image": wrist,
                                   "observation/state": state,
                                   "prompt": row["task_description"],
                                   "_policy_noise": noise})
            plan = np.asarray(result["actions"], dtype=np.float32)
            if plan.ndim != 2 or plan.shape[0] < int(row["replan_steps"]) or plan.shape[1] != 7:
                raise ValueError(f"invalid action plan shape {plan.shape}")
            if not np.isfinite(plan).all():
                raise FloatingPointError("non-finite policy action")
            queue.extend(plan[:int(row["replan_steps"])])
            inference_ms.append(float(result["server_timing"]["infer_ms"]))
        action = np.asarray(queue.popleft(), dtype=np.float32)
        actions.append(action)
        observation, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break

    episode_dir = output / "episodes" / row["suite"] / f"task_{task_id:02d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    stem = f"state_{state_index:02d}"
    trajectory = episode_dir / f"{stem}.npz"
    raw_actions = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(trajectory, actions=raw_actions,
                        motion_controller_inputs=np.clip(raw_actions[:, :6], -1.0, 1.0),
                        gripper_sign=np.sign(raw_actions[:, 6]),
                        inference_ms=np.asarray(inference_ms, dtype=np.float32))
    video = None
    video_hash = None
    if not success:
        video = output / "videos" / row["suite"] / f"task_{task_id:02d}_{stem}_failure.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=10)
        video_hash = sha256(video)
    return {**row, "task_id": task_id, "init_state_index": state_index,
            "env_seed": env_seed, "policy_seed": policy_seed,
            "status": "valid", "success": success, "steps": len(actions),
            "policy_inferences": len(inference_ms),
            "mean_inference_ms": float(np.mean(inference_ms)),
            "elapsed_seconds": time.monotonic() - start,
            "max_abs_action": float(np.max(np.abs(actions))),
            "action_out_of_nominal_bounds": bool(np.max(np.abs(actions)) > 1.000001),
            "raw_bound_strict_success": bool(success and np.max(np.abs(actions)) <= 1.000001),
            "motion_bound_exceeding_steps": int(np.sum(np.any(np.abs(raw_actions[:, :6]) > 1.000001, axis=1))),
            "gripper_bound_exceeding_steps": int(np.sum(np.abs(raw_actions[:, 6]) > 1.000001)),
            "action_audit_note": "motion_controller_inputs and gripper_sign are controller-input proxies, not actuator torques",
            "checkpoint": checkpoint, "config": config,
            "manifest_sha256": sha256(MANIFEST),
            "trajectory": str(trajectory), "trajectory_sha256": sha256(trajectory),
            "video": str(video) if video else None, "video_sha256": video_hash}


def aggregate(output, rows, filename="summary.json"):
    records = []
    for row in rows:
        path = output / "episodes" / row["suite"] / f"task_{int(row['task_id']):02d}" / f"state_{int(row['init_state_index']):02d}.json"
        if path.is_file():
            records.append(json.loads(path.read_text()))
    table = []
    keys = sorted({(r["suite"], r["task_id"]) for r in records})
    for suite, task_id in keys:
        subset = [r for r in records if r["suite"] == suite and r["task_id"] == task_id]
        table.append({"suite": suite, "task_id": task_id, "completed": len(subset),
                      "successes": sum(r["success"] for r in subset),
                      "success_rate": sum(r["success"] for r in subset) / len(subset)})
    suite_rates = {}
    for suite in sorted({r["suite"] for r in records}):
        task_rates = [x["success_rate"] for x in table if x["suite"] == suite]
        suite_rates[suite] = sum(task_rates) / len(task_rates) if task_rates else None
    summary = {"complete": len(records) == len(rows), "completed": len(records),
               "planned": len(rows), "valid": sum(r["status"] == "valid" for r in records),
               "successes": sum(r["success"] for r in records),
               "raw_bound_strict_successes": sum(bool(r["success"] and not r["action_out_of_nominal_bounds"]) for r in records),
               "suite_rates": suite_rates, "tasks": table,
               "manifest_sha256": sha256(MANIFEST)}
    atomic_json(output / filename, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", default="development")
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_10"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    rows = [r for r in load_rows(args.phase, set(args.suites))
            if int(r["task_id"]) % args.shard_count == args.shard_index]
    output = ROOT / "evaluations" / args.run_id / args.phase
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"shard_{args.shard_index}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_name = "summary.json" if args.shard_count == 1 else f"summary.shard_{args.shard_index}.json"
    heartbeat_path = output / f"heartbeat.shard_{args.shard_index}.json"

    def heartbeat(**kwargs):
        atomic_json(heartbeat_path, {"pid": os.getpid(), "timestamp": time.time(),
                                    "shard_index": args.shard_index, **kwargs})

    heartbeat(stage="connecting")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("config") != args.config or metadata.get("checkpoint") != args.checkpoint or not metadata.get("explicit_noise"):
        raise ValueError(f"wrong policy server metadata: {metadata}")
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        for suite in args.suites:
            task_suite = benchmark_dict[suite]()
            suite_rows = [r for r in rows if r["suite"] == suite]
            for task_id in range(task_suite.n_tasks):
                task_rows = [r for r in suite_rows if int(r["task_id"]) == task_id]
                if not task_rows:
                    continue
                task = task_suite.get_task(task_id)
                heartbeat(stage="creating_environment", row=task_rows[0])
                env = make_env(task, int(task_rows[0]["env_seed"]))
                try:
                    for row in task_rows:
                        path = output / "episodes" / suite / f"task_{task_id:02d}" / f"state_{int(row['init_state_index']):02d}.json"
                        if path.is_file():
                            previous = json.loads(path.read_text())
                            if (previous.get("status") != "valid" or previous.get("checkpoint") != args.checkpoint
                                    or previous.get("config") != args.config
                                    or previous.get("manifest_sha256") != sha256(MANIFEST)
                                    or any(str(previous.get(key)) != str(value) for key, value in row.items())):
                                raise ValueError(f"incompatible existing episode: {path}")
                            continue
                        attempt_dir = output / "attempts" / suite / f"task_{task_id:02d}" / path.stem
                        attempt_dir.mkdir(parents=True, exist_ok=True)
                        attempt_number = len(list(attempt_dir.glob("attempt_*.json"))) + 1
                        if attempt_number > 2:
                            raise RuntimeError(f"episode retry limit reached: {path}")
                        attempt_path = attempt_dir / f"attempt_{attempt_number}.json"
                        attempt = {"row": row, "attempt": attempt_number, "started_at": time.time(),
                                   "pid": os.getpid(), "status": "running"}
                        atomic_json(attempt_path, attempt)
                        heartbeat(stage="starting_episode", row=row)
                        try:
                            record = run_episode(client, row, task_suite, env, output,
                                                 args.checkpoint, args.config, heartbeat)
                        except Exception:
                            error = traceback.format_exc()
                            failure = {**attempt, "status": "error", "error": error, "ended_at": time.time()}
                            atomic_json(attempt_path, failure)
                            atomic_json(output / f"last_error.shard_{args.shard_index}.json", failure)
                            raise
                        atomic_json(path, record)
                        atomic_json(attempt_path, {**attempt, "status": "complete", "ended_at": time.time()})
                        heartbeat(stage="episode_complete", row=row)
                        print(json.dumps({"event": "episode_complete", "run_id": args.run_id,
                                          "suite": suite, "task_id": task_id,
                                          "state": row["init_state_index"], "success": record["success"]}), flush=True)
                        aggregate(output, rows, summary_name)
                finally:
                    env.close()
    finally:
        summary = aggregate(output, rows, summary_name)
        heartbeat(stage="finished", complete=summary["complete"])
        print(json.dumps({"event": "evaluation_status", **summary}), flush=True)


if __name__ == "__main__":
    main()

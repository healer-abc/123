"""Record per-seed evaluation results without changing native task dynamics."""
import json
import os
from pathlib import Path
import threading
import types

from robotwin_pi0_ext import build_openpi


def build_eval_model(cfg, torch_dtype):
    import torch
    model = build_openpi(cfg, torch_dtype)
    native_predict = model.predict_action_batch
    model._audit_generator = None

    def predict(self, *args, **kwargs):
        if self._audit_generator is None:
            device = next(self.parameters()).device
            self._audit_generator = torch.Generator(device=device).manual_seed(
                20261026 + int(os.environ.get("RANK", "0")))
        kwargs.setdefault("rng", self._audit_generator)
        return native_predict(*args, **kwargs)

    model.predict_action_batch = types.MethodType(predict, model)
    return model


def register():
    from rlinf.models import register_model
    register_model("openpi", build_eval_model, category="embodied", force=True)
    import robotwin.envs.vector_env as vector
    import rlinf.envs.sim.robotwin.robotwin_env as module

    if getattr(module.RoboTwinEnv, "_pi0_audited", False):
        return
    context = threading.local()
    native_factory = vector.class_decorator
    native_setup_task = vector.SubEnv.setup_task

    def factory(task_name):
        task = native_factory(task_name)
        setup = task.setup_demo

        def setup_demo(*args, **kwargs):
            result = setup(*args, **kwargs)
            task._audit_actual_seed = int(kwargs["seed"])
            context.seed = task._audit_actual_seed
            return result

        task.setup_demo = setup_demo
        return task

    def setup_task(sub):
        native_setup_task(sub)
        sub._audit_metadata_seed = context.seed

    vector.class_decorator = factory
    vector.SubEnv.setup_task = setup_task
    native_env = module.RoboTwinEnv

    class AuditedRoboTwinEnv(native_env):
        _pi0_audited = True

        def __init__(self, *args, **kwargs):
            self._audit_scenes = {}
            self._audit_completed = set()
            super().__init__(*args, **kwargs)

        def _write(self, row):
            folder = Path(os.environ["ROBOTWIN_EVAL_AUDIT_DIR"])
            folder.mkdir(parents=True, exist_ok=True)
            row.update(worker=self.seed_offset, pid=os.getpid())
            with (folder / f"episodes-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        def reset(self, env_idx=None, env_seeds=None):
            result = super().reset(env_idx=env_idx, env_seeds=env_seeds)
            indices = range(self.num_envs) if env_idx is None else env_idx
            for index in indices:
                sub = self.venv.envs[index]
                requested = int(self.reset_state_ids[index])
                actual = sub.task._audit_actual_seed
                metadata_seed = sub._audit_metadata_seed
                metadata_matches = sub.episode_info_list[0] == sub.task.get_info()
                self._audit_scenes[index] = {
                    "env_index": index, "requested_seed": requested,
                    "actual_seed": actual, "metadata_seed": metadata_seed,
                    "metadata_matches_scene": metadata_matches,
                    "valid_scene": requested == actual == metadata_seed and metadata_matches,
                    "instruction": sub.get_instruction(), "task_info": sub.task.get_info(),
                }
                self._write({"event": "reset", **self._audit_scenes[index]})
            return result

        def chunk_step(self, actions):
            import torch
            assert torch.isfinite(torch.as_tensor(actions)).all(), "Nonfinite policy actions"
            result = super().chunk_step(actions)
            done = torch.logical_or(result[2], result[3]).any(dim=1)
            for index in torch.where(done)[0].tolist():
                scene = self._audit_scenes[index]
                key = (index, scene["requested_seed"])
                if key in self._audit_completed:
                    continue
                self._write({"event": "episode", **scene,
                             "success_once": bool(self.success_once[index]),
                             "episode_len": int(self.elapsed_steps[index]),
                             "return": float(self.returns[index])})
                self._audit_completed.add(key)
            return result

    module.RoboTwinEnv = AuditedRoboTwinEnv

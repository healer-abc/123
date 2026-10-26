"""Fresh official SimpleDP3 prefix: stop200, snapshot completed50/100/200, DDIM5."""
import difflib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0, str(BASE))
from runtime import atomic_json, bootstrap, sha256


def main():
    task, attempt = sys.argv[1], int(sys.argv[2])
    plan = json.loads((HERE / 'plan.json').read_text())
    for name, digest in plan['execution_files'].items():
        assert sha256(HERE / name) == digest
    assert sha256(BASE / 'worker.py') == plan['baseline_worker_sha256']
    source = BASE / 'sources/dp3/3D-Diffusion-Policy-master/3D-Diffusion-Policy/train.py'
    assert sha256(source) == plan['official_training_sha256']
    out = HERE / 'training' / task / f'attempt_{attempt:02d}'
    out.mkdir(parents=True, exist_ok=False)
    bootstrap()
    spec = importlib.util.spec_from_file_location('rollout', HERE / 'rollout.py')
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules['rollout'] = evaluator
    spec.loader.exec_module(evaluator)
    spec = importlib.util.spec_from_file_location('original_baseline_worker', BASE / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    original_load = worker.load_file
    def load_file(name, path):
        module = original_load(name, path)
        if name != 'official_dp3_train':
            return module
        cls = module.TrainDP3Workspace
        before = textwrap.dedent(inspect.getsource(cls.run))
        assert before.count('range(cfg.training.num_epochs)') == 1
        after = before.replace('range(cfg.training.num_epochs)', 'range(200)')
        increment = '        self.epoch += 1\n'
        assert after.count(increment) == 1
        after = after.replace(increment, increment + '''        if self.epoch in (50,100,200):
            if self._saving_thread is not None:
                self._saving_thread.join()
            snapshot=self.save_checkpoint(tag=f'epoch_{self.epoch:04d}',use_thread=False)
            snapshot_atomic_json(pathlib.Path(self.output_dir)/'snapshot_receipts'/f'epoch_{self.epoch:04d}.json',
                {'checkpoint':str(snapshot),'sha256':snapshot_sha256(snapshot),
                 'completed_epochs':self.epoch,'optimizer_updates':self.global_step,
                 'lr_schedule_epochs':1000,'num_inference_steps':5,'horizon':16,
                 'includes_model_ema_optimizer':True})
''')
        module.__dict__.update(snapshot_atomic_json=atomic_json, snapshot_sha256=sha256)
        (out / 'original_run_source.py').write_text(before)
        (out / 'training_run_source.py').write_text(after)
        (out / 'loop_override.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='official_training_loop', tofile='same_LR_schedule_stop200_snapshots50_100_200')))
        ns = {}
        exec(compile(after, str(path) + '::prefix200', 'exec'), module.__dict__, ns)
        loop = ns['run']
        def checked_run(self):
            cfg = self.cfg
            assert cfg.training.num_epochs == 1000 and cfg.training.seed == 0
            assert not cfg.training.resume and not cfg.training.debug
            assert cfg.dataloader.batch_size == 128 and cfg.training.gradient_accumulate_every == 1
            assert cfg.training.lr_warmup_steps == 500 and cfg.policy.num_inference_steps == 5
            assert cfg.horizon == 16 and cfg.n_obs_steps == 2 and cfg.n_action_steps == 8
            loop(self)
            assert self.epoch == 200 and self.global_step == 12200
        cls.run = checked_run
        return module
    worker.load_file = load_file
    # Keep the original config/optimizer/model construction. Change inference steps before instantiation.
    before = textwrap.dedent(inspect.getsource(worker.train_diffusion))
    anchor = '    cfg.training.num_epochs = 1000\n'
    assert before.count(anchor) == 1
    after = before.replace(anchor, anchor + '    cfg.policy.num_inference_steps = 5\n')
    cut = after.index('    policy = ws.ema_model\n')
    after = after[:cut] + "    wandb.finish()\n"  # Screening and selected final evaluation are supervised separately.
    (out / 'worker_override.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='baseline_worker.train_diffusion', tofile='fresh200_DDIM5_training_only')))
    ns = {}
    exec(compile(after, str(BASE / 'worker.py') + '::train200', 'exec'), worker.__dict__, ns)

    reference_target = json.loads((HERE.parent / 'metaworld_simpledp3_epoch500_ddim10_20261020' / 'targets' / (task+'.json')).read_text())
    with (Path(reference_target['output']) / 'metrics.jsonl').open() as f:
        reference = [json.loads(next(f)) for _ in range(20)]
    original_install = worker.install_logging
    observed = []
    def install_logging(path):
        original_install(path)
        import wandb
        original_init = wandb.init
        def init(*args, **kwargs):
            run = original_init(*args, **kwargs)
            original_log = run.log
            def log(data, *args, **kwargs):
                step = int(data.get('global_step', -1))
                if 0 <= step < 20 and len(observed) == step:
                    a, b = float(data['train_loss']), float(reference[step]['train_loss'])
                    observed.append({'step': step, 'reference_loss': b, 'actual_loss': a, 'abs_difference': abs(a-b)})
                    assert abs(a-b) <= 1e-5 + 1e-4*abs(b)
                    assert abs(float(data['lr']) - float(reference[step]['lr'])) < 1e-12
                    if len(observed) == 20:
                        atomic_json(out / 'prefix_audit.json', {'passed': True, 'first_updates': 20, 'checks': observed,
                            'note': 'Initial optimization prefix only. Training starts from scratch; inference DDIM5 fixed for development and final evaluation.'})
                return original_log(data, *args, **kwargs)
            run.log = log
            return run
        wandb.init = init
    worker.install_logging = install_logging
    atomic_json(out / 'training_identity.json', {'created': time.time(), 'task': task,
        'training_seed': 0, 'initialization': 'fresh; no prior checkpoint loaded', 'stop_completed_epochs': 200,
        'snapshot_epochs': [50, 100, 200], 'lr_schedule_epochs': 1000,
        'num_inference_steps': 5, 'horizon': 16, 'source_plan_sha256': sha256(HERE / 'plan.json')})
    ns['train_diffusion']('SimpleDP3', task, out, smoke=False)
    assert len(observed) == 20
    import torch
    import dill
    for epoch in [50, 100, 200]:
        receipt = json.loads((out / 'snapshot_receipts' / f'epoch_{epoch:04d}.json').read_text())
        assert receipt['completed_epochs'] == epoch and receipt['optimizer_updates'] == 61*epoch
        assert sha256(receipt['checkpoint']) == receipt['sha256']
        payload = torch.load(receipt['checkpoint'], pickle_module=dill, map_location='cpu', weights_only=False)
        assert {'model','ema_model','optimizer'} <= set(payload['state_dicts'])
        assert dill.loads(payload['pickles']['epoch']) == epoch and dill.loads(payload['pickles']['global_step']) == 61*epoch
        assert payload['cfg'].policy.num_inference_steps == 5 and payload['cfg'].horizon == 16
        del payload
    atomic_json(out / 'training_complete.json', {'completed_epochs': 200, 'optimizer_updates': 12200,
        'snapshot_epochs_verified': [50,100,200], 'training_seed': 0, 'num_inference_steps': 5, 'horizon': 16})


if __name__ == '__main__':
    main()

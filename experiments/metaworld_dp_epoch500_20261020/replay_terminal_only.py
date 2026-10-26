"""DP terminal-only save override for jobs not yet started; original scientific configuration retained."""
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
from runtime import atomic_json, sha256, bootstrap


def save_resume_context(workspace, scheduler, ema, train_loader, val_loader, sample_batch, runner):
    """Save local training state missing from the upstream workspace checkpoint."""
    import random
    import numpy as np
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    path = Path(workspace.output_dir) / 'checkpoints' / f'epoch_{workspace.epoch:04d}_resume.pt'
    context = {'format_version': 1, 'completed_epochs': workspace.epoch,
        'global_step': workspace.global_step, 'lr_schedule_epochs': 1000,
        'lr_scheduler': scheduler.state_dict(),
        'ema_state': {k: v for k, v in ema.__dict__.items() if k != 'averaged_model'},
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()},
        'train_sampling_batch': dict_apply(sample_batch, lambda x: x.detach().cpu()),
        'runner_state': {'round_id': runner.round_id, 'scores': runner.scores},
        'train_loader_generator': train_loader.generator.get_state() if train_loader.generator is not None else None,
        'val_loader_generator': val_loader.generator.get_state() if val_loader.generator is not None else None,
        'resume_notes': 'Epoch boundary; dataset __getitem__ is deterministic. Recreate persistent loaders without consuming restored global RNG for a new worker base seed. Restore scheduler, EMA counters, sampling batch, runner and RNG before the next epoch. No continuation authorized yet.'}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(context, temporary)
    temporary.replace(path)
    return {'resume_context': str(path), 'resume_context_sha256': sha256(path)}


def main():
    task, attempt = sys.argv[1], int(sys.argv[2])
    override = json.loads((HERE / 'operations/no_intermediate_snapshots/manifest.json').read_text())
    assert sha256(__file__) == override['terminal_driver_sha256']
    assert task in override['tasks']
    plan = json.loads((HERE / 'plan.json').read_text())
    for rel, digest in plan['execution_files'].items():
        assert sha256(HERE / rel) == digest, 'Changed execution source: ' + rel
    assert sha256(BASE / 'worker.py') == plan['baseline_worker_sha256']
    assert sha256(BASE / 'protocol.json') == plan['baseline_protocol_sha256']
    source = BASE / plan['official_training_relative_path']
    assert sha256(source) == plan['official_training_sha256']
    out = HERE / 'runs' / task / f'attempt_{attempt:02d}'
    out.mkdir(parents=True, exist_ok=False)
    bootstrap()
    spec = importlib.util.spec_from_file_location('rollout', HERE / 'rollout500.py')
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules['rollout'] = evaluator
    spec.loader.exec_module(evaluator)
    from diffusion_policy.workspace import train_diffusion_unet_image_workspace as official
    cls = official.TrainDiffusionUnetImageWorkspace
    before = textwrap.dedent(inspect.getsource(cls.run))
    assert before.count('range(cfg.training.num_epochs)') == 1
    after = before.replace('range(cfg.training.num_epochs)', 'range(500)')
    increment = '            self.epoch += 1\n'
    assert after.count(increment) == 1
    after = after.replace(increment, increment + '''            if self.epoch == 500:
                if self._saving_thread is not None:
                    self._saving_thread.join()
                snapshot = self.save_checkpoint(tag=f'epoch_{self.epoch:04d}', use_thread=False)
                resume_context = snapshot_resume_writer(self, lr_scheduler, ema, train_dataloader,
                    val_dataloader, train_sampling_batch, env_runner)
                snapshot_atomic_json(pathlib.Path(self.output_dir) / 'snapshot_receipts' / f'epoch_{self.epoch:04d}.json',
                    {'checkpoint': str(snapshot), 'sha256': snapshot_sha256(snapshot),
                     'completed_epochs': self.epoch, 'optimizer_updates': self.global_step,
                     'lr_schedule_epochs': 1000, 'includes_model_ema_optimizer': True, **resume_context})
''')
    official.__dict__.update(snapshot_atomic_json=atomic_json, snapshot_sha256=sha256,
                            snapshot_resume_writer=save_resume_context)
    (out / 'original_run_source.py').write_text(before)
    (out / 'replay_run_source.py').write_text(after)
    (out / 'loop_override.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile='official1000_training_loop', tofile='same_schedule_stop500_save300_400_500')))
    ns = {}
    exec(compile(after, str(source) + '::prefix500', 'exec'), official.__dict__, ns)
    replay = ns['run']
    def checked_run(self):
        cfg = self.cfg
        cfg.checkpoint.save_last_ckpt = False
        cfg.checkpoint.save_last_snapshot = False
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, out / 'config.yaml', resolve=True)
        assert cfg.training.num_epochs == 1000 and cfg.training.seed == 0
        assert not cfg.training.debug and not cfg.training.resume
        assert cfg.dataloader.batch_size == 128 and cfg.training.gradient_accumulate_every == 1
        assert cfg.training.lr_warmup_steps == 500 and cfg.policy.num_inference_steps == 10
        assert cfg.horizon == 16 and cfg.n_obs_steps == 2 and cfg.n_action_steps == 8
        replay(self)
        assert self.epoch == 500 and self.global_step == 30500
        # Validate snapshots before evaluating, using the serialized counters and original schema.
        import torch
        import dill
        for epoch in [500]:
            receipt = json.loads((out / 'snapshot_receipts' / f'epoch_{epoch:04d}.json').read_text())
            assert receipt['completed_epochs'] == epoch and receipt['optimizer_updates'] == 61 * epoch
            assert sha256(receipt['checkpoint']) == receipt['sha256']
            payload = torch.load(receipt['checkpoint'], pickle_module=dill, map_location='cpu', weights_only=False)
            assert {'model', 'ema_model', 'optimizer'} <= set(payload['state_dicts'])
            assert dill.loads(payload['pickles']['epoch']) == epoch
            assert dill.loads(payload['pickles']['global_step']) == 61 * epoch
            del payload
            assert sha256(receipt['resume_context']) == receipt['resume_context_sha256']
            context = torch.load(receipt['resume_context'], map_location='cpu', weights_only=False)
            assert context['completed_epochs'] == epoch and context['global_step'] == 61 * epoch
            assert context['lr_scheduler']['last_epoch'] == 61 * epoch
            assert context['ema_state']['optimization_step'] == 61 * epoch
            assert {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= set(context['rng'])
            assert {'obs', 'action'} <= set(context['train_sampling_batch'])
            del context
        last = json.loads((out / 'metrics.jsonl').read_text().splitlines()[-1])
        assert 4e-5 < float(last['lr']) < 6e-5, 'Original1000-epoch LR schedule was not retained'
        atomic_json(out / 'training_audit.json', {'passed': True, 'completed_epochs': 500,
            'optimizer_updates': 30500, 'snapshot_epochs': [500],
            'lr_schedule_epochs': 1000, 'lr_at_stop': float(last['lr']), 'resume_contexts_verified': True,
            'continue_training_only_on_future_user_instruction': True})
    cls.run = checked_run
    original_save = cls.save_checkpoint
    def save_checkpoint(self, *args, **kwargs):
        if kwargs.get('tag') == 'final':
            checkpoint = Path(self.output_dir) / 'checkpoints/epoch_0500.ckpt'
            final = Path(self.output_dir) / 'checkpoints/final.ckpt'
            assert checkpoint.is_file() and not final.exists()
            final.hardlink_to(checkpoint)
            return str(final.absolute())
        return original_save(self, *args, **kwargs)
    cls.save_checkpoint = save_checkpoint

    spec = importlib.util.spec_from_file_location('frozen_baseline_worker', BASE / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    reference_job = next(j for j in json.loads((HERE / 'preserved_baseline/status_before_pause.json').read_text())['jobs']
                         if j['key'] == 'DP_' + task)
    reference = []
    if reference_job.get('out'):
        with (Path(reference_job['out']) / 'metrics.jsonl').open() as f:
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
                if reference and 0 <= step < 20 and len(observed) == step:
                    a, b = float(data['train_loss']), float(reference[step]['train_loss'])
                    observed.append({'step': step, 'reference_loss': b, 'replay_loss': a, 'abs_difference': abs(a-b)})
                    assert abs(a-b) <= 1e-5 + 1e-4 * abs(b), 'Replay prefix differs from reference baseline'
                    assert abs(float(data['lr']) - float(reference[step]['lr'])) < 1e-12
                    if len(observed) == 20:
                        atomic_json(out / 'prefix_audit.json', {'passed': True, 'first_updates': 20, 'checks': observed,
                            'note': 'Initial prefix check; no claim of bitwise equivalence for all500 epochs.'})
                return original_log(data, *args, **kwargs)
            run.log = log
            return run
        wandb.init = init
    worker.install_logging = install_logging
    atomic_json(out / 'replay_identity.json', {'created': time.time(), 'task': task, 'method': 'DP',
        'training_seed': 0, 'initialization': 'fresh, resume=False; no old weights loaded',
        'completed_epoch_target': 500, 'snapshot_epochs': [500], 'lr_schedule_epochs': 1000,
        'source_plan_sha256': sha256(HERE / 'plan.json'), 'reference': reference_job})
    worker.train_diffusion('DP', task, out, smoke=False)
    assert not reference or len(observed) == 20


if __name__ == '__main__':
    main()

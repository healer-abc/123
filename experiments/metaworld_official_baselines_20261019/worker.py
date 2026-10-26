"""Official training entrypoints plus explicit MetaWorld I/O adapters."""
import argparse
import ast
import copy
import importlib.util
import json
import math
import numbers
import os
from pathlib import Path
import pickle
import sys
import time
from runtime import ROOT, DP3, DP, ACT, atomic_json, bootstrap, install_render_fix, sha256

bootstrap()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def heartbeat(out, **kw):
    atomic_json(Path(out) / 'heartbeat.json', dict(time=time.time(), pid=os.getpid(), **kw))


def install_logging(out):
    import wandb
    old_init = wandb.init
    def init(*a, **kw):
        run = old_init(*a, **kw)
        original = run.log
        def log(data, *args, **kwargs):
            values = {}
            for k, v in data.items():
                if isinstance(v, numbers.Real):
                    if not math.isfinite(float(v)):
                        raise FloatingPointError(f'Nonfinite metric {k}={v}')
                    values[k] = float(v)
            with open(Path(out) / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(dict(time=time.time(), **values)) + '\n')
            heartbeat(out, stage='training', **values)
            return original(data, *args, **kwargs)
        run.log = log
        return run
    wandb.init = init


def collect(task, out):
    import numpy as np
    import torch
    import zarr
    from rollout import seed_all
    install_render_fix()
    target = ROOT / 'datasets' / f'metaworld_{task}_expert.zarr'
    receipt = ROOT / 'datasets' / f'{task}.json'
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite dataset {target}')
    heartbeat(out, stage='collection', task=task)
    seed_all(42)
    module = load_file('official_generator', DP3 / 'third_party/Metaworld/gen_demonstration_expert.py')
    module.main(argparse.Namespace(env_name=task, num_episodes=10, root_dir=str(target.parent)))
    data = zarr.open(str(target), 'r')
    assert len(data['meta/episode_ends']) == 10
    assert all(np.isfinite(data['data'][k][:]).all() for k in ['action', 'state', 'point_cloud'])
    hashes = {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}
    atomic_json(receipt, {'task': task, 'collection_seed': 42, 'episodes': 10,
        'source': 'official DP3 MetaWorld expert generator, unmodified success filter',
        'episode_ends': data['meta/episode_ends'][:].tolist(),
        'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in data['data'].arrays()},
        'file_hashes': hashes})
    convert_act(task)
    atomic_json(Path(out) / 'complete.json', {'stage': 'collect', 'task': task, 'receipt': str(receipt)})


def convert_act(task):
    """Lossless ACT view of exactly the same officially generated demonstrations."""
    import numpy as np
    import zarr
    import h5py
    data = zarr.open(str(ROOT / 'datasets' / f'metaworld_{task}_expert.zarr'), 'r')
    act_dir = ROOT / 'datasets' / ('act_' + task)
    act_dir.mkdir(exist_ok=False)
    ends = data['meta/episode_ends'][:]
    for i, end in enumerate(ends):
        start = int(ends[i-1]) if i else 0
        with h5py.File(act_dir / f'episode_{i}.hdf5', 'w') as f:
            f.attrs['sim'] = True
            q = data['data/state'][start:end]
            f.create_dataset('observations/qpos', data=q)
            f.create_dataset('observations/qvel', data=np.zeros_like(q))
            images = data['data/img'][start:end]
            assert images.shape[1:] == (128, 128, 3)
            f.create_dataset('observations/images/corner2', data=images, compression='gzip')
            f.create_dataset('action', data=data['data/action'][start:end])


def train_diffusion(method, task, out, smoke=False):
    import hydra
    from omegaconf import OmegaConf, open_dict
    import torch
    import wandb
    install_render_fix()
    install_logging(out)
    if method == 'SimpleDP3':
        config_dir = DP3 / '3D-Diffusion-Policy/diffusion_policy_3d/config'
        with hydra.initialize_config_dir(str(config_dir), version_base=None):
            cfg = hydra.compose(config_name='simple_dp3', overrides=['task=metaworld_' + task])
        module = load_file('official_dp3_train', DP3 / '3D-Diffusion-Policy/train.py')
        Workspace = module.TrainDP3Workspace
        with open_dict(cfg):
            cfg.task.env_runner._target_ = 'simple_adapter.Runner'
            cfg.task.dataset.zarr_path = str(ROOT / 'datasets' / f'metaworld_{task}_expert.zarr')
            cfg.checkpoint.save_ckpt = True
    else:
        config_dir = DP / 'diffusion_policy/config'
        with hydra.initialize_config_dir(str(config_dir), version_base=None):
            cfg = hydra.compose(config_name='train_diffusion_unet_image_workspace')
        from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
        Workspace = TrainDiffusionUnetImageWorkspace
        with open_dict(cfg):
            cfg.task = {'name': task, 'shape_meta': {
                'obs': {'image': {'shape': [3, 128, 128], 'type': 'rgb'},
                        'agent_pos': {'shape': [9], 'type': 'low_dim'}},
                'action': {'shape': [4]}},
                'env_runner': {'_target_': 'dp_adapter.Runner', 'task_name': task},
                'dataset': {'_target_': 'dp_adapter.Dataset',
                    'zarr_path': str(ROOT / 'datasets' / f'metaworld_{task}_expert.zarr'),
                    'horizon': '${horizon}', 'pad_before': "${eval:'${n_obs_steps}-1'}",
                    'pad_after': "${eval:'${n_action_steps}-1'}", 'seed': 42, 'val_ratio': 0.02}}
            cfg.task_name = task
    cfg.training.seed = 0  # published example seed, common to all methods
    cfg.training.device = 'cuda:0'
    cfg.training.resume = False  # attempts are immutable; retries start fresh, same seed/config
    cfg.logging.mode = 'offline'
    cfg.logging.name = method + '_' + task
    cfg.logging.project = 'metaworld_official_baselines'
    cfg.exp_name = method + '_' + task
    OmegaConf.save(cfg, Path(out) / 'config.yaml', resolve=True)
    ws = Workspace(cfg, output_dir=str(out))
    if smoke:
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        ws.model.set_normalizer(dataset.get_normalizer())
        ws.model.cuda()
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.batch_size,
            shuffle=True, num_workers=0)))
        from diffusion_policy_3d.common.pytorch_util import dict_apply
        batch = dict_apply(batch, lambda t: t.cuda())
        ws.model.train()
        result = ws.model.compute_loss(batch)
        loss = result[0] if isinstance(result, tuple) else result
        assert torch.isfinite(loss)
        loss.backward()
        ws.optimizer.step()
        ws.model.eval()
        with torch.no_grad():
            pred = ws.model.predict_action(batch['obs'])['action']
        assert pred.shape[-1] == 4 and torch.isfinite(pred).all()
        atomic_json(Path(out) / 'complete.json', {'stage': 'smoke', 'method': method,
            'task': task, 'loss': loss.item(), 'action_shape': list(pred.shape),
            'batch_size': cfg.dataloader.batch_size,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})
        return
    heartbeat(out, stage='training', epoch=0)
    ws.run()  # unmodified upstream training loop, optimizer, scheduler, EMA and losses
    # Upstream periodic checkpoints do not necessarily include terminal epoch.
    ws.save_checkpoint(tag='final', use_thread=False)
    if getattr(ws, '_saving_thread', None):
        ws._saving_thread.join()
    checkpoint = Path(out) / 'checkpoints/final.ckpt'
    atomic_json(Path(out) / 'trained.json', {'checkpoint': str(checkpoint),
        'sha256': sha256(checkpoint), 'completed_epochs': ws.epoch,
        'global_step': ws.global_step, 'selection': 'fixed terminal epoch, EMA'})
    policy = ws.ema_model
    policy.cuda().eval()
    from rollout import evaluate
    result = evaluate(policy, task, method, Path(out) / 'final_rollouts', final=True, checkpoint=checkpoint)
    atomic_json(Path(out) / 'complete.json', result)
    wandb.finish()


def act_functions():
    # Load official functions without importing its ALOHA-only simulator at module scope.
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import policy
    import utils
    selected = {'forward_pass', 'train_bc', 'plot_history', 'make_policy', 'make_optimizer'}
    tree = ast.parse((ROOT / 'sources/act/act-main/imitate_episodes.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    ns = {'np': np, 'torch': torch, 'plt': plt, 'tqdm': tqdm, 'os': os,
          'deepcopy': copy.deepcopy, 'ACTPolicy': policy.ACTPolicy, 'CNNMLPPolicy': policy.CNNMLPPolicy,
          **{n: getattr(utils, n) for n in ['set_seed', 'compute_dict_mean', 'detach_dict']}}
    exec(compile(tree, str(ROOT / 'sources/act/act-main/imitate_episodes.py'), 'exec'), ns)
    return ns


def act_args(out, task):
    return ['act', '--ckpt_dir', str(out), '--policy_class', 'ACT', '--task_name', task,
            '--seed', '0', '--num_epochs', '2000', '--lr', '1e-5', '--kl_weight', '10',
            '--chunk_size', '100', '--hidden_dim', '512', '--dim_feedforward', '3200', '--batch_size', '8']


def train_act(task, out, smoke=False):
    import numpy as np
    import torch
    import utils
    sys.argv = act_args(out, task)
    utils.set_seed(1)  # official main() seeds dataset split at 1, train_bc seeds model at 0
    train, val, stats, _ = utils.load_data(str(ROOT / 'datasets' / ('act_' + task)), 10, ['corner2'], 8, 8)
    config = {'num_epochs': 2000, 'ckpt_dir': str(out), 'seed': 0, 'policy_class': 'ACT',
        'policy_config': {'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10, 'hidden_dim': 512,
            'dim_feedforward': 3200, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
            'enc_layers': 4, 'dec_layers': 7, 'nheads': 8, 'camera_names': ['corner2']}}
    atomic_json(Path(out) / 'config.json', config)
    with open(Path(out) / 'dataset_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    ns = act_functions()
    original = ns['forward_pass']
    steps = [0]
    def forward(data, policy):
        result = original(data, policy)
        values = {k: float(v.detach().cpu()) for k, v in result.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite ACT loss')
        steps[0] += 1
        heartbeat(out, stage='training', batches=steps[0], **values)
        return result
    ns['forward_pass'] = forward
    if smoke:
        utils.set_seed(0)
        policy = ns['make_policy']('ACT', config['policy_config'])
        optim = policy.configure_optimizers()
        loss = forward(next(iter(train)), policy)['loss']
        loss.backward()
        optim.step()
        policy.eval()
        image, q, _, _ = next(iter(val))
        with torch.no_grad():
            actions = policy(q.cuda(), image.cuda())
        assert actions.shape[-2:] == (100, 4) and torch.isfinite(actions).all()
        atomic_json(Path(out) / 'complete.json', {'stage': 'smoke', 'method': 'ACT',
            'task': task, 'loss': loss.item(), 'action_shape': list(actions.shape),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})
        return
    best_epoch, min_val_loss, state = ns['train_bc'](train, val, config)
    checkpoint = Path(out) / 'policy_best.ckpt'
    torch.save(state, checkpoint)
    atomic_json(Path(out) / 'trained.json', {'checkpoint': str(checkpoint),
        'sha256': sha256(checkpoint), 'completed_epochs': 2000,
        'best_validation_epoch': best_epoch, 'best_validation_loss': float(min_val_loss),
        'selection': 'official best validation loss'})
    policy = ns['make_policy']('ACT', config['policy_config'])
    policy.load_state_dict(state)
    from rollout import evaluate
    result = evaluate(policy, task, 'ACT', Path(out) / 'final_rollouts', final=True,
        checkpoint=checkpoint, stats=stats)
    atomic_json(Path(out) / 'complete.json', result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['collect', 'smoke', 'train'], required=True)
    p.add_argument('--method', default='SimpleDP3')
    p.add_argument('--task', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    heartbeat(args.out, stage='initializing')
    if args.stage == 'collect':
        collect(args.task, args.out)
    elif args.method == 'ACT':
        train_act(args.task, args.out, args.stage == 'smoke')
    else:
        train_diffusion(args.method, args.task, args.out, args.stage == 'smoke')


if __name__ == '__main__':
    main()

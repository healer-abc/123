"""Prepare read-only upstream copies and a documented ACT dimension adapter."""
import difflib
import json
from pathlib import Path
import shutil
from runtime import ROOT, TASKS, METHODS, sha256, atomic_json, DEMONSTRATIONS, TRAIN_EPISODES, ADDITIONAL_SEED


def main():
    source = ROOT / 'sources/act/act-main'
    dest = ROOT / 'runtime_act'
    if not dest.exists():
        shutil.copytree(source, dest)
        path = dest / 'detr/models/detr_vae.py'
        original = path.read_text()
        # MetaWorld has 9 robot-state values and 4 action values; all hidden layers,
        # latent variables, transformer layers and training objectives stay upstream.
        updated = original.replace('state_dim = 14 # TODO hardcode', 'state_dim = 4 # MetaWorld action dimension')
        updated = updated.replace('self.input_proj_robot_state = nn.Linear(14, hidden_dim)',
                                  'self.input_proj_robot_state = nn.Linear(9, hidden_dim)')
        updated = updated.replace('self.encoder_action_proj = nn.Linear(14, hidden_dim)',
                                  'self.encoder_action_proj = nn.Linear(4, hidden_dim)')
        updated = updated.replace('self.encoder_joint_proj = nn.Linear(14, hidden_dim)',
                                  'self.encoder_joint_proj = nn.Linear(9, hidden_dim)')
        path.write_text(updated)
        (ROOT / 'act_metaworld_dimensions.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), updated.splitlines(True),
            fromfile='upstream/detr/models/detr_vae.py', tofile='runtime_act/detr/models/detr_vae.py')))
    import numpy as np
    ids = np.random.RandomState(1).permutation(DEMONSTRATIONS).tolist()
    atomic_json(ROOT / 'manifests/dataset_split.json', {
        'episodes': DEMONSTRATIONS, 'seed': 1, 'algorithm': 'numpy.RandomState.permutation, same as official ACT load_data',
        'train_episode_ids': ids[:TRAIN_EPISODES], 'validation_episode_ids': ids[TRAIN_EPISODES:]})
    protocol = {'tasks': TASKS, 'methods': METHODS, 'training_seed': 0,
        'data': {'previous_verified_successful_demonstrations': 10, 'previous_collection_seed': 42,
                 'additional_successful_demonstrations': 40, 'additional_collection_seed': ADDITIONAL_SEED,
                 'training_episodes': TRAIN_EPISODES, 'validation_episodes': DEMONSTRATIONS-TRAIN_EPISODES,
                 'shared_split_sha256': sha256(ROOT / 'manifests/dataset_split.json')},
        'successful_demonstrations_per_task': DEMONSTRATIONS,
        'final_episodes_per_group': 20, 'final_episode_seeds': list(range(2026101900, 2026101920)),
        'development_episode_seeds': list(range(1026101900, 1026101905)),
        'development_rollout_episodes': 5,
        'max_episode_steps': 200, 'success': 'any official info.success within full episode',
        'DP': {'source': 'real-stanford/diffusion_policy', 'config': 'train_diffusion_unet_image_workspace.yaml',
            'num_epochs': 8000, 'batch_size': 64, 'horizon': 16, 'n_action_steps': 8,
            'checkpoint': 'fixed terminal epoch EMA'},
        'SimpleDP3': {'source': 'YanjieZe/3D-Diffusion-Policy', 'config': 'simple_dp3.yaml',
            'num_epochs': 3000, 'batch_size': 128, 'horizon': 16, 'n_action_steps': 8,
            'checkpoint': 'fixed terminal epoch EMA'},
        'ACT': {'source': 'tonyzhaozh/act', 'config': 'README simulated training command',
            'num_epochs': 2000, 'batch_size': 8, 'chunk_size': 100, 'kl_weight': 10,
            'hidden_dim': 512, 'dim_feedforward': 3200, 'lr': 1e-5,
            'temporal_aggregation': False, 'checkpoint': 'official best validation loss'},
        'scope': 'official algorithms with documented MetaWorld I/O adaptation; no model improvements',
        'runtime_adaptations': {'development_rollout_every_epochs': 500,
            'diffusion_checkpoint_every_epochs': 500, 'DP_validation_every_epochs': 20,
            'diffusion_sampling_diagnostic_every_epochs': 100,
            'diffusion_loader_workers': 2, 'persistent_data_loader_workers': True,
            'development_rollouts_restore_training_rng': True,
            'all_policy_models_trained_fresh': True},
        'limitations': ['one training seed per method/task; 20 final episodes is not a multi-seed benchmark',
            'DP and ACT have no official MetaWorld recipe; robot dimensions, data and rollout I/O are adapted',
            'training workspaces periodically run additional development rollouts; only final holdout 20 enter final table',
            'DP/SimpleDP3 terminal checkpoint holdout score differs from DP3 paper largest-K development-score reporting',
            'all methods share the official ACT 40/10 split; DP/SimpleDP3 dataset split adapted accordingly',
            'pinned source archives; existing compatible PyTorch 2.7.1 CUDA12.8 runtime on RTX6000D',
            'software EGL rendering due unavailable hardware-render device permissions']}
    atomic_json(ROOT / 'protocol.json', protocol)
    (ROOT / 'datasets').mkdir(exist_ok=True)
    execution = list(ROOT.glob('*.py')) + [ROOT / 'protocol.json', ROOT / 'manifests/dataset_split.json']
    for dirname in ['sources/dp3/3D-Diffusion-Policy-master/3D-Diffusion-Policy',
                    'sources/dp3/3D-Diffusion-Policy-master/third_party/Metaworld',
                    'sources/dp/diffusion_policy-main/diffusion_policy', 'runtime_act']:
        execution += [p for p in (ROOT / dirname).rglob('*') if p.is_file()
                      and p.suffix in ['.py', '.yaml', '.xml']]
    atomic_json(ROOT / 'source_manifest.json', {
        'archives': {p.name: sha256(p) for p in (ROOT / 'sources').glob('*.tar.gz')},
        'execution_files': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(execution))}})
    print('Prepared source and protocol manifests')


if __name__ == '__main__':
    main()

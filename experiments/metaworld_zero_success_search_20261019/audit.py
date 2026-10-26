"""Independent artifact and scoring audit, after the search worker has finished."""
import difflib
import csv
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('zero_search', HERE / 'search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
plan = search.get_plan()
state = json.loads((HERE / 'status.json').read_text())
assert state['state'] == 'complete', state
checked = []
trials = []
paired_reports = []
model_action_reconstructions = 0
for row in plan['targets']:
    target = HERE / (row['method'] + '_' + row['task'])
    trained = json.loads((Path(row['output']) / 'trained.json').read_text())
    assert search.sha256(trained['checkpoint']) == row['checkpoint_sha256']
    complete = json.loads((target / 'complete.json').read_text())
    stats = None
    if row['method'] == 'ACT':
        with (Path(row['output']) / 'dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
    for p in sorted(target.glob('**/episode_*.json')):
        record = json.loads(p.read_text())
        with np.load(p.with_suffix('.npz')) as data:
            assert np.isfinite(data['actions']).all()
            assert np.isfinite(data['predicted_chunks']).all()
            assert len(data['actions']) == record['steps'] == len(record['infos']) == 200
            assert abs(float(data['rewards'].sum()) - record['return']) < 1e-8
            assert record['success'] == any(bool(x['success']) for x in record['infos'])
            expected_first = next((i+1 for i,x in enumerate(record['infos']) if x['success']), None)
            assert record['first_success_step'] == expected_first
            assert p.with_suffix('.mp4').stat().st_size > 1000
            identity = record['identity']
            assert identity['checkpoint_sha256'] == row['checkpoint_sha256']
            if stats is not None and not identity['expert_positive_control'] and not identity['config']['temporal_agg']:
                query_indices = np.searchsorted(data['query_steps'], np.arange(200), side='right') - 1
                offsets = np.arange(200) - data['query_steps'][query_indices]
                expected = data['predicted_chunks'][query_indices, offsets] * stats['action_std'] + stats['action_mean']
                assert np.allclose(expected, data['actions'], rtol=1e-6, atol=1e-6), p
                model_action_reconstructions += 1
        checked.append({'path': str(p), 'success': record['success'], 'seed': record['seed']})
    for p in sorted(target.glob('**/summary.json')):
        episodes = [json.loads(e.read_text()) for e in sorted(p.parent.glob('episode_*.json'))]
        summary = json.loads(p.read_text())
        assert len(episodes) == summary['episodes']
        assert sum(x['success'] for x in episodes) == summary['successes']
        if p.parent.parent.name == 'search':
            trials.append({'task': row['task'], 'method': row['method'], 'trial': p.parent.name,
                'config': json.dumps(summary['identity']['config'], sort_keys=True),
                'seeds': json.dumps(summary['seeds']), 'successes': summary['successes'],
                'episodes': summary['episodes'], 'summary': str(p)})
    lock_path = target / 'locked.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        assert lock['screening_result']['successes'] > 0
        holdout = complete['holdout']
        assert holdout['seeds'] == plan['holdout_seeds'] and holdout['episodes'] == 20
        searched = [r['seed'] for r in checked if '/search/' in r['path']]
        assert set(searched).isdisjoint(holdout['seeds'])
        diff = json.loads((target / 'parameter_diff.json').read_text())
        before = json.dumps(diff['baseline'], indent=2, sort_keys=True).splitlines(keepends=True)
        after = json.dumps(diff['locked'], indent=2, sort_keys=True).splitlines(keepends=True)
        (target / 'parameters.diff').write_text(''.join(difflib.unified_diff(before, after, fromfile='original_inference.json', tofile='locked_inference.json')) + '\n')
    else:
        (target / 'parameters.diff').write_text('No parameter configuration was selected. All attempted configurations and seeds are retained in search/trial_*/summary.json and ../plan.json.\n')
    paired = target / 'paired_original20/summary.json'
    if paired.exists():
        control = json.loads(paired.read_text())
        holdout = complete['holdout']
        assert control['seeds'] == holdout['seeds'] == plan['holdout_seeds']
        pairs = []
        for i in range(20):
            a = json.loads((target / 'paired_original20' / f'episode_{i:02d}.json').read_text())
            b = json.loads((target / 'holdout20' / f'episode_{i:02d}.json').read_text())
            assert a['seed'] == b['seed']
            assert a['initial_state_sha256'] == b['initial_state_sha256'], (i,a,b)
            pairs.append({'episode': i, 'seed': a['seed'], 'same_initial_state': True,
                'original_success': a['success'], 'locked_success': b['success']})
        paired_report = {'task': row['task'], 'method': row['method'],
            'same_checkpoint': True, 'same_twenty_seeds': True, 'all_initial_states_equal': True,
            'original': control, 'locked': holdout, 'pairs': pairs,
            'success_rate_difference_percentage_points': 100*(holdout['success_rate']-control['success_rate'])}
        search.atomic_json(target / 'paired_comparison.json', paired_report)
        paired_reports.append(paired_report)
search.report(plan)
if paired_reports:
    with (HERE / 'summary.md').open('a') as f:
        f.write('\n## 同checkpoint、同seed配对对照\n\n| 任务 | 方法 | 原始执行100步 | 锁定执行1步 | 差值 |\n|---|---|---:|---:|---:|\n')
        for r in paired_reports:
            a,b = r['original'],r['locked']
            f.write(f"| {r['task']} | {r['method']} | {a['successes']}/20 ({a['success_rate']*100:.0f}%) | {b['successes']}/20 ({b['success_rate']*100:.0f}%) | {r['success_rate_difference_percentage_points']:+.0f}个百分点 |\n")
        f.write('\n两组20个初始状态哈希逐局相同。配对对照不参与参数筛选。历史基线仍保留原始seed上的0/20。\n')
    with (HERE / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        r = next((r for r in paired_reports if r['method']==row['method'] and r['task']==row['task']), None)
        row['paired_original20'] = f"{r['original']['successes']}/20" if r else ''
    with (HERE / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
if trials:
    with (HERE / 'search_trials.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
search.atomic_json(HERE / 'artifact_audit.json', {'passed': True, 'episodes_checked': len(checked),
    'checkpoint_unchanged': True, 'score_recomputed_from_official_infos': True,
    'finite_actions_and_predictions': True, 'all_episodes_200_steps': True,
    'episodes_with_actions_reconstructed_from_model_predictions': model_action_reconstructions,
    'episodes': checked})
print(json.dumps({'audit_passed': True, 'episodes': len(checked)}))

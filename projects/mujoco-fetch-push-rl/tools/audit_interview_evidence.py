"""Read existing artifacts for interview preparation; never train or overwrite results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    baseline = {int(r['episode_seed']): r for r in rows(ROOT / 'checks/zero_action_diagnostic_20261023.csv')}
    near = {seed for seed, r in baseline.items() if int(r['initial_success'])}
    records = {}
    for group in 'ABCDE':
        for seed in range(3):
            key = f'{group}_seed{seed}'
            run = ROOT / 'runs/formal_20261022' / key
            evaluation = ROOT / 'evaluations/formal_20261022' / key
            episodes = rows(evaluation / 'episodes.csv')
            gain_records = {}
            for gain in sorted({float(r['gain']) for r in episodes}):
                subset = [r for r in episodes if float(r['gain']) == gain]
                assert len(subset) == 100
                assert {int(r['episode_seed']) for r in subset} == set(baseline)
                initial_failures = [r for r in subset if int(r['episode_seed']) not in near]
                initial_successes = [r for r in subset if int(r['episode_seed']) in near]
                gain_records[str(gain)] = {
                    'episodes': len(subset),
                    'successes': sum(int(r['success']) for r in subset),
                    'initial_failure_scenes': len(initial_failures),
                    'initial_failure_final_successes': sum(int(r['success']) for r in initial_failures),
                    'initial_success_final_successes': sum(int(r['success']) for r in initial_successes),
                    'initial_success_scenes_lost': [int(r['episode_seed']) for r in initial_successes if not int(r['success'])],
                    'max_distance_difference_from_zero_action_m': max(abs(float(r['distance_m']) - float(baseline[int(r['episode_seed'])]['zero_action_distance_m'])) for r in subset),
                    'scenes_final_distance_exceeds_zero_action_by_1cm': [int(r['episode_seed']) for r in subset if float(r['distance_m']) > float(baseline[int(r['episode_seed'])]['zero_action_distance_m']) + .01],
                }
            with zipfile.ZipFile(run / 'final/model.zip') as archive:
                data = json.loads(archive.read('data'))
            validation = rows(run / 'validation.csv')
            peak = max(validation, key=lambda r: float(r['success_rate']))
            records[key] = {
                'gains': gain_records,
                'num_timesteps': data['num_timesteps'],
                'n_updates': data['_n_updates'],
                'validation_peak': {k: peak[k] for k in ['timesteps', 'success_rate']},
                'validation_last': {k: validation[-1][k] for k in ['timesteps', 'success_rate']},
                'resumed_from': json.loads((run / 'config.json').read_text()).get('resumed_from'),
                'model_sha256': hashlib.sha256((run / 'final/model.zip').read_bytes()).hexdigest(),
            }
    table = rows(ROOT / 'report/results_per_seed.csv')
    ood = {}
    for group in 'DE':
        per_seed = [statistics.mean(float(r['success_rate']) for r in table if r['group'] == group and int(r['training_seed']) == seed and float(r['gain']) in (.6, 1.4)) for seed in range(3)]
        ood[group] = {'per_seed': per_seed, 'mean': statistics.mean(per_seed), 'sample_sd': statistics.stdev(per_seed)}
    b = [float(r['success_rate']) for r in table if r['group'] == 'B']
    result = {
        'purpose': 'Post-hoc read-only interview evidence; no training or model selection',
        'new_information': 'A seed1/2 zero-action distance comparisons and conditional counts computed from existing CSVs, not new rollouts',
        'initial_success_scene_seeds': sorted(near),
        'total_evaluation_rows': sum(g['episodes'] for item in records.values() for g in item['gains'].values()),
        'unique_scene_seeds': len(baseline), 'records': records, 'ood': ood,
        'B_mean': statistics.mean(b), 'B_sample_sd': statistics.stdev(b),
        'ood_E_minus_D_pp': 100 * (ood['E']['mean'] - ood['D']['mean']),
    }
    assert result['total_evaluation_rows'] == 3900
    assert all(r['resumed_from'] is None for r in records.values())
    destination = ROOT / 'checks/interview_evidence_20261023.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'episodes': result['total_evaluation_rows'], 'A_max_differences': [records[f'A_seed{s}']['gains']['1.0']['max_distance_difference_from_zero_action_m'] for s in range(3)], 'B_validation': records['B_seed0']['validation_peak'], 'B_validation_last': records['B_seed0']['validation_last'], 'C_initial_success_lost': [records[f'C_seed{s}']['gains']['1.0']['initial_success_scenes_lost'] for s in range(3)]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Freeze demonstration/event/control and discordant rollout samples before inference."""
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from common import ROOT, PROJECT, DATA, SEED, events_and_mask, image_value, sha, write_json


def inspect(path):
    table = pq.read_table(path, columns=['actions', 'episode_index', 'task_index'], use_threads=False)
    actions = np.asarray(table['actions'].to_pylist(), dtype=np.float32)
    assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
    events, mask = events_and_mask(actions)
    return {'path': str(path), 'episode': int(table['episode_index'][0].as_py()),
            'task': int(table['task_index'][0].as_py()), 'frames': len(actions),
            'events': events, 'event_directions': ['close_command' if actions[k,6] > 0 else 'open_command' for k in events],
            'covered_frames': int(mask.sum()), 'actions': actions, 'mask': mask}


def round_robin(groups, count, rng):
    keys = sorted(groups)
    rng.shuffle(keys)
    groups = {key: [v for v in groups[key]] for key in keys}
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        changed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                changed = True
        if not changed:
            raise ValueError('not enough eligible samples')
    return selected


def main():
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    if (ROOT/'artifacts/samples.json').exists():
        raise SystemExit('frozen diagnostic samples already exist; refusing replacement')
    tasks = {v['task_index']: v['task'] for v in map(json.loads, (DATA/'meta/tasks.jsonl').read_text().splitlines())}
    paths = sorted((DATA/'data').glob('*/*.parquet'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, paths))
    by_episode = {r['episode']: r for r in rows}
    stats = {'seed': SEED, 'episodes': len(rows), 'frames': sum(r['frames'] for r in rows),
             'command_events': sum(len(r['events']) for r in rows),
             'covered_frames': sum(r['covered_frames'] for r in rows),
             'event_window_fraction': sum(r['covered_frames'] for r in rows)/sum(r['frames'] for r in rows),
             'per_task': []}
    for task in sorted(tasks):
        subset = [r for r in rows if r['task'] == task]
        frames = sum(r['frames'] for r in subset)
        covered = sum(r['covered_frames'] for r in subset)
        stats['per_task'].append({'task_index': task, 'description': tasks[task], 'episodes': len(subset),
                                  'frames': frames, 'events': sum(len(r['events']) for r in subset),
                                  'covered_frames': covered, 'fraction': covered/frames})
    write_json(ROOT/'artifacts/dataset_statistics.json', stats)
    serial_rows = [{k: v for k,v in r.items() if k not in ('actions','mask')} for r in rows]
    write_json(ROOT/'artifacts/episode_events.json', serial_rows)
    norm = json.loads((PROJECT.parents[1]/'projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json').read_text())
    # Statistical scale is used for matching diagnostics only; model transforms use original quantile normalization.
    scale = np.asarray(norm['norm_stats']['actions']['std'][:6]) if 'norm_stats' in norm else np.asarray(norm['actions']['std'][:6])
    rng = np.random.default_rng(SEED)
    chosen = []
    for direction in ('close_command', 'open_command'):
        groups = collections.defaultdict(list)
        for row in rows:
            valid_controls = [t for t in range(row['frames']-9) if not row['mask'][t:t+10].any()]
            if not valid_controls:
                continue
            for k,d in zip(row['events'],row['event_directions']):
                if d == direction and 5 <= k and k+5 <= row['frames']:
                    groups[row['task']].append((row['episode'],k))
        for episode,k in round_robin(groups,24,rng):
            row = by_episode[episode]
            anchor = k-5
            energy = float(np.mean(np.square(row['actions'][anchor:anchor+10,:6]/scale)))
            controls = [t for t in range(row['frames']-9) if not row['mask'][t:t+10].any()]
            energies = np.array([np.mean(np.square(row['actions'][t:t+10,:6]/scale)) for t in controls])
            distance = np.abs(np.log(energies+1e-6)-np.log(energy+1e-6))
            # Pick among up to three closest motion-energy controls, independently of model loss.
            best = np.argsort(distance)[:min(3,len(controls))]
            control = controls[int(rng.choice(best))]
            chosen.append({'index': len(chosen), 'episode': episode, 'task_index': row['task'],
                           'task_description': tasks[row['task']], 'event_frame': k, 'event_direction': direction,
                           'event_anchor': anchor, 'control_anchor': control, 'path': row['path'],
                           'event_motion_energy': energy, 'control_motion_energy': float(np.mean(np.square(row['actions'][control:control+10,:6]/scale)))})
    write_json(ROOT/'artifacts/samples.json', {'seed': SEED, 'sample_count':48,'samples':chosen})
    for page in range(12):
        sheet = Image.new('RGB',(8*128,4*288),'white')
        draw = ImageDraw.Draw(sheet)
        for r,entry in enumerate(chosen[page*4:page*4+4]):
            table = pq.read_table(entry['path'],columns=['image','wrist_image'],use_threads=False)
            ids = np.clip(np.array([-5,-2,0,2,5,8,10,15])+entry['event_frame'],0,len(table)-1)
            draw.text((3,r*288+2),f"#{entry['index']} ep{entry['episode']} task{entry['task_index']} {entry['event_direction']} k={entry['event_frame']}",fill='black')
            for column, frame in enumerate(ids):
                sheet.paste(Image.fromarray(image_value(table['image'][int(frame)].as_py())).resize((128,128)),(column*128,r*288+25))
                sheet.paste(Image.fromarray(image_value(table['wrist_image'][int(frame)].as_py())).resize((128,128)),(column*128,r*288+153))
                draw.text((column*128+2,r*288+14),f"{int(frame)}",fill='black')
        sheet.save(ROOT/f'artifacts/demo_events_{page:02d}.jpg')
    # Choose exposed development identities in both discordant directions, task-balanced.
    records = {}
    for run in ('E1-s42','E2-s42'):
        records[run] = {(v['suite'],v['task_id'],v['init_state_index']):v for v in map(lambda p:json.loads(p.read_text()),
                       (PROJECT/'evaluations'/run/'development/episodes').glob('*/*/*.json'))}
    selected_rows = []
    for e1_success in (True,False):
        groups = collections.defaultdict(list)
        for key,v in records['E1-s42'].items():
            other = records['E2-s42'][key]
            if v['success'] == e1_success and other['success'] != e1_success:
                groups[key[:2]].append(key)
        for key in round_robin(groups,6,rng):
            selected_rows.append({'suite':key[0],'task_id':key[1],'init_state_index':key[2],
                                  'original_e1_success':records['E1-s42'][key]['success'],
                                  'original_e2_success':records['E2-s42'][key]['success']})
    write_json(ROOT/'artifacts/rollout_sample.json', {'seed':SEED,'identities':selected_rows,'planned_rollouts':24})
    write_json(ROOT/'artifacts/preparation_complete.json', {'dataset_metadata_sha256':sha(DATA/'meta/info.json'),
               'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'sample_sha256':sha(ROOT/'artifacts/samples.json'),
               'rollout_sample_sha256':sha(ROOT/'artifacts/rollout_sample.json'),'sampling_complete':True,
               'task_count_in_demo_sample':len({s['task_index'] for s in chosen})})
    print(json.dumps({k:v for k,v in stats.items() if k!='per_task'},indent=2))


if __name__ == '__main__':
    main()

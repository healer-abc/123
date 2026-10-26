"""Apply the same preregistered finite search to later completed zero baselines."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('zero_search', HERE / 'search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
template = search.get_plan()
covered = {(r['method'], r['task']) for r in template['targets']}
search.atomic_json(HERE / 'followup_pid.json', {'pid': os.getpid(), 'started': time.time()})


def update_index(rows, entries, state):
    search.atomic_json(HERE / 'followup_status.json', {'time': time.time(), 'pid': os.getpid(),
        'state': state, 'baseline_groups_completed': len(rows), 'baseline_groups_total': 15, 'searches': entries})
    lines = ['# 后续零成功率组的自动处理', '',
        '只接收已经完成20局、原始结果为0/20的50示教基线。训练中checkpoint不参与。每个新增组在首局测试前独立冻结同一有限参数网格与seed计划。', '',
        f'已完成基线组：{len(rows)}/15；监控状态：{state}。', '',
        f'- 当前已知零组：{HERE / "summary.md"}']
    for entry in entries:
        lines.append(f"- {entry['method']} / {entry['task']}: {entry['state']}；{entry['root']}/summary.md")
    (HERE / 'followup_index.md').write_text('\n'.join(lines) + '\n')


entries = []
while True:
    try:
        rows = list(csv.DictReader((search.BASE / 'report/results.csv').open()))
        for row in rows:
            key = row['method'], row['task']
            if key in covered or int(row['episodes']) != 20 or int(row['successes']) != 0:
                continue
            suffix = row['method'] + '_' + row['task']
            root = HERE.parent / ('metaworld_zero_success_followup_' + suffix + '_20261019')
            root.mkdir(exist_ok=True)
            for name in ['search.py', 'audit.py', 'PARAMETERS.md']:
                dest = root / name
                if not dest.exists():
                    source = HERE / ('search_followup.py' if name == 'search.py' else name)
                    shutil.copy2(source, dest)
            plan_path = root / 'plan.json'
            if not plan_path.exists():
                plan = copy.deepcopy(template)
                plan.update(created_at=time.time(), targets=[dict(row)],
                    baseline_results_sha256=search.sha256(search.BASE / 'report/results.csv'),
                    script_sha256=search.sha256(root / 'search.py'),
                    followup_origin=str(HERE / 'plan.json'))
                search.atomic_json(plan_path, plan)
                (root / 'baseline_results_snapshot.csv').write_text((search.BASE / 'report/results.csv').read_text())
            entry = {'method': row['method'], 'task': row['task'], 'root': str(root), 'state': 'running'}
            entries.append(entry)
            update_index(rows, entries, 'running_followup')
            with (root / 'supervisor.log').open('a') as log:
                child = subprocess.Popen([str(search.PYTHON), str(root / 'search.py')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = child.pid
                while child.poll() is None:
                    update_index(rows, entries, 'running_followup')
                    time.sleep(15)
            if child.returncode == 0:
                result = subprocess.run([str(search.PYTHON), str(root / 'audit.py')], capture_output=True, text=True)
                (root / 'audit.log').write_text(result.stdout + result.stderr)
                entry['state'] = 'complete' if result.returncode == 0 else 'audit_error'
            else:
                entry['state'] = 'infrastructure_error'
            covered.add(key)
        done = len(rows) == 15
        update_index(rows, entries, 'complete' if done else 'waiting_for_completed_baselines')
        if done:
            break
    except Exception:
        search.atomic_json(HERE / ('followup_error_' + str(int(time.time())) + '.json'),
            {'time': time.time(), 'traceback': traceback.format_exc()})
    time.sleep(30)

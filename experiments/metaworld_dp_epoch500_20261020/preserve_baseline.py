"""Pause the existing DP campaign without discarding its live training state."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0, str(BASE))
from runtime import atomic_json, sha256


def verify_process(pid, expected):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert expected in command, (pid, command)
    return command


def main():
    receipt = HERE / 'preserved_baseline/receipt.json'
    assert not receipt.exists(), 'Already preserved; inspect receipt before repeating'
    status = json.loads((BASE / 'logs/status.json').read_text())
    supervisor = status['pid']
    command = verify_process(supervisor, str(BASE / 'baseline_watchdog.py'))
    os.kill(supervisor, signal.SIGSTOP)
    time.sleep(.2)
    # Re-read only after supervisor is stopped to close the scheduling race.
    status = json.loads((BASE / 'logs/status.json').read_text())
    atomic_json(HERE / 'preserved_baseline/status_before_pause.json', status)
    jobs = []
    for job in status['jobs']:
        if job.get('method') != 'DP' or job['stage'] != 'train':
            continue
        saved = dict(job)
        if job['status'] == 'running':
            pid = job['pid']
            verify_process(pid, str(BASE / 'worker.py'))
            assert os.getpgid(pid) == pid
            os.killpg(pid, signal.SIGSTOP)
            saved['state'] = 'paused_live_state_in_memory'
        else:
            saved['state'] = job['status']
        jobs.append(saved)
    time.sleep(.3)
    # Persist process identities immediately, before copying multi-GB checkpoints.
    record = {'time': time.time(), 'supervisor_pid': supervisor,
              'supervisor_command': command, 'jobs': jobs,
              'mode': 'SIGSTOP preserves exact live state; disk checkpoints are older periodic saves',
              'resume_rule': 'Resume workers first; wait for fresh heartbeats, then resume supervisor. Never resume supervisor first.',
              'unrelated_jobs_touched': False}
    atomic_json(receipt, record)
    for job in jobs:
        if not job.get('out'):
            continue
        out = Path(job['out'])
        target = HERE / 'preserved_baseline' / job['key']
        target.mkdir(parents=True, exist_ok=True)
        for name in ['heartbeat.json', 'config.yaml', 'trained.json', 'complete.json']:
            source = out / name
            if source.exists():
                shutil.copy2(source, target / name)
        if (out / 'heartbeat.json').exists():
            job['paused_heartbeat'] = json.loads((out / 'heartbeat.json').read_text())
        job['disk_checkpoints'] = []
        for source in sorted((out / 'checkpoints').glob('*.ckpt')):
            dest = target / source.name
            shutil.copy2(source, dest)  # Separate inode: a future latest save cannot overwrite the archive.
            assert dest.stat().st_size == source.stat().st_size
            digest = sha256(dest)
            assert digest == sha256(source)
            job['disk_checkpoints'].append({'source': str(source), 'archive': str(dest),
                                            'sha256': digest, 'bytes': dest.stat().st_size})
        atomic_json(receipt, record)
    paused = dict(status, time=time.time(), status='paused_for_independent_DP500',
                  preservation_receipt=str(receipt))
    for job in paused['jobs']:
        if job.get('method') == 'DP' and job['status'] == 'running':
            job['status'] = 'paused'
    atomic_json(BASE / 'logs/status.json', paused)
    (BASE / 'PAUSED_FOR_DP500.md').write_text(
        'DP1000 campaign paused by user request. Exact running states retained via SIGSTOP; '
        'independent copies of periodic disk checkpoints are archived. '
        'Live state requires these processes and host to remain alive. '\
        f'\nPreservation receipt: {receipt}\n'
        'Resume DP workers first and wait for their heartbeat files to advance before SIGCONT to the supervisor.\n'
        'Independent fresh DP500 run: ../metaworld_dp_epoch500_20261020/progress.md\n')
    print(json.dumps({'preserved': True, 'receipt': str(receipt),
                      'jobs': [{'key': j['key'], 'state': j['state'],
                                'epoch': j.get('paused_heartbeat', {}).get('epoch')}
                               for j in jobs]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

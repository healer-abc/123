"""Adopt live DP workers and apply terminal-only saves to newly launched remaining jobs."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

OPS = Path(__file__).resolve().parent
ROOT = OPS.parent.parent
BASE = ROOT.parent/'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0,str(BASE))
from runtime import PYTHON,atomic_json,environment,sha256

spec=importlib.util.spec_from_file_location('original_dp_supervisor_helpers',ROOT/'run.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


class AdoptedProcess:
    def __init__(self,pid,out):
        self.pid=pid;self.out=out;self.returncode=None
        self.identity=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    def poll(self):
        p=Path(f'/proc/{self.pid}/stat')
        if p.exists():
            fields=p.read_text().split(') ')[1].split()
            if fields[19]==self.identity and fields[0]!='Z':return None
        self.returncode=0 if (self.out/'complete.json').exists() else -1
        return self.returncode
    def wait(self,timeout):
        until=time.time()+timeout
        while self.poll() is None and time.time()<until:time.sleep(.1)
        if self.poll() is None:raise subprocess.TimeoutExpired(str(self.pid),timeout)
        return self.returncode


def report(jobs,state='running'):
    original.report(jobs,state)
    p=ROOT/'progress.md';s=p.read_text()
    s=s.replace('完成300、400、500轮分别保存模型、EMA与优化器快照；500轮EMA以10步去噪评测每任务20局。',
        '已运行任务保留原保存逻辑；新启动的stick-push/door-open仅保存500轮最终权重及续训状态，最终评测仍各20局。')
    s=s.replace('；SimpleDP3实验继续运行。','；SimpleDP3评测已完成。')
    p.write_text(s)
    status=json.loads((ROOT/'status.json').read_text())
    status['snapshot_override']=str(OPS/'manifest.json')
    status['snapshot_override_note']='Existing workers retain already-loaded save code; new remaining-task workers save terminal500 only.'
    atomic_json(ROOT/'status.json',status)


def main():
    manifest=json.loads((OPS/'manifest.json').read_text())
    assert sha256(__file__)==manifest['supervisor_sha256']
    assert sha256(ROOT/'replay_terminal_only.py')==manifest['terminal_driver_sha256']
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=original.plan()
    for name,digest in plan['execution_files'].items():assert sha256(ROOT/name)==digest
    status=json.loads((OPS/'status_at_handover.json').read_text())
    jobs=status['jobs'];active={}
    for job in jobs:
        if original.result(job['task']):job['state']='complete';continue
        if job['state']!='running':continue
        pid=job['pid'];command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert str(ROOT/'replay500.py') in command and job['task'] in command and os.getpgid(pid)==pid
        proc=AdoptedProcess(pid,original.output(job['task']))
        active[job['task']]={'proc':proc,'job':job,'log':Path(job['log']).open('a')}
    def handler(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    atomic_json(ROOT/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time(),'adopted_live_workers':list(active),'override':str(OPS/'manifest.json')})
    try:
        while True:
            free={int(g):int(f) for g,f in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            for task,item in list(active.items()):
                proc,job=item['proc'],item['job']
                activity=[job['started'],Path(job['log']).stat().st_mtime]
                activity += [p.stat().st_mtime for p in original.output(task).rglob('heartbeat.json')]
                reason=None
                if proc.poll() is None and (free[job['gpu']]<3500 or time.time()-max(activity)>1200):
                    reason='low_free_memory' if free[job['gpu']]<3500 else 'no_progress_1200s';original.stop(proc)
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode==0 and original.result(task):job['state']='complete'
                    else:
                        job['failures']+=reason!='low_free_memory';job['state']='pending' if job['failures']<3 else 'error';job['not_before']=time.time()+60
                        atomic_json(ROOT/f'alert_{task}_attempt_{job["attempt"]}.json',{'time':time.time(),'job':job,'reason':reason or 'worker_error','exit_code':proc.returncode})
                    del active[task]
            occupied={x['job']['gpu'] for x in active.values()}
            for job in jobs:
                if job['state']!='pending' or len(active)>=plan['max_parallel'] or time.time()<job['not_before']:continue
                candidates=[(f,g) for g,f in free.items() if g not in occupied and f>=plan['training_budget_mib']+plan['reserve_mib']]
                if not candidates:continue
                _,gpu=max(candidates);job.update(state='running',attempt=job['attempt']+1,gpu=gpu,started=time.time())
                path=ROOT/'logs'/f'{job["task"]}_attempt_{job["attempt"]:02d}.log';log=path.open('a');job['log']=str(path)
                driver='replay_terminal_only.py' if job['task'] in manifest['tasks'] else 'replay500.py'
                proc=subprocess.Popen([str(PYTHON),'-B',str(ROOT/driver),job['task'],str(job['attempt'])],cwd=BASE,
                    env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job['pid']=proc.pid;active[job['task']]={'proc':proc,'job':job,'log':log};occupied.add(gpu)
            report(jobs)
            if all(j['state']=='complete' for j in jobs):
                original.audit(plan)
                audit=json.loads((ROOT/'audit.json').read_text());audit.pop('snapshot_epochs_verified',None)
                audit['snapshot_epochs_verified_by_task']={j['task']:json.loads((original.output(j['task'])/'training_audit.json').read_text())['snapshot_epochs'] for j in jobs}
                audit['operational_override_manifest_sha256']=sha256(OPS/'manifest.json')
                atomic_json(ROOT/'audit.json',audit)
                report(jobs,'complete');atomic_json(ROOT/'complete.json',{'state':'complete','episodes':100,'audited':True});return
            if all(j['state'] in ['complete','error'] for j in jobs):report(jobs,'partial_failure');return
            time.sleep(5)
    finally:
        for item in active.values():original.stop(item['proc']);item['log'].close()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OPS/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()});raise

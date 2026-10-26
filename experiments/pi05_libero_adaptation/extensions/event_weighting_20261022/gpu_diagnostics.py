"""Bounded read-only GPU diagnostic supervisor; never launches training."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from common import ROOT, PROJECT, OPENPI, write_json

sys.path.insert(0,str(PROJECT))
import development_watchdog as helpers

STOP=False
CHILDREN=[]
STARTED=None


def stop(*_):
    global STOP
    STOP=True


def check(gpu):
    global STARTED
    if STOP:
        raise InterruptedError('user stopped diagnostic queue')
    if STARTED is not None and time.monotonic()-STARTED>1800:
        raise TimeoutError('30-minute GPU diagnostic budget exhausted')
    memory=helpers.gpu_memory()[gpu]
    with (ROOT/'artifacts/gpu_samples.csv').open('a',newline='') as f:
        csv.writer(f).writerow([time.time(),gpu,memory['used_mib'],memory['free_mib']])
    if memory['free_mib']<4096:
        raise RuntimeError('GPU free memory below 4096 MiB; release diagnostic processes')
    return memory


def wait_gpu():
    started=time.monotonic()
    while time.monotonic()-started<600:
        if STOP:
            raise InterruptedError('user stopped diagnostic queue')
        choices=[(v['free_mib'],gpu) for gpu,v in helpers.gpu_memory().items() if v['free_mib']>=24576]
        if choices:
            return max(choices)[1]
        write_json(ROOT/'artifacts/gpu_status.json',{'state':'waiting_for_memory','pid':os.getpid(),'required_free_mib':24576})
        time.sleep(5)
    raise TimeoutError('no GPU admitted within ten minutes; CPU diagnostics remain available')


def launch(command,gpu,name):
    global STARTED
    if STARTED is None:
        STARTED=time.monotonic()
    env=helpers.environment(gpu)
    env.update(OPENPI_DATA_HOME='/home/zhsu/jyluo/myopenpi/dataset/ckpt',HF_LEROBOT_HOME='/home/zhsu/jyluo/dataset_root',HF_HUB_OFFLINE='1')
    log=ROOT/'artifacts'/f'{name}.log'
    child=helpers.spawn(command,env,log)
    CHILDREN.append(child)
    write_json(ROOT/'artifacts/gpu_status.json',{'state':'running','stage':name,'pid':os.getpid(),'child_pid':child.pid,'gpu':gpu,'log':str(log)})
    return child


def wait(child,gpu):
    while child.poll() is None:
        check(gpu)
        time.sleep(1)
    if child.returncode:
        raise RuntimeError(f'diagnostic process exited {child.returncode}')


def main():
    lock=(ROOT/'artifacts/gpu_diagnostics.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'artifacts/gpu_diagnostics.pid').write_text(str(os.getpid())+'\n')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    runs=json.loads((PROJECT/'manifests/runs.json').read_text())['runs']
    try:
        for run_id in ('E1-s42','E2-s42'):
            if (ROOT/'artifacts'/f'{run_id}_loss_probe.json').exists():
                continue
            gpu=wait_gpu()
            child=launch([str(OPENPI/'.venv/bin/python'),'-u',str(ROOT/'loss_probe.py'),'--run-id',run_id],gpu,f'{run_id}_loss_probe')
            wait(child,gpu)
        for run_id in ('E1-s42','E2-s42'):
            out=ROOT/'rollouts'/run_id/'summary.json'
            if out.exists() and json.loads(out.read_text()).get('complete'):
                continue
            gpu=wait_gpu()
            run=next(r for r in runs if r['run_id']==run_id)
            if not helpers.free_port(8125):
                raise RuntimeError('diagnostic port 8125 is already occupied')
            server=launch([str(OPENPI/'.venv/bin/python'),'-u',str(PROJECT/'seeded_policy_server.py'),
                           '--config',run['config'],'--checkpoint',run['final_checkpoint'],'--port','8125'],gpu,f'{run_id}_replay_server')
            start=time.monotonic()
            while not helpers.healthy(8125):
                check(gpu)
                if server.poll() is not None or time.monotonic()-start>180:
                    raise RuntimeError('diagnostic server startup failed')
                time.sleep(1)
            client=launch([str(OPENPI/'examples/libero/.venv/bin/python'),'-u',str(ROOT/'replay.py'),
                           '--run-id',run_id,'--port','8125'],gpu,f'{run_id}_replay_client')
            wait(client,gpu)
            helpers.terminate(server)
        write_json(ROOT/'artifacts/gpu_status.json',{'state':'completed','pid':os.getpid(),'training_updates':0,
                   'gpu_elapsed_seconds':time.monotonic()-STARTED if STARTED else 0})
    except Exception as e:
        write_json(ROOT/'artifacts/gpu_status.json',{'state':'stopped_by_signal' if STOP else 'blocked','pid':os.getpid(),'error':str(e)})
        raise
    finally:
        for child in reversed(CHILDREN):
            helpers.terminate(child)


if __name__=='__main__':
    main()

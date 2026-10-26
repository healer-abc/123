"""Two independent target searches, with isolated heartbeat paths and fixed final seeds."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_grid_engine',HERE/'search.py')
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def report(plan,jobs):
    engine.report(plan)
    lines=['# 当前ACT零成功组的网格搜索','','原始100步基线、本轮固定1步评测、网格搜索后的独立评测分别保留。后者不能当作未调参基线。',
        '每任务预定8个配置、3个seed轮次，每配置每轮3局，最多72局筛选；首个命中后锁定，并且只跑一次新的独立20局。',
        '筛选seeds从3126102000递增；最终seeds固定2126102000–2126102019。', '',
        '| 任务 | 历史100步 | 固定1步 | 新网格命中配置 | 新独立20局 | 状态 |','|---|---:|---:|---|---:|---|']
    data=[]
    for job in jobs:
        row=plan['targets'][job['index']]
        folder=HERE/(row['method']+'_'+row['task'])
        complete=json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else None
        lock=json.loads((folder/'locked.json').read_text()) if (folder/'locked.json').exists() else None
        config=json.dumps(lock['config'],sort_keys=True) if lock else '尚未命中'
        final='—'
        if complete:
            if complete['state']=='holdout_complete':
                h=complete['holdout'];final=f"{h['successes']}/20 ({h['success_rate']*100:.0f}%)"
                status='独立复评完成' if h['successes']>0 else '独立复评仍0；保留并报告未达目标'
            else:status='有限网格穷尽，仍0；已停手并生成排错报告'
        else:status=job['state']
        old=f"{row['successes']}/20"
        lines.append(f"| {row['task']} | {old} | 0/20 | {config} | {final} | {status} |")
        data.append({'task':row['task'],'historical':old,'fixed_interval_one':'0/20','locked_config':config,'independent20':final,'state':status})
    lines += ['', '所有搜索试验、失败记录、原始动作、模型输出与视频保留于ACT_<task>/search/；最终复评在holdout20/。',
        '若锁定配置的独立20局仍0，不再根据这20局结果选择下一配置或更换seed。',
        '原始五任务训练完成后新出现的0组，由此前的全方法watch_future.py继续接续处理。','']
    (HERE/'grid_summary.md').write_text('\n'.join(lines))
    with (HERE/'grid_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main():
    plan=engine.get_plan()
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        index=int(sys.argv[2]);row=plan['targets'][index]
        original=engine.atomic_json
        def routed(path,value):
            if Path(path)==HERE/'heartbeat.json':
                path=HERE/'progress'/row['task']/'heartbeat.json'
            original(path,value)
        engine.atomic_json=routed
        engine.run_target(row,plan)
        return
    engine.atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'time':time.time()})
    jobs=[{'index':i,'task':r['task'],'state':'complete' if (HERE/(r['method']+'_'+r['task'])/'complete.json').exists() else 'pending','attempt':0}
          for i,r in enumerate(plan['targets'])]
    active={}
    while any(j['state']!='complete' for j in jobs):
        for task,item in list(active.items()):
            proc,job=item['process'],item['job']
            hb=HERE/'progress'/task/'heartbeat.json'
            last=max(job['started'],hb.stat().st_mtime if hb.exists() else job['started'],Path(job['log']).stat().st_mtime)
            if proc.poll() is None and time.time()-last>600:
                engine.atomic_json(HERE/f'alert_{task}_{job["attempt"]}.json',{'time':time.time(),'job':job,'reason':'600s without progress'})
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
            if proc.poll() is not None:
                item['log'].close()
                row=plan['targets'][job['index']]
                cp=HERE/(row['method']+'_'+task)/'complete.json'
                if proc.returncode==0 and cp.exists():job['state']='complete'
                else:
                    engine.atomic_json(HERE/f'error_{task}_{job["attempt"]}.json',{'job':job,'time':time.time(),'exit_code':proc.returncode})
                    job['state']='pending' if job['attempt']<3 else 'error'
                    if job['state']=='error':raise RuntimeError(f'Repeated infrastructure failure: {task}')
                del active[task]
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(g):int(f) for g,f in (line.split(',') for line in out.splitlines())}
        occupied={i['job']['gpu'] for i in active.values()}
        for job in jobs:
            if job['state']!='pending' or len(active)>=2:continue
            choices=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
            if not choices:continue
            _,gpu=max(choices)
            job.update(state='running',attempt=job['attempt']+1,gpu=gpu,started=time.time())
            p=HERE/f'{job["task"]}_attempt_{job["attempt"]:02d}.log'
            log=p.open('a')
            proc=subprocess.Popen([str(engine.PYTHON),str(Path(__file__).resolve()),'--worker',str(job['index'])],
                env=engine.environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(pid=proc.pid,log=str(p));active[job['task']]={'process':proc,'job':job,'log':log};occupied.add(gpu)
        report(plan,jobs)
        engine.atomic_json(HERE/'status.json',{'time':time.time(),'state':'running','pid':os.getpid(),'jobs':jobs,'gpu_free_mib':free})
        if all(j['state']=='complete' for j in jobs):break
        time.sleep(5)
    report(plan,jobs)
    engine.atomic_json(HERE/'status.json',{'time':time.time(),'state':'complete','pid':os.getpid(),'jobs':jobs})
    result=subprocess.run([str(engine.PYTHON),str(HERE/'audit.py')],capture_output=True,text=True)
    (HERE/'audit.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Artifact audit failed; see audit.log')
    report(plan,jobs)
    engine.atomic_json(HERE/'finalized.json',{'time':time.time(),'audit_passed':True,'jobs':jobs})


if __name__=='__main__':
    try:main()
    except Exception:
        engine.atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()})
        traceback.print_exc()
        raise

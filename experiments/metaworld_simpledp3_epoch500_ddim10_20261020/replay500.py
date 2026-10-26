"""Reconstruct a500-epoch prefix of the frozen1000-epoch training schedule."""
import difflib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0,str(BASE))
from runtime import atomic_json,sha256,bootstrap


def main():
    task=sys.argv[1];attempt=int(sys.argv[2]);p=json.loads((HERE/'plan.json').read_text())
    assert sha256(__file__)==p['replay_sha256']
    assert sha256(HERE/'rollout500.py')==p['evaluator_sha256']
    assert sha256(BASE/'worker.py')==p['baseline_worker_sha256']
    assert sha256(BASE/'protocol.json')==p['baseline_protocol_sha256']
    source=BASE/'sources/dp3/3D-Diffusion-Policy-master/3D-Diffusion-Policy/train.py'
    assert sha256(source)==p['official_training_sha256']
    out=HERE/'runs'/task/f'attempt_{attempt:02d}';out.mkdir(parents=True,exist_ok=False)
    target=json.loads((HERE/'targets'/(task+'.json')).read_text())
    bootstrap()
    # Only the isolated replay process uses this evaluator; the original modules stay unchanged.
    spec=importlib.util.spec_from_file_location('rollout',HERE/'rollout500.py')
    evaluator=importlib.util.module_from_spec(spec);sys.modules['rollout']=evaluator;spec.loader.exec_module(evaluator)
    spec=importlib.util.spec_from_file_location('frozen_baseline_worker',BASE/'worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    original_load=worker.load_file
    def load_file(name,path):
        module=original_load(name,path)
        if name=='official_dp3_train':
            cls=module.TrainDP3Workspace
            before=textwrap.dedent(inspect.getsource(cls.run))
            assert before.count('range(cfg.training.num_epochs)')==1
            after=before.replace('range(cfg.training.num_epochs)','range(500)')
            increment='        self.epoch += 1\n'
            assert after.count(increment)==1
            after=after.replace(increment,increment+'''        if self.epoch in (300,400):
            if self._saving_thread is not None:
                self._saving_thread.join()
            snapshot=self.save_checkpoint(tag=f'epoch_{self.epoch:04d}',use_thread=False)
            snapshot_atomic_json(pathlib.Path(self.output_dir)/'snapshot_receipts'/f'epoch_{self.epoch:04d}.json',
                {'checkpoint':str(snapshot),'sha256':snapshot_sha256(snapshot),
                 'completed_epochs':self.epoch,'optimizer_updates':self.global_step,
                 'lr_schedule_epochs':1000,'includes_model_ema_optimizer':True})
''')
            module.__dict__['snapshot_atomic_json']=atomic_json
            module.__dict__['snapshot_sha256']=sha256
            (out/'original_run_source.py').write_text(before)
            (out/'replay_run_source.py').write_text(after)
            (out/'loop_override.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                fromfile='original1000_schedule',tofile='same_schedule_stop_after500')))
            ns={};exec(compile(after,str(path)+'::prefix500','exec'),module.__dict__,ns)
            replay=ns['run']
            def checked_run(self):
                cfg=self.cfg
                assert cfg.training.num_epochs==1000 and cfg.training.seed==0
                assert not cfg.training.debug and not cfg.training.resume
                assert cfg.dataloader.batch_size==128 and cfg.training.gradient_accumulate_every==1
                assert cfg.training.lr_warmup_steps==500 and cfg.policy.num_inference_steps==10
                assert cfg.horizon==16 and cfg.n_obs_steps==2 and cfg.n_action_steps==8
                replay(self)
                assert self.epoch==500 and self.global_step==30500
            cls.run=checked_run
        return module
    worker.load_file=load_file

    reference=[]
    with (Path(target['output'])/'metrics.jsonl').open() as f:
        for _ in range(20):reference.append(json.loads(next(f)))
    original_install=worker.install_logging
    observed=[]
    def install_logging(path):
        original_install(path)
        import wandb
        original_init=wandb.init
        def init(*args,**kw):
            run=original_init(*args,**kw);original_log=run.log
            def log(data,*args,**kw):
                step=int(data.get('global_step',-1))
                if 0<=step<20 and len(observed)==step:
                    a=float(data['train_loss']);b=float(reference[step]['train_loss'])
                    observed.append({'step':step,'reference_loss':b,'replay_loss':a,'abs_difference':abs(a-b)})
                    assert abs(a-b)<=1e-5+1e-4*abs(b), 'Replay prefix differs from frozen baseline'
                    assert abs(float(data['lr'])-float(reference[step]['lr']))<1e-12
                    if len(observed)==20:
                        atomic_json(out/'prefix_audit.json',{'passed':True,'first_updates':20,'checks':observed,
                            'note':'Prefix-loss check supports same initialization/data path; not a claim of bitwise identity through all500 epochs.'})
                return original_log(data,*args,**kw)
            run.log=log
            return run
        wandb.init=init
    worker.install_logging=install_logging
    atomic_json(out/'replay_identity.json',{'created':time.time(),'task':task,'training_seed':0,
        'completed_epoch_target':500,'lr_schedule_epochs':1000,'reference_baseline':target,
        'classification':p['classification'],'source_plan_sha256':sha256(HERE/'plan.json')})
    worker.train_diffusion('SimpleDP3',task,out,smoke=False)
    trained=json.loads((out/'trained.json').read_text())
    assert trained['completed_epochs']==500 and trained['global_step']==30500
    assert sha256(trained['checkpoint'])==trained['sha256']
    alias=out/'checkpoints/epoch_0500.ckpt'
    if not alias.exists():alias.hardlink_to(Path(trained['checkpoint']))
    assert len(observed)==20
    import torch
    import dill
    for epoch in [300,400]:
        receipt=json.loads((out/'snapshot_receipts'/f'epoch_{epoch:04d}.json').read_text())
        assert receipt['completed_epochs']==epoch and receipt['optimizer_updates']==61*epoch
        assert sha256(receipt['checkpoint'])==receipt['sha256']
        payload=torch.load(receipt['checkpoint'],pickle_module=dill,map_location='cpu',weights_only=False)
        assert {'model','ema_model','optimizer'}<=set(payload['state_dicts'])
        assert dill.loads(payload['pickles']['epoch'])==epoch
        assert dill.loads(payload['pickles']['global_step'])==61*epoch
        del payload
    last=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
    assert 4e-5<float(last['lr'])<6e-5, 'Original1000-epoch LR schedule was not retained'
    atomic_json(out/'replay_audit.json',{'passed':True,'completed_epochs':500,'optimizer_updates':30500,
        'lr_schedule_epochs':1000,'lr_at_stop':float(last['lr']),'baseline_prefix_first20_updates_match':True,
        'checkpoint_sha256':trained['sha256'],'checkpoint_source':'Fresh replay prefix; not an originally retained500-epoch file'})


if __name__=='__main__':main()

"""Continue the complete saved DP state from500 through1000 completed epochs."""
import copy
import difflib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import textwrap
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0,str(BASE))
from runtime import atomic_json,bootstrap,sha256,install_render_fix


def restore_context(ws,context,scheduler,ema,train_loader,val_loader,runner,device):
    import numpy as np
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    assert ws.epoch==context['completed_epochs']==500 and ws.global_step==context['global_step']==30500
    assert context['lr_schedule_epochs']==1000 and context['lr_scheduler']['last_epoch']==30500
    scheduler.load_state_dict(context['lr_scheduler'])
    for group,lr in zip(ws.optimizer.param_groups,context['lr_scheduler']['_last_lr']):group['lr']=lr
    for key,value in context['ema_state'].items():setattr(ema,key,value)
    assert ema.optimization_step==30500
    runner.round_id=context['runner_state']['round_id'];runner.scores=context['runner_state']['scores']
    assert context['train_loader_generator'] is None and context['val_loader_generator'] is None
    assert train_loader.sampler.generator is None
    # The old persistent loaders already existed at epoch500. A fresh loader must not draw
    # an extra worker base seed from the restored global RNG. Keep RandomSampler.generator=None.
    train_loader.generator=torch.Generator().manual_seed(0)
    val_loader.generator=torch.Generator().manual_seed(1)
    sampling_batch=dict_apply(context['train_sampling_batch'],lambda x:x.to(device,non_blocking=True))
    random.setstate(context['rng']['python']);np.random.set_state(context['rng']['numpy'])
    torch.set_rng_state(context['rng']['torch_cpu'])
    torch.cuda.set_rng_state_all(context['rng']['torch_cuda'])
    atomic_json(Path(ws.output_dir)/'restore_audit.json',{'passed':True,'epoch':ws.epoch,'global_step':ws.global_step,
        'lr_scheduler_last_epoch':scheduler.last_epoch,'optimizer_lr':[x['lr'] for x in ws.optimizer.param_groups],
        'ema_updates':ema.optimization_step,'runner_round_id':runner.round_id,
        'sampling_batch_restored':True,'all_rng_states_restored':True,'extra_loader_base_seed_rng_draw_avoided':True,
        'random_sampler_generator_remains_none':train_loader.sampler.generator is None})
    return sampling_batch


def save_snapshot(ws,scheduler,ema,train_loader,val_loader,sampling_batch,runner):
    import numpy as np
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    assert ws.epoch in (600,700,800,900,1000)
    assert ws.global_step==61*ws.epoch and ema.optimization_step==ws.global_step
    if ws._saving_thread is not None:ws._saving_thread.join()
    checkpoint=Path(ws.save_checkpoint(tag=f'epoch_{ws.epoch:04d}',use_thread=False))
    final=checkpoint.parent/'final.ckpt'
    if ws.epoch==1000:final.hardlink_to(checkpoint)
    context={'format_version':1,'completed_epochs':ws.epoch,'global_step':ws.global_step,'lr_schedule_epochs':1000,
        'lr_scheduler':scheduler.state_dict(),'ema_state':{k:v for k,v in ema.__dict__.items() if k!='averaged_model'},
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
               'torch_cuda':torch.cuda.get_rng_state_all()},
        'train_sampling_batch':dict_apply(sampling_batch,lambda x:x.detach().cpu()),
        'runner_state':{'round_id':runner.round_id,'scores':runner.scores},
        'train_loader_generator':train_loader.generator.get_state(),'val_loader_generator':val_loader.generator.get_state(),
        'resume_notes':'Continuation from500; cumulative100-epoch snapshot. Stop at1000 unless user authorizes more.'}
    path=checkpoint.parent/f'epoch_{ws.epoch:04d}_resume.pt';tmp=path.with_suffix('.pt.tmp');torch.save(context,tmp);tmp.replace(path)
    assert scheduler.last_epoch==ws.global_step
    receipt={'completed_epochs':ws.epoch,'optimizer_updates':ws.global_step,
        'checkpoint':str(checkpoint),'sha256':sha256(checkpoint),'resume_context':str(path),'resume_context_sha256':sha256(path),
        'scheduler_last_epoch':scheduler.last_epoch,'ema_updates':ema.optimization_step,'optimizer_lr':[x['lr'] for x in ws.optimizer.param_groups]}
    atomic_json(Path(ws.output_dir)/'snapshot_receipts'/f'epoch_{ws.epoch:04d}.json',receipt)
    if ws.epoch==1000:
        assert all(abs(x['lr'])<1e-12 for x in ws.optimizer.param_groups)
        atomic_json(Path(ws.output_dir)/'terminal_receipt.json',dict(receipt,checkpoint=str(final)))


def main():
    task,attempt=sys.argv[1],int(sys.argv[2])
    plan=json.loads((HERE/'plan.json').read_text())
    for name,digest in plan['execution_files'].items():assert sha256(HERE/name)==digest
    assert sha256(BASE/'worker.py')==plan['baseline_worker_sha256']
    assert sha256(BASE/'protocol.json')==plan['baseline_protocol_sha256']
    source=BASE/'sources/dp/diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py'
    assert sha256(source)==plan['official_training_sha256']
    t=json.loads((HERE/'targets'/(task+'.json')).read_text())
    assert sha256(HERE/'targets'/(task+'.json'))==plan['targets_sha256'][task]
    assert sha256(t['checkpoint'])==t['checkpoint_sha256'] and sha256(t['resume_context'])==t['resume_context_sha256']
    out=HERE/'runs'/task/f'attempt_{attempt:02d}';out.mkdir(parents=True,exist_ok=False)
    bootstrap()
    spec=importlib.util.spec_from_file_location('rollout',HERE/'rollout.py')
    evaluator=importlib.util.module_from_spec(spec);sys.modules['rollout']=evaluator;spec.loader.exec_module(evaluator)
    import dill
    import torch
    from omegaconf import OmegaConf
    from hydra.core.utils import setup_globals
    setup_globals()  # Register the standard now/hydra/python_version resolvers for serialized configs.
    from diffusion_policy.workspace import train_diffusion_unet_image_workspace as official
    spec=importlib.util.spec_from_file_location('baseline_worker',BASE/'worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    install_render_fix();worker.install_logging(out)
    payload=torch.load(t['checkpoint'],pickle_module=dill,map_location='cpu',weights_only=False)
    context=torch.load(t['resume_context'],map_location='cpu',weights_only=False)
    assert dill.loads(payload['pickles']['epoch'])==500 and dill.loads(payload['pickles']['global_step'])==30500
    cfg=copy.deepcopy(payload['cfg'])
    assert cfg.training.num_epochs==1000 and cfg.training.seed==0 and cfg.training.use_ema
    assert cfg.dataloader.batch_size==128 and cfg.training.gradient_accumulate_every==1
    assert cfg.horizon==16 and cfg.n_obs_steps==2 and cfg.n_action_steps==8 and cfg.policy.num_inference_steps==10
    cfg.training.resume=False  # All state is restored explicitly, rather than auto-loading a new output's latest.
    cfg.training.device='cuda:0';cfg.logging.mode='offline';cfg.logging.name='DP_resume1000_'+task
    cfg.checkpoint.save_last_ckpt=False;cfg.checkpoint.save_last_snapshot=False;cfg.checkpoint.topk.k=0
    OmegaConf.save(cfg,out/'config.yaml',resolve=True)
    cls=official.TrainDiffusionUnetImageWorkspace
    before=textwrap.dedent(inspect.getsource(cls.run))
    assert before.count('range(cfg.training.num_epochs)')==1
    after=before.replace('range(cfg.training.num_epochs)','range(self.epoch, cfg.training.num_epochs)')
    anchor="    log_path = os.path.join(self.output_dir, 'logs.json.txt')\n"
    assert after.count(anchor)==1
    after=after.replace(anchor,"    train_sampling_batch=continuation_restore(self,continuation_context,lr_scheduler,ema,train_dataloader,val_dataloader,env_runner,device)\n"+anchor)
    increment='            self.epoch += 1\n';assert after.count(increment)==1
    after=after.replace(increment,increment+'''            if self.epoch in (600,700,800,900,1000):
                continuation_save(self,lr_scheduler,ema,train_dataloader,val_dataloader,train_sampling_batch,env_runner)
''')
    official.__dict__.update(continuation_restore=restore_context,continuation_context=context,continuation_save=save_snapshot)
    ns={};exec(compile(after,str(source)+'::resume500_to1000','exec'),official.__dict__,ns)
    cls.run=ns['run']
    (out/'original_run_source.py').write_text(before);(out/'continuation_run_source.py').write_text(after)
    (out/'training_loop.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='official_loop',tofile='restore500_and_continue_to1000')))
    ws=cls(cfg,output_dir=str(out))
    ws.load_payload(payload,include_keys=['global_step','epoch'])
    del payload
    assert ws.epoch==500 and ws.global_step==30500 and Path(ws.output_dir)==out
    assert ws.model.horizon==16 and ws.ema_model.num_inference_steps==10
    worker.heartbeat(out,stage='restoring500',epoch=500,global_step=30500)
    atomic_json(out/'continuation_identity.json',{'task':task,'source':t,'source_epoch':500,'target_epoch':1000,
        'additional_epochs':500,'initialization':'restore model, EMA, optimizer, scheduler, EMA counters, RNG, sampling batch and runner state',
        'source_plan_sha256':sha256(HERE/'plan.json')})
    ws.run()
    assert ws.epoch==1000 and ws.global_step==61000
    for epoch in [600,700,800,900,1000]:
        receipt=json.loads((out/'snapshot_receipts'/f'epoch_{epoch:04d}.json').read_text())
        assert receipt['completed_epochs']==epoch and receipt['optimizer_updates']==61*epoch
        assert sha256(receipt['checkpoint'])==receipt['sha256']
        assert sha256(receipt['resume_context'])==receipt['resume_context_sha256']
        saved=torch.load(receipt['checkpoint'],pickle_module=dill,map_location='cpu',weights_only=False,mmap=True)
        assert {'model','ema_model','optimizer'}<=set(saved['state_dicts'])
        assert dill.loads(saved['pickles']['epoch'])==epoch and dill.loads(saved['pickles']['global_step'])==61*epoch
        del saved
    trained=json.loads((out/'terminal_receipt.json').read_text())
    assert sha256(trained['checkpoint'])==trained['sha256']
    atomic_json(out/'trained.json',dict(trained,source_checkpoint_sha256=t['checkpoint_sha256'],selection='fixed terminal1000 EMA'))
    atomic_json(out/'training_audit.json',{'passed':True,'start_epoch':500,'completed_epochs':1000,'additional_epochs':500,
        'start_updates':30500,'optimizer_updates':61000,'additional_updates':30500,'scheduler_last_epoch':61000,'ema_updates':61000,
        'resume_context_restored':True,'final_lr':0.0,'snapshot_epochs_verified':[600,700,800,900,1000]})
    ws.ema_model.cuda().eval()
    result=evaluator.evaluate(ws.ema_model,task,'DP',out/'final_rollouts',final=True,checkpoint=trained['checkpoint'])
    atomic_json(out/'complete.json',result)
    import wandb
    wandb.finish()


if __name__=='__main__':main()

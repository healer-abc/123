"""Metadata-only workspace atlas; inspect small source ASTs, never import project code.

Writes only this new atlas directory. Does not read credentials or hash datasets,
checkpoints, archives, virtual environments, or arbitrary binary files.
"""
import ast
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import warnings

ROOT=Path('/home/zhsu/jyluo')
OUT=Path(__file__).resolve().parent
OUT_REL=OUT.relative_to(ROOT).as_posix()
CONFIG_REL='projects/openpi/src/openpi/training/config.py'
CONFIRMED={
 'experiments/pi05_libero_adaptation/manual_workflow/lab.sh',
 'experiments/pi05_libero_adaptation/manual_workflow/results.py',
 *['experiments/pi05_libero_adaptation/report/'+n for n in (
 'INTERVIEW_TRAINING_EVAL_QA_20261022.md','INTERVIEW_72_ANSWERS_20261023.md',
 'OPENPI_ENGINEERING_RUNBOOK_20261023.md')],
 OUT_REL+'/build_map.py',
 OUT_REL+'/lookup.py',
 'OPENPI_REPOSITORY_MAP.md',
}
EXACT={
 'openpi_env.sh':('历史项目环境变量入口；只作用当前shell，含编译器、缓存、LIBERO路径；不等同激活venv','每次使用前审查绝对路径/镜像/解释器'),
 'README_部署说明.md':('历史部署说明，含离线依赖迁移、旧环境和清理命令','历史步骤非当前状态；不得盲跑旧删除/覆盖命令'),
 'pyproject.server.toml':('部署时把Git依赖改成本地路径的配置模板，不是当前生效配置','与projects/openpi/pyproject.toml对比，不要覆盖冻结源码'),
 'tools/add_pi05_lora_config.py':('历史补丁工具：向训练配置追加本地pi05 LoRA配方','会修改源码；已有配置不要重复执行'),
 'tools/patch_libero_video_names.py':('历史补丁工具：修改LIBERO示例视频命名/成功视频保存选项','会修改官方来源示例；先检查是否已经打补丁'),
 'tools/auto_eval_after_train.sh':('旧E0训练完成后自动服务/四套测评调度','写死旧路径/GPU/进程匹配；禁止作为新实验通用入口'),
 'tools/gpu_monitor.sh':('设备显存、利用率、功耗采样工具','确认采样设备和进程归属；不是显存互斥锁'),
 'tools/parse_train_log.py':('训练文本日志解析为step/吞吐CSV及JSON','确认日志阶段/恢复窗口/原始值与区间均值'),
 'tools/parse_eval_log.py':('历史LIBERO文本评测日志解析与成功率汇总','旧日志身份不足，不能直接替代新协议配对episode'),
 'scripts/train.py':('JAX/Flax训练主入口：初始化、数据装载、求导、优化、日志、checkpoint','训练预算、trainable集合、LR、恢复、EMA与保存完成'),
 'scripts/train_pytorch.py':('PyTorch训练入口；不是本轮JAX主线实际训练器','不要把另一后端的默认行为套用到本轮'),
 'scripts/serve_policy.py':('普通模型WebSocket服务入口，可指定config/checkpoint','默认专家与自己训练checkpoint不同；不接显式噪声评测'),
 'scripts/compute_norm_stats.py':('从训练数据计算归一化资产','会写统计文件；新数据须独立资产，原统计不要覆盖'),
 'src/openpi/training/config.py':('官方来源配置注册表，本地新增E1/E2/E3并配置数据、冻结、LR与资产','核心人工审查：默认值与真实argv、冻结集合和资产路径'),
 'src/openpi/models/pi0_config.py':('模型变体、动作维度、state开关和get_freeze_filter','返回冻结集合不是训练集合；LoRA不自动等于冻结视觉'),
 'src/openpi/models/pi0.py':('视觉语言前缀、动作suffix、flow loss与动作积分采样','state是否实际进入模型、mask、时间方向、padding loss'),
 'src/openpi/models/gemma.py':('语言/动作双分支Transformer，attention、FFN、remat及LoRA配置','真实LoRA位置/rank、分支路径、注意力与梯度区别'),
 'src/openpi/models/lora.py':('本地实际使用的低秩Einsum/FFN算子','A/B初始化、缩放及scan轴计数，勿套PEFT默认'),
 'src/openpi/models/siglip.py':('图像编码器及其输出映射/重算','img子树范围、计算与冻结边界'),
 'src/openpi/models/model.py':('观测/动作定义、图像预处理与训练增强','双相机、颜色/空间增强、有效mask和state实际使用'),
 'src/openpi/models/tokenizer.py':('指令/state token化与tokenizer资源加载','当前discrete_state_input=False；确认资源缓存'),
 'src/openpi/training/data_loader.py':('LeRobot/RLDS装载、变换、shuffle与batch','帧采样不等于task均衡；尾部动作mask有无传入loss'),
 'src/openpi/training/checkpoints.py':('训练恢复状态、推理参数导出和保存完成','EMA导出、finalize、恢复优化器不等于精确恢复数据顺序'),
 'src/openpi/training/optimizer.py':('LR schedule和AdamW/裁剪定义','实际LR、beta、weight decay、裁剪与受训dtype'),
 'src/openpi/policies/policy.py':('模型前后变换、JIT与动作推理','输入/输出dtype、显式noise、推理计时边界'),
 'src/openpi/policies/policy_config.py':('按配置、checkpoint及资产构造policy','推理权重类型、归一化和训练/推理精度差别'),
 'src/openpi/policies/libero_policy.py':('LIBERO输入映射/双相机mask和32→7动作输出适配','图像方向/通道、state存在不等于使用、动作维度'),
 'src/openpi/shared/download.py':('当前源码的外部权重/tokenizer缓存与下载实现','先查缓存，避免重下覆盖；具体来源以Git索引对比为准'),
 'examples/libero/main.py':('官方来源、已本地改过的视频/任务评测示例','--args.*参数；普通演示不等于固定manifest正式测评'),
 'pyproject.toml':('生效的包依赖/uv本地路径源声明','已本地适配；重建新环境，不同步覆盖旧venv'),
 'uv.lock':('依赖解析锁，不是训练日志','与生效pyproject、平台/Python和实际freeze共同审查'),
}
LOCAL={
 'audit_parameters.py':'逐叶统计总参数与训练参数，输出CSV',
 'build_eval_manifest.py':'生成固定任务/初始状态/hash/环境与策略seed清单；不可重建覆盖已冻结清单',
 'eval_manifest.py':'按冻结身份运行闭环、保存动作/失败视频/attempt，拒绝不兼容旧结果',
 'seeded_policy_server.py':'显式高斯噪声服务；接受10×32 noise并核验形状',
 'development_watchdog.py':'开发/确认服务和客户端资源准入、锁、心跳、限次恢复',
 'training_watchdog.py':'历史E3训练/恢复监督；带旧实验路径，不是任意run启动器',
 'cost_profiles_watchdog.py':'监督成本短测和历史数据恢复，使用旧run约束',
 'profile_training.py':'独立220步训练短测；20预热+200测量',
 'profile_hooks.py':'W&B初始化后安装observer，避免log重绑导致缺失',
 'compare_development.py':'旧E1/E2开发配对bootstrap；组名/输出路径硬编码',
 'complete_e3_analysis.py':'等待E3开发完成后与E1/E2比较；旧路径',
 'review_development.py':'旧E1/E2动作边界审计和固定失败视频接触表生成',
 'review_final_development.py':'E3终审审计与固定失败样本，不等于全体人工逐帧标注',
 'verify_e2_smoke.py':'E2冻结/训练叶差分与checkpoint双重读取检查，seed42专用',
 'verify_e3_smoke.py':'E3冻结/训练叶检查及两次加载推理，配方专用',
 'freeze_confirmation.py':'原E2候选冻结，记录权重/源码/资产/统计哈希；拒绝覆盖',
 'confirmation_gate.py':'原候选身份/内容核验，原E0/E1/E2约束',
 'run_confirmation_pipeline.py':'原确认+E0开发补齐+最终分析调度；不要盲启动',
 'analyze_confirmation.py':'原确认记录验收、suite等权/Wilson/成对bootstrap',
 'run_development_pipeline.sh':'历史开发看门狗外壳；非当前运行状态证明',
 'common.py':'前置诊断路径、命令事件提取/哈希等CPU工具',
 'prepare.py':'该实验的数据/身份准备；具体行为见所属目录说明和模块摘要',
 'analyze_cpu.py':'命令/指关节响应与既有rollout的只读统计',
 'loss_probe.py':'E1/E2固定配对示范窗口flow loss分维度前向，不更新参数',
 'gpu_diagnostics.py':'限预算GPU前向/短重放监督；不会开始加权训练',
 'replay.py':'已暴露开发状态诊断重放，另存所有成功/失败视频',
 'prepare_rollout_review.py':'准备同policy-step对齐重放接触表',
 'first_input_probe.py':'保存精确首输入/noise并重复推理，对照reset/构造差异',
 'fixed_input_reload.py':'同输入包跨新模型进程重载检查',
 'run_reload_checks.py':'原诊断预算内两次新进程重载调度',
 'render_failure_review.py':'当前失败粗/细时间表生成，供人工审阅',
 'summarize_loss.py':'诊断分维度loss汇总与探索性task抽样区间',
 'finalize_diagnosis.py':'验收诊断身份/hash/动作与视频，不是新训练',
 'lab.sh':'本会话新增：手动四组环境/选择/训练/服务/评测封装；source不启动实验',
 'results.py':'本会话新增：新manual run审计/冻结/配对分析；复用旧数学不执行旧main',
}


def parse_index(worktree,gitdir):
    b=(gitdir/'index').read_bytes()
    if b[:4]!=b'DIRC':raise ValueError('not git index')
    version,n=struct.unpack('>II',b[4:12])
    if version not in (2,3):raise ValueError(f'unsupported index v{version}')
    out={};pos=12
    for _ in range(n):
        start=pos;mode=struct.unpack('>I',b[pos+24:pos+28])[0]
        oid=b[pos+40:pos+60].hex();flags=struct.unpack('>H',b[pos+60:pos+62])[0]
        pos+=62
        if flags&0x4000:pos+=2
        end=b.index(0,pos);name=b[pos:end].decode('utf-8','surrogateescape')
        pos=start+((end+1-start+7)//8)*8
        out[(worktree/name).relative_to(ROOT).as_posix()]=(oid,mode)
    head=(gitdir/'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref=head[5:]
        if (gitdir/ref).exists():head=(gitdir/ref).read_text().strip()
        else:
            for line in (gitdir/'packed-refs').read_text().splitlines():
                if line.endswith(' '+ref):head=line.split()[0]
    return out,{'worktree':str(worktree.relative_to(ROOT)),'gitdir':str(gitdir.relative_to(ROOT)),
                'recorded_HEAD':head,'index_version':version,'index_entries':n,
                'caveat':'comparison is against local Git index, not fetched current upstream; index may contain staged changes'}


def scope(path):
    if path.startswith(('projects/openpi/','experiments/pi05_libero_adaptation/')):return '主线'
    if path.startswith(('experiments/pi0','experiments/pi05')):return 'MetaWorld-VLA旁线'
    if path.startswith('experiments/metaworld'):return 'MetaWorld关联基线'
    if path.startswith(('dataset_root/','myopenpi/','projects/offline-deps/','tools/','metrics/','metrics_lora/','eval_auto/')):return '共享输入/部署/历史产物'
    if path.split('/')[0] in ('miniforge3',):return '共享环境基础设施'
    if path.startswith('.codex/'):return '敏感基础设施'
    if path.startswith(('gemma-exp/','models/','projects/LLaMA-Factory/','projects/mujoco-fetch-push-rl/')):return '相邻独立项目'
    if '/' not in path and (path in ('OPENPI_REPOSITORY_MAP.md','openpi_env.sh','README_部署说明.md','pyproject.server.toml','auto_eval.log','metrics_lora_steps.txt') or path.endswith('.tar.gz')):return '共享输入/部署/历史产物'
    return '其他/关联待核实'


def is_dependency(path):
    parts=Path(path).parts
    return any(x.startswith('.venv') for x in parts) or path.startswith('miniforge3/') or '/site-packages/' in path


def is_sensitive(path):
    n=Path(path).name.lower()
    return path.startswith('.codex/') or n in ('auth.json','.env','id_rsa','id_ed25519','credentials','token.json') or n.endswith(('.pem','.key'))


def info(path,kind,index):
    name=Path(path).name;suffix=Path(path).suffix.lower()
    origin='来源待核实';evidence='路径/类型推断，非作者证明';review='按用途核查';creator='未知';code=False
    if is_sensitive(path):
        return ['敏感配置/凭据','仅文件名/metadata，未读取内容','认证或私密材料；不可公开原文','不得上传/纳入模型实验交付','认证工具/用户配置',False]
    if kind=='symlink':return ['链接/引用','未跟随链接，不重复扫描目标','路径别名或环境链接；不代表独立数据副本','审查目标与可移植性','安装器/本地配置',False]
    if is_dependency(path):
        return ['环境安装产物/第三方包','虚拟环境/发行版路径；未逐包确认作者','解释器/依赖内部文件：'+name,'通常不手改；审版本/lock/freeze与环境隔离','uv/pip/conda安装或缓存；不是train.py生成',False]
    if '/.git/' in path or name=='.git':
        return ['版本控制元数据','Git管理路径','Git对象/索引/引用/子模块记录：'+name,'只用于追溯；不手改index/对象','Git clone/checkout/index操作',False]
    if path.startswith('dataset_root/'):
        if '/meta/' in path:role='LeRobot数据元信息：'+{'info.json':'格式/帧率/总数/字段','episodes.jsonl':'每条轨迹任务与长度','tasks.jsonl':'训练task_index与指令','stats.json':'数据发布方统计'}.get(name,name)
        elif suffix=='.parquet':role='LeRobot单episode数据（RGB、wrist、state、action等），'+Path(path).stem
        else:role='数据下载缓存/记录：'+name
        return ['外部下载数据','数据仓库physical-intelligence/libero与本地路径','输入，不是训练器采集的新演示；'+role,'审查版本/帧率/任务/长度/变换；不得覆盖原数据','数据发布方转换；HF下载落盘',False]
    if path.startswith('myopenpi/'):
        return ['外部预训练资产/下载缓存','OPENPI_DATA_HOME映射路径','base权重/OCDBT分片或tokenizer资产：'+name,'查完整性/模型用途；不是本地微调结果','openpi下载器/GCS或离线迁移',False]
    if '/checkpoints/' in path:
        external=not path.startswith('projects/openpi/checkpoints/')
        role=('公开模型下载/运行时权重，未据此认定本地训练' if external else '本地训练checkpoint：恢复状态/推理权重/归一化资产')
        if name=='_CHECKPOINT_METADATA':role+='；目录保存完成的重要标记'
        elif '/params/' in path or '/opt_state/' in path:role+='；数组/分片必须与索引配套，不能逐片删改'
        elif name=='norm_stats.json':role+='；推理资产，审查是否同源'
        return [('下载/运行时模型资产' if external else '训练器自动产物'), '路径类型；具体下载来源看所属manifest' if external else '官方来源train/checkpoint实现，经本地配置/包装调用',role,'审查run/config/step/EMA/finalize/hash；不手改权重','外部下载/本地封装准备' if external else 'scripts/train.py + Orbax/checkpoints.py',False]
    if '/wandb/' in path:
        return ['训练追踪自动产物','W&B目录模式','W&B离线运行/历史事件/元信息：'+name,'审查真实argv、指标来源、step与恢复历史；勿公开可能的私密运行配置','训练器调用W&B；不是人工书写算法',False]
    if '__pycache__' in path or suffix=='.pyc':return ['Python自动缓存','路径/扩展名','编译缓存；不是实验配置或结果','不应审此文件判断源代码，查看对应.py','Python import/compile',False]
    if name.endswith(('.log','.pid','.lock')) or name.endswith('.launch.json') or '/logs/' in path:
        return ['运行/调度/重定向自动产物','扩展名/命名模式','日志、历史PID、互斥或启动快照：'+name,'查时间/状态/真实进程；PID文件不证明仍活动；错误保留','shell重定向/本地监督器/训练与服务输出',False]
    if '/evaluations/' in path or '/rollouts/' in path or path.startswith('eval_auto/'):
        identity='/'.join(Path(path).parts[-4:])
        return ['测评管线自动产物','所属目录与文件类型','身份化评测/轨迹/失败视频/汇总：'+identity,'审phase、state、seed、成功口径、完整分母、hash；失败不可挑最好重跑','eval_manifest.py/历史本地调度器；不是官方训练器自动建立此目录',False]
    if path.startswith(('metrics/','metrics_lora/')) or '/metrics/' in path:
        return ['指标解析/采样自动产物','metrics路径','训练loss/吞吐/资源/成本恢复记录：'+name,'审原始日志、预热、同步、恢复来源与共享干扰','本地parse脚本/profile工具/nvidia-smi采样',False]
    if path.startswith('experiments/') and any('/'+x+'/' in path for x in ('artifacts','results','results_servo','results_droid_units','videos','runs','training','evaluation','progress','paired_episodes','operations')) and suffix not in ('.py','.sh','.bash','.md','.yaml','.yml','.toml'):
        return ['本地实验管线产物/输入快照','实验运行目录；生成者需结合所属worker/prepare/rollout','本地准备/运行/统计产物：'+path,'审查数据是否输入/输出、任务身份、原始动作、计分与hash；不认定为官方训练器原生输出','所属本地prepare/worker/rollout/统计工具（可能调用外部官方实现）',False]
    if '/sources/' in path or '/deps/' in path or '/runtime_act/' in path:
        origin='外部源码/运行时副本，可能含本地适配';evidence='来源隔离目录；应核对source_manifest/补丁，不凭目录认定未改'
        creator='下载/复制及本地prepare脚本'
    elif path in index:
        oid,mode=index[path]
        if mode==0o160000:return ['Git子模块引用','Git index mode160000','上游子模块引用，不是整个目录的代码作者标记','核对子模块自己的索引与本地修改','Git',False]
        p=ROOT/path
        if p.is_file() and p.stat().st_size<16*1024*1024:
            raw=p.read_bytes();blob=lambda x:hashlib.sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()
            if blob(raw)==oid:origin='上游来源：本地Git索引一致';evidence='当前blob SHA1与对应仓库index相同'
            elif b'\r\n' in raw and blob(raw.replace(b'\r\n',b'\n'))==oid:origin='上游路径：仅换行不同';evidence='CRLF→LF后blob与本地index相同'
            else:origin='上游路径：本地内容修改';evidence='当前blob SHA1不同于本地Git index'
        creator='对应上游仓库；本地差异的个人作者未逐项验证'
    elif path in CONFIRMED or path.startswith(OUT_REL+'/'):
        origin='本会话可确认由Agent新增';evidence='当前对话有创建/补丁记录';creator='本会话文档/工具创建'
    elif path.startswith(('tools/','experiments/')):
        origin='本地项目扩展/文档/材料';evidence='项目结构及脚本行为；并非上游openpi原生文件，个人作者未逐项证明';creator='本地工具/人工或Agent辅助'
    elif path.startswith('projects/offline-deps/'):
        origin='离线引入的第三方源码';evidence='pyproject本地依赖源+源码包；没有Git索引保证完全未改';creator='外部项目+离线打包/迁移'
    elif path.startswith('projects/openpi/'):
        origin='未列入本地上游索引：待核对';evidence='Git index无此路径；不据此直接认定作者';creator='本地新增/复制/更新，待核实'
    if path.startswith('projects/openpi/'):
        key=path[len('projects/openpi/'):]
    else:key=path
    if key in EXACT:role,review=EXACT[key]
    elif path in EXACT:role,review=EXACT[path]
    elif path.startswith('experiments/pi05_libero_adaptation/') and name in LOCAL:
        role=LOCAL[name];review='确认入口参数/旧run硬编码、输出路径；审阅后才执行'
    elif name in ('candidate_freeze.json','freeze.json'):
        role='候选/参考/权重/源码/统计规则冻结登记，不是权重本身';review='最高优先级审查；冻结后不手改，不覆盖'
        creator='本地冻结工具生成，需本人先完成审阅'
    elif '/manifests/' in path:
        role='实验身份/数据/运行/暴露/seed清单：'+name;review='最高优先级：实际argv/样本身份/暴露与统计规则；不覆盖已冻结清单'
    elif '/audits/' in path or '/audit/' in path:
        role='审计/参数差分/完整性证据：'+name;review='核对检查范围与passed含义；文件存在不等于人工批准';creator='本地审计程序/人工整理'
    elif name in ('HANDOFF.md','decision_log.md','protocol.md','PROTOCOL.md','MEASUREMENT_NOTES.md'):
        role='交接/决策/预登记或测量边界：'+name;review='本人必读：区分历史快照、当前授权、已做与待做；存在不等于仍在运行'
    elif 'labels' in name and suffix=='.csv':
        role='视频/事件人工或Agent辅助标注表';review='本人复核可见现象、uncertain、取样偏差；不得当接触真值';creator='人工/Agent审阅或辅助生成'
    elif suffix=='.py':
        role=('回归/单元测试：' if name.startswith('test_') or name.endswith('_test.py') else '程序模块：')+name
        review='看具体数据流/入口/副作用；测试文件是代码，不是测试已通过的证明'
    elif suffix in ('.sh','.bash'):role='Shell入口/运行封装：'+name;review='先审路径、解释器、设备、kill/覆盖操作，勿整段盲跑'
    elif suffix in ('.yaml','.yml','.toml','.in'):role='配置/依赖/CI声明：'+name;review='核对生效文件、CLI覆盖与所用后端，不凭默认值报告实际实验'
    elif suffix=='.md':role='说明/交接/研究报告：'+name;review='审数字来源、日期、实验口径与未完成项；不是所有报告都自动生成'
    elif suffix=='.csv':role='结构化表/清单：'+name;review='确认字段、分母、来源程序/人工标签区别'
    elif suffix in ('.json','.jsonl'):role='结构化配置/审计/结果：'+name;review='按所属目录看来源；核对schema/状态/身份，不公开私密值'
    elif suffix in ('.npz','.npy','.hdf5','.h5','.zarr'):role='数据数组/示范/轨迹：'+name;review='核对字段、形状、动作定义和生成程序，不手改二进制'
    elif suffix in ('.mp4','.gif','.jpg','.png','.jpeg','.svg','.pdf'):role='可视化/视频/报告附件：'+name;review='核对采样、帧率、任务与原始记录；图片不代替计数证据'
    elif name.endswith(('.tar.gz','.zip','.whl')):role='源码/依赖离线分发包：'+name;review='审来源、清单/哈希与解压目标；不是当前生效源码';creator='离线打包/下载'
    elif '.bak' in name:role='历史备份：'+name;review='只作对照；运行入口通常不读取此备份'
    else:role='辅助/资产文件：'+name;review='用途按所属目录解释，来源未充分核实时不宣称原创'
    code=suffix in ('.py','.sh','.bash','.yaml','.yml','.toml','.md','.in','.ipynb') and not is_dependency(path)
    if suffix=='.py' and code and (ROOT/path).stat().st_size<500000:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',SyntaxWarning)
                node=ast.parse((ROOT/path).read_text(errors='replace'))
            doc=(ast.get_docstring(node) or '').split('\n')[0].strip()
            funcs=[n.name for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            if doc and '程序模块：' in role:role+='；模块说明：'+doc[:150]
            if funcs:role+='；顶层定义：'+', '.join(funcs[:10])
        except (SyntaxError,ValueError,OSError):pass
    return [origin,evidence,role,review,creator,code]


def folder_info(path):
    exact={
      '.':('整个共享工作区；不是单一Git仓库','按项目边界阅读，不能统一称作本项目原创'),
      'projects':('源代码项目集合：openpi、离线依赖及独立Gemma/Fetch-Push工程','区分主线与相邻项目'),
      'experiments':('LIBERO主线、VLA MetaWorld探索及关联对照的实验目录集合','不能将不同协议/模型结果合并'),
      'projects/openpi':('实际生效的官方来源openpi工作树，含3个实质内容修改路径和大量运行产物','上游来源不等于整个目录未经本地修改'),
      'projects/openpi/src':('Python包源代码根目录','安装路径与import来源'),
      'projects/openpi/src/openpi':('模型/训练/变换/推理包','算法逻辑与数据处理审查'),
      'projects/openpi/src/openpi/models':('JAX/Flax模型、flow与LoRA实现','本轮实际后端；冻结与loss维度'),
      'projects/openpi/src/openpi/models_pytorch':('PyTorch模型实现分支','不是本轮JAX实验的实际训练实现'),
      'projects/openpi/src/openpi/training':('实验配置、数据加载、优化器、保存与恢复','最高优先级人工审查'),
      'projects/openpi/src/openpi/policies':('机器人输入输出适配、policy构造和推理','训练/推理语义一致性'),
      'projects/openpi/src/openpi/serving':('WebSocket策略服务','服务端口、metadata和模型身份'),
      'projects/openpi/src/openpi/shared':('数组类型、归一化、图像、下载、NNX工具','不要凭辅助模块名认定本地作者'),
      'projects/openpi/scripts':('官方来源训练/服务/统计入口','参数解析、真实argv和输出路径'),
      'projects/openpi/examples':('机器人示例入口集合，含LIBERO及其他机器人','示例不等于本项目全部实际执行过'),
      'projects/openpi/examples/libero':('LIBERO示例客户端及其独立Python3.8环境','main.py已本地修改；正式manifest客户端另在experiments'),
      'projects/openpi/packages':('工作区子包，主要openpi-client','与完整模型服务包区分'),
      'projects/openpi/packages/openpi-client':('轻量网络客户端/图像工具包','跨Python环境通信，非完整模型'),
      'projects/openpi/third_party':('Git子模块：LIBERO与ALOHA','各自有上游来源/索引，不能算openpi本人原创'),
      'projects/openpi/docs':('上游来源部署、归一化与远程推理说明','区别新项目报告和上游文档'),
      'projects/openpi/assets':('训练与推理所需归一化资产','有输出也有复用输入；不是可随意删除缓存'),
      'projects/openpi/checkpoints':('正式/调试/smoke/profile训练权重和恢复状态','不要混用checkpoint用途，禁止挑删分片'),
      'projects/openpi/wandb':('离线W&B运行记录','保留指标来源和训练argv'),
      'projects/offline-deps':('lerobot/dlimp本地源码依赖，供pyproject引用','第三方来源，未做完整上游差分认证'),
      'dataset_root':('HF/LeRobot数据根目录，由HF_LEROBOT_HOME指向','外部输入，非本地训练产物'),
      'dataset_root/physical-intelligence/libero':('LIBERO四套合并1693条示范/273465帧','核对版本、任务、帧率与动作定义'),
      'dataset_root/physical-intelligence/libero/meta':('数据schema、统计、episode和task映射','本人必审；不能将训练ID与套件ID混用'),
      'dataset_root/physical-intelligence/libero/data':('按chunk分组的episode Parquet输入','有图像嵌入；不因total_videos=0认定无图像'),
      'myopenpi':('历史下载缓存根目录包装','不是第二份正在执行的openpi源码'),
      'myopenpi/dataset/ckpt':('OPENPI_DATA_HOME缓存：pi05_base与tokenizer','预训练输入，不是本地微调checkpoint'),
      'tools':('早期本地部署/补丁/日志解析/监控/自动评测脚本','带历史硬编码，先审副作用'),
      'metrics':('早期全量训练日志解析结果','历史解析，不是新协议评测'),
      'metrics_lora':('早期LoRA训练日志解析结果','核对对应train_lora.log'),
      'eval_auto':('历史自动LIBERO测评、失败视频与旧协议解析','身份信息不足，不替代新manifest配对结果'),
      'experiments/pi05_libero_adaptation':('E0–E3主线科学协议、调度、审计与结果','当前最重要实验工作目录'),
      'experiments/pi05_libero_adaptation/manual_workflow':('本会话创建的学习用新run命令/结果工具','不是官方入口、不是已运行新训练的证据'),
      'experiments/pi05_libero_adaptation/manifests':('数据、run、暴露、评测身份、候选冻结登记','最高优先级审查；冻结后不手改'),
      'experiments/pi05_libero_adaptation/audits':('参数、smoke、环境和最终记录审计证据','检查范围不等于所有机制已证明'),
      'experiments/pi05_libero_adaptation/report':('人工/Agent整理的交付、审阅与学习文档','核对日期、分母、口径、结论边界'),
      'experiments/pi05_libero_adaptation/extensions':('主线结束后的探索拓展','不计入旧6400主结果'),
      'experiments/pi05_libero_adaptation/extensions/event_weighting_20261022':('事件加权前置诊断，未开始加权训练','保留抽样/复现局限，不能说新方法已提升'),
      'gemma-exp':('独立Gemma/LLaMA-Factory语言模型LoRA工程','非openpi；当前IDE打开不改变本次地图范围'),
      'models':('独立Gemma模型存储','不是pi05_base默认加载来源'),
      'miniforge3':('共享conda发行版/编译器/环境/包缓存','不是项目源代码，勿全目录删除'),
      '.codex':('认证等敏感基础设施','仅记录路径，不读取/公开凭据'),
      'projects/LLaMA-Factory':('独立LLM训练框架，关联gemma-exp','不是本轮openpi训练器'),
      'projects/mujoco-fetch-push-rl':('独立Fetch-Push强化学习工程','不能将其report/RESUME当openpi报告'),
    }
    if path in exact:return exact[path]
    if is_sensitive(path+'/x'):return ('认证/私密基础设施子目录','不要公开内容')
    if is_dependency(path+'/x'):return ('环境安装内部目录：'+path,'不手改包内部；审lock/freeze与解释器')
    if '/.git' in path:return ('Git索引/对象/子模块管理：'+path,'版本追溯，不手动修改内部对象')
    if path.startswith('experiments/') and path.count('/')==1:
        p=ROOT/path/'HANDOFF.md'
        title=''
        if p.is_file():
            title=next((s.strip('# ').strip() for s in p.read_text(errors='replace').splitlines() if s.strip()),'')
        return ('实验分支：'+(title or Path(path).name),'先读该目录HANDOFF/PROTOCOL/source_manifest；不能由文件夹名推导成功率')
    if path.startswith('projects/openpi/checkpoints/'):
        return ('训练权重/恢复状态层级：'+path,'对应config/exp_name/step；smoke/profile不是正式权重')
    if any('/'+x+'/' in path+'/' for x in ('episodes','records','attempts','videos','all_videos','rollouts','paired_episodes')):
        return ('逐任务/状态评测或尝试/视频目录：'+path,'按suite/task/state核对身份、原始动作、计分和重试')
    if any('/'+x+'/' in path+'/' for x in ('logs','wandb','metrics','progress')):
        return ('日志/指标/心跳/历史状态目录：'+path,'看时间与真实进程，不能由PID文件推断正在运行')
    if any('/'+x+'/' in path+'/' for x in ('sources','deps','runtime_act')):
        return ('外部源码/依赖隔离副本：'+path,'可能含本地适配，核对source_manifest/patch')
    if '/report/' in path or '/artifacts' in path:
        return ('报告、诊断/可视化或审阅证据子目录：'+path,'区分机器自动产物和本人审阅标签')
    return ('所属项目的子目录实例：'+path,'逐文件用途见文件清单；不凭目录存在认定运行或作者')


def main():
    indexes={};metadata=[];errors=[]
    for work,gdir in [('projects/openpi','projects/openpi/.git'),
                      ('projects/openpi/third_party/libero','projects/openpi/.git/modules/third_party/libero'),
                      ('projects/openpi/third_party/aloha','projects/openpi/.git/modules/third_party/aloha')]:
        try:
            idx,meta=parse_index(ROOT/work,ROOT/gdir);indexes.update(idx);metadata.append(meta)
        except (OSError,ValueError) as e:errors.append([gdir,str(e)])
    headers=['path','scope','kind','bytes','origin','evidence','purpose','review','created_by']
    groups={};writers={};counts=collections.Counter();origins=collections.Counter();dirs={};code_rows=[];changed=[]
    for label in ('project','dependencies','adjacent'):
        f=gzip.open(OUT/f'{label}_files.csv.gz','wt',encoding='utf-8',newline='')
        groups[label]=f;writers[label]=csv.writer(f);writers[label].writerow(headers)
    def err(e):errors.append([getattr(e,'filename',''),str(e)])
    for current,children,files in os.walk(ROOT,followlinks=False,onerror=err):
        children.sort();files.sort();p=Path(current);rel=p.relative_to(ROOT).as_posix()
        if rel==OUT_REL:
            # Do not recursively inventory this generator's changing outputs.
            children[:]=[];files=[x for x in ('build_map.py','lookup.py') if (p/x).exists()]
        direct_bytes=0;direct_count=0
        entries=[(name,False) for name in files]
        for name in list(children):
            if (p/name).is_symlink():entries.append((name,True));children.remove(name)
        for name,isdirlink in entries:
            path=p/name;relative=path.relative_to(ROOT).as_posix()
            try:
                st=path.lstat();kind='symlink' if stat.S_ISLNK(st.st_mode) else ('file' if stat.S_ISREG(st.st_mode) else 'special')
                sc=scope(relative);origin,evidence,purpose,review,creator,code=info(relative,kind,indexes)
                label='dependencies' if is_dependency(relative) else ('adjacent' if sc in ('相邻独立项目','敏感基础设施','其他/关联待核实') else 'project')
                row=[relative,sc,kind,st.st_size,origin,evidence,purpose,review,creator]
                writers[label].writerow(row);counts[label]+=1;origins[origin]+=1
                direct_bytes+=st.st_size;direct_count+=1
                if code and label=='project':code_rows.append(row)
                if origin in ('上游路径：本地内容修改','上游路径：仅换行不同'):changed.append(row)
            except (OSError,ValueError) as e:errors.append([relative,str(e)])
        dirs[rel]={'path':rel,'scope':scope(rel+'/'),'direct_files':direct_count,'direct_bytes':direct_bytes,
                   'recursive_files':direct_count,'recursive_bytes':direct_bytes,'children':len(children)}
    for f in groups.values():f.close()
    for path in sorted(dirs,key=lambda x:x.count('/'),reverse=True):
        if path=='.':continue
        parent=Path(path).parent.as_posix()
        if parent in dirs:
            dirs[parent]['recursive_files']+=dirs[path]['recursive_files']
            dirs[parent]['recursive_bytes']+=dirs[path]['recursive_bytes']
    with gzip.open(OUT/'directories.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['path','scope','direct_files','recursive_files','logical_bytes','purpose','review'])
        for path,v in sorted(dirs.items()):
            role,review=folder_info(path)
            writer.writerow([path,v['scope'],v['direct_files'],v['recursive_files'],v['recursive_bytes'],role,review])
    def clean(s):return str(s).replace('|','/').replace('\n',' ')
    for label,rows in [('SOURCE_FILES',code_rows),('UPSTREAM_LOCAL_CHANGES',changed)]:
        with (OUT/f'{label}.md').open('w') as f:
            f.write(f'# {label}：逐文件索引\n\n生成日期2026-10-23。对比本地Git索引，不等于在线上游未修改认证；个人作者不从mtime推断。源码摘要只静态读取小文件，不执行代码。\n\n')
            f.write('| 路径 | 来源/证据 | 文件用途 | 本人审查点 |\n|---|---|---|---|\n')
            for row in sorted(rows):
                f.write('|['+clean(row[0])+']('+str(ROOT/row[0])+')|'+clean(row[4]+'；'+row[5])+'|'+clean(row[6])+'|'+clean(row[7])+'|\n')
    result={'date':'2026-10-23','root':str(ROOT),'scope':'openpi/pi0/pi05 + LIBERO + MetaWorld VLA and associated baselines; neighboring projects inventoried as boundaries',
            'files_by_catalog':dict(counts),'origins':dict(origins),'directories':len(dirs),'readable_source_entries':len(code_rows),
            'git_index_metadata':metadata,'upstream_path_changes':len(changed),'errors':errors,
            'size_caveat':'logical lstat sizes; no symlink following, no hardlink deduplication, not physical du usage',
            'self_exclusion':'atlas generated outputs excluded except build_map.py/lookup.py; prevents recursive self-inventory',
            'missing_git_index_paths':[p for p in indexes if not (ROOT/p).exists()],
            'top_level':[v for k,v in sorted(dirs.items()) if k!='.' and '/' not in k]}
    (OUT/'scan_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('files_by_catalog','directories','readable_source_entries','upstream_path_changes','git_index_metadata','errors')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

# MetaWorld 五任务规范基线：50条示教，训练seed0

用户于2026-10-20授权按有出处的常用设置重新开始。本轮共15个独立训练组、300局正式评测；全部从头训练，不加载上轮policy权重。所有历史数据与失败结果保留。

| 方法 | 固定训练与推理参数 | 主要依据 |
|---|---|---|
| ACT | chunk10，KL10，hidden512，FF3200，batch8，lr1e-5，2000epochs；时间聚合开启，query_frequency=1，k=0.01；官方best-validation checkpoint | [官方ACT训练命令](https://github.com/tonyzhaozh/act/blob/main/README.md)、[官方时间聚合](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)、[BAKU附录A.4中MetaWorld的MT-ACT chunk10](https://arxiv.org/html/2406.07539v2#A1.SS4) |
| DP | 官方图像UNet、horizon16/obs2/action8；DDIM，训练噪声100步、推理10步，sample预测；batch128，1000epochs，lr1e-4，EMA；固定terminal checkpoint | [官方DP架构配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)、[DP3论文MetaWorld预算与采样设置](https://arxiv.org/html/2403.03954v4) |
| SimpleDP3 | 官方simple_dp3模型和配置，horizon16/obs2/action8，DDIM10，sample预测；batch128，1000epochs，lr1e-4，EMA；固定terminal checkpoint | [官方SimpleDP3配置](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/config/simple_dp3.yaml)、DP3论文MetaWorld预算 |

这是文献和官方源码支持的单任务适配配置，不声称是整个社区唯一默认值或某篇论文的完全复现。特别是：ACT chunk10来源于多任务MT-ACT；DP3论文主实验的短horizon4/action3与发布代码的16/8不同，本轮固定发布代码的16/8；SimpleDP3发布通用配置3000epochs，本轮依MetaWorld论文预算固定1000epochs。

## 统一实验条件

- 任务：shelf-place、pick-place-wall、pick-place、stick-push、door-open；沿用已固定MetaWorld v2源码及MuJoCo版本。
- 每任务共享同一50条真实专家示教，固定40训练/10验证划分，原始数据通过只读使用方式复用，不重新采集、删减或筛选。
- 训练seed统一0。ACT每epoch5次更新，2000epochs共10000次；扩散模型更新次数以数据采样器和训练日志实际记录，约61000次/任务。不同算法不以相同epoch数假装预算相同。
- 正式评测seed：2026102000–2026102019，三方法、五任务顺序相同。此前训练/参数选择未使用这组seed。
- 开发评测seed：1026102000–1026102004；周期500epochs。开发评测保存并恢复训练RNG，不用于挑选正式测试seed。
- 每组20局，每局200步，成功定义为任意一步原生info.success为真；即使成功也跑满episode，不修改环境判定。
- 共享128×128图像数据和9维本体状态、4维动作；SimpleDP3用512点点云。数值归一化保持各自官方方法。原始官方loader对50示教拟合normalizer的做法保留并明确披露，最终rollout独立。
- checkpoint规则预先固定，不依据正式测试结果选checkpoint。不再做“破0”参数/seed搜索，所有0/20也进入主表。
- 这是单训练seed初步基线，不提供训练seed间均值/方差。20局评测seed不能替代多个独立训练seed。

## 运行与交付

入口为baseline_watchdog.py。先做完整批量训练/反传/推理预检，再自动派发15个训练组。调度只操作本目录已核对身份的进程，不终止其他项目GPU任务。按实测显存限制并发并保留显存余量；训练日志或heartbeat超过1200秒无进展自动报警；基础设施错误最多三次重试，失败尝试留档。科学异常和参数变动不会被静默改配置后继续统计。

代码和协议SHA256冻结于source_manifest.json，数据收据和共享划分同样固定。训练日志保存在runs下；正式每局保存MP4、NPZ（动作、预测chunk、query时间、初始状态、奖励）与完整原生info JSON。JSON最后提交，便于恢复和核验。

进度：progress.md、logs/status.json。异常：logs/latest_alert.json。汇总：report/results.md、report/results.csv。协议差异：protocol.diff。

原始旧实验已按“重新开始”的授权停止，目录../metaworld_official_baselines_50demo_20261019保持原始记录；旧破0监控与汇总跟踪也已停用，避免把探索结果混入本轮。

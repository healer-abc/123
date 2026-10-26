# MetaWorld 基线协议核验与建议

本文件是配置核验和下一轮协议建议，未修改正在运行的训练、模型或冻结的历史结果。

## 结论

正式比较 DP、ACT、SimpleDP3 时，每个任务应对三种方法使用完全相同、预先冻结的评测 seed 序列和初始条件。五任务也可共享 seed 序列，但同一 seed 不表示不同任务具有相同的物理场景。

若只改 rollout seed，只需重新评测，不需要重新训练。若改训练 seed，则对应一个新的独立训练实验，不能通过重新评测代替。

当前 ACT 的4026101900–4026101919集合已经参与执行间隔选择，属于开发/探索性验证。将其他方法改用这组 seed 可以对齐场景，但不能恢复它作为独立未见测试集的性质。正式新主表宜在配置冻结后，统一使用一组从未参与选择的评测 seed；所有失败均计入分母。

## 已核验的一手来源

1. [官方 ACT README](https://github.com/tonyzhaozh/act/blob/main/README.md)：chunk_size=100、KL=10、2000 epochs 是 Transfer Cube 示例；同一仓库还提供 Bimanual Insertion。它不是官方 MetaWorld 配方。时间聚合通过单独评测 flag 启用。
2. [BAKU 论文附录 A.4、表4和附录D](https://arxiv.org/html/2406.07539v2)：MetaWorld 的 action chunk length 为10，A.4明确说明对 MT-ACT 按 benchmark 更换该长度。该基线是多任务 MT-ACT，为本项目单任务 ACT 选择10提供文献参照，但不能声称配置与实验条件完全相同。
3. [官方 ACT 推理代码](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)：开启 temporal_agg 后 query_frequency=1；聚合指数系数 k=0.01。chunk长度和执行间隔不同。
4. [官方 DP 图像 UNet 配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)：horizon=16、观测2步、执行8步、DDPM100步、epsilon预测。这是官方通用任务配置，默认任务为lift，不能标注为官方 MetaWorld 专用配方。
5. [官方 SimpleDP3 配置](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/config/simple_dp3.yaml)：horizon=16、观测2步、执行8步、DDIM10步、sample预测。

以上来源支持具体配置参照；不足以证明整个 MetaWorld 社区统一采用“ACT chunk=10–20且必须开启聚合”。

## 建议的配置起点（尚未执行）

| 方法 | 配置起点 | 说明 |
|---|---|---|
| ACT | chunk_size=10，KL=10；采用官方 temporal_agg=true、query_frequency=1、k=0.01 | chunk10参照BAKU的MT-ACT；聚合参照官方ACT。组合是本项目明确声明的适配配置，不称为社区统一默认。20可作为开发集候选，不能在正式测试集上反复挑选。 |
| DP | horizon16，obs2，action8，DDPM100，epsilon预测 | 当前官方配置可继续作为起点；若改用某篇MetaWorld评测的DDIM方案，需要整体核对训练目标和调度器，不能仅替换推理步数。 |
| SimpleDP3 | horizon16，obs2，action8，DDIM10，sample预测 | 保持SimpleDP3身份，避免与完整DP3配置混称。 |

同一算法的五任务采用同一套默认超参数。不同算法按各自有依据的推理机制运行；公平不要求三种算法拥有相同chunk长度或执行间隔。如另做相同控制频率对照，应单独标注为受控实验。

## 重训边界

- ACT chunk_size 从100改成10/20：训练目标序列长度和query embedding数量变化；正式该配置应从头训练，不把旧checkpoint只执行前10步称为chunk10重训结果。
- KL权重变化：训练目标改变，需要重新训练。现有KL已经为10。
- 仅开启 temporal_agg 或改变执行间隔：属于推理配置改变，可用旧模型做开发集消融；不能替代短chunk模型的训练。
- 仅换环境评测seed：只重跑rollout；训练seed和checkpoint不变。

## 下一轮正式实验应预先冻结

1. 相同50条示教预算（当前40训练/10验证），固定共享数据划分、任务版本、相机和状态接口、动作缩放、200步时限、原生success定义。
2. 每种方法的配置出处、源码版本、实际更新次数、批大小、训练预算与checkpoint选择规则；不能仅用epoch数量比较训练充分性。
3. 配置选择只使用开发集/开发seed；正式测试seed独立冻结，三种算法逐任务使用同一序列，每组20局。
4. 资源允许时建议训练seed0/1/2，报告各训练seed的成功率和均值、标准差；单训练seed的20局结果应明确标注为初步基线。
5. 0/20是有效观测，规范配置无法保证所有组非零；正式主表完整报告。历史探索与主表分开，不覆盖或丢弃失败。

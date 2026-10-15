# 训练端 Lab QV256 正式训练交接

整理日期：2026-10-16。四卡正式训练已完整结束于20,000步；完整FP32导出、72层LoRA合并和独立动作重载审计通过。

本轮是独立的Lab实验，从固定官方基座重新训练；不加载旧Lab15000权重或optimizer。它同时改变语言q/v LoRA、视觉训练范围、动作表示、norm、loss、教师和优化器，不能作为H0/E/F/G的单变量消融。

## 1. 交付位置与身份

模型私有仓库：[HumaLog/Lab-QV256](https://huggingface.co/HumaLog/Lab-QV256)。按仓库[归档守则](Checkpoint与模型归档守则.md)，只为本实验建立独立目录，不覆盖其它实验或步数。

```text
experiments/lab_qv256_20261012/
  README.md
  training_config.yaml
  training_runtime_config.json
  run_metadata.json
  assets/                       # 原manifest、norm/provenance、robot config、输入身份
  processor/                    # 配套Qwen tokenizer/processor文件
  source/                       # 本轮冻结适配代码
  inference/                    # 训练原CLI、参考审计CLI与实际生产CLI分开保存
  evidence/                     # 四rank最终结果、完成证明与门检/映射收据
  checkpoints/global_step_20000/
    training_state/              # 4×rank*.pt + 4×data_rank*.pt及各自读回收据
    hf_ckpt_fp32/                # 14个完整模型分片、index、config与action audits
  SHA256SUMS.txt
```

<!-- HF_ARCHIVE_RECEIPT -->
上传状态：**已归档并通过固定revision核验**。私有HF revision：`84afa4f2fbef16ad6d41dbb1c3b27967f2d7a8e6`。

固定下载目录：[experiments/lab_qv256_20261012](https://huggingface.co/HumaLog/Lab-QV256/tree/84afa4f2fbef16ad6d41dbb1c3b27967f2d7a8e6/experiments/lab_qv256_20261012)。
共84个文件，45,161,047,200字节；所有大文件LFS SHA256与实际源文件一致，小文件已按该revision实际下载读回SHA。
详细收据见[HF归档核验](train/evidence/lab_qv256_20261012/huggingface-archive.json)。
<!-- /HF_ARCHIVE_RECEIPT -->

| 身份字段 | 实际值 |
| --- | --- |
| 实验ID / run | `lab_qv256_20261012` / `formal_20000_save2000` |
| 官方代码revision | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` |
| 原6B基座revision | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` |
| clean数据revision | `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |
| 原manifest SHA256 | `911e7ea61baecd505368fe6d5eb70052827b989ba91f93b6d6d13d331df28e78` |
| norm SHA256 | `b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d` |
| robot config SHA256 | `1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b` |
| 最终complete.json SHA256 | `a47653ee9e0835d9fc11c298fd7d76968779dcf64074df1e0166d686c2dfea13` |
| 训练完成核验SHA256 | `d5e0a495f8f183c51a9276ba38a591050472073929e73469dd3909b1cf4269f1` |

本轮适配代码启动时没有单独记录Git commit；以[冻结源码快照](train/lab_qv256/source/)及checkpoint identity中的内容SHA追溯。本次交接Git提交不冒充训练时的源码提交。官方源码仍以上述固定revision为准。

正式入口是`train_fresh.py`。`train.py`仅提供本轮复用的保存/恢复、教师等辅助函数；其中旧模型构建、旧优化器分组及随机有放回reader不在本轮正式路径中执行。实际采样与配置以fresh入口和本轮运行收据为准。

## 2. 数据、norm与seed

- 50个RoboTwin任务，每任务50条clean示范，共2500条；全部用于训练，没有45/5验证划分。可索引起始帧549,787；没有下一动作的末帧不作为起点。
- 官方`StatefulDistributedSampler(shuffle=True, seed=42)`；每个索引对应确定的真实帧，不在reader里随机有放回抽帧。分片尾部补齐及`drop_last=True`保留，不能宣称每个epoch恰好无重复覆盖所有帧。
- Python、NumPy、Torch模型初始化seed均为42；DataLoader generator为`42+rank`；样本指令/增强使用`42 + epoch×549787 + sampler_index`。每rank实际2个spawn worker、prefetch2、非persistent worker；实际worker数以运行参数为准，不把CLI内未使用的`data.num_workers=8`当成真实值。
- 12维机械臂动作是`future_target[h] - q_now`；按每个horizon分别统计MeanStd，shape为`[50,12]`。推理还原时每个horizon都只加同一次`q_now`，不沿时间累加。
- 两个夹爪保持绝对值，使用MinMax；实际JSON中的action夹爪统计为`[50,2]`，各horizon的min/max行相同，不要误写成文件存储shape为`[2]`。state机械臂/夹爪统计分别为`[12]`/`[2]`。经官方FeatureTransform映射到55维，padding和未使用维度由mask排除。
- norm不含padding统计；数据来源、2500条clean范围及SHA见[原provenance的脱敏副本](train/evidence/lab_qv256_20261012/assets/norm_stats.provenance.json)。本模型不得使用H0/E0/E1的norm。

RGB三路256×256。启用官方配对颜色增强：brightness0.7–1.3、contrast0.6–1.4、saturation0.5–1.5；当前/未来帧和三路相机共用同一组增强参数。HDF5输入为float RGB `[0,255]`：增强前转`[0,1]`，增强后恢复`[0,255]`；学生processor和教师入口各按其约定缩放一次。修复在实际spawn worker中生效；不以转uint8损失浮点精度来规避问题。

## 3. 模型训练范围

| 模块 | 范围 | 参数量 |
| --- | --- | ---: |
| Action Expert | 全量，包括36层、MoE/router/shared expert | 1,786,669,824 |
| 视觉编码器 | 全量训练 | 415,347,712 |
| 语言主干 | 冻结原参数，仅q/v加LoRA | 冻结4,022,468,096 |
| 语言LoRA | 36层×q/v=72模块；r256、alpha512、dropout0，初始B=0 | 94,371,840 |
| 对齐投影、动作输入输出等 | 全量训练 | 151,420,727 |
| 总可训练 | FP32 master | 2,447,810,103 |

语言k/o、MLP、embedding不加LoRA；视觉和Action Expert不加LoRA。`alpha/r=2`是本轮选择，不声称论文规定该值。完整推理模型已合并72个LoRA，不需要额外adapter；合并后为6,375,906,359个参数、1672个参数tensor，另有36个persistent buffer，共1708个保存tensor。

## 4. 实际训练参数

| 参数 | 本轮值 |
| --- | --- |
| 硬件 / 分布式 | 4×RTX4090 48GB / 官方FSDP2 full shard |
| 每卡batch / global batch / 累积 | 32 / 128 / 1 |
| optimizer steps / 处理样本量 | 20,000 / 2,560,000 |
| checkpoint | 每2000步；本次上传最终20k |
| 正式段恢复点 | 同配方gate第3步；此前3步计入20k总数 |
| Action loss | `fm`，官方实现为L2 MSE，不是`L1_fm` |
| 教师 | 冻结在线MoGe2+MoRGBD深度与DINO-video，`no_grad`、BF16 autocast、逐样本生成 |
| 对齐监督 | 当前/未来深度及当前/未来视频；depth、future-depth、video权重各0.004 |
| MoE | 32 experts、top-k4、sigmoid、routed scale4、官方fused实现 |
| MoE auxiliary | sequence-wise `1e-3`、router-z `1e-4`、per_sequence；手动bias更新速度0 |
| optimizer | 官方Muon + AdamW fallback；二维LoRA也沿用官方Muon分组 |
| LR / 最低LR / 调度 | `1e-4` / `5e-5` / 20k cosine，无warmup |
| 专家LR | routed experts乘`sqrt(32/4)`；初始`2.828427e-4`，最终`1.414214e-4` |
| weight decay / clipping | 0 / max norm1 |
| Muon | momentum0.95、Nesterov、NS steps5、match_rms_adamw |
| AdamW | beta(0.9,0.95)、eps1e-8、foreach=true、fused=false |
| 训练attention / compile | action `flex_cached`、vision SDPA、外层compile=false |
| gradient checkpointing | true，non-reentrant |

全部实际值与配置分组见[training_runtime_config.json](train/evidence/lab_qv256_20261012/training_runtime_config.json)和[training_config.yaml](train/evidence/lab_qv256_20261012/training_config.yaml)。教师是冻结目标生成器，不进入optimizer；本次不归档其公开预训练大权重。

## 5. 数值精度与恢复边界

本轮基座、冻结参数和可训练master都保留原FP32值，保存/完整导出也为FP32。官方模块混合精度为Action Expert blocks FP32，vision/language/root BF16 compute，gradient reduction FP32。不能简写为“全程FP32计算”，也不能称为团队E/F/G的“BF16基座协议”。本交接如实记录已有独立实验，没有回改训练或评测dtype。

训练状态由4份`rank*.pt`保存trainable参数分片、optimizer、scheduler及Torch CPU/CUDA RNG；4份`data_rank*.pt`保存loader、generator和epoch。恢复需要原FP32基座、相同官方/适配代码、数据、norm、在线教师和四卡拓扑。冻结基座不重复写入训练state；完整HF推理模型则已包含冻结权重。

最终20k已达到原训练目标。本次仅做归档，不继续训练、不延长schedule。原8k/14k/16k/18k/20k checkpoint及评测文件均不覆盖。

## 6. 最终完成与导出验证

四份最终result均`passed`且`final_step=20000`；`complete.json`为step20000/world4；4份模型状态和4份数据状态的`readback_equal`全true。本次归档另对实际源文件逐SHA读回。

| 最终训练指标 | 值 |
| --- | ---: |
| total loss | 0.013946533203125 |
| flow matching L2 | 0.0092315673828125 |
| depth / future depth | 0.0014095306396484375 / 0.002071380615234375 |
| video current+future | 0.00012290477752685547 |
| sequence-wise / router-z | 0.0010738372802734375 / 0.00001703202724456787 |
| gradient norm before clip | 0.0646437332034111 |
| 最后一step耗时 | 约14.81秒 |

上述是训练损失，不是任务成功率。根输出BF16舍入会使日志分项重求和与total存在小差，不改官方反向图。

| 固定输入、固定噪声审计 | 实测 |
| --- | --- |
| 72层q/v merge，物理动作max abs差 | `2.4383886154666357e-7` |
| 独立进程reload，物理动作max abs差 | `1.1561431190187932e-7` |
| 独立reload canonical max abs差 | 0 |
| 形状 / 有限 / 容差 | 50×14 / true / atol=rtol=1e-6 |
| 审计后端 | FP32 highest、eager attention、FA2 vision、reference MoE、compile关闭 |

原始[merge审计](train/evidence/lab_qv256_20261012/checkpoints/global_step_20000/hf_ckpt_fp32/export_audit.json)、[独立reload审计](train/evidence/lab_qv256_20261012/checkpoints/global_step_20000/hf_ckpt_fp32/action_reload_audit.json)、[实际保存参数值绑定](train/evidence/lab_qv256_20261012/checkpoints/global_step_20000/hf_ckpt_fp32/runtime-parameter-audit.json)均保留。这些审计只证明共同参考后端下的保存/合并/重载一致，不冒充与训练混合精度或正式官方后端逐位相同。

## 7. 评测端接手

1. 按本页固定HF revision下载本实验`hf_ckpt_fp32/`、norm/provenance、robot config、processor和inference配置；逐文件校验`SHA256SUMS.txt`。只做推理不需要下载19.6GB训练恢复状态。
2. 模型、tokenizer、norm、robot config路径改到本地下载位置；其余动作语义、精度和参数保持不变。生产CLI与数值参考审计CLI不得混淆。
3. 实际验证12关节相对MeanStd、两夹爪绝对MinMax、50预测/执行50、10flow、输入三路RGB和当前14关节；不能只复制YAML却未确认进程真正使用它。
4. 本轮soarm正式开发评测已沿用E0固定50×2×5输入、仿真专家/规划检查关闭、FP32、FA2/official MoE、compile/high、256输入、head+observer10fps、cache=false。关闭的是仿真专家，不是模型的MoE。门检4项完整通过且不进入500分母；不得重复已结算回合。沿用900秒墙钟/240秒无完整动作块的策略超时0分继续下一原键规则；真实输入或基础设施错误另行诊断，不能伪造成策略0分。
5. 跨主机实际RGB与E0不同，原生模型RNG流也未连续恢复，E0另有历史精度差异。保留这些限制；本交接不登记尚未完成全量审计的20k最终成功率，不将开发评测称为赛事成绩。

## 8. Git证据与归档校验

[轻量证据目录](train/evidence/lab_qv256_20261012/)包含配置、run metadata、norm、完成收据、导出/动作审计、checkpoint清单、路径脱敏报告和SHA清单；[训练适配源码](train/lab_qv256/source/)用于追溯本轮实现。大权重、训练状态、原始HDF5及任何账号凭据不进入Git。

脱敏文件的distributed SHA与原source SHA分列；norm原字节不改。`HF-SHA256SUMS.txt`对应私有HF包，`SHA256SUMS.txt`对应Git证据目录，二者用途不同。固定HF revision与上传核验结果由实际远端回读后登记。

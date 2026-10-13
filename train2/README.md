# H0 主力训练：Clean-only 完整 Action Expert

## 1. 结论

这一轮不再做 E0/E1 式的小变量试验，而是训练一个可正式评测的主力模型：

```text
LingBot-VLA-v2-6b 原始基座
+ 官方 50 任务 Clean demonstration（仅 Clean）
+ absolute joint action
+ Clean absolute-action norm / bounds_99_woclip
+ L1 flow-matching
+ 完整 36 层 Action Expert
+ state/action/time projections
+ 合法训练期图像增广
```

本方案刻意不使用 Randomized 数据，不用 Randomized 计算 norm、筛选样本、生成 observation 或选择 checkpoint。Randomized 只用于方案冻结后的测试。

## 2. 为什么这样改

E0 只训练 Action Expert 最后 2 层和 projections，共约 101M 参数；F1 在相同训练范围上增加 photometric augmentation，最终为 49/500（9.8%），没有超过 E0 的 54/500（10.8%）。E1 改为 relative arm action，但训练范围仍是最后 2 层。因此下一轮的主要矛盾不应继续押在单个动作表示或单独增广上，而应提高动作模型的可训练容量。

完整 Action Expert 是现有自建训练入口已经能表达的最大可靠扩展：`mode: expert`、`expert_last_n_layers: 36`。当前证据仓库没有 `/data/lingbot-vla-robotwin` 的 `lrvla` 源码，不能确认“VLM 后 8 层分组解冻”、状态扰动或 seen 指令替换已有真实实现；未确认前不得通过增加无效 YAML 字段假装启用。

## 3. 正式配置

- 两张 A100 80G：[`configs/train_h0_full_expert_2gpu.yaml`](configs/train_h0_full_expert_2gpu.yaml)
- 八卡接力：[`configs/train_h0_full_expert_8gpu.yaml`](configs/train_h0_full_expert_8gpu.yaml)
- absolute robot config：[`configs/robotwin_absolute.yaml`](configs/robotwin_absolute.yaml)

两套配置保持：

```text
global batch = per-rank batch × world size × gradient accumulation = 64
```

安全起点均为每卡 batch 1：

- 2 卡：`1 × 2 × 32 = 64`
- 8 卡：`1 × 8 × 8 = 64`

冒烟确认显存充足后，可以提高每卡 batch，同时等比例降低梯度累积；global batch 必须仍为 64。不要因为换成 8 卡而直接改变优化口径。

## 4. 六天安排

1. 第 0–0.5 天：复制本目录到训练机，固定代码 commit，运行 preflight 和 100-step 冒烟；保存并重载一次 checkpoint。
2. 第 0.5–5.5 天：只运行 H0，不再启动其它配方。正式目标先设 15,000 optimizer steps，即与 E0 相同的 960,000 processed samples。
3. 保存 1k、2.5k、5k、7.5k、10k、12.5k、15k 中可实现的里程碑；现有入口若只能固定周期，使用 `save_every: 1000`。
4. 第 5.5–6 天：导出最终 FP32 HF 模型、动作审计、固定 Clean 小开发集评测。Randomized 只在模型和 checkpoint 冻结之后测试。
5. 若 15k 提前完成且固定 Clean 验证仍改善，保留 15k 后另开 continuation run；不要覆盖 15k。

## 5. 启动前必须通过

在训练工程根目录可用且数据就位后：

```bash
export TRAIN_ROOT=/data/lingbot-vla-robotwin
export H0_NORM_PROVENANCE=/absolute/path/to/clean_norm_stats.provenance.json

python train2/scripts/preflight_h0.py \
  --training-root "$TRAIN_ROOT" \
  --config train2/configs/train_h0_full_expert_2gpu.yaml \
  --norm-provenance "$H0_NORM_PROVENANCE" \
  --world-size 2
```

随后先检查而不启动：

```bash
bash train2/scripts/launch_h0.sh --check 2
```

确认输出和记录目录正确后再启动：

```bash
bash train2/scripts/launch_h0.sh --launch 2
```

八卡接力前，必须先用 2 卡产生的早期 checkpoint 做一次“8 卡加载后继续 20–50 step”的恢复测试。DCP/FSDP 理论能力不能替代本工程实测；当前自建入口历史上使用 DDP，若 8 卡工程改用 FSDP2，需要训练工程提供并验收相应入口，不能直接把 `data_parallel_mode: fsdp2` 写进当前 YAML 后默认生效。

## 6. 100-step 冒烟的通过条件

- 两卡均参与训练，CUDA/NCCL 无错误；
- trainable inventory 包含 Expert 的 `layers.0` 到 `layers.35`，而不只是 34、35；
- `trainable_parameters` 明显大于 E0 的 `101,159,479`；
- VLM/vision 参数保持冻结，除非后续有经过代码审计的新实现；
- 峰值显存不超过每卡约 75 GiB，留出 allocator/保存余量；
- loss、grad norm 全部有限；
- 100 step 后 checkpoint 可在新进程恢复；
- manifest 为 50 个 Clean 任务，norm SHA 与 Clean provenance 相符；
- robot config 中 arm 和 gripper 均为 `subtract_state: false`；
- 模型物理动作仍为完整 `50×14`。

如果每卡 batch 1 仍 OOM，不缩短 chunk、不改变动作语义；先确认 gradient checkpointing 生效，再转 FSDP2。不要悄悄退回最后两层后继续把 run 叫作 H0。

## 7. 选模规则

训练期只用 Clean validation BC loss 和固定 Clean 小开发集选 checkpoint。建议只闭环评测三个候选：早期、中期、最终。最终 checkpoint 冻结后再报告 Clean 和 Randomized；不得用 Randomized 反复选择增强强度、训练步数或 checkpoint。

## 8. 暂不并入本轮的功能

以下方向规则允许，但现有证据仓库没有可审计实现：

- 输入 robot state 小噪声；
- 同任务 `seen_instructions` 池内随机替换；
- Action Expert 与 VLM 后层的参数组学习率；
- FSDP2 下跨 2 卡/8 卡 world-size 恢复。

若训练工程补齐实现，必须先提交代码、单元测试和实际生效日志，再决定是否并入；不要在正式 run 中临时修改。

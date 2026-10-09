# 训练端交付

`train/` 只保留双 A100 阶段仍有效的交付内容：正式 checkpoint 备份工具和已经脱敏、
带 SHA-256 的真机冒烟证据。训练实现、启动脚本和默认 YAML 均以完整训练仓库
`/data/lingbot-vla-robotwin` 为准，不在两个仓库中维护重复副本。

所有正式训练产物必须遵守仓库根目录的
[`Checkpoint与模型归档守则.md`](../Checkpoint与模型归档守则.md)：按实验 ID 和
`global_step` 使用不可变目录，禁止用新模型覆盖旧 checkpoint。

## 当前内容

| 路径 | 用途 |
| --- | --- |
| `backup_checkpoint.py` | 将最新可恢复 `training.pt` 及运行身份文件上传到 HF 私有模型仓库 |
| `backup_checkpoint.sh` | 激活训练环境并调用备份脚本 |
| `evidence/two_a100_2026-10-07/` | global batch 2/32/64 的日志、配置、metadata、SwanLab ID 与哈希清单 |

历史单卡 wrapper、单任务 smoke YAML 和独立 SwanLab patch 已删除。双卡训练代码通过
`server_patch/ddp_two_a100/0001`–`0005` 交付，其中 `0001` 是后续 DDP 补丁的基线依赖，
不能单独跳过。

## 权威训练入口

应用全部补丁后，在完整训练仓库执行：

```bash
cd /data/lingbot-vla-robotwin
export CONDA_ROOT=/home/linux/anaconda3
source scripts/activate.sh

# 已验证默认：2×A100，每卡 batch 32，global batch 64
bash scripts/train_ddp.sh
```

默认配置为完整训练仓库的 `configs/train_expert_ddp.yaml`：

- `batch_size: 32`（每卡）
- `gradient_accumulation: 1`
- `world_size: 2`
- `global_batch_size: 64`
- `max_steps: 15000`（约 96 万 processed samples 的第一候选预算）

恢复训练直接向 DDP 入口传入完整训练状态：

```bash
bash scripts/train_ddp.sh \
  --resume runs/e0_expert_partial_ddp/checkpoints/global_step_N/training.pt
```

同一实验恢复时必须保留原输出目录及其中的 `swanlab_run_id.txt`，并保持 world size、
每卡 batch 和 gradient accumulation 与 checkpoint 一致。

## 正式 checkpoint 备份

本轮 3/10-step 冒烟不上传 HF。正式训练产生有恢复价值的 checkpoint 后执行：

```bash
cd /data/lingbot_challenge
export PROJECT_ROOT=/data/lingbot-vla-robotwin
export CONDA_ROOT=/home/linux/anaconda3
export HF_ENDPOINT=https://huggingface.co

bash train/backup_checkpoint.sh \
  /data/lingbot-vla-robotwin/runs/e0_expert_partial_ddp \
  HumaLog/lingbot-vla-checkpoints \
  experiments/E0_EXPERIMENT_NAME
```

脚本会拒绝向公开仓库上传，只备份最新 `training.pt` 和必要的配置、源码身份、数据身份、
episode split、SwanLab run ID、metrics 与哈希清单。最终候选的完整 `hf_ckpt/` 应另行
上传；不要重复上传官方基座或 RoboTwin 原始数据。

## 证据

`evidence/two_a100_2026-10-07/` 包含：

- `global_b2`：10-step DDP 正确性冒烟；
- `global_b32`：每卡 batch 16 的 3-step 短测；
- `global_b64`：每卡 batch 32 的 3-step 短测，最终默认；
- `SHA256SUMS`：所有证据文件的内容哈希。

详细结论见仓库根目录 `训练端_双A100_DDP冒烟交接.md`。

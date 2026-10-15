# 训练实验与交付

`train/` 保存各轮训练实验的冻结设计、配置、启动门禁、交付工具和脱敏证据。可复用的模型、
数据与分布式训练实现统一维护在 [`training/lingbot-vla-robotwin/`](../training/lingbot-vla-robotwin/)；
实验目录只保留为复现实验所需的薄层，不复制运行时源码。

所有正式训练产物必须遵守仓库根目录的
[`Checkpoint与模型归档守则.md`](../Checkpoint与模型归档守则.md)：按实验 ID 和
`global_step` 使用不可变目录，禁止用新模型覆盖旧 checkpoint。

## 当前内容

| 路径 | 用途 |
| --- | --- |
| [`h0/`](h0/) | H0/G2 完整 Action Expert：设计、冻结配置、启动门禁、测试与最终训练记录 |
| [`evidence/`](evidence/) | E0、E1 和双 A100 smoke 的轻量脱敏证据与统一索引 |
| [`tools/`](tools/) | checkpoint 私有备份等训练交付工具 |

历史单卡 wrapper、单任务 smoke YAML 和独立 SwanLab patch 已删除。双卡训练代码通过
`server_patch/ddp_two_a100/0001`–`0005` 交付，其中 `0001` 是后续 DDP 补丁的基线依赖，
不能单独跳过。

## H0 权威入口

H0 已完成 15k 正式训练并停止加训。复现、检查、恢复和导出流程以
[`h0/README.md`](h0/README.md) 为准，实验依据和最终判定见 [`h0/DESIGN.md`](h0/DESIGN.md)。
最终 FP32 模型归档在 [`HumaLog/H0`](https://huggingface.co/HumaLog/H0)。

```bash
cd /data/lingbot_challenge
python -m unittest discover -s train/h0/tests -v
bash train/h0/scripts/launch_h0.sh --check
```

## 历史 E0 双卡入口

以下内容记录 H0 之前的 E0 双 A100 DDP 交付路径。应用全部补丁后，在完整训练仓库执行：

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

bash train/tools/backup_checkpoint.sh \
  /data/lingbot-vla-robotwin/runs/e0_expert_partial_ddp \
  HumaLog/lingbot-vla-checkpoints \
  experiments/E0_EXPERIMENT_NAME
```

脚本会拒绝向公开仓库上传，只备份最新 `training.pt` 和必要的配置、源码身份、数据身份、
episode split、SwanLab run ID、metrics 与哈希清单。最终候选的完整 `hf_ckpt/` 应另行
上传；不要重复上传官方基座或 RoboTwin 原始数据。

## 证据

全部历史训练证据、规模、完整性和新增实验的入库规则见
[`evidence/README.md`](evidence/README.md)。当前 Git 内证据覆盖 E0、E1 和早期双 A100 smoke；
H0 的设计与最终结论已经归档，独立轻量 evidence 包仍待从训练机收集。

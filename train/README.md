# 训练端

## 状态

当前路径已经在 A100 40GB 上验证：真实 6B 权重加载、partial Action Expert 更新、
checkpoint 保存、断点恢复、完整 HF 权重导出、SwanLab 续接和 HF 私有仓库备份。

训练器目前是**单 GPU 实现**。计划使用的 4 张 A800 到位后，必须先实现并验证
DDP/FSDP，再增加正式多卡配置。不要直接用 `torchrun` 启动当前脚本，否则多个进程
会各自训练并争用同一输出目录。

## 使用

```bash
bash train/setup_train.sh /data/lingbot-vla-robotwin
PROJECT_ROOT=/data/lingbot-vla-robotwin \
  bash train/train.sh train/configs/train_smoke.yaml --allow-subset
PROJECT_ROOT=/data/lingbot-vla-robotwin \
  bash train/train.sh train/configs/train_single_gpu.yaml
```

`train_single_gpu.yaml` 中的 batch 32 已通过 A100 40GB 纯训练计算 profiler，但尚未完成
全量数据端到端短跑；它是正式候选配置，不是未经复核即可长训的最终配置。

断点恢复：

```bash
PROJECT_ROOT=/data/lingbot-vla-robotwin bash train/resume.sh \
  train/configs/train_single_gpu.yaml \
  runs/expert_partial_clean/checkpoints/global_step_100/training.pt
```

同一次实验必须沿用原输出目录，使训练器读取 `swanlab_run_id.txt` 并续接同一条曲线。
新实验必须使用新目录。

## 外部备份

先在 Hugging Face 登录，并保证目标仓库是 private：

```bash
hf auth login
export HF_ENDPOINT=https://huggingface.co
export HTTP_PROXY=http://proxy.mornai.cn:7890
export HTTPS_PROXY=http://proxy.mornai.cn:7890
export HF_HUB_DISABLE_XET=1

PROJECT_ROOT=/data/lingbot-vla-robotwin bash train/backup_checkpoint.sh \
  runs/expert_partial_clean \
  HumaLog/lingbot-vla-checkpoints \
  experiments/EXPERIMENT_NAME
```

脚本仅上传可恢复训练状态和运行元数据，不上传基座、原始数据或每次约 13 GB 的完整
HF 导出。最终候选模型另行上传完整合并权重。token 只保存在机器凭据目录或环境变量中，
不得写入配置和仓库。

## 正式训练前检查

- `python data/verify_clean_data.py` 通过；
- manifest、norm stats、provenance sidecar 的 SHA-256 已锁定；
- 训练与推理端确认使用同一份 norm stats；
- SwanLab 页面能收到短跑指标；
- checkpoint 能在新进程中恢复；
- 多卡实现完成后，先做端到端短跑、保存、恢复和单进程导出测试。

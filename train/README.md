# 训练端

## 这个模块做什么

`train/` 负责把已经准备好的 clean 数据送入 LingBot-VLA 训练器，并覆盖一轮实验从
启动到可恢复交接的完整生命周期：

- 在新机器上建立训练环境并应用已验证补丁；
- 以统一配置启动真实模型训练或短冒烟；
- 将指标持续上传 SwanLab；
- 保存和恢复 optimizer、scheduler 与训练 step；
- 导出评测所需 HF 权重；
- 将可恢复 checkpoint 和运行元数据备份到 HF 私有仓库。

这个模块**不负责**下载/转换训练数据，也不负责 RoboTwin 仿真评测；前者属于
`data/`，后者属于 `eval/`。

## 输入与输出

训练输入：

- LingBot-VLA v2 6B 基座与 Qwen3-VL 资源；
- `data/clean_manifest.json`；
- 与该 manifest 绑定的 `data/norm_stats_clean.json` 和 provenance sidecar；
- 训练 YAML，以及可选的 `training.pt` 恢复点。

主要输出：

| 输出 | 用途 |
| --- | --- |
| `metrics.jsonl` | 本地逐 step 指标 |
| `swanlab_run_id.txt` | 断点续训时续接同一 SwanLab run |
| `checkpoints/global_step_*/training.pt` | 包含优化器等状态的可恢复 checkpoint |
| `checkpoints/global_step_*/hf_ckpt/` | 评测或提交使用的完整 HF 权重 |
| `lingbotvla_cli.yaml` | 推理端配置 |

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `setup_train.sh` | 调用参考工程 bootstrap、应用 SwanLab patch、安装日志依赖并跑合成冒烟 |
| `train.sh` | 单 GPU 训练统一入口，并阻止误用多进程启动 |
| `resume.sh` | 从指定 `training.pt` 恢复 |
| `backup_checkpoint.py/.sh` | 向 HF 私有仓库上传最新可恢复状态及元数据 |
| `configs/train_smoke.yaml` | 单任务真实模型链路测试 |
| `configs/train_single_gpu.yaml` | 单卡正式候选配置，长训前仍需端到端复核 |

## 已验证范围与限制

当前路径已经在 A100 40GB 上验证：真实 6B 权重加载、partial Action Expert 更新、
checkpoint 保存、断点恢复、完整 HF 权重导出、SwanLab 续接和 HF 私有仓库备份。

`setup_train.sh` 会安装已验证的 SwanLab `0.10.1`，但登录凭据不会随仓库迁移。每台新
训练机仍需由使用者执行一次 `swanlab login`，密钥不得写入脚本或 YAML。

训练器目前是**单 GPU 实现**。计划使用的 4 张 A800 到位后，必须先实现并验证
DDP/FSDP，再增加正式多卡配置。不要直接用 `torchrun` 启动当前脚本，否则多个进程
会各自训练并争用同一输出目录。

## 操作流程

以下命令均在本交付仓库根目录执行，通过 `PROJECT_ROOT` 指向实际训练工程：

```bash
bash train/setup_train.sh /data/lingbot-vla-robotwin
swanlab login
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

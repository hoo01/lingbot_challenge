# 训练交付工具

本目录放跨实验复用、但不属于模型训练运行时的交付工具。模型、数据和 FSDP2 实现仍位于
[`training/lingbot-vla-robotwin/`](../../training/lingbot-vla-robotwin/)。

## 私有 checkpoint 备份

[`backup_checkpoint.sh`](backup_checkpoint.sh) 激活 `lingbot-vla` 环境并调用
[`backup_checkpoint.py`](backup_checkpoint.py)，将最新的可恢复 `training.pt` 与必要身份文件上传到
Hugging Face 私有模型仓库。脚本会查询目标仓库可见性并拒绝上传到公开仓库。

```bash
cd /data/lingbot_challenge
export PROJECT_ROOT=/data/lingbot-vla-robotwin
export CONDA_ROOT=/home/linux/anaconda3

bash train/tools/backup_checkpoint.sh \
  /data/lingbot-vla-robotwin/runs/<experiment_id> \
  HumaLog/lingbot-vla-checkpoints \
  experiments/<experiment_id>
```

该工具只负责可恢复状态备份，不替代最终完整 HF 模型导出、逐文件哈希验收或评测机交付。

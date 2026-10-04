# LingBot-VLA × RoboTwin 训练交付

本仓库保存比赛准备期的训练、数据审计和复现材料。训练实现基于参考工程
[`EclipseaHime017/lingbot-vla-robotwin`](https://github.com/EclipseaHime017/lingbot-vla-robotwin)。

## 当前状态

- A100 40GB 上已完成真实 6B 单卡训练、保存、断点恢复和 HF 权重导出。
- SwanLab 初始化、指标上传及同一 run 断点续接已验证。
- Hugging Face 私有仓库备份、回下载和 SHA-256 一致性校验已验证。
- `open_microwave` 的 50 条 clean demonstrations 已完成端到端冒烟。
- 50 任务 × 50 条 clean 数据的归一化统计已完成独立审计。
- 全量数据转换、最终 provenance 和多卡训练尚未执行。

详细记录见 [`杨_进度.md`](./杨_进度.md)，比赛约束和分工见
[`队内任务书_1008.md`](./队内任务书_1008.md)。

## 目录

```text
train/                       训练环境、启动、恢复和备份入口
data/                        clean 数据准备、校验和归一化统计
robotwin_clean_data_audit/   50 任务 clean 数据专项审计
server_patch/                已在云端验证的 SwanLab 接入补丁和完整文件
官方文档/                   赛题规则与常见问题
```

## 推荐流程

```bash
# 在本交付仓库根目录执行，训练工程位于另一目录
bash train/setup_train.sh /data/lingbot-vla-robotwin
bash data/prepare_data.sh /data/lingbot-vla-robotwin
PROJECT_ROOT=/data/lingbot-vla-robotwin \
  bash train/train.sh train/configs/train_single_gpu.yaml
```

正式训练前必须先完成 README 中标注的检查，尤其是：

1. manifest 必须包含 50 个任务、每任务 50 个 clean episodes；
2. norm stats、provenance sidecar 和 manifest 必须互相绑定并记录 SHA-256；
3. 训练与推理必须使用同一份 norm stats；
4. 当前训练器只验证过单 GPU，不能直接用 `torchrun` 冒充四卡训练；
5. token、模型权重、原始数据、checkpoint 和 SwanLab 本地日志不得提交到 Git。

## 最终交接

每个候选 checkpoint 至少要一起交接：训练配置、clean manifest、norm stats、
provenance sidecar、上述文件的 SHA-256、推理配置、SwanLab run ID 和恢复说明。
LoRA 路线如被采用，提交前必须合并回基座并导出全量权重。

# LingBot-VLA × RoboTwin 比赛工程

本仓库是队内的工程交付仓，用来组织 RoboTwin clean 数据准备、LingBot-VLA 后训练、
评测接口以及复现文档。实际模型代码基于参考工程
[`EclipseaHime017/lingbot-vla-robotwin`](https://github.com/EclipseaHime017/lingbot-vla-robotwin)，
两个仓库在训练机上分开放置，本仓库提供脚本、配置、审计结果和交接说明。

## 工程流程

```text
官方 RoboTwin clean 数据
        │
        ▼
data/：下载、转换、划分、归一化统计与完整性校验
        │
        ├── clean_manifest.json
        ├── norm_stats_clean.json
        └── norm_stats_clean.provenance.json
        │
        ▼
train/：环境、训练、日志、断点恢复、导出与异地备份
        │
        ├── training.pt（可恢复状态）
        └── hf_ckpt/（评测/提交权重）
        │
        ▼
eval/：仿真评测与官方复测入口（由评测负责人维护）
```

## 模块分工

| 模块 | 负责内容 | 当前状态 | 详细说明 |
| --- | --- | --- | --- |
| `data/` | clean 数据下载、LeRobot v3.0 转换、manifest、norm stats、provenance 和校验 | 脚本已交付；全量转换等 A800 平台执行 | [`data/README.md`](./data/README.md) |
| `train/` | 环境安装、训练入口、恢复、SwanLab 日志、HF checkpoint 备份 | 单卡链路已验证；4 卡 DDP/FSDP 待实现 | [`train/README.md`](./train/README.md) |
| `eval/` | 仿真环境、子集评测、`eval.sh` 和结果记录 | 由队友夏维护，当前尚未合入 | `eval/README.md`（待添加） |
| `robotwin_clean_data_audit/` | 50×50 clean 数据兼容性、关节角和归一化统计审计 | 已完成 | 目录内报告 |
| `server_patch/` | 已验证但尚未进入上游训练工程的补丁 | SwanLab 补丁已保存 | patch 文件 |

## 两个仓库的关系

建议在训练机上使用以下布局：

```text
/data/lingbot_challenge/        # 本交付仓库
/data/lingbot-vla-robotwin/     # 实际训练工程、模型和数据
```

本仓库不保存基座权重、原始数据、转换后视频、checkpoint、token 或 SwanLab 本地日志。
脚本通过参数或 `PROJECT_ROOT` 操作实际训练工程。

## 从新机器开始

```bash
git clone https://github.com/hoo01/lingbot_challenge.git /data/lingbot_challenge
# 另行取得参考训练工程到 /data/lingbot-vla-robotwin

cd /data/lingbot_challenge
bash train/setup_train.sh /data/lingbot-vla-robotwin
swanlab login

# A800 正式平台：下载并转换完整 50×50 clean 数据
bash data/prepare_data.sh /data/lingbot-vla-robotwin

# 数据验收通过后才能训练
PROJECT_ROOT=/data/lingbot-vla-robotwin \
  bash train/train.sh train/configs/train_single_gpu.yaml
```

当前训练器只验证过单 GPU。上面的单卡配置用于链路验证和候选参数测试；正式 4 卡长训
必须先实现并验证 DDP/FSDP，不能直接用 `torchrun` 启动当前脚本。

## 关键约束

1. 训练数据只能包含 50 个任务、每任务 50 条的 clean demonstrations。
2. manifest、norm stats 和 provenance sidecar 必须绑定并记录 SHA-256。
3. 训练与推理必须使用同一份 norm stats。
4. 正式长训前必须验证 SwanLab、checkpoint 恢复和外部备份。
5. LoRA 路线如被采用，提交前必须合并回基座并导出全量权重。
6. `eval.sh` 和最终提交目录结构仍需结合官方说明完成。

## 文档入口

- 比赛分工与交付清单：[`队内任务书_1008.md`](./队内任务书_1008.md)
- 杨在临时 A100 上完成的冒烟测试及 A800 交接：[`杨_进度.md`](./杨_进度.md)
- 个人研究思路与待验证方向：`自查.md`（不属于工程 README）
- 官方规则整理：`官方文档/`

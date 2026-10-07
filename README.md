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
| `data/` | clean 数据下载、LeRobot v3.0 转换、manifest、norm stats、provenance 和校验 | 双 A100 机器已完成全量 50×50 数据及 SHA 验收 | [`data/README.md`](./data/README.md) |
| `train/` | 环境安装、训练入口、恢复、SwanLab 日志、HF checkpoint 备份 | 单卡全链路与双 A100 DDP 真实数据冒烟均已验证 | [`train/README.md`](./train/README.md) |
| `eval/` | 仿真环境、子集评测、`eval.sh` 和结果记录 | 已合入；当前 52/500（10.4%）仅为内部固定开发协议，不是官方成绩 | [`eval/README.md`](./eval/README.md) |
| `robotwin_clean_data_audit/` | 50×50 clean 数据兼容性、关节角和归一化统计审计 | 已完成 | 目录内报告 |
| `server_patch/` | 已验证或待目标机器验证、但尚未进入上游训练工程的补丁 | 单卡基线及双 A100 DDP 补丁已保存 | [`server_patch/ddp_two_a100/README.md`](./server_patch/ddp_two_a100/README.md) |

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
git clone https://github.com/EclipseaHime017/lingbot-vla-robotwin.git /data/lingbot-vla-robotwin

# 在完整训练工程上依次应用已验证单卡基线和双卡 DDP 补丁
cd /data/lingbot-vla-robotwin
git checkout 02ba532
git am /data/lingbot_challenge/server_patch/ddp_two_a100/0001-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0002-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0003-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0004-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0005-*.patch

cd /data/lingbot_challenge
bash train/setup_train.sh /data/lingbot-vla-robotwin
swanlab login

# 正式双 A100 平台：下载并转换完整 50×50 clean 数据
bash data/prepare_data.sh /data/lingbot-vla-robotwin

# 数据验收后先做 10 step 双卡冒烟，不直接启动长训
cd /data/lingbot-vla-robotwin
bash scripts/train_ddp.sh \
  --max-steps 10 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --no-export \
  --output-dir artifacts/ddp_smoke
```

当前训练器已经在单张 A100 40GB 上完成真实数据训练、恢复、SwanLab 和 HF 异地备份
验证，并在 2×A100-SXM4-40GB 上用完整 50-task clean 数据完成 10-step DDP 冒烟。
本阶段目标是验证单卡代码到双卡 DDP 的可移植性，不包含正式长训。详细结果见
[`训练端_双A100_DDP冒烟交接.md`](./训练端_双A100_DDP冒烟交接.md)。DDP 不会合并
两张卡的显存，后续扩大解冻范围若单卡放不下，仍需 FSDP/ZeRO。

## 关键约束

1. 训练数据只能包含 50 个任务、每任务 50 条的 clean demonstrations。
2. manifest、norm stats 和 provenance sidecar 必须绑定并记录 SHA-256。
3. 训练与推理必须使用同一份 norm stats。
4. 正式长训前必须验证 SwanLab、checkpoint 恢复和外部备份。
5. LoRA 路线如被采用，提交前必须合并回基座并导出全量权重。
6. 正式实验按处理样本数对齐，而不是机械固定 optimizer step；例如 global batch 32、
   30,000 step 约处理 96 万样本。
7. `eval.sh` 的内部开发集结果不能当作官方比赛成绩，最终提交目录仍需按官方说明验收。

## 文档入口

- 比赛分工与交付清单：[`队内任务书_1008.md`](./队内任务书_1008.md)
- 正式训练路线与实验矩阵：[`实验设计文档.md`](./实验设计文档.md)
- 临时 A100 冒烟测试及训练端交接：[`训练端_A100冒烟交接.md`](./训练端_A100冒烟交接.md)
- 双 A100 全量数据与 DDP 真机冒烟交接：[`训练端_双A100_DDP冒烟交接.md`](./训练端_双A100_DDP冒烟交接.md)
- 双 A100 代码补丁与真机验收：[`server_patch/ddp_two_a100/README.md`](./server_patch/ddp_two_a100/README.md)
- 个人研究思路与待验证方向：`自查.md`（不属于工程 README）
- 官方规则整理：`官方文档/`

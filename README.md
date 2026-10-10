# LingBot-VLA × RoboTwin 比赛工程

本仓库是队内的工程交付仓，用来组织 RoboTwin clean 数据准备、LingBot-VLA 后训练、
评测接口以及复现文档。实际模型代码基于参考工程
[`EclipseaHime017/lingbot-vla-robotwin`](https://github.com/EclipseaHime017/lingbot-vla-robotwin)，
两个仓库在训练机上分开放置，本仓库提供脚本、配置、审计结果和交接说明。

## 最新评测与交付（2026-10-10）

- [原始基座与 E0 同种子对照](eval/results/baseline-e0-same-seed-20261010.md)：500个相同 task/setting/seed，基座49/500（9.8%）、E0 54/500（10.8%）。基座含4个900秒超时0分，双方输入与协议差异在报告中单列。
- [E0 15000步完整报告](eval/results/e0-15000-final.md)：500个完整策略回合、54成功；这是分阶段开发评测，不是统一赛事成绩。实际评测模型为BF16保存、FP32执行版本，不等同于训练者保留FP32原值的导出。
- [评测结果索引](eval/results/README.md)：成绩、成功seed、失败案例、精度审计和轻量证据。
- [评测端GitHub交付清单](eval/交付与提交清单.md)：按队内任务书A8/B6列出脚本、文档、结果和验证材料，区分已完成项与缺口。权重按仓库归档守则存放。F1尚未训练完成，本仓库没有F1评测结果。

## 工程流程

```text
官方 RoboTwin clean 数据
        │
        ▼
data/：下载、转换、划分、归一化统计与完整性校验
        │
        ├── clean_manifest.json
        ├── clean_norm_stats.json
        └── clean_norm_stats.provenance.json
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
| `eval/` | 仿真环境、评测入口、结果与数值链路验收 | 基座/E0同seed开发对照完成；空机重建、子集/推理对比和数值验收缺口见交付清单 | [`eval/README.md`](./eval/README.md) |
| `robotwin_clean_data_audit/` | 历史数据审计证据：兼容性、关节角与归一化统计对比 | 已完成；非正式训练输入 | 目录内报告 |
| `server_patch/` | 已验证或待目标机器验证、但尚未进入上游训练工程的补丁 | 单卡基线及双 A100 DDP 补丁已保存 | [`server_patch/ddp_two_a100/README.md`](./server_patch/ddp_two_a100/README.md) |

> `robotwin_clean_data_audit/` 仅用于追溯早期审计结论，不得将其中的
> `data/analysis/norm_stats_clean.json` 作为正式训练输入。正式训练以
> `/data/lingbot-vla-robotwin/data/clean_manifest.json`、`clean_norm_stats.json`
> 及与之匹配的 provenance 校验结果为准。

## 两个仓库的关系

建议在训练机上使用以下布局：

```text
/data/lingbot_challenge/        # 本交付仓库
/data/lingbot-vla-robotwin/     # 实际训练工程、模型和数据
```

本仓库不保存基座权重、原始数据、转换后视频、checkpoint、token 或 SwanLab 本地日志。
脚本通过参数或 `PROJECT_ROOT` 操作实际训练工程。

归档原则：可复现代码、配置、文档和脱敏短测证据进 GitHub；正式训练的可恢复状态与
候选权重进 HF 私有模型仓库；官方公开基座和 RoboTwin 原始数据只记录固定 revision，
不重复上传。转换数据仅在许可证允许且确有跨机器复用需求时进入私有 Dataset 仓库或
平台快照。详细矩阵见 `训练端_双A100_DDP冒烟交接.md`。

### 当前归档状态（2026-10-07）

- 双 A100 的 global batch 2、32、64 三组真实数据短测日志、配置、运行元数据、
  episode split、SwanLab run ID 和 SHA-256 清单已经归档到
  [`train/evidence/two_a100_2026-10-07/`](./train/evidence/two_a100_2026-10-07/)。
- 最终默认值为每卡 batch 32、两卡 global batch 64；补丁链为 `0001`–`0005`。
- 三组短测均为 3/10-step smoke，使用 `--no-export`；即使本地存在短测
  `training.pt`，也不作为可恢复候选上传 Hugging Face。
- 本阶段 **Hugging Face 无新增上传项**。正式训练开始后，才将重要
  `training.pt`、候选 `hf_ckpt/` 及其配置/数据身份/哈希清单上传到私有仓库
  `HumaLog/lingbot-vla-checkpoints`。
- 3.9 GB LeRobot 转换数据优先随平台数据盘或快照保存；在未核实再分发许可前，
  不上传公开 HF Dataset。基座、官方 raw archives 与解压 raw 均不重复上传。

## 从新机器开始

```bash
git clone https://github.com/hoo01/lingbot_challenge.git /data/lingbot_challenge
git clone https://github.com/EclipseaHime017/lingbot-vla-robotwin.git /data/lingbot-vla-robotwin

# 在完整训练工程上依次应用双卡补丁链（0001 是后续 DDP 补丁的基线依赖）
cd /data/lingbot-vla-robotwin
git checkout 02ba532
git am /data/lingbot_challenge/server_patch/ddp_two_a100/0001-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0002-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0003-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0004-*.patch \
       /data/lingbot_challenge/server_patch/ddp_two_a100/0005-*.patch

export CONDA_ROOT=/home/linux/anaconda3
bash scripts/bootstrap.sh
source scripts/activate.sh
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

- 比赛分工与阶段交付清单：[`队内任务书_10.7前第一阶段.md`](./队内任务书_10.7前第一阶段.md)
- 当前评测端GitHub交付清单：[`eval/交付与提交清单.md`](./eval/交付与提交清单.md)
- 正式训练路线与实验矩阵：[`实验设计文档.md`](./实验设计文档.md)
- 双 A100 全量数据与 DDP 真机冒烟交接：[`训练端_双A100_DDP冒烟交接.md`](./训练端_双A100_DDP冒烟交接.md)
- 双 A100 代码补丁与真机验收：[`server_patch/ddp_two_a100/README.md`](./server_patch/ddp_two_a100/README.md)
- 个人研究思路与待验证方向：`自查.md`（不属于工程 README）
- 官方规则整理：`官方文档/`

# 数据端

## 这个模块做什么

`data/` 负责把官方 RoboTwin 2.0 clean demonstrations 变成训练器可以安全读取、可以
追溯的数据输入。它不仅做格式转换，还负责阻止以下静默错误：混入 randomized 数据、
任务或 episode 缺失、LeRobot 版本不兼容、训练/推理归一化统计不一致，以及 provenance
和实际文件不匹配。

处理流程：

```text
Hugging Face 官方 clean 原始包
        ↓ 下载和校验
原始 episode 数据
        ↓ prepare_data.py
LeRobot v3.0 数据集 + 训练/验证划分
        ↓ 全量统计
manifest + norm stats + provenance sidecar
        ↓ verify_clean_data.py
通过验收后交给 train/
```

## 输入与输出

输入是官方 `TianxingChen/RoboTwin2.0` 中 Aloha-AgileX 的 50×50 clean demonstrations。

| 输出 | 作用 | 是否进入 Git |
| --- | --- | --- |
| `data/lerobot/` | 转换后的图像、状态、动作和指令 | 否，体积大 |
| `clean_manifest.json` | 任务、episode、路径及 clean-only 状态 | 正式版本需要交接 |
| `norm_stats_clean.json` | 动作/状态归一化与反归一化 | 是 |
| `norm_stats_clean.provenance.json` | 绑定 manifest、验证划分和 norm SHA | 正式生成后交接 |
| `SHA256SUMS` | 锁定交接文件版本 | 是 |

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `prepare_data.sh` | 在实际训练工程中下载并转换完整 clean 数据，同时生成统计与 provenance |
| `verify_clean_data.py` | fail-closed 验收任务数、episode 数、格式、clean-only 和 SHA 绑定 |
| `norm_stats_clean.json` | 基于 50×50 HF clean 包得到的独立审计统计副本 |
| `SHA256SUMS` | 当前审计统计副本的哈希；正式重算后必须更新 |

## 数据硬约束

- 只允许 RoboTwin 2.0 Aloha-AgileX clean demonstrations；禁止混入 randomized 数据。
- 完整规格是 50 tasks × 50 episodes。
- 使用 LeRobot v3.0；训练环境锁定 `lerobot==0.4.2`。
- 训练和推理必须使用同一份 clean norm stats。
- norm stats 必须和最终 manifest、验证集划分及 provenance sidecar 一起锁定。

## 操作流程

在本交付仓库根目录运行；脚本通过参数操作实际训练工程：

```bash
bash data/prepare_data.sh /data/lingbot-vla-robotwin
```

脚本不带任务过滤器，因此面向完整 clean 数据。正式启动长训前必须再运行：

```bash
python data/verify_clean_data.py \
  --manifest /data/lingbot-vla-robotwin/data/clean_manifest.json \
  --norm-stats /data/lingbot-vla-robotwin/data/norm_stats_clean.json
```

校验器采用 fail-closed 策略：任务数、episode 数、clean-only、LeRobot 版本、provenance
或 norm SHA 任一不符合都会退出非零。

## 历史审计副本：`norm_stats_clean.json`

仓库中的统计量来自 Hugging Face `RoboTwin_lerobot_v30.zip` 的全部 2500 条 clean
episodes，共 548,893 帧；计算口径和详细对比见
`robotwin_clean_data_audit/data/analysis/lerobot_compat_and_norm_report.md`。

该文件仅作为独立历史审计参考，**不是当前正式训练输入**。正式训练应由 `prepare_data.py` 针对最终 manifest 和
验证集划分重新生成 norm stats 及 `.provenance.json`，然后运行校验器并记录 SHA-256。
不要手工伪造 provenance，也不要回退到基于 randomized 数据的官方 `robotwin.json`。
双 A100 正式流程当前使用 `/data/lingbot-vla-robotwin/data/clean_norm_stats.json`
及与之匹配的 provenance。

当前审计副本的 SHA-256 记录在 `SHA256SUMS`。正式重新生成后必须同步更新该清单。

原始 zip、转换后视频、parquet 和缓存不进入 Git。

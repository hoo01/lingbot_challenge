# 训练证据索引

本目录只保存可公开进入 Git 的轻量、脱敏证据，用于回答“这次训练使用了什么、是否完整结束、
产物如何验收”。模型权重、可恢复 checkpoint、原始数据、token 和 SwanLab 本地缓存不进入这里。

## 已归档证据

| 路径 | 实验与范围 | 规模 | 完整性 |
| --- | --- | ---: | --- |
| [`two_a100_2026-10-07/`](two_a100_2026-10-07/) | 早期 2×A100 DDP；global batch 2/32/64 的 3/10-step smoke | 0.13 MiB | 配置、日志、metadata、SwanLab ID 和总 SHA 清单 |
| [`e0_2026-10-08/`](e0_2026-10-08/) | E0；absolute action、最后两层 Action Expert，15k 正式训练 | 4.07 MiB | 25 项分发证据 SHA-256 已核对；不含权重 |
| [`e1_shuweilin_20261011_080143/`](e1_shuweilin_20261011_080143/) | E1；relative arm、absolute gripper，15k 正式训练 | 7.70 MiB | 25 项分发证据 SHA-256 已核对；不含权重 |

E0 的完整交接见 [`训练端_E0正式训练交接.md`](../../训练端_E0正式训练交接.md)，E1 见
[`训练端_E1正式训练交接.md`](../../训练端_E1正式训练交接.md)。目录名保留原始实验 ID 或归档日期；
已有目录被评测报告和哈希记录引用，不为统一外观而重命名。

## H0 状态

H0-A 15k 已完成，设计、实测摘要、停止决策和复现入口位于 [`train/h0/`](../h0/)，最终模型位于
[`HumaLog/H0`](https://huggingface.co/HumaLog/H0)。目前尚未从训练机归档独立的 H0 轻量 evidence 包；
在拿到原始文件前只登记该缺口，不根据聊天记录重建或伪造收据。

建议后续补入：`preflight_receipt.json`、最终 `metrics_summary.json`、`action-audit.json`、
`export.json`、`exit_code`、源码/配置身份和覆盖这些文件的 SHA-256 清单。完整 metrics 或日志只有在
确实承担曲线复核、恢复诊断或错误审计作用时才进入 Git。

## 新实验入库规则

1. 每个正式实验使用一个不可变目录，并提供 `README.md` 说明实验 ID、训练范围和证据边界。
2. 至少记录配置、源码 commit、数据/Norm 身份、指标摘要、退出状态和 checkpoint/export 清单。
3. 原始文件如经过路径脱敏，分别记录 source SHA-256 与 distributed SHA-256。
4. 不上传模型权重、`training.pt`、原始数据、凭据或含隐私的机器配置。
5. `SHA256SUMS*` 必须覆盖实际提交的字节；修改受清单保护的文件时同步更新清单。

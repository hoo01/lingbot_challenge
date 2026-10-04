# 数据端

## 硬约束

- 只允许 RoboTwin 2.0 Aloha-AgileX clean demonstrations；禁止混入 randomized 数据。
- 完整规格是 50 tasks × 50 episodes。
- 使用 LeRobot v3.0；训练环境锁定 `lerobot==0.4.2`。
- 训练和推理必须使用同一份 clean norm stats。
- norm stats 必须和最终 manifest、验证集划分及 provenance sidecar 一起锁定。

## 准备

在 `lingbot-vla-robotwin` 工程根目录运行：

```bash
bash data/prepare_data.sh /data/lingbot-vla-robotwin
```

脚本不带任务过滤器，因此面向完整 clean 数据。正式启动长训前必须再运行：

```bash
python data/verify_clean_data.py
```

校验器采用 fail-closed 策略：任务数、episode 数、clean-only、LeRobot 版本、provenance
或 norm SHA 任一不符合都会退出非零。

## `norm_stats_clean.json`

仓库中的统计量来自 Hugging Face `RoboTwin_lerobot_v30.zip` 的全部 2500 条 clean
episodes，共 548,893 帧；计算口径和详细对比见
`robotwin_clean_data_audit/data/analysis/lerobot_compat_and_norm_report.md`。

该文件可作为独立审计参考。正式训练仍应由 `prepare_data.py` 针对最终 manifest 和
验证集划分重新生成 norm stats 及 `.provenance.json`，然后运行校验器并记录 SHA-256。
不要手工伪造 provenance，也不要回退到基于 randomized 数据的官方 `robotwin.json`。

当前审计副本的 SHA-256 记录在 `SHA256SUMS`。正式重新生成后必须同步更新该清单。

原始 zip、转换后视频、parquet 和缓存不进入 Git。

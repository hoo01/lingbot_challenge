# E0 2026-10-08 脱敏证据

本目录对应仓库根目录的 [训练端_E0正式训练交接.md](../../../训练端_E0正式训练交接.md)。证据来自用户提供的 `e0_handoff_evidence.zip`，辅以本轮已提供的导出/审计脚本副本。包内不含训练权重或原始数据。

## 放入仓库

将交接 ZIP 的内容解压到团队仓库根目录，保留根目录 Markdown 和本目录结构。如果已有同名文件，先比较内容再合并。

仅暂存指定交接内容：

```bash
git add -- "训练端_E0正式训练交接.md" train/evidence/e0_2026-10-08/
git diff --cached --stat
```

本包没有替你执行 Git 提交或推送，也没有启动训练、注册模型或排队评测。

## 脱敏与哈希

训练记录中的绝对训练根路径统一替换为 `${TRAIN_ROOT}`。保留训练 step、指标、run ID、模型张量名、任务名、哈希、GPU 型号和进程 ID，以便核对过程。辅助脚本保持本轮提供版本的字节，默认目录不是登录凭据。

`redaction-report.json` 对每份原始证据记录 source_sha256、distributed_sha256 和修改类型。`metrics.jsonl`、`assets/norm_stats.json` 等无路径替换的文件保持原始字节。配置和 manifest 的原始文件 SHA 用于确认训练来源，不应拿来校验已经脱敏的副本。

在本目录检查分发证据：

```bash
sha256sum -c SHA256SUMS.evidence
```

`SHA256SUMS.evidence` 不包含自身。模型权重哈希来自 A100 审计记录，本包不含模型字节，整理过程中未重新计算模型分片哈希。

## 完整模型验收

每个 `checkpoints/global_step_XXXXX/hf_ckpt_fp32/SHA256SUMS.hf_ckpt_fp32` 包含 50 个 safetensors 分片、config 和索引文件的 SHA。将对应清单复制到待验收的完整模型目录后，在该目录运行：

```bash
sha256sum -c SHA256SUMS.hf_ckpt_fp32
```

该检查覆盖清单中的模型文件，不包含 tokenizer/processor 辅助文件的完整验收。完整模型仍应整体传输，归一化单独按交接文档的 SHA 核对。目录不存在或尚未上传时，不应将其写成验收通过。

## 结论范围

三份 `action-audit.json` 均为 local FP32 SDPA / reference MoE 后端的独立进程重载比较。两条测试输入的物理动作和归一化输出均完全一致。报告同时明确官方评测后端未测试，未确认 native 或 bf16_then_fp32 构建方式，不可据此填报任务成功率。

导出回执中的 `action_consistency_audit=not_run` 记录较早时间点，后续动作结果见同目录审计报告。SwanLab 实际 run ID 见 `swanlab_run_id.txt`，不是初始化配置中的 null。

本包缺少完整 stdout/stderr、恢复事件日志、HF 远端上传回执和仿真评测报告。对应状态在主文档中已注明，未据此推断训练从未重启、全部权重已远端归档或正式评测已完成。

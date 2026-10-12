# E1 正式训练脱敏证据

本目录对应仓库根目录的 [训练端_E1正式训练交接.md](../../../训练端_E1正式训练交接.md)，来源为 2A100 上的正式运行 `runs/e1_shuweilin_20261011_080143/`。这里只存放小型训练记录、配置、指标和校验清单；**不含** `training.pt`、safetensors、原始数据或观测/动作 fixture。

## 入库范围

只暂存交接文档和本目录，不要顺手加入其他未跟踪文件：

```bash
git add -- "训练端_E1正式训练交接.md" train/evidence/e1_shuweilin_20261011_080143/
git diff --cached --stat
```

本目录未代替 Hugging Face 私有归档、评测机 `incoming/` 交付或正式评测。

## 脱敏与身份

记录文件中的 `/data/lingbot-vla-robotwin` 和 `/home/linux/anaconda3` 分别替换为 `${TRAIN_ROOT}`、`${CONDA_ROOT}`。`scripts/launch_e1_screen.sh` 保留训练机原始字节，其默认路径是代码行为记录，不含凭据。`redaction-report.json` 同时记录各原始文件与本地分发副本的 SHA-256；不能把脱敏文件的 SHA 与训练机原始 SHA 混用。运行指标、norm 数值、实验 ID、代码 commit 和训练 step 均未改写。

核对本地分发包：

```bash
cd train/evidence/e1_shuweilin_20261011_080143
shasum -a 256 -c SHA256SUMS.evidence
```

训练机上的最终导出目录名为 `hf_ckpt/`，归档约定中称为 `hf_ckpt_fp32/`；其 `export.json` 记录 `torch.float32`。`checkpoints/global_step_15000/hf_ckpt_fp32/SHA256SUMS.hf_ckpt_fp32` 是训练机原始完整模型的逐文件哈希清单，**不是**模型本体。接收完整模型后，在其目录中用 `sha256sum -c SHA256SUMS.hf_ckpt_fp32` 校验；本地 Mac 上未复制权重，不能在本证据目录运行该命令并声称模型通过。

## 证据边界

- `metrics_summary.json` 从原始 `metrics.jsonl` 重新计算；连续 step 和启动器 `exit_code=0` 不能证明从未重启。
- 根目录的 `eval-protocol.json` 是训练前协议原件的同字节副本，只冻结 E0/E1 配对评测的计划口径，不证明评测实现或模型注册已通过。训练机上的历史原路径仍为 `runs/e1_preflight/eval-protocol.json`。
- 当前未收到 E1 导出前后/独立新进程动作审计、HF 固定 revision 与远端逐文件校验回执、评测机接收或闭环成功率报告。这些字段在主交接文档中保持待补。
- 脱敏证据中的 YAML/JSON 不应直接代替训练机原始文件部署；评测端必须核对原始文件 SHA、E1 有效 robot config 和专属 norm。

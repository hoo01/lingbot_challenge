# Checkpoint 与模型归档守则

**适用范围**：LingBot-VLA 2.0 × RoboTwin 2.0 的所有正式训练、短训候选、恢复检查点和评测模型

**存储位置**：代码与脱敏证据进入 GitHub；模型产物进入团队 Hugging Face 私有仓库

**核心原则**：模型产物按“实验 ID + global step”不可变归档，禁止用新实验或新 step 覆盖旧产物。

## 1. 强制规则

1. 每次训练启动前必须确定唯一 `experiment_id`，开始训练后不得复用给另一组实验。
2. 同一实验的不同 `global_step` 必须使用不同目录；不得维护一个被反复覆盖的 `training.pt` 或 `hf_ckpt/`。
3. 不同实验组必须使用不同目录，即使模型结构和文件名相同也不得覆盖。
4. `training.pt` 与完整 HF 导出模型必须分开标识：前者用于续训，后者用于评测。
5. 模型文件上传到既有路径时，只允许 SHA-256 完全相同的幂等重传；内容不同必须创建新 step、实验目录或修订目录。
6. `main`、`latest` 和网页当前文件列表都不是可复现版本。交接必须记录固定 commit revision、仓库内路径和 SHA-256。
7. 未完成远端文件列表和 SHA 校验前，不得在交接文档中写“模型已归档”或“可直接评测”。
8. 禁止强推、清空历史或直接删除疑似被覆盖的旧版本。发现错误时先冻结上传，再按第 8 节恢复。

## 2. 两类模型产物

| 产物 | 典型目录 | 用途 | 能否直接评测 |
| --- | --- | --- | --- |
| 可恢复训练状态 | `checkpoints/global_step_N/training.pt` | 恢复可训练参数、optimizer、scheduler、step 和 RNG | 否；通常还依赖原始基座和冻结配置 |
| 完整评测模型 | `checkpoints/global_step_N/hf_ckpt_fp32/` | 独立加载、注册评测、模型交付 | 是；必须包含完整 safetensors、索引和配置 |

完整模型如果包含 LoRA，上传前必须完成合并；不得把 adapter-only 权重当成完整评测模型。

## 3. Hugging Face 标准目录

所有人统一使用同一个私有仓库时，目录必须如下：

```text
HumaLog/lingbot-vla-checkpoints
└── experiments/
    └── <experiment_id>/
        ├── training_config.yaml
        ├── run_metadata.json
        ├── episode_splits.json
        ├── metrics.jsonl
        ├── swanlab_run_id.txt
        ├── latest.json
        └── checkpoints/
            ├── global_step_5000/
            │   ├── training.pt
            │   ├── training.pt.sha256
            │   └── hf_ckpt_fp32/
            ├── global_step_10000/
            │   ├── training.pt
            │   ├── training.pt.sha256
            │   └── hf_ckpt_fp32/
            └── global_step_15000/
                ├── training.pt
                ├── training.pt.sha256
                └── hf_ckpt_fp32/
```

`latest.json` 只能保存最新 step、路径和 revision 的指针，不得存放唯一一份模型，也不得代替历史目录。

### 实验 ID 命名

```text
<group>_<owner>_<YYYYMMDD>_<HHMMSS>
```

示例：

```text
e0_jiali_20261008_004327
f1_name_20261009_093000
f_low_name_20261010_101500
g1_name_20261011_080000
```

同一实验恢复训练时继续使用原 `experiment_id` 和新的 `global_step`；改变数据、norm、动作表示、损失、增广组合或可训练参数范围时，必须创建新实验 ID。

## 4. 保存与保留策略

训练程序可以按较短间隔在本机保存，但 HF 不要求上传每一个 checkpoint。正式实验至少保留：

- 预先登记的阶段点，例如 5k、10k、15k；
- 用于断点恢复的最新有效 `training.pt`；
- 进入开发集比较的每个候选完整模型；
- 最终选中模型及其前一个稳定 checkpoint；
- 出现异常前后的诊断 checkpoint（如确有分析价值）。

比赛结束或技术报告冻结前，不得删除已经参与比较、产生报告或被他人引用的 checkpoint。确需清理时，由实验负责人列出路径、revision、SHA 和删除理由，经队内确认后执行。

## 5. 上传流程

### 5.1 上传前

- 确认目标仓库为私有仓库；
- 确认 `experiment_id`、`global_step` 和目标路径；
- 检查目标路径不存在，或远端同名文件 SHA 与本地完全一致；
- 计算 `training.pt` 或所有 safetensors、索引、配置、norm 的 SHA-256；
- 确认配置中不含 token、密码和不应公开的绝对路径；
- 完整模型先通过独立进程重载和动作一致性审计。

可恢复检查点使用仓库脚本，并把唯一实验目录作为第三个参数：

```bash
bash train/backup_checkpoint.sh \
  /data/lingbot-vla-robotwin/runs/<experiment_id> \
  HumaLog/lingbot-vla-checkpoints \
  experiments/<experiment_id>
```

脚本会按运行目录中的 `checkpoints/global_step_N/training.pt` 上传最新恢复点。后续 step 仍使用同一个实验目录，不能改成仓库根目录。

### 5.2 上传后

上传人必须生成或补充 `hf_upload_receipt.json`，至少记录：

```json
{
  "repo_id": "HumaLog/lingbot-vla-checkpoints",
  "path_in_repo": "experiments/<experiment_id>/checkpoints/global_step_N",
  "revision": "完整的 Hugging Face commit hash",
  "experiment_id": "<experiment_id>",
  "global_step": 15000,
  "artifact_kind": "training_state 或 full_merged_model",
  "uploaded_at": "ISO-8601 时间",
  "sha256_manifest": "SHA256SUMS 文件路径"
}
```

随后完成：

1. 在固定 revision 下列出远端文件，确认数量、路径和大小；
2. 按固定 revision 下载校验，至少完整核对 `training.pt`；完整模型应核对全部分片清单；
3. 为正式候选创建易读 tag，例如 `<experiment_id>-step-15000`；
4. 将 revision、tag、路径和 SHA 回填 GitHub 交接文档；
5. 由另一名队员复核后，才将归档状态标记为 `verified`。

## 6. 允许更新与禁止覆盖

同一实验运行期间，以下运行级文件可以在原路径更新：

- 持续增长的 `metrics.jsonl`；
- 指向最新不可变 checkpoint 的 `latest.json`；
- 不改变训练身份的状态说明。

以下内容一旦上传并被引用，视为不可变：

- `training.pt`；
- safetensors 分片、索引和 `config.json`；
- norm stats 及 provenance；
- 已冻结的训练配置、episode split 和动作审计结果；
- SHA 清单和上传回执。

需要纠正不可变文件时，新建 `revision_2/` 或新的实验 ID，并在说明中记录旧版本问题；不得静默替换。

## 7. GitHub 与 Hugging Face 的边界

GitHub 可以保存：

- 训练与导出脚本；
- YAML/JSON 配置和脱敏 metadata；
- loss 汇总、动作审计、文件清单和 SHA；
- HF 仓库路径、固定 revision、tag 和上传回执。

GitHub 禁止保存：

- `training.pt`、safetensors 和其他大权重；
- Hugging Face token、SSH 私钥、密码；
- 原始 RoboTwin 数据、转换视频和未确认许可的素材；
- SwanLab 私密凭据及未脱敏的机器信息。

## 8. 发生覆盖时的处理

1. 立即停止对该 HF 仓库继续上传、删除或强推；
2. 在 Hugging Face commit 历史中定位覆盖前和覆盖后的 revision；
3. 给两个 revision 分别创建 tag，防止后续人员继续只引用 `main`；
4. 从旧 revision 将旧模型恢复或复制到正确的实验/step 目录；
5. 对恢复后的文件重新计算并核对 SHA；
6. 为当前模型建立独立目录，不再复用旧路径；
7. 在交接文档记录事故、影响范围、恢复 revision 和验证结果。

旧文件在 commit 历史中仍可访问，不代表归档已经合格；只有独立路径、固定 revision 和 SHA 都补齐后，才算完成修复。

## 9. 训练到评测的最小交接信息

每个送评 checkpoint 必须同时交付：

- 实验 ID、实验组和 global step；
- HF repo、仓库内路径、固定 revision 和 tag；
- checkpoint 类型（恢复状态或完整模型）；
- 权重、配置、norm、manifest 的 SHA-256；
- 训练代码 commit、基座 revision 和实际训练配置；
- SwanLab run ID、processed samples 和 global batch；
- 导出/重载动作审计结果；
- 评测端登记后的 `MODEL_ID` 和评测 run ID。

评测端只接收完整、已合并且通过哈希校验的模型，不从 `main` 或 `latest` 猜测应加载哪个版本。

## 10. 提交前检查表

- [ ] 实验 ID 唯一，未被其他实验使用；
- [ ] 不同 global step 位于不同目录；
- [ ] `training.pt` 与完整评测模型标识清楚；
- [ ] 没有上传到仓库根目录覆盖同名模型；
- [ ] 目标路径不存在，或幂等上传前后 SHA 相同；
- [ ] 已记录固定 revision，而不是只写 `main`；
- [ ] 已生成并核对 SHA-256 清单；
- [ ] 已补充上传回执和交接文档；
- [ ] 无 token、密码、私钥或未脱敏敏感信息；
- [ ] 另一名队员已完成复核。

## 参考

- [Hugging Face：上传文件与 `path_in_repo`](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Hugging Face：仓库 branch、tag 与 revision](https://huggingface.co/docs/huggingface_hub/en/guides/repository)

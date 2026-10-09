# Checkpoint 与模型归档守则

适用于所有 E/F/G 组训练。代码和脱敏记录进入 GitHub，权重进入团队 Hugging Face 私有仓库。

## 1. 一条核心规则

**每个实验、每个 global step 使用独立目录，禁止用新模型覆盖旧 checkpoint。**

## 2. 统一目录

```text
experiments/
  <experiment_id>/
    training_config.yaml
    run_metadata.json
    checkpoints/
      global_step_5000/
        training.pt
        hf_ckpt_fp32/
      global_step_10000/
        training.pt
        hf_ckpt_fp32/
      global_step_15000/
        training.pt
        hf_ckpt_fp32/
```

实验 ID 建议使用：

```text
<实验组>_<负责人>_<日期时间>
```

例如：`e0_jiali_20261008_004327`、`f1_name_20261009_093000`。

- 同一实验恢复训练：继续使用原实验 ID，保存到新的 `global_step_N`。
- 改变增广、动作表示、loss、norm 或可训练范围：新建实验 ID。
- `latest` 只能作为指针，不能代替历史 checkpoint。

## 3. 两类文件不要混淆

| 文件 | 用途 |
| --- | --- |
| `training.pt` | 断点续训，通常还需要原始基座、optimizer、scheduler 和配置 |
| `hf_ckpt_fp32/` | 完整、已合并的模型，用于独立加载和评测 |

LoRA 必须先合并再作为完整评测模型上传，不能只交 adapter。

## 4. 上传后必须记录

每个正式 checkpoint 至少记录：

- Hugging Face 仓库内路径；
- 固定 commit revision，不只写 `main`；
- 实验 ID 和 global step；
- `training.pt` 或完整模型的 SHA-256；
- 对应训练配置、norm SHA 和训练代码 commit。

上传完成后按固定 revision 核对远端文件和 SHA，确认无误才能写“已归档”。

## 5. 禁止事项

- 不把模型直接反复上传到仓库根目录；
- 不覆盖其他实验或其他 step 的同名权重；
- 不把 `training.pt`、safetensors、token、密码或原始数据提交到 GitHub；
- 不删除或强推已经参与评测、报告或交接的历史版本。

## 6. 如果已经覆盖

先停止继续上传和删除操作，在 Hugging Face 历史中找到覆盖前后的 revision；分别保留 revision/tag，再把旧模型和当前模型整理到各自的实验/step 目录，并重新核对 SHA。

## 7. 上传前快速检查

- [ ] 实验 ID 唯一；
- [ ] global step 使用独立目录；
- [ ] 没有覆盖已有不同 SHA 的文件；
- [ ] 已记录固定 revision 和 SHA-256；
- [ ] 没有上传凭据或未脱敏信息。

# 双 A100 训练代码交接

这组补丁基于完整训练仓库 `EclipseaHime017/lingbot-vla-robotwin` 的
`02ba532`，按顺序包含：

1. `0001-*.patch`：已经在单张 A100 40GB 上验证过的冒烟训练基线（含 SwanLab）。
2. `0002-*.patch`：单机双卡 DDP、分布式采样、全局 loss 归一化、多 rank RNG
   断点恢复、rank 0 日志/导出以及双卡配置。
3. `0003-*.patch`：将 SwanLab 改成上游固定依赖安装完成后的独立安装步骤，避免
   protobuf 版本在同一次 pip 求解中冲突。
4. `0004-*.patch`：让资产下载脚本同时用 `HF_ENDPOINT` 构造元数据 API 与文件
   下载地址，可切换到 Hugging Face 镜像并续传已有 `.part` 文件。

本地实现最新提交为 `3f836f3`，分支名为 `feat/ddp-two-a100`。由于当前 GitHub
账号对完整训练仓库没有写权限，补丁是这次交接的可移植来源；不要只复制单个
`training.py`。

## 在新双卡机器应用

```bash
git clone https://github.com/EclipseaHime017/lingbot-vla-robotwin.git
cd lingbot-vla-robotwin
git checkout 02ba532
git am /path/to/ddp_two_a100/0001-*.patch \
       /path/to/ddp_two_a100/0002-*.patch \
       /path/to/ddp_two_a100/0003-*.patch \
       /path/to/ddp_two_a100/0004-*.patch
git log -4 --oneline
git status --short
```

应看到最新提交标题为 `fix(download): honor configurable Hugging Face endpoint`，
且工作树为空。不同机器重新应用补丁后
commit SHA 可能保持不变；正式训练以机器上的 `git rev-parse HEAD` 为准。

## 租卡后先做双卡冒烟

先准备与单卡阶段相同的模型、Qwen、50 任务 clean 数据、manifest 和 clean norm
stats/provenance。然后执行：

```bash
source scripts/activate.sh
nvidia-smi
python -m py_compile \
  src/lrvla/training.py \
  src/lrvla/training_data.py \
  src/lrvla/training_checkpoint.py \
  src/lrvla/training_distributed.py
pytest -q tests/test_training.py tests/test_training_distributed.py

bash scripts/train_ddp.sh \
  --max-steps 10 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --allow-subset \
  --no-export \
  --output-dir artifacts/ddp_smoke \
  2>&1 | tee artifacts/train_ddp_smoke.log
```

通过标准：

- 日志显示 `world_size: 2` 和 `global_batch_size: 2`；
- 两张 GPU 都有显存和利用率；
- 10 step 正常结束，无 NCCL hang、OOM 或 nonfinite；
- `run_metadata.json` 的 `source.dirty` 为 `false`；
- SwanLab 只出现一个 run，不是两个重复 run。

## 再测实际 batch

冒烟通过后依次测试每卡 batch 16、24、32，每组约 20 个 step。不要直接假定
40GB A100 能放下每卡 32；DDP 不会合并两张卡的显存。正式预设在
`configs/train_expert_ddp.yaml`，默认每卡 batch 16、global batch 32。

固定约 96 万处理样本时：

- global batch 32：约 30,000 optimizer steps；
- global batch 48：约 20,000 optimizer steps；
- global batch 64：约 15,000 optimizer steps。

正式长训前还需确认：50 个任务均已转换、`complete_50_tasks=true`、全量 clean
norm stats 的 provenance 匹配当前 manifest、SwanLab 登录成功、HF 私有仓库上传
和恢复路径可用。

## 2026-10-07 双 A100 真机验收结果

补丁链已经在单机 `2×NVIDIA A100-SXM4-40GB`（GPU 间拓扑为 PHB）上完成真实数据
验收。本阶段只验证双卡迁移，不启动正式长训。

- 全量数据：50 tasks、2500 clean episodes，训练 episode 0–44，验证 episode 45–49；
- norm 输入：494,328 个训练 frame；manifest 与 norm provenance SHA-256 均匹配；
- 自动化测试：训练与 DDP 相关测试共 31 项通过；
- DDP 冒烟：10/10 step 完成，`world_size=2`、`global_batch_size=2`；
- 稳态 step time 约 2.05 秒，单卡峰值显存约 14.15 GiB；
- loss 与 grad norm 全部有限，无 OOM、NCCL hang 或 rank 不同步；
- SwanLab 只创建一个 run，run ID 为 `f0unq1gg`，上传完成。

PyTorch 关于 `init_process_group` 未显式传入 device id 的输出属于警告，本次 current
device、双 rank 训练和 collective 均正常；torchvision video deprecation 同样不影响
本次验收。完整操作和结果见仓库根目录 `训练端_双A100_DDP冒烟交接.md`。

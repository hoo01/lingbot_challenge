# 训练端｜双 A100 DDP 冒烟交接

更新时间：2026-10-07（Asia/Shanghai）

> 文档定位：记录单卡实现迁移到双卡 DDP 后的真机验收。该阶段目标是确认代码、
> 全量 clean 数据和日志链路能在两张 A100 上正确工作，不包含正式长训。

## 结论

双 A100 DDP 迁移已经完成并通过真实数据冒烟，可以作为后续实验的已验证基线。
补丁以 `server_patch/ddp_two_a100/0001` 至 `0004` 保存；不要只复制某个 Python 文件。

## 机器与环境

- GPU：2×NVIDIA A100-SXM4-40GB；GPU 间拓扑 `PHB`
- CPU/RAM：32 核、125 GiB
- 数据盘：295 GB；验收后使用 88 GB、剩余 195 GB
- Driver/CUDA：550.107.02 / CUDA 12.4 driver capability
- PyTorch：2.8.0+cu128；两卡 BF16 矩阵乘测试通过
- SwanLab：0.10.1
- 训练工程：`/data/lingbot-vla-robotwin`
- 交付工程：`/data/lingbot_challenge`

## 代码与环境验收

- 依次应用 `0001`–`0004` 补丁；
- `training.py`、`training_data.py`、`training_checkpoint.py`、
  `training_distributed.py` 均通过 `py_compile`；
- `tests/test_training.py`、`tests/test_training_distributed.py`、
  `tests/test_bc_accumulation.py`：`31 passed`；
- CPU 合成训练冒烟通过：loss `1.133752 → 0.006771`，恢复、冻结权重检查和导出重载均通过。

## 全量 clean 数据验收

固定来源版本：

- 基座：`robbyant/lingbot-vla-v2-6b@11c703bf6a5c1f45b3b69168482da11fdbba53d7`
- Qwen：`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`
- 数据：`TianxingChen/RoboTwin2.0@3dc3b798668feb99ac61cc9086d84cbcc3d79186`

验收结果：

| 项目 | 结果 |
| --- | --- |
| manifest 状态 | `full_clean_data` |
| 完整任务 | `complete_50_tasks=true` |
| tasks / episodes | 50 / 2500 |
| train / validation episode | 0–44 / 45–49（每任务） |
| norm 训练样本 | 494,328 frames |
| source provenance | 50 份 |
| manifest SHA | 与 norm provenance 匹配 |
| norm SHA | 与 norm provenance 匹配 |

磁盘占用：raw archives 约 23 GB、解压 raw 约 34 GB、LeRobot 数据约 3.9 GB。

## 双卡真实数据冒烟

执行命令：

```bash
bash scripts/train_ddp.sh \
  --max-steps 10 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --val-episodes 5 \
  --no-export \
  --output-dir artifacts/ddp_smoke \
  2>&1 | tee artifacts/train_ddp_smoke.log
```

运行身份：

- `world_size=2`
- 每卡 batch 1，`global_batch_size=2`
- `training_samples=494328`
- `validation_samples=55459`
- Action Expert 最后 2 层，可训练参数 101,159,479

| step | loss | grad norm | step time | 单卡峰值显存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.277332 | 1.599860 | 4.249 s | 13.729 GiB |
| 2 | 0.337289 | 2.299791 | 1.773 s | 14.143 GiB |
| 3 | 0.240751 | 2.254878 | 2.271 s | 14.143 GiB |
| 4 | 0.273158 | 2.313334 | 2.097 s | 14.148 GiB |
| 5 | 0.412395 | 1.635576 | 2.047 s | 14.148 GiB |
| 6 | 0.487272 | 1.907217 | 2.070 s | 14.148 GiB |
| 7 | 0.396249 | 1.899057 | 2.062 s | 14.148 GiB |
| 8 | 0.322648 | 1.834818 | 2.048 s | 14.148 GiB |
| 9 | 0.407041 | 1.723181 | 2.048 s | 14.148 GiB |
| 10 | 0.315089 | 1.370212 | 2.055 s | 14.148 GiB |

10 步全部完成，无 OOM、NCCL hang、nonfinite 或 rank 不同步。首步包含初始化，后续
稳态约 2.05 秒/step。SwanLab 只创建一个 run：`f0unq1gg`，并显示上传 142 条记录完成。

### 目标全局 batch 32 短测

在相同全量数据上进一步使用每卡 batch 16、`world_size=2`、gradient accumulation 1，
即 `global_batch_size=32`，完成 3-step 短测：

| step | loss | grad norm | step time | 单卡峰值显存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.354039 | 0.599217 | 4.812 s | 22.015 GiB |
| 2 | 0.370527 | 0.793681 | 2.657 s | 23.153 GiB |
| 3 | 0.342442 | 0.691312 | 2.388 s | 23.153 GiB |

排除首步初始化后，稳态约 2.52 秒/step，即约 12.7 samples/s。3 步均正常完成，
没有 OOM、NCCL hang、nonfinite 或 rank 不同步；显存仍有约 16.8 GiB 余量。
SwanLab run ID 为 `9kwghjty`，上传 75 条记录完成。由此确认双卡目标全局 batch 32
在当前 partial Action Expert 配置与完整 clean 数据上可运行。

`init_process_group` 未显式传 device id 和 torchvision video deprecation 均为警告，
不影响本次验收结论。

## 交付边界

GitHub 保存：补丁、配置、复现说明和本交接文档。GitHub 不保存基座、原始数据、
转换视频、token、SwanLab 本地目录或大 checkpoint。本轮使用 `--no-export`，没有为
10-step DDP 冒烟生成需要异地备份的正式 checkpoint。

后续若决定正式训练，应另立实验输出目录并重新确定训练预算、checkpoint 策略和
评测门槛；全局 batch 32 已完成短测，但正式长训仍不属于本次双卡迁移验收。

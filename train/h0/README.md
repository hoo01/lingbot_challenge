# H0/G2：双卡 Clean-only 完整 Action Expert

## 当前结论

`H0` 是本仓库的实验名称；训练运行时代码将同一配方称为 `G2`。本目录维护同一实验的三个冻结阶段：H0-A（0–15k）、H0-C1（15k–25k）和 H0-C2（25k–40k）。

设计假设、变量控制、效果判定和失败后的决策见 [`DESIGN.md`](DESIGN.md)；本文只维护可执行操作步骤。

H0-A 已于 2026-10-15 正常完成 15,000 step。完整 FP32 导出及独立 action audit 均通过；冻结的
validation 窗口改善为 `-0.77%`，未达到继续训练所需的 `+1%`，因此结论为 `STOP_AT_15K`，
不启动 H0-C1。最终模型归档在 [`HumaLog/H0`](https://huggingface.co/HumaLog/H0)。C1/C2 配置仅作为
已验证的续训预案保留，不表示已经执行。

15k原固定500回合闭环评测现已完成：145/500（29.0%），500完整、0超时零分。
Clean102/250、Randomized43/250；与Lab q/v-r256的联合结果、原始小回执、配对CSV及比较边界见
[`../../eval/results/lab-qv256-h0-20261016.md`](../../eval/results/lab-qv256-h0-20261016.md)。

```text
LingBot-VLA-v2-6b 原始基座
+ 官方 50 任务 Clean demonstration
+ absolute joint action
+ Clean absolute-action norm / bounds_99_woclip
+ L1 flow-matching
+ 完整 Action Expert
+ state/action/time projections
+ 两卡 upstream FSDP2
```

第一轮不加入 Randomized 数据、图像增广、动态 seen-instruction 采样或 VLM 解冻。这样相对 E0 的主要变化只有 Action Expert 从最后两层扩大到完整 Expert。

## 冻结配置

H0-A 正式配置为 [`configs/train_h0_full_expert_2gpu.yaml`](configs/train_h0_full_expert_2gpu.yaml)：

- `expert_last_n_layers: -1`：训练完整 Action Expert；
- `distributed_strategy: upstream_fsdp2`：使用已接入的上游 FSDP2；
- `batch_size: 32`、`gradient_accumulation: 1`、两张 80G GPU：global batch 为 64；
- `image_augment: false`：第一轮不混入新的视觉变量；
- `save_every: 500`：H0-A 滚动保存并永久保留 5k、10k、15k；
- 最终保存 FP32 HF 导出并执行 action audit。

只有第 10.2 节的冻结判据通过时才启用 continuation：

- [`configs/train_h0_c1_2gpu.yaml`](configs/train_h0_c1_2gpu.yaml)：从 H0-A 15k DCP 恢复到累计 25k；
- [`configs/train_h0_c2_2gpu.yaml`](configs/train_h0_c2_2gpu.yaml)：从 H0-C1 25k DCP 恢复到累计 40k；
- 两段 continuation 都沿用 optimizer/RNG 状态，并将 H0-A 末端的 `5e-6` 学习率无 warmup 恒定延续；每个阶段使用新的 run 目录，不覆盖父阶段。

运行时代码当前明确要求 `world_size == 2`，DCP 恢复也要求保存和恢复时 world size 相同。因此不提供 8 卡配置，也不安排 2 卡到 8 卡接力。

## 仓库布局

训练运行时已并入 `main`，新机器直接克隆主分支：

```bash
git clone \
  --branch main \
  --single-branch \
  https://github.com/hoo01/lingbot_challenge.git \
  /data/lingbot_challenge
```

随后目录关系为：

```text
/data/lingbot_challenge/                         Git 根目录
/data/lingbot_challenge/train/h0/                H0 配方、设计文档和启动器
/data/lingbot_challenge/training/lingbot-vla-robotwin/  训练运行时
```

启动器默认按这个布局寻找训练运行时；若位置不同，再显式设置 `TRAIN_ROOT`。

## 空机器初始化

建议数据盘至少提供 300 GiB 空间；正式输出所在文件系统在模型和数据准备完成后仍须保留至少 120 GiB。300 GiB 是 H0-A 的开机下限，不是 H0-C 的容量证明；首次 `--resume-test` 后必须按设计文档第 8.3 节用实测 DCP 大小复核 continuation 空间。以下命令只准备训练环境、固定版本源码、原始基座、Qwen processor 和 50 任务 Clean 数据，不下载 teacher 权重或仿真资源：

```bash
cd /data/lingbot_challenge/training/lingbot-vla-robotwin

bash scripts/bootstrap.sh
source scripts/activate.sh lingbot-vla

python scripts/download_assets.py --asset all --plan
python scripts/download_assets.py --asset all --workers 3
python scripts/prepare_data.py
```

下载和数据转换都支持复用已经验证的文件；中断后重新执行同一命令即可。`prepare_data.py` 完成后应生成：

```text
data/clean_manifest.json
data/clean_manifest.txt
data/clean_norm_stats.json
data/clean_norm_stats.provenance.json
```

这些目录均被 Git 忽略，不得把模型、数据、环境或训练产物提交到仓库。环境和数据准备不等于训练验收，随后仍须按下文顺序完成 preflight、DCP、真实 smoke 和真实恢复测试。

## preflight 检查什么

[`scripts/preflight_h0.py`](scripts/preflight_h0.py) 会 fail closed 地检查：

- 配方确实为双卡 upstream FSDP2、完整 Expert、global batch 64；
- 运行时所需源码存在；
- 原始基座的 safetensors 文件与索引一致，并逐字节匹配 E0/F1 已验证使用的
  `robbyant/lingbot-vla-v2-6b@11c703bf...` 六分片身份；
- Qwen processor 必需文件齐全，`config.json` 匹配固定的
  `Qwen/Qwen3-VL-4B-Instruct@ebb281ec...`；
- LingBot vendor 固定在 `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`；
- 实际被训练代码读取的 vendor `robotwin.yaml` 中 arm 和 gripper 都是 absolute action；
- manifest 是完整 50 任务 Clean-only 数据，并逐项确认数据目录、LeRobot v3.0
  `meta/info.json` 和 manifest 声明的 parquet 文件实际存在；
- norm provenance 与 manifest、episode 0–44/45–49 划分及 norm SHA 一致；Norm
  本身必须是有限、分位数有序的 absolute 14-D state/action 结构，样本数也必须与 provenance 一致；
- checkpoint、FP32 export 和 action audit 已开启。
- 目标输出文件系统至少有 120 GiB 可用空间且可写；该门槛覆盖最多 4 份同时存在的
  full-Expert DCP、最终约 23.75 GiB FP32 导出及保存临时余量。首次真实
  `--resume-test` 后应记录实际 DCP 大小再复核，但不得未经测量向下调门槛。
- continuation 还会核对父 DCP 的阶段边界、两卡拓扑、原始基座 identity、完成标记、FP32 导出和 action audit，并按实测父 DCP/导出大小检查新阶段空间。

它不把 [`action_mode`](configs/train_h0_full_expert_2gpu.yaml) 字符串当作动作语义证据，而是检查运行时真正读取的 vendor robot config。

preflight 的无 GPU 单元测试可在本机运行：

```bash
python -m unittest discover -s train/h0/tests -v
```

## 执行顺序

以下命令都从 Git 根目录执行。启动器会拒绝在 tracked files dirty 时运行。

### 1. 静态和数据检查

```bash
cd /data/lingbot_challenge
bash train/h0/scripts/launch_h0.sh --check
```

`--check` 不查询或占用 GPU；它会读取基座分片计算哈希，但不会加载模型。默认检查
`$TRAIN_ROOT/runs` 所在文件系统；若正式输出使用其它盘，设置 `H0_OUTPUT_DIR` 后重新执行。

默认 provenance 路径为：

```text
training/lingbot-vla-robotwin/data/clean_norm_stats.provenance.json
```

不同位置可显式覆盖：

```bash
H0_NORM_PROVENANCE=/absolute/path/to/provenance.json \
  bash train/h0/scripts/launch_h0.sh --check
```

静态检查通过后，可以在仍无空闲 GPU 时预演正式启动。该命令同时确认目标 run
目录尚不存在，只打印最终命令，不创建文件：

```bash
bash train/h0/scripts/launch_h0.sh --dry-run
```

### 2. 两卡 toy DCP save/resume 单元验收

该测试使用 toy model，不需要正式模型和数据：

```bash
bash train/h0/scripts/launch_h0.sh --state-test
```

### 3. 真实模型和真实 Clean 数据 smoke

默认 10 optimizer steps，不写 checkpoint、不导出、不连接 SwanLab：

```bash
bash train/h0/scripts/launch_h0.sh --smoke
```

在正式开训前再跑 100 steps，并显式保存路径：

```bash
SMOKE_RUN=/data/lingbot_challenge/training/lingbot-vla-robotwin/runs/h0_smoke100
H0_OUTPUT_DIR="$SMOKE_RUN" bash train/h0/scripts/launch_h0.sh --smoke 100
```

通过条件包括：两卡都参与、FSDP2 无错误、Expert 0–35 全部被包裹、loss/grad norm 有限，并记录峰值 allocated/reserved 显存与 step time。启动器会用 step 21–99 自动写出 `h0_timing_budget_initial.json/.md`；此时尚无真实 DCP 保存耗时，所以报告明确标记为 provisional，不能据此正式开训。

### 4. 真实模型 upstream FSDP2 恢复验收

该测试先训练到 step 1 并保存 DCP，再启动两个新进程恢复并训练到 step 2：

```bash
H0_SMOKE_RUN="$SMOKE_RUN" bash train/h0/scripts/launch_h0.sh --resume-test
```

这一步和 toy DCP 测试不同：它会加载正式模型与 Clean 数据，并经过生产训练入口的 composable FSDP2 包装。设置 `H0_SMOKE_RUN` 后，启动器会把真实 DCP 保存耗时并入 smoke 统计，覆盖生成完整的 `h0_timing_budget.json/.md`；只有其中 `timing_gate_complete: true` 才通过时间门槛。

### 5. 正式 H0-A：15k 对齐阶段

建议在外层 `screen` 或 `tmux` 中前台运行启动器：

```bash
screen -S h0
cd /data/lingbot_challenge
bash train/h0/scripts/launch_h0.sh --launch
```

启动器创建唯一 run 目录，记录 source commit、配置 SHA、provenance 路径、日志和退出码。不要复用已有输出目录。

到 step 500 后，使用正式 8-batch validation 和首次正式 DCP 耗时更新预算：

```bash
python train/h0/scripts/plan_h0_timing.py \
  --metrics /data/lingbot_challenge/training/lingbot-vla-robotwin/runs/<H0-A-run>/metrics.jsonl \
  --current-step 500 \
  --elapsed-rental-hours <开机至今小时数> \
  --output /data/lingbot_challenge/training/lingbot-vla-robotwin/runs/<H0-A-run>/h0_timing_budget_step500.json
```

正式启动前的所有身份检查会原子写入 `preflight_receipt.json`，包括源码 commit、
基座 identity、Qwen config、vendor revision、manifest、Norm、robot config、world size
和 global batch。恢复训练会另存带时间戳的 `preflight_resume_*.json`，不会覆盖原始收据。

### 6. 同为两卡的 DCP 恢复

```bash
bash train/h0/scripts/launch_h0.sh --resume \
  /data/lingbot_challenge/training/lingbot-vla-robotwin/runs/<run>/checkpoints/global_step_<N>
```

恢复会回到原 run 目录，并由训练器核对 frozen base、manifest、norm、trainable selection、模型几何、采样配置和双卡拓扑。

### 7. 15k 后的 continuation 决策

15k 是与 E0 对齐的第一个完整里程碑，不是预设的最终终点。如果设计文档第 10.2 节预先冻结的 Clean validation 判据在 15k 达到至少 1% 的窗口改善，并且六天时间与实测 DCP 存储预算允许，应进入独立的 H0-C continuation 阶段，预留 25k、40k 累计 step 候选。

H0-C 的独立配置、恒定 `5e-6` scheduler、父 checkpoint 身份/容量检查和独立输出协议已经实现。达到冻结判据后启动 C1：

```bash
bash train/h0/scripts/launch_h0.sh --continue-c1 \
  /data/lingbot_challenge/training/lingbot-vla-robotwin/runs/<H0-A-run>/checkpoints/global_step_15000
```

C1 中断后用 `--resume-c1 <C1-checkpoint>` 原目录恢复。25k 仍达到相同改善判据且预算允许时启动 C2：

```bash
bash train/h0/scripts/launch_h0.sh --continue-c2 \
  /data/lingbot_challenge/training/lingbot-vla-robotwin/runs/<H0-C1-run>/checkpoints/global_step_25000
```

C2 中断后用 `--resume-c2 <C2-checkpoint>`。不得直接修改 `max_steps` 后复用 H0-A 的 `--resume`，否则 cosine 学习率会在恢复后向上跳变。

## 正式开训门槛

在 15k run 前必须全部满足：

1. `--check` 通过；
2. `--state-test` 的 toy DCP 保存与新进程恢复完全一致；
3. 10-step smoke 通过；
4. 100-step smoke 通过，显存和吞吐可接受；
5. 100-step 稳态 step time 与真实 DCP 耗时已生成六天时间预算，且 `timing_gate_complete: true`；
6. `--resume-test` 的真实模型 upstream FSDP2 恢复通过；
7. 启动日志确认完整 36 层 Expert 和 projections 可训练、VLM/vision 冻结；
8. 数据、norm、源码和配置 SHA 已记录；
9. 团队确认本轮仍从原始基座启动，而不是临时改为 E0 warm start。

## 暂不并入本轮

- 动态 `seen_instructions` 采样；
- 图像或状态增广；
- VLM 后层解冻和参数组学习率；
- E0 权重 warm start；
- 8 卡训练或跨 world-size 恢复。

这些可以分别实现和验收，但不得在正式 run 中临时加入。

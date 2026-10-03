# HF RoboTwin LeRobot 包 × LingBot data loader 兼容性 + clean 数据 norm stats 重算

对象：HF `TianxingChen/RoboTwin2.0` 下的 `lerobot_dataset/RoboTwin_lerobot_v21.zip` 和 `RoboTwin_lerobot_v30.zip`，内容为 Aloha-AgileX clean_50，50 任务 × 50 条。对照：

- LingBot 端：`lingbot-vla-v2`，环境由 `tools/create_train_env.sh` 安装，lerobot 锁定 **v0.4.2**，torchcodec 0.6.0。
- RoboTwin 端：pi0 转换链路，LingBot 指定 commit 13c3c47，包括 `process_data.py` 和 `convert_aloha_data_to_lerobot_robotwin.py`。

本次只做静态分析加数据比对，没有在 LingBot 环境里实际跑 loader。

## 结论速览

| 项 | 结论 |
| --- | --- |
| 字段名 / 维度 | ✅ 两个包都满足 `robotwin.yaml` 需要的全部 origin key |
| 格式版本 | ❌ **v2.1 包在 LingBot 锁定的 lerobot 0.4.2 下无法加载**（`BackwardCompatibilityError`）。✅ 改用 v3.0 包，它的轨迹与 v2.1 逐帧一致 |
| 帧率 15 vs 50 | ⚠️ 动作 chunk 和 future image 按帧取，不受影响；**但 DINO future-video 对齐的时间 RoPE 步长相差 3.3 倍**，只影响辅助损失 |
| 图像 | ⚠️ HF 包是 av1 视频、320×240，即原生分辨率；LingBot 链路存的是 480×640 图像帧，由原生图像放大而来。需要确认环境里的 FFmpeg 能解码 av1 |
| seen 指令 | ✅ 和 LingBot 链路一样，每条 episode 只保留 1 条从 `seen` 里随机抽取的指令，没有用 unseen。两条链路都**不保留**完整的 100 条 seen 列表 |
| norm stats | clean 和 randomized 的 q01/q99 差异不大：Δ ≤ 0.29 rad，区间比 0.88–1.18，主要集中在 L5、L6、R1、R5、R6 |

## 1. 兼容性检查

### 1.1 字段名与维度

`configs/robot_configs/robotwin.yaml` 需要以下 origin key。两个包都有，dtype 和 shape 也一致：

| key | 用途 | HF v2.1 / v3.0 |
| --- | --- | --- |
| `observation.state` | `[0:6]+[7:13]` → state.arm，`[6]`、`[13]` → state.effector | float32 [14] ✅ |
| `action` | 同上切片，`subtract_state: False` | float32 [14] ✅ |
| `observation.images.cam_high` | camera_top | video [3, 240, 320] ✅ |
| `observation.images.cam_left_wrist` | camera_wrist_left | video [3, 240, 320] ✅ |
| `observation.images.cam_right_wrist` | camera_wrist_right | video [3, 240, 320] ✅ |

两边的 motor name 不同：HF 包是 `left_joint_0…6`，LingBot 链路是 `left_waist…left_gripper`。但 FeatureTransform 是按 start/end **下标切片**取值的，不读 name，所以不受影响。

两个包的维度布局相同：`[0:6]` 是左臂，`[6]` 是左夹爪（0–1），`[7:13]` 是右臂，`[13]` 是右夹爪。

### 1.2 格式版本（阻塞项）

**lerobot 版本**

- `create_train_env.sh` 安装的是 `lerobot @ v0.4.2`，其中 `CODEBASE_VERSION = "v3.0"`。
- `LeRobotDatasetMetadata.load_metadata()` 会调用 `check_version_compatibility`。如果数据集的主版本号低于 3，就抛出 `BackwardCompatibilityError`。

**LingBot 自带的兼容分支**

`base_dataset.py` 里有一段 v2 fallback：先尝试 `import lerobot.datasets`，失败才退回 `lerobot.common.datasets`。这个分支只有在装的是旧版 lerobot 时才会生效。在锁定的环境下，**v2.1 包会直接报错**。README 中“v2.1/v3.0 都直接支持”的说法，在默认环境下不成立。

**v3.0 包与 v2.1 包的比对结果**

| 项 | 结果 |
| --- | --- |
| state/action 轨迹 | 按内容匹配，**2500/2500 逐帧一致**，误差 < 1e-6 |
| 3.3 节那 37 条被替换的轨迹 | 两个包相同 |
| episode 顺序 | v3.0 按任务名字母序排列，同编号下只有 50 条重合 |
| 指令 | 每条 episode 重新随机抽取，只有 30/2500 与 v2.1 相同 |
| fps / 视频格式 | 同为 15 fps、av1、240×320 |

**建议**：直接用 **v3.0 包**。如果坚持用 v2.1，需要先跑 lerobot 自带的 v2.1→v3.0 转换脚本。

另外，LingBot 推荐的 RoboTwin 转换脚本使用的是旧版 lerobot（`lerobot.common.*`），产物同样是 v2.x 格式，按理也会遇到这个问题。这一点没有实测。

### 1.3 帧率：15 fps（HF）vs 50 fps（LingBot 链路）

**来源**

- RoboTwin 的 `convert_aloha_data_to_lerobot_robotwin.py` 中写死了 `fps=50`。
- 原始 hdf5 没有时间戳，所以这两个数字都只是元信息里的标签，不代表真实的控制频率。

**LingBot loader 中 fps 出现的三个位置**（`lingbotvla/data/vla_data/base_dataset.py`）：

| 用途 | 写法 | fps 的影响 |
| --- | --- | --- |
| action chunk | `delta_timestamps = [t/fps for t in range(50)]`，LeRobot 内部用 `round(ts·fps)` 换回帧号 | **无**，始终是未来 50 帧 |
| future image（`use_future_image: true`） | 偏移 `(chunk−1)/fps`，即第 49 帧之后 | **无**，按帧取 |
| `future_video_effective_fps` | `fps / (chunk−1)`，经 `get_video_target(effective_fps=…)` 传给 DINO 视频教师的时间 RoPE：`step = base_fps(24) / effective_fps` | **有**：fps=50 时 eff=1.02、step≈23.5；fps=15 时 eff=0.31、step≈78.4 |

**第三项的影响**

- 只作用于 future-video 对齐的 target，权重 `future_video_loss_weight: 0.004`。动作、状态和归一化都不受影响。
- 如果想和 LingBot 官方链路保持一致，需要让 effective_fps 按 50/49 计算。
- 不能直接把 `meta/info.json` 里的 fps 改成 50：视频是按时间戳解码的，`tolerance_s=1e-4`，改了会导致取帧错位或报错。只能在代码里覆盖 `future_video_effective_fps`。这属于训练代码改动，本次没有动。

### 1.4 图像

| | 原始 hdf5 | LingBot 链路产物 | HF 包 |
| --- | --- | --- | --- |
| 分辨率 | 320×240 JPEG（4 路相机实测） | 480×640，由 `process_data.py` 放大得到 | 240×320，即原生分辨率 |
| 存储 | — | `generate.sh` 没传 `--mode`，默认 `mode="image"`，图像帧存在 parquet 里 | av1 mp4 |
| 进入模型 | — | Resize 到 224×224 | Resize 到 224×224 |

HF 包没有损失分辨率，只多了一层 av1 有损压缩。

HF 包是视频格式，需要用 torchcodec 0.6.0 解码，而 torchcodec 依赖系统 FFmpeg。**这套 FFmpeg 必须带 av1 解码器（dav1d 或 libaom）**。建议在训练环境里先确认：

```bash
ffmpeg -decoders | grep -i av1
```

### 1.5 指令（seen_instructions 是否保留）

**两条链路的做法相同**：每条 episode 只保留 1 条指令。

- **LingBot 链路**：`process_data.py` 先把 `instructions/episode{i}.json["seen"]` 的完整列表（100 条）写入中间文件；`convert_*_robotwin.py` 再用 `np.random.choice` **只取 1 条**，写成该 episode 的 `task`。
- **HF 包**：每条 episode 对应 1 个 task。v2.1 共有 2413 条不同的指令，v3.0 共有 2411 条。

LingBot loader 只读取 `item["task"]` 这一个字符串，没有指令增广。所以完整的 seen 列表在两条链路里都**没有保留**；如果要做指令增广，需要额外从原始 `instructions/*.json` 读取。

**HF v2.1 指令的来源核对**：把每条 HF 指令和原始 `aloha-agilex_clean_50/instructions/episode{i}.json` 对照，结果如下。

| episode 类型 | 在本条 episode 的 seen 列表中 | 在同任务其他 episode 的 seen 列表中 | 都不在 |
| --- | ---: | ---: | ---: |
| 与原始数据一致（2463 条） | 2411 | 0 | 52 |
| 被替换（37 条） | 2 | 15 | 20 |

- **没有任何一条指令来自 unseen 列表。**
- 那 52 条“都不在”的，是 seen 指令的语法修正版，与 seen 指令的最高相似度 ≥ 0.91，中位数 0.97。例如原始的 “Lift the bottle **the long thin bottle** up…” 在 HF 包里是 “Lift the long thin bottle up…”。
- 被替换 episode 的指令对应的是替换后的轨迹，所以对不上原始列表是正常的。

v3.0 包的核对结果类似：2422 条完全命中本 episode 的 seen 列表，没有一条命中本 episode 的 unseen 列表。

明细见 `lerobot_instruction_check.csv`。

## 2. clean 数据的 bounds_99_woclip 统计量 vs 仓库自带版本

**选定数据**：HF LeRobot clean 包。v2.1 和 v3.0 的轨迹相同，共 2500 条、548,893 帧。

**计算口径**：复现 `scripts/compute_norm_stats.py`。

- 每帧一个样本。
- state 取当前帧。
- action 取从当前帧开始的 50 步 chunk，越过 episode 末尾的部分用最后一帧补齐；chunk 内所有元素合并统计，对应 `norm_merge_chunk_dim=True`，共 2744 万个样本。
- 分位数用精确值。LingBot 的 5000 bin 直方图版本也复现了，与精确值相差 ≤ 0.0015 rad。直方图版本得到的夹爪 q01/q99 是 −1e-10 / 0.9998，与仓库文件完全一致，说明口径复现正确。

**对照**：仓库 `assets/norm_stats/robotwin.json`。它在 `aloha-agilex_randomized_500` 上计算，count 为 6,062,592 帧。

**表中各列的含义**

- **区间比** = (q99 − q01)<sub>clean</sub> / (q99 − q01)<sub>rand</sub>。
- **|x|>1.05** = 用对应统计量归一化后，clean 数据中超出 ±1.05 的比例。用 1.05 而不是 1，是为了避开夹爪 0.9998 带来的伪越界。

### 2.1 action.arm.position（单位 rad）

| 关节 | rand q01 | clean q01 | Δq01 | rand q99 | clean q99 | Δq99 | 区间比 | \|x\|>1.05（rand 统计量） | \|x\|>1.05（clean 统计量） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | −1.019 | −0.956 | +0.062 | 0.172 | 0.212 | +0.040 | 0.981 | 1.22% | 1.49% |
| L2 | −0.001 | −0.000 | +0.001 | 2.602 | 2.572 | −0.030 | 0.988 | 0.38% | 0.53% |
| L3 | −0.004 | −0.000 | +0.004 | 2.451 | 2.461 | +0.011 | 1.003 | 0.75% | 0.71% |
| L4 | −1.566 | −1.614 | −0.048 | 1.352 | 1.341 | −0.010 | 1.013 | 1.22% | 0.95% |
| L5 | −0.651 | −0.444 | **+0.207** | 1.237 | 1.256 | +0.019 | 0.901 | 0.77% | 1.32% |
| L6 | −2.233 | −2.265 | −0.032 | 1.600 | 1.451 | **−0.149** | 0.969 | 1.23% | 1.53% |
| R1 | −0.172 | −0.162 | +0.010 | 0.995 | 1.199 | **+0.204** | 1.166 | 2.17% | 1.24% |
| R2 | −0.003 | 0.000 | +0.003 | 2.619 | 2.635 | +0.017 | 1.005 | 0.66% | 0.47% |
| R3 | −0.002 | −0.000 | +0.002 | 2.453 | 2.524 | +0.070 | 1.028 | 1.04% | 0.74% |
| R4 | −1.645 | −1.710 | −0.065 | 1.290 | 1.340 | +0.050 | 1.039 | 1.65% | 0.92% |
| R5 | −1.023 | −1.078 | −0.055 | 0.875 | 1.162 | **+0.287** | 1.180 | 2.52% | 0.70% |
| R6 | −1.648 | −1.635 | +0.013 | 2.264 | 2.285 | +0.021 | 1.002 | 1.24% | 1.27% |

### 2.2 observation.state.arm.position（单位 rad）

| 关节 | rand q01 | clean q01 | Δq01 | rand q99 | clean q99 | Δq99 | 区间比 | \|x\|>1.05（rand 统计量） | \|x\|>1.05（clean 统计量） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | −1.019 | −0.954 | +0.064 | 0.172 | 0.173 | +0.001 | 0.947 | 0.97% | 1.36% |
| L2 | −0.001 | −0.000 | +0.001 | 2.602 | 2.571 | −0.031 | 0.988 | 0.36% | 0.51% |
| L3 | −0.004 | −0.000 | +0.004 | 2.450 | 2.447 | −0.002 | 0.997 | 0.72% | 0.72% |
| L4 | −1.565 | −1.585 | −0.020 | 1.350 | 1.325 | −0.025 | 0.998 | 1.02% | 1.04% |
| L5 | −0.648 | −0.439 | **+0.210** | 1.234 | 1.209 | −0.026 | 0.875 | 0.67% | 1.73% |
| L6 | −2.225 | −2.081 | +0.144 | 1.600 | 1.431 | **−0.169** | 0.918 | 1.03% | 1.67% |
| R1 | −0.172 | −0.147 | +0.024 | 0.995 | 1.065 | +0.069 | 1.039 | 1.80% | 1.70% |
| R2 | −0.003 | 0.000 | +0.003 | 2.617 | 2.593 | −0.024 | 0.990 | 0.44% | 0.59% |
| R3 | −0.002 | −0.000 | +0.002 | 2.451 | 2.481 | +0.030 | 1.012 | 0.84% | 0.72% |
| R4 | −1.644 | −1.697 | −0.054 | 1.288 | 1.304 | +0.016 | 1.024 | 1.18% | 0.74% |
| R5 | −1.022 | −1.027 | −0.004 | 0.874 | 1.053 | **+0.179** | 1.097 | 2.10% | 1.72% |
| R6 | −1.645 | −1.568 | +0.078 | 2.261 | 2.107 | −0.154 | 0.941 | 1.05% | 1.74% |

### 2.3 effector.position（夹爪，state 和 action 相同）

| | rand q01 / q99 | clean q01 / q99 |
| --- | --- | --- |
| L_grip、R_grip | −1e-10 / 0.9998 | 0 / 1.0（直方图口径为 −1e-10 / 0.9998，与 rand 完全相同） |

### 2.4 解读

**差异幅度不大**

- 12 个臂关节中，q01/q99 的最大偏移是 0.29 rad（R5 action q99），区间比在 0.88–1.18 之间。
- 无论用哪套统计量，clean 数据归一化后超出 ±1.05 的比例都在 0.4%–2.5%。
- 夹爪统计量完全相同。

**差异最大的维度是 L5、L6、R1、R5、R6**

- clean 数据上 L5 的负向活动范围更窄（q01 从 −0.65 变为 −0.44）。
- R1 和 R5 的 action q99 更大。
- action 和 state 的差别（例如 R1 action q99 = 1.199，而 state 只有 1.065）来自 chunk 末尾补帧：靠近 episode 末尾的姿态被重复计数，最多重复 49 次。randomized 统计量也是同样的口径。

**两套统计量都消除不了离群值**

- R1 有极端值（clean 中最小 −2.946 rad）。归一化后的最大 |x|：rand 统计量下为 5.75，clean 统计量下为 5.09。
- bounds_99 的 q01/q99 对这 1% 以外的尾部不敏感。要处理离群值，得剔除对应的 episode，见上一份报告 `joint_angle_report.md`。

**适用范围**：本节统计的是 HF clean 包。如果改用原始 hdf5 自行转换，会多出 62 条越界轨迹（|θ| 最大 7.34 rad）。按上一份报告，它们在所有点中的占比 < 0.4%，预计对 q01/q99 影响很小，但会把 min/max 和归一化后的最大 |x| 推高到约 ±11。这一点没有按 hdf5 帧重算。

**是否换用 clean 统计量**：如果只用 clean 数据训练，可以用 `norm_stats_clean.json`，格式与 `robotwin.json` 相同。替换方法是修改 `robotwin.yaml` 中的 `norm_stats` 路径。本次没有修改任何配置。

## 附：文件

| 文件 | 说明 |
| --- | --- |
| `lerobot_compat_check.py` / `.log` | 字段比对、v2.1 与 v3.0 逐帧比对、指令来源核对 |
| `lerobot_instruction_check.csv` | 每条 episode 的指令归属 |
| `norm_stats_compare.py` / `.log` / `.csv` | norm stats 重算与对比 |
| `norm_stats_clean.json` | clean 数据的统计量，LingBot 格式，可直接替换 |
| `fetch_meta_extra.py` | 拉取 v3.0 包的 parquet/meta 和原始 instructions（不含图像） |
| `../ref/lingbot-vla-v2/` | LingBot 仓库的只读参考副本（不含 .git） |

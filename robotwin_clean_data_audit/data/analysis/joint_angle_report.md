# RoboTwin 2.0 clean 数据关节角 ±π 边界检查报告

范围：Aloha-AgileX `clean_50`，50 任务 × 50 条 = 2500 条轨迹，只统计 12 个机械臂关节（夹爪 dim 6/13 排除）。randomized 数据未触碰，训练代码未修改。

## 结论（按判定规则）

**命中第三行「有越界、无跳变」**：LingBot 实际使用的原始 hdf5 中有 62/2500 条轨迹（2.5%）的关节角越过 ±π，最大 7.34 rad（> 2π），但所有轨迹内 |Δθ| 都 < π。数据侧处理有意义，但 naive wrap 会在这 62 条轨迹里制造 113 次 2π 跳变，**不能单独做 wrap，必须配合 wrap-aware loss**。更便宜的替代方案是直接剔除或替换这 62 条，官方 LeRobot 包看起来已经这样做了，见 §3.3。

---

## 1. 数据格式与维度布局（第 0 步）

**仓库结构**

`TianxingChen/RoboTwin2.0` 的数据按任务打包，每个任务一个 `dataset/{task}/aloha-agilex_clean_50.zip`，大小 0.2–0.6 GB。zip 内包含四个子目录：

| 目录 | 内容 |
| --- | --- |
| `data/episode{i}.hdf5` | 主数据，每条约 6 MB，绝大部分是 4 路相机 JPEG |
| `_traj_data/episode{i}.pkl` | 规划器输出的关节轨迹 |
| `video/` | 视频 |
| `instructions/` | 语言指令 |

另有 `lerobot_dataset/RoboTwin_lerobot_v21.zip`。它是已转好的 50 任务 × 50 条 clean 数据（2500 条、548,893 帧、15 fps），parquet 和 mp4 分开存放。

**zip 里的图像无法用 `allow_patterns` 排除**

zip 是 deflate 压缩的；而在 hdf5 文件内部，关节数据又排在所有图像之后（offset 约 8.78 MB / 8.84 MB）。所以想读到关节数据，就得把整条 hdf5 完整传下来。实际采用的读取方式：

- 用 HTTP Range 远程读 zip 目录，只提取需要的成员，见 `remote_zip.py`。
- LeRobot 包只取 parquet 和 meta，约 60 MB，视频不下载。
- 原始 hdf5 只取 3 个任务（adjust_bottle、rotate_qrcode、open_microwave）的 150 条，外加 2 条单独抽查。图像只在内存里，不落盘。全量 hdf5（约 15 GB）有意不拉：它只会让占比数字更精确，不改变结论。
- 50 个任务全部的 `_traj_data/*.pkl` 都已取下，约 280 MB。它们作为原始 hdf5 的代理，见 §3.1。

**hdf5 字段**（以 adjust_bottle/episode0 为例，T=140）

| 字段 | shape | dtype |
| --- | --- | --- |
| `joint_action/vector` | (T, 14) | float64 |
| `joint_action/left_arm`, `right_arm` | (T, 6) | float64 |
| `joint_action/left_gripper`, `right_gripper` | (T,) | float64 |
| `endpose/left_endpose`, `right_endpose` | (T, 7) | float64 |
| `observation/{head,front,left,right}_camera/rgb` | (T,) | 变长 JPEG bytes |
| `observation/*/{intrinsic_cv, extrinsic_cv, cam2world_gl}` | — | float32 |

**没有单独的 state 字段。** RoboTwin 的转换脚本 `policy/pi0/scripts/process_data.py`（LingBot 指定的 commit 13c3c47）是这样取的：

- state = `vector[0:T-1]`
- action = `vector[1:T]`

所以 state 和 action 是同一条序列错开一步。

**LeRobot parquet 字段**

- `observation.state` 和 `action`：(14,) float32，`action[t] == state[t+1]`。全量验证误差为 0。

**维度布局**（已按实际数据确认）

| 维度 | 含义 |
| --- | --- |
| `[0:6]` | 左臂 fl_joint1–6 |
| `[6]` | 左夹爪，取值 0–1 |
| `[7:13]` | 右臂 fr_joint1–6 |
| `[13]` | 右夹爪 |

**单位**：弧度。

**URDF 关节限位**

URDF（`arx5_description_isaac.urdf`）里 12 个臂关节的限位都是 **±10 rad**；curobo 配置里也没有更紧的限位。所以仿真和规划器都允许关节连续转过 ±π，数据本身**不会被 wrap**。

## 2. LingBot 在 RoboTwin 上的动作表示（第 1 步）

信息来源：`lingbot-vla-v2` 的以下文件：

- `configs/robot_configs/robotwin.yaml`
- `configs/vla/robotwin/robotwin.yaml`
- `lingbotvla/data/vla_data/transform.py`
- `experiment/robotwin/README.md`

**动作表示：绝对关节角**

- `action.arm.position` 取 action 的 `[0:6] + [7:13]`，并设置了 `subtract_state: False`，也就是不减 state、不做 delta。
- 夹爪单独作为 `effector.position`。
- 没有使用末端位姿：`end.position` 在 robot config 里没有映射。

**归一化：`bounds_99_woclip`**

- 公式为 `(x − q01) / (q99 − q01) · 2 − 1`，**不截断**。
- `assets/norm_stats/robotwin.json` 中 12 个关节的 q01/q99 都在 [−2.24, 2.62] 以内。注意这份统计是在 randomized_500 数据上算的。
- 按这份统计，原始 clean 数据里 7.34 rad 的点归一化后约为 **±11.6**，而正常范围是 [−1, 1]。|x| > 3 的点占 0.03%。

**数据来源：从官方 hdf5 自行转换**

README 要求先下载官方 hdf5，再跑 `process_data_pi0.sh` 和 `generate.sh` 生成 LeRobot 格式。它没有用 HF 上现成的 LeRobot 包。所以**原始 hdf5 是 LingBot 实际训练的数据**，这是本报告的主口径。

**已有的相关选项**：`transform.py` 支持 `sincos` 归一化，但代码限定只能用于 state key。

**action chunk**：`chunk_size` 默认为 50，robotwin 配置没有覆盖。

结论：用的是绝对关节角，不是 delta，也不是末端位姿，所以 wrap 问题需要认真检查。

## 3. 统计结果（第 2 步）

### 3.1 主口径：原始数据源（hdf5 / 规划器轨迹），2500 条

**为什么能用 pkl 代替 hdf5**

hdf5 中的 `joint_action` 就是回放 `_traj_data/*.pkl` 里的规划轨迹得到的。两者的对应关系已验证：

- adjust_bottle 全部 50 条 episode，pkl 与 hdf5 的每关节 max|θ| 相差都 < 0.05。
- 单独抽查的 grab_roller ep38 与 place_shoe ep45，hdf5 实测结果与 pkl 一致。

**口径说明**

- pkl 的采样密度比 hdf5 帧高。所以下表中的「占比」只能作为近似；max、越界轨迹数、跳变、跨界这几项是可靠的。
- state 和 action 来自同一序列，只差一步，因此原始口径下两者的统计相同，不再分列。

| 关节 | min | max | 1% | 99% | \|θ\|>0.9π 占比 | \|θ\|>0.95π 占比 | 越界点数 \|θ\|>π | 越界轨迹数 | max\|Δθ\| | \|Δθ\|>π | 跨 ±0.9π 轨迹 | naive wrap 制造的跳变 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | −7.341 | 5.682 | −0.964 | 0.246 | 0.17% | 0.17% | 7291 | 7 | 0.25 | 0 | 0 | 10 |
| L2 | −0.000 | 3.888 | 0.000 | 2.572 | 0.16% | 0.12% | 4186 | 6 | 0.38 | 0 | 0 | 11 |
| L3 | −0.119 | 4.501 | 0.000 | 2.564 | 0.36% | 0.16% | 5527 | 8 | 0.40 | 0 | 0 | 15 |
| L4 | −1.927 | 1.790 | −1.702 | 1.340 | 0 | 0 | 0 | 0 | 0.15 | 0 | 0 | 0 |
| L5 | −1.422 | 1.563 | −0.439 | 1.217 | 0 | 0 | 0 | 0 | 0.09 | 0 | 0 | 0 |
| L6 | −6.232 | 4.390 | −2.114 | 1.407 | 0.30% | 0.22% | 6938 | 15 | 0.07 | 0 | 0 | 20 |
| R1 | −6.317 | 1.389 | −0.158 | 1.021 | 0.04% | 0.03% | 1058 | 2 | 2.51* | 0 | 0 | 3 |
| R2 | −0.574 | 3.243 | 0.000 | 2.671 | 0.06% | 0.03% | 524 | 5 | 0.39 | 0 | 0 | 9 |
| R3 | −0.005 | 3.659 | 0.000 | 2.608 | 0.38% | 0.07% | 2114 | 9 | 0.83 | 0 | 0 | 17 |
| R4 | −1.989 | 1.953 | −1.726 | 1.316 | 0 | 0 | 0 | 0 | 0.16 | 0 | 0 | 0 |
| R5 | −2.129 | 1.397 | −1.028 | 0.928 | 0 | 0 | 0 | 0 | 0.07 | 0 | 0 | 0 |
| R6 | −6.270 | 3.504 | −1.715 | 2.061 | 0.34% | 0.25% | 9033 | 23 | 0.08 | 0 | 0 | 28 |

**汇总**

- 任一关节越界的轨迹：**62 / 2500**，其中 7 条超过 2π。
- 轨迹内 |Δθ| > π：**0**。
- 同时出现 > 0.9π 和 < −0.9π 的轨迹：**0**。
- naive wrap 到 [−π, π] 后会出现 **113 次** 2π 跳变，62 条越界轨迹全部受影响。

越界值都是从 ±0.9π 一侧**连续**转过去的。例如 grab_roller ep38 的 L1 在 hdf5 中是 −2.80 → −2.97 → −3.14 → −3.31 → −3.48 → …，一直到 7.34 rad。这正是任务书里提醒的「连续不 wrap」陷阱。

\* place_shoe ep45 的 R1 在 hdf5 中有一帧从 −5.576 跳到 −3.070（2.5 rad，规划段接缝处）。它小于 π，不计入跳变，但属于原始数据异常。

### 3.2 次口径：HF 现成的 LeRobot 包（2500 条 / 548,893 帧）

**action**

| 关节 | min | max | 1% | 99% | \|θ\|>0.9π | \|θ\|>0.95π | \|θ\|>π 数 | max\|Δθ\| | \|Δθ\|>π | 跨 ±0.9π 轨迹 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | −1.395 | 0.436 | −0.954 | 0.173 | 0 | 0 | 0 | 0.25 | 0 | 0 |
| L2 | −0.000 | 3.389 | 0.000 | 2.571 | 0.055% | 0.034% | 44 | 0.38 | 0 | 0 |
| L3 | −0.119 | 3.451 | 0.000 | 2.448 | 0.20% | 0.032% | 87 | 0.40 | 0 | 0 |
| L4 | −1.927 | 1.790 | −1.586 | 1.325 | 0 | 0 | 0 | 0.15 | 0 | 0 |
| L5 | −1.422 | 1.563 | −0.440 | 1.213 | 0 | 0 | 0 | 0.11 | 0 | 0 |
| L6 | −3.469 | 2.904 | −2.095 | 1.433 | 0.26% | 0.18% | 564 | 0.16 | 0 | 0 |
| R1 | −2.946 | 1.389 | −0.149 | 1.068 | 0.011% | 0 | 0 | 0.12 | 0 | 0 |
| R2 | −0.574 | 2.995 | 0.000 | 2.594 | 0.033% | 0.009% | 0 | 0.39 | 0 | 0 |
| R3 | −0.005 | 3.333 | 0.000 | 2.484 | 0.24% | 0.025% | 69 | 0.83 | 0 | 0 |
| R4 | −1.989 | 1.953 | −1.698 | 1.308 | 0 | 0 | 0 | 0.16 | 0 | 0 |
| R5 | −1.634 | 1.397 | −1.027 | 1.061 | 0 | 0 | 0 | 0.11 | 0 | 0 |
| R6 | −2.940 | 3.425 | −1.570 | 2.112 | 0.22% | 0.13% | 409 | 0.17 | 0 | 0 |

**state**：与 action 只差一步。各项指标几乎相同，越界点数为 L2 44、L3 87、L6 555、R3 69、R6 406，其余指标与上表一致。完整数值见 `joint_angle_check.log`。

**汇总**

- 越界轨迹：**25 / 2500**，最大 |θ| 为 3.47 rad。
- 轨迹内跳变：**0**。
- naive wrap 后会出现 44 次跳变，影响 25 条轨迹。
- 50 步 action chunk 中有跳变的比例：单关节最多 0.14%（R6）。

### 3.3 LeRobot 包与原始 hdf5 不一致

有 37 条 episode 的 LeRobot 版本与原始数据不同：长度不同，max|θ| 也不同。原始版本的 max|θ| 在 3.50–7.34 rad 之间，LeRobot 版本都 ≤ 3.05 rad。看起来发布 LeRobot 包时，**多圈旋转的异常轨迹被换成了另一条正常轨迹**。

hdf5 实测交叉校验结果（误差 < 3e-8 视为一致）：

| 任务 | 一致的 episode | 不一致的 episode |
| --- | --- | --- |
| adjust_bottle | 49 / 50 | 16 |
| rotate_qrcode | 47 / 50 | 17, 25, 38 |
| open_microwave | 50 / 50 | — |

不一致的 episode 正好就是 pkl 预测出来的那几条。open_microwave 在 pkl 中没有越界轨迹（max 2.909 rad），hdf5 实测也是 50 条全部一致、max 2.909 rad。

结论：**两份数据要区分开**。如果队里直接用 HF 的 LeRobot 包训练，那么问题只有 25 条轻微越界（≤ 3.47 rad）；如果按 LingBot README 从 hdf5 自行转换，那么面对的是 62 条、最大 7.34 rad 的情况。

## 4. 出现边界问题的任务（原始口径）

| 任务 | max\|θ\| | 关节 | 越界轨迹 | >1.5π 轨迹 | naive wrap 制造的跳变 |
| --- | ---: | --- | ---: | ---: | ---: |
| grab_roller | 7.34 | L1 | 7 | 1 | 12 |
| place_dual_shoes | 7.18 | L1 | 9 | 4 | 12 |
| turn_switch | 6.38 | L1 | 3 | 3 | 5 |
| place_shoe | 5.58 | R1 | 6 | 1 | 8 |
| adjust_bottle | 5.57 | L6 | 1 | 1 | 2 |
| place_can_basket | 4.50 | L3 | 6 | 0 | 14 |
| rotate_qrcode | 4.47 | R6 | 3 | 0 | 3 |
| shake_bottle_horizontally | 4.30 | L3 | 1 | 0 | 2 |
| move_stapler_pad | 4.23 | R6 | 1 | 0 | 1 |
| place_mouse_pad | 3.84 | L6 | 7 | 0 | 7 |
| hanging_mug | 3.79 | R6 | 1 | 0 | 1 |
| handover_block | 3.66 | L3 | 1 | 0 | 4 |
| stack_bowls_two | 3.66 | R3 | 1 | 0 | 4 |
| stack_bowls_three | 3.66 | R3 | 1 | 0 | 4 |
| stamp_seal | 3.60 | L3 | 1 | 0 | 4 |
| blocks_ranking_size | 3.57 | R3 | 3 | 0 | 10 |
| press_stapler | 3.55 | R3 | 3 | 0 | 4 |
| put_bottles_dustbin | 3.39 | L2 | 1 | 0 | 2 |
| move_can_pot | 3.34 | L3 | 3 | 0 | 6 |
| place_object_basket | 3.26 | R3 | 2 | 0 | 6 |
| place_bread_skillet | 3.20 | R3 | 1 | 0 | 2 |

另有 13 个任务的最大值在 0.9π–π 之间（2.90–3.14 rad），没有越界：

- 多数是 L3/R3 在约 2.90 rad 处的常见折叠姿态。
- 具体任务包括 put_object_cabinet、open_microwave、stack_bowls 等。

其余 16 个任务的所有关节都 < 0.9π。

## 5. 对后续方向的建议

- **不要单独做 naive wrap。** 原始数据是连续的，wrap 之后反而会在 62 条轨迹中引入 113 次 2π 跳变，对 flow matching 的 action chunk 伤害更大。
- **最便宜的方案：剔除或替换这 62 条异常轨迹。** 这是推断，没有实测：这些轨迹在 `bounds_99_woclip` 下会产生归一化后约 ±11 的离群目标，VLAct 消融中 wrap 带来的 +3.1，可能主要就来自消除这类离群值。直接剔除这些轨迹，或者改用 HF LeRobot 包（已替换为正常轨迹），可能拿到大部分收益，而且不需要改 loss。
- **如果要完整复现 VLAct**：wrap 和 wrap-aware loss 必须一起做；同时要确认推理时的 action 也能 unwrap 回连续的关节指令。仿真关节限位是 ±10 rad，如果直接下发 wrap 后的目标，关节会反向转一大圈。

## 附：文件

| 文件 | 说明 |
| --- | --- |
| `joint_angle_check.py` | 统计脚本。首次运行用 `--download` 拉取 LeRobot parquet |
| `remote_zip.py` | 基于 HTTP Range 的远程 zip 读取，走 `HTTP(S)_PROXY` |
| `fetch_traj_pkl.py` | 拉取 50 个任务的 `_traj_data/*.pkl` |
| `fetch_hdf5_joints.py` | 流式读取指定任务的 hdf5，只保存关节和末端位姿 |
| `joint_angle_check.log` | 完整输出 |
| `joint_angle_stats.json` | 全部统计表 |
| `joint_angle_hist.png` | 12 个关节的直方图。蓝色为原始数据，黑线为 LeRobot 包；红线为 ±π，橙色虚线为 ±0.9π |

原始数据存放在 `data/raw/`（不含图像）。

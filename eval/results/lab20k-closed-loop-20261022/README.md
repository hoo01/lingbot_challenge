# Lab20k 6k–8k 闭环测评归档

归档日期：2026-10-22（Asia/Shanghai）。本目录保存四个已完成 RoboTwin 闭环测评的轻量、可离线复核证据；整理过程没有重新运行模型或仿真。

|归档|权重|协议|Clean|Randomized|总成绩|初始化失败|运行错误|
|---|---|---|---:|---:|---:|---:|---:|
|`step6000-official`|Lab20k FullSFT 6k|官方固定执行 50|162/250|87/250|249/500（49.8%）|81|0|
|`step6000-pace`|Lab20k FullSFT 6k|PACE 动态 horizon|164/250|94/250|258/500（51.6%）|81|0|
|`step7000-official`|Lab20k FullSFT 7k|官方固定执行 50|170/250|79/250|249/500（49.8%）|81|0|
|`step8000-pace`|Lab20k FullSFT 8k|PACE 动态 horizon|172/250|79/250|251/500（50.2%）|81|0|

每项均为 50 任务 × 2 setting × 每组 5 个有效回合，共 500 个有效回合。初始化失败不进入有效分母，按从 100010 开始的连续后备 seed 补足；因此每项实际尝试 581 次，其中 81 次初始化失败。四项运行错误均为 0。

## 文件说明

每个子目录保留：

- `job.json`：任务 ID、标题、创建/完成时间和平台配置。
- `request.json`：run 启动时实际请求；判断 checkpoint ID 和 seed 策略时以它为准。
- `checkpoint.json`：运行时 checkpoint 登记、权重分片 SHA、norm 与 robot config 身份。
- `official-defaults-manifest.json`：官方源码 revision 与关键源文件 SHA。
- `server-runtime.json`：实际推理精度、执行 horizon、flow 步数、编译、MoE、动作表示和 PACE 身份。
- `report.json`：平台原始汇总、100 个 group 和全部尝试记录。
- `scores.csv`：平台原始 100 组成绩。
- `timing.json`：权重核验、服务启动和测评墙钟时间。
- `summary.json`、`by-group.csv`、`by-task.csv`、`episodes.csv`：由本目录脚本确定性派生。

顶层 `comparison.csv` 和 `summary.json` 汇总四项结果；`verification.json` 记录完整性断言；`manifest.json` 记录轻量归档文件的 SHA256。权重、视频、图像、轨迹和多 GB 日志未复制进 Git。

## 离线复核

在仓库根目录执行：

```bash
python3 eval/results/lab20k-closed-loop-20261022/build_report_tables.py
```

脚本会从四份原始 JSON 重建 CSV 和汇总，并检查：任务已完成、500 个有效回合、100 个任务/setting 组、每组 5 个有效回合、成功/初始化失败计数闭合、attempt 数与 episode 数一致、FP32、机械臂相对关节/夹爪绝对动作，以及 checkpoint、runtime、source manifest 的 norm/robot config SHA 绑定一致。

该命令是离线证据复算，不下载权重、不连接服务器、不执行模型或仿真。完整重跑仍需要 `checkpoint.json` 中列出的权重文件、对应 norm/robot config、官方源码 revision `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`、RoboTwin 环境和 GPU。

## 推理条件

四项共同条件：FP32、图像 256、10 个 flow step、official MoE、FlashAttention 2、compile 开启、进程初始 sampling seed 42、机械臂 `subtract_state=true / relative_type=joint`、夹爪绝对值。模型预测 chunk 为 50。

官方协议每次执行完整 50 步。PACE 使用 `pace_v1`，运行记录绑定 commit `b1b61224a1219041ef3c8efc9db9577ae0beb798` 和 50 个任务的 calibration SHA；6k PACE 平均实际 horizon 为 21.9708，8k PACE 为 22.1317，范围均为 2–50，fallback 为 0。

6k 官方 run 的原始 `request.json` 使用 checkpoint 登记 `641d8d031051`；后续平台数据库将同一 6k 来源登记为 `9a64d1b0cf84`。本归档保留两份身份，但以 run 内原始 request、checkpoint 和 runtime 路径作为实际运行事实。

7k 官方 run 的 robot config SHA 为 `f4a1de74470ae9d7353bbfe7bdd173305bf391cecd05f4dd7f5276968e426c5e`；其余三项为 `1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b`。因此 6k 官方与 7k 官方之间不应解释为只改变训练步数的严格单变量对照。

## 结果解释边界

6k 官方与 6k PACE 使用相同 6k 权重来源及 norm，最接近执行协议对照：PACE 多 9 次成功（+1.8 个百分点）。但它们在不同运行时段/服务实例执行，模型原生噪声流没有逐回合锁定，不能把差异完全归因于 PACE。

8k PACE 比 6k PACE 少 7 次成功（−1.4 个百分点）。这同时改变权重训练步数，不能用于判断 PACE 本身是否提升。7k 官方与 6k 官方总成功数相同，但 Clean/Randomized 分布不同，且上述 robot config SHA 不同。

这些成绩是队内开发测评，不代表竞赛官方复核或最终提交成绩。

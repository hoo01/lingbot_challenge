# Lab r256 / H0 离线报告证据

主报告：[训练及固定500回合结果](../lab-qv256-h0-20261016.md)。冻结日期2026-10-16。

- `evidence/`：从本地已完成归档逐字节复制的六轮verification、paired-outcomes、receipt-manifest；原E0 reference；Lab训练完成回执与H0原发布/实际启动小证据。8k原manifest列出的by-group和CPU审计源码也保留，仅哈希校验，不重新执行它。
- `sources.json`：本包29份输入文件SHA、原本地归档位置、正式job/model身份。此文件是本次组包索引，不冒充原任务生成的manifest。
- `summary.json` / `summary.csv`：六轮最终结果和审计计数；`by-task.csv`：全部50任务；`paired-lab20k-h0.csv`：相同原500键的两模型结果。
- `results.png`：由上述表格生成的静态图；没有从网页抄录或估计成绩。
- `evidence/retain-status-snapshot.json`：RETAIN进行中小状态，明确`not_final=true`；不进入最终成绩表。

```bash
python eval/results/lab-qv256-h0-20261016/recompute.py --check
```

复算只需要Python标准库。`--plot`可选，需要matplotlib；不带`--check`会重写派生表格，不修改原证据。原始文件和本目录按Git `-text`保留字节，避免换行变化使CSV/JSON的原SHA失效。

核验范围是已完成小回执与表格的离线重算。原图像/动作/终帧/视频元数据CPU全量审计已在各任务完成时进行，本包保留其回执；不含大权重、训练state、完整视频或任何账户凭据，也不承诺据此独立重跑仿真。绝对机器路径是历史索引，不是当前访问或恢复指令。

H0原HF preflight与旧训练交接中的source/config SHA存在不同口径，原字节保留并在报告列出；norm/data/robot/基座身份一致，不擅自把两个源码/config记录改成同一值。

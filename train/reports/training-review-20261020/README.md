# 训练策略文档附件

日期：2026年10月20日。主文档为[训练策略与实验复盘](../training-strategies-20261012-20261020.md)。

## 配方与采样

- `recipes.json`记录原Lab20k、4090全量SFT、AMD八卡及新增8000迁移的配置摘要，不是直接执行的启动器。
- `sampling-initial.csv`记录4090前7000步的任务采样概率。
- `sampling-after7000.csv`记录后5000步的概率及整个12000步的预期累计倍率。

## 曲线与统计

| 路线 | 逐步记录 | Global batch | 数据采样 | Action计算精度 |
| --- | --- | ---: | --- | --- |
| 4×4090重采样，`route=four4090` | [loss-4090.csv](loss-4090.csv)，1…12000，共12000行 | 128（4×16×2） | 前7000步原重采样；之后5000步使用新表，前200步渐变 | FP32 |
| 8×W7900D原帧比例，`route=amd8` | [loss-amd8.csv](loss-amd8.csv)，1…8000，共8000行 | 256（8×16×2） | 官方逐帧shuffle，原有效帧比例 | BF16 |

两条路线都从未做RETAIN插值的同一Lab20k开始，分别新建AdamW。主参数、Adam状态与保存均为FP32，视觉和语言计算均为BF16；上表的Action精度仅指动作模块。后续“AMD8000迁移到4090”继承的是AMD模型和Adam，不是第一条4090重采样路线，其工程或续训更新不在本次两份CSV中。

4090保留父阶段1…7000和切换阶段7001…12000；AMD保留完整2000点之前的1…2000和恢复后的2001…8000。回退丢弃的旧2001…4000、Flex试验前87步、8000后的未保存更新不混入。每个文件的步数连续且唯一，原始尖峰全部保留。

CSV每行对应一次完整optimizer更新，字段定义如下：

| 字段 | 含义 |
| --- | --- |
| `route` | 固定实验标识：`four4090`或`amd8` |
| `step` | 原20k之后新增的optimizer更新次数，从1开始 |
| `total` | 日志中记录的总loss |
| `flow_matching` | 动作flow-matching loss |
| `depth`、`future_depth`、`video` | 日志中的对应教师对齐loss |
| `sequence_wise`、`router_z` | 日志中的对应MoE正则loss |
| `gradient_norm_before_clip` | 全局裁剪前的梯度norm |

loss先在梯度累积的microbatch间平均，再在全部rank间平均；CSV保留原日志数值，不重新从舍入后的分项拼总loss。单步值不是平滑值，也不是单样本loss。

[loss-curves.png](loss-curves.png)和[action-loss-smoothed.png](action-loss-smoothed.png)分别用左右面板展示4090与AMD。浅线保留全部尖峰，深线使用含当前步的向后100步均值；平滑图从第100步开始。[loss-summary.json](loss-summary.json)提供各路线配置标识、来源SHA及聚合统计。采样、batch、动作计算精度和后端存在差异，不能按两条曲线的绝对高低直接排名闭环能力。

## 来源与版本

`rules-source.json`记录2026年10月20日核对的官方公开FAQ来源与响应SHA。`sources.json`只列原始本地资料的身份哈希；详细回执与过滤索引保留本地。发布文件按`SHA256SUMS.json`校验，目录内Git属性保留文件原始字节。

主文档中的新迁移是截稿快照：源8000的16份SHA交付和CPU/真实输入门检通过；CPU重分片进行中，GPU保存恢复与正式启动尚未通过。曲线下降不等于闭环成功率上升，各套开发评测结果也不能混作官方成绩。

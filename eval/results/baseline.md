# LingBot-VLA 基座评测结果

原始 LingBot-VLA v2 6B 在 RoboTwin 2.0 的50任务开发场景中，完成 **500个有效回合，成功52个，成功率10.4%**。评测于2026-10-05 08:44:26（北京时间）结束。

|环境|成功 / 有效|成功率|初始化失败|
|---|---:|---:|---:|
|Clean|27/250|10.8%|20|
|Randomized|25/250|10.0%|61|
|合计|52/500|10.4%|81|

本结果使用官方默认推理参数，但场景未经过官方专家筛选，属于开发基线，**不是官方完整榜单成绩**。

## 模型与评测设置

- 模型：`robbyant/lingbot-vla-v2-6b`，revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`；原始预训练权重，未做本任务SFT或LoRA微调。
- 推理源码：`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`，使用上游原始归一化。
- 推理参数：256×256、FP32/high、chunk/执行50、flow10、compile开启、官方 fused MoE / 视觉 FlashAttention 2、全CUDA、原生噪声。
- 模型服务进程启动seed42，回合间不重设模型seed；故障恢复形成独立的随机数进程段。
- 场景：50任务 × Clean/Randomized × 每组5有效。起始seed为100010、100011，仅初始化失败按未尝试整数seed顺序补测；策略失败正常计入有效分母。

100组均完成5个有效回合。共581条回合记录，另有3次运行中断（两次SIGSEGV、一次900秒超时），合计584次尝试；结束时当前运行错误为0。448个有效策略失败均到任务动作上限。

## 各任务结果

表中分数为成功数 / 有效回合数，初始化失败为两个环境的合计。

|任务|Clean|Randomized|合计|初始化失败|
|---|---:|---:|---:|---:|
|adjust_bottle|0/5|0/5|0/10|3|
|beat_block_hammer|0/5|0/5|0/10|0|
|blocks_ranking_rgb|0/5|0/5|0/10|1|
|blocks_ranking_size|0/5|0/5|0/10|0|
|click_alarmclock|1/5|2/5|3/10|0|
|click_bell|1/5|2/5|3/10|0|
|dump_bin_bigbin|0/5|0/5|0/10|7|
|grab_roller|0/5|0/5|0/10|2|
|handover_block|0/5|0/5|0/10|1|
|handover_mic|0/5|0/5|0/10|1|
|hanging_mug|0/5|0/5|0/10|1|
|lift_pot|0/5|0/5|0/10|3|
|move_can_pot|0/5|0/5|0/10|3|
|move_pillbottle_pad|0/5|0/5|0/10|1|
|move_playingcard_away|1/5|1/5|2/10|0|
|move_stapler_pad|0/5|0/5|0/10|1|
|open_laptop|1/5|2/5|3/10|1|
|open_microwave|5/5|4/5|9/10|0|
|pick_diverse_bottles|0/5|0/5|0/10|13|
|pick_dual_bottles|0/5|0/5|0/10|1|
|place_a2b_left|2/5|0/5|2/10|0|
|place_a2b_right|0/5|0/5|0/10|0|
|place_bread_basket|0/5|0/5|0/10|0|
|place_bread_skillet|0/5|0/5|0/10|1|
|place_burger_fries|0/5|0/5|0/10|2|
|place_can_basket|0/5|0/5|0/10|0|
|place_cans_plasticbox|0/5|0/5|0/10|0|
|place_container_plate|0/5|0/5|0/10|1|
|place_dual_shoes|0/5|0/5|0/10|1|
|place_empty_cup|0/5|0/5|0/10|0|
|place_fan|0/5|0/5|0/10|0|
|place_mouse_pad|0/5|0/5|0/10|1|
|place_object_basket|0/5|0/5|0/10|4|
|place_object_scale|0/5|0/5|0/10|0|
|place_object_stand|0/5|0/5|0/10|0|
|place_phone_stand|0/5|0/5|0/10|1|
|place_shoe|0/5|0/5|0/10|1|
|press_stapler|5/5|5/5|10/10|0|
|put_bottles_dustbin|0/5|0/5|0/10|4|
|put_object_cabinet|0/5|0/5|0/10|0|
|rotate_qrcode|0/5|0/5|0/10|2|
|scan_object|0/5|0/5|0/10|2|
|shake_bottle|4/5|2/5|6/10|8|
|shake_bottle_horizontally|3/5|3/5|6/10|8|
|stack_blocks_three|0/5|0/5|0/10|0|
|stack_blocks_two|0/5|0/5|0/10|0|
|stack_bowls_three|0/5|0/5|0/10|4|
|stack_bowls_two|0/5|0/5|0/10|1|
|stamp_seal|0/5|0/5|0/10|1|
|turn_switch|4/5|4/5|8/10|0|

## 结果解读

成功主要集中在少数任务：press_stapler为10/10、open_microwave为9/10、turn_switch为8/10。每个环境只有5个有效回合，属于小样本结果。

成功标签采用环境判定。例如open_microwave按门角判定，并不检查指令指定的手臂，因此不能直接把该分数解释为语言遵从程度。比较后续模型时，应固定推理配置、实际seed、指令与初始三相机，并同时报告新增成功与退化。

## 耗时

从2026-10-04 01:24:47开始到结束，日历耗时31小时19分38秒，包含模型启动、故障处理和等待。有效回合耗时合计24.23小时，初始化失败合计0.41小时。

|录像模式|有效回合数|平均秒 / 回合|中位秒 / 回合|
|---|---:|---:|---:|
|不录像|441|164.0|132.9|
|仿真时间25fps|27|351.7|307.8|
|逐动作/缓存帧录像|32|169.1|137.9|
|全部|500|174.5|141.0|

各组任务和seed不同，不能据此计算录像的因果开销。当前平台可选录像使用10fps仿真原速，以上为这次基座运行的实际记录。

## 结果文件

- [汇总JSON](baseline_official.json)：推理配置、100组统计、耗时和运行中断。
- [分任务CSV](baseline_by_task.csv)：便于汇总与比较。
- [回合记录](baseline_episodes.jsonl)：581条精简记录及随机数进程分段。
- [固定场景清单](baseline_valid_scenes.json)：500个有效场景的实际seed、指令和初始图像哈希。

平台中可查看对应报告、三相机、动作和视频。运行方法见[使用说明](../README.md)。

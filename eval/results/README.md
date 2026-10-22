# 评测结果索引

最新核对：2026-10-22。下列成绩用于队内开发评测与交接，具体分母和协议差异见各报告。

|报告|结果|使用范围|
|---|---|---|
|[Lab20k 6k–8k 四项闭环测评](lab20k-closed-loop-20261022.md)|6k official 249/500；6k PACE 258/500；7k official 249/500；8k PACE 251/500；四项均500有效、81初始化失败补seed、0运行错误|RoboTwin 50×2×5；含原始请求、runtime、checkpoint/source SHA、逐组/逐任务/逐回合轻量证据和离线复算；官方50步与PACE协议需分开解释|
|[Lab q/v-LoRA r256 与 H0](lab-qv256-h0-20261016.md)|Lab20k 249/500（49.8%）；H0-15k 145/500（29.0%），各500完整、0超时零分；含Lab8k/14k/16k/18k趋势|相同原E0固定500键，各自own norm/FP32；训练配方及硬件/RGB/RNG差异单列，附逐任务/配对CSV和离线复算；RETAIN尚在进行|
|[E1 15000步](e1-15000-completed-20261013.md)|51/500（10.2%）；500完整执行，0超时零分|E0冻结50×2×5；相对关节、E1专属norm、FP32；原输入及1500张初始RGB一致，训练端完整动作审计仍待补齐|
|[F1 15000步](f1-15000-completed-20261012.md)|49/500（9.8%）；499完整执行 + 1超时零分|E0冻结50×2×5；绝对关节、FP32；精度/场景/RNG限制见报告|
|[基座与E0同seed对照](baseline-e0-same-seed-20261010.md)|基座49/500（9.8%）；E0 54/500（10.8%）|500相同task/setting/seed，优先用于本轮汇报；基座含4超时0分，输入/RNG/专家协议并非完全一致|
|[E0 15000步](e0-15000-final.md)|54/500完整策略回合|保留旧385+新115阶段，说明精度、专家封装与历史中断|
|[历史原始基座](baseline.md)|52/500完整策略回合|原始500场景；与E0有88键不同，不用它直接替代同seed对照|

## 查任务、seed和录像

- [逐任务成绩JSON](baseline-e0-same-seed-20261010/by-task.json)、[成功任务图](baseline-e0-same-seed-20261010/successful-tasks.png)。
- [500个配对结果](baseline-e0-same-seed-20261010/paired-outcomes.md)、[103条模型成功记录](baseline-e0-same-seed-20261010/successful-episodes.md)（基座49+E0 54，双方成功的相同键可各出现一次）。
- [E0失败案例与成功参照](e0-15000-final/error-analysis/index.html)、[4个基座超时部分证据](baseline-e0-same-seed-20261010/timeouts.json)。
- [E0人工视频复核表（2026-10-12）](e0-15000-final/error-analysis/manual-review-20261012/README.md)：14个失败案例已复核，6个成功对照视频待核；含关键截图、日志核对与候选训练建议。
- [对照验收](baseline-e0-same-seed-20261010/verification.json)、[交付链接核验](baseline-e0-same-seed-20261010/delivery-verification.json)、[E0完整验收](e0-15000-final/verification.json)。

视频/远端关键帧链接需本机18788隧道在线。旧基座361个复用回合原本未录像，仅提供已保存的关键帧和轨迹；超时视频只承诺部分记录。

## 方法与数值审计

- [权重精度与交接](e0-15000-handoff-audit.md)：实际BF16保存、FP32执行与训练者FP32原值导出的区别。
- [评测脚本审计](e0-15000-evaluator-audit.md)、[专家配额协议](e0-15000-resume-protocol-plan.md)、[专家外层异常处理](e0-15000-expert-exception-stop.md)。
- [迭代子集验证状态](subset_validation.md)、[推理策略对比状态](inference_sweep.md)：未完成的研究项明确留空，不伪造结论。
- [GitHub交付清单](../交付与提交清单.md)：对应队内任务书的文件清单与验收状态。

`baseline_official.json`是早期任务书规定的文件名，表示我们运行官方基座得到的历史开发结果，不表示主办方已经审核。F1训练和500键开发评测已完成，详见上方最终报告；不同报告的分母与协议差异分别保留。

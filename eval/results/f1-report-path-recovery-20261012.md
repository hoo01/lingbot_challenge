# F1：报告路径与完成摘要修复（2026-10-12）

双卡HF绝对动作F1任务 `a263b1d335d2` 当前仍为 `failed`，不能称为续跑成功。本轮仅排队一次恢复，启动再次失败后未重复排队；两小时定时任务保留，下一轮按默认 `paired_fixed_resume` 做一次安全续跑。

## 已定位的两个问题

1. 分片1完成250条、仿真退出码0；runner对分片目录再次套用全局分片计划，把真实结果误读为0条，误判失败并取消另一分片。已修复为分片runner只读取自己的输出，全局聚合仍读取全部分片。
2. 默认续跑段 `report-path-20261012-0547` 保留原471条后，已完成分片启动时找不到 `summary.json`。准备工具复制了逐回合结果和协议却遗漏完成摘要；CLI在任何新仿真开始前报FileNotFoundError，另一分片随之取消，未新增计分。

两处修复在本地和服务器分别通过5项CPU回归。报告路径还直接核验了真实分片221/250条；独立复制真实完整分片元数据运行生产CLI，缺少摘要时退出2，补齐原摘要后退出0，未启动模型、仿真或专家。

## 结果与证据

- 保留471个唯一计分：470完整、1原超时0分、49成功。所有原记录字节不变，已结算普通失败和超时没有重跑。
- 当前分片1的摘要由原完成源目录按字节复制，SHA为 `24cae8675f554edbea6e3a2e3aff658414356146b2e141d8a683504490b5eac3`，没有重建或伪造成绩。
- 分片0下一未结算固定键仍是 `stack_blocks_three/demo_clean/100011`，旧500动作后中断证据保留。分片1已完成，无剩余键。
- 模型、固定E0 500键、instruction、parameters、policy_episode_index、绝对关节norm、FP32、attention、渲染、两路录像及超时规则均未更改；专家仍关闭。原模型RNG流未恢复。

当前active-resume为 `resume-requests/report-path-20261012-0547/manifest.json`，后续仍用Store读取实时状态。下一轮使用已修复的 `incoming/F1-step-15000/frozen-entry-20261011/f1_fixed_resume.py` prepare新唯一segment，核对receipt和verify_resume后queue同一segment；不加 `--transition`。新工具仅在完整分片的原摘要协议、计数与记录完全匹配时复制摘要，并把源摘要SHA绑定manifest。

远端 `artifacts/runner-report-repair-20261012-0547/` 保留原runner、原准备工具、两组CPU日志、`diagnosis-verified.json`、`summary-repair-verified.json`；旧段和失败新段及所有录像未删除。本地轻量回执在 `artifacts/f1-15000-eval-20261010/report-path-recovery-20261012/receipt.json`。

独立Lab新训练本轮四卡均587步、loss有限、图像修复标记与自己的norm正确、磁盘约3.03TB空闲；soarm101仍保持maintenance和19条计分，没有清除维护或恢复它。


## 2026-10-12 07:50 定时检查续跑（覆盖前段快照）

本轮先实时确认 F1 仍 failed、471 个唯一计分（470 完整、1 原超时0分、49成功），无其他活动任务或 GPU 计算进程。核对报告修复与完成摘要修复的代码 SHA、CPU 回执，并取得两卡租约后，使用已修复的默认 `paired_fixed_resume` 创建 `report-fixed-20261012-0750`，通过 `verify_resume` 后只排队一次，没有再修改评测代码或配置。

两分片原221/250条记录逐字节不变，完整分片的原 `summary.json` SHA `24cae8675f554edbea6e3a2e3aff658414356146b2e141d8a683504490b5eac3` 已复制并绑定。下一未结算键仍为 `stack_blocks_three/demo_clean/100011`，分片1没有剩余场景。固定E0清单、instruction、parameters、policy_episode_index、绝对动作norm、FP32、录像和超时规则不变，专家关闭；原模型RNG流未恢复。

启动短检时任务已 running，两个当前服务进程仍在初始化模型。55秒有界核验结束时尚无新 runtime 或动作推进回执，不能宣称评测已正常推进，也没有证据把正常加载直接判成运行失败。保留这个限制，下一次仅查询实时 `Store.job/shards/shard_folder`；当前 active-resume 是 `report-fixed-20261012-0750`，不复用旧段或再排队本段。

轻量回执：本地 `artifacts/f1-15000-eval-20261010/report-fixed-20261012-0750/receipt.json`，远端 `runs/eval-a263b1d335d2/resume-requests/report-fixed-20261012-0750/heartbeat-startup-receipt.json`。独立Lab新训练四卡本轮均1109步、loss有限、图像边界修复和自己的norm正确，仍未到首次2000步保存；soarm保持维护与19条计分，不清除标记或恢复。定时任务继续保留，三项目标都完成或被用户明确结束后才删除。


## 2026-10-12 F1最终完成（覆盖全部旧快照）

实时任务completed，500唯一键严格覆盖冻结E0 50×2×5，100组各5计分，无重复及未处置运行错误。499完整执行+1原超时0分，49成功，9.8%；Clean29/250=11.6%，Randomized20/250=8.0%。结束于北京时间10:12:51。实际FP32/own norm、原instruction/parameters/初始关节和471条原结果字节前缀均已核验；原模型RNG流未恢复，10场景初始像素SHA差异及历史超时缺少index字段均保留，不宣称500完整回合或正式竞赛成绩。

完成报告：[f1-15000-completed-20261012.md](f1-15000-completed-20261012.md)。本地回执 `artifacts/f1-15000-eval-20261010/completed-20261012-1156/receipt.json`，远端 `artifacts/f1-completed-20261012-1156/verification.json`。F1目标结束，不再排队或增加场景。Lab新训练仍2124步正常且2000步保存8份读回通过；soarm维护保留，定时任务继续检查这两个未完成目标。

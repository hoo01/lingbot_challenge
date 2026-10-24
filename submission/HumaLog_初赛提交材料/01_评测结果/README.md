# 评测结果待填

results.pending.json保留官方schema_version=1、team_id、results.clean/randomized以及各50任务attempts/successes结构，team_id为字符串1380507。null表示待完成，不是0次测试，也不是0成功。

当前没有本次提交可用的16000模型完整评测计数，故不生成results.json。后续仅从同一HF revision、冻结协议和已核验逐回合账本汇总。计划为每组30，共3000；若未测满按真实次数披露，不伪造或外推。

官网建议每组尽量测100；FAQ允许时间或算力不足时按实测提交，最终以官方复测为准，覆盖更充分者有优先考虑。JSON只写次数，不写百分比，不附加自定义成功率字段。

校验器：在02_代码材料目录运行`python src/validate_results.py ../01_评测结果/results.json`。它检查字段、50任务、团队ID、整数计数及0<=successes<=attempts；不替代逐回合来源和模型身份审计。pending文件必须被它拒绝。

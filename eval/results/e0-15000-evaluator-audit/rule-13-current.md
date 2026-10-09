Source: https://tianchi.aliyun.com/v3/proxy/competition/api/race/getDetail?raceId=532514
Checked: 2026-10-09T11:46:59.636938+00:00

### 13. 专家的五次限制是同一 seed 重试专家，还是失败后换 seed 筛选场景？达到五次上限后应如何处理、是否计入 JSON 的attempts？

专家数据采集的五次限制按采集发起次数计算，与 seed 无关：每次尝试均可更换 seed 以筛选合适的场景随机化，也可同 seed 重试，但每次重试均计入配额。达到五次上限后，该场景不得再发起专家采集，但策略评估不受影响，应继续完成 policy 测试并如实记录成绩。

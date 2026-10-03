# 蚂蚁灵波具身大模型挑战赛 — 常见 Q&A

> 来源：天池大赛 › AI大模型赛 › 大赛详情
> 即日起开始初赛，**10 月 26 日 24:00 初赛作品提交截止**

---

## 💡 赛题相关

### 1. 请问训练和评测的具体要求是什么？

选手需基于 LingBot-VLA 2.0 预训练模型完成训练与调优，训练数据统一使用 RoboTwin 2.0 中 Aloha-AgileX 的 50 个任务 clean 数据，每个任务 50 条，禁止使用 randomized 数据参与训练。

选手可通过调整模型训练参数、优化训练策略、改进数据处理方式、调整模型结构或推理策略提升性能。最终需按照官方评测标准，提交 Aloha-AgileX 50 个任务在 clean 和 randomized 两个 setting 下的测试结果，每个任务测试 100 次。

> **注意：randomized 数据只能用于测试，不能参与训练，也不能用外部数据！**

### 2. 初赛阶段，真机创新任务分值占比高吗？

真机创新 Demo 作为初赛评审的补充材料，不设置固定分值，主办方将结合 Demo 完成质量作为筛选参考。有真机落地能力的同学都非常建议大家做可选任务，提交真机 Demo。

### 3. 决赛真机是否需要自己准备？

决赛阶段由主办方统一提供。

### 4. 最终提交的模型权重是否必须为全量权重，可以提交 LoRA 训练后合并回基座模型的权重吗？

选手可以上传 LoRA 训练后 Merge 回基座的全量权重，不过类似这样的改动需要在 Config 文件或者 Readme 文件中说明清楚，方便评委进行评测。

### 5. 评测结果 JSON 文件是否必须在 CUDA + cuRobo 环境下生成，如果使用 AMD 赛事算力，在 ROCm + MPLib 环境下生成评测结果，应该如何提交？

正常提交，在 Readme 中说明清楚评测环境等细节以保证评委可以复现即可。

### 6. 是否可以对 LingBot-VLA 2.0 进行较大幅度的模型结构改造？是否有必须保留的模块或参数比例要求？

可以。只要基于 LingBot-VLA 2.0 进行改进，即主干模型从 LingBot-VLA 2.0 初始化即可。赛事没有要求必须保留特定模块或满足特定参数比例。

### 7. 是否允许使用其他公开的预训练模块或权重，例如 vision encoder、depth/segmentation encoder？

允许使用其他公开的 vision encoder、depth/segmentation encoder。为便于主办方审核和复现验证、保证比赛公平性，需在 README 中注明相关模块或权重的来源及使用方式。

### 8. 是否允许针对不同任务使用 task-specific 的 LoRA、adapter、head 或 checkpoint，并在推理时根据 task instruction 或 task ID 进行路由？

不允许根据任务 ID 提前进行路由。这种方式在 instruction following 方面不公平。

### 9. 推理阶段是否可以使用多个 checkpoint/model 做 ensemble，或加入 temporal ensemble、replanning、action smoothing、confidence gating 等模块？

不允许根据题目相关信息（如任务 ID、instruction 等）进行确定性的路由等推理阶段增强。最终推理时只能使用一份权重。

### 10. 测试次数"按实测填写"是否意味着每个任务、每个 setting 不足 100 次也可以提交，还是仍必须完成各 100 次？

建议尽量完成每个任务、每个 setting 的 100 次测试。若受时间或算力限制未能测满，最终以官方复测成绩为准。

### 11. Clean 和 randomized 分别使用 seen 还是 unseen 指令？

最终提交一份在 50 个任务上进行 cotrain 的权重，并分别在 `clean2clean` 和 `clean2random` 两种 setting 下测试。其中，`clean2clean` 对应 seen，`clean2random` 对应 unseen。

### 12. Clean/randomized、各任务如何加权？最终排名采用自测成绩还是官方复测成绩？

最终成绩以官方复测为准。

### 13. 专家的五次限制是同一 seed 重试专家，还是失败后换 seed 筛选场景？达到五次上限后应如何处理、是否计入 JSON 的 attempts？

专家数据采集的五次限制按采集发起次数计算，与 seed 无关：每次尝试均可更换 seed 以筛选合适的场景随机化，也可同 seed 重试，但每次重试均计入配额。达到五次上限后，该场景不得再发起专家采集，但策略评估不受影响，应继续完成 policy 测试并如实记录成绩。

### 14. 能否用官方 Robotwin 权重作起点？

参赛选手需要统一使用 lingbot-VLA-v2-6b。

### 15. 初赛评测方式：是由官方用我们提交的 checkpoint 重新跑评测，还是直接认可我们提交的 results.json？若重跑，用的是什么接口——固定单一 ckpt 路径，还是允许我们自己的 eval.sh？

在 Readme 中说明清楚评测环境等细节以保证评委可以复现即可。

### 16. 推理限制：初赛仿真评测是否限制单次推理时间或显存上限？（决赛边缘算力有显式限制，初赛这边想确认是否有同样约束）

没有限制

### 17. 测试时适应（TTA）：无梯度、不改权重的测试时适应（如多步滚动预测、多 seed 投票）是否属于被允许的"推理策略"范畴？

不允许多seed投票

### 18. 关于选手在数据处理上的规则边界以及具体Case

训练过程中的visual/state augmentation是可以的，但不能重新生成数据或者做额外的rollout。

1. 视觉上对亮度、颜色等加上噪声（可以）
2. 机器人状态扰动（可以）
3. 模型训练机制中的噪声（类似diffusion类模型训练时的加噪再去噪过程，不额外生成demonstration）（可以）
4. 改变光照，桌面高度等外观后重新生成observation，不使用官方的randomization（不可以）
5. 基于simulator做额外的rollout训练还是所有训练supervision都必须严格按照clean demonstration（不可以基于 simulator 做额外 rollout 训练）

---

## 🔗 资源相关

### 1. 数据集在哪里？

训练数据统一使用 RoboTwin 2.0 中 Aloha-AgileX 的 50 个任务 clean 数据，每个任务 50 条，禁止使用 randomized 数据参与训练。

数据集链接：https://huggingface.co/datasets/TianxingChen/RoboTwin2.0

### 2. 如何领取 AMD 算力？

详细情况可以参考链接：

GitHub - AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09: Embodied-AI-Challenge-AMD-Platform-2026-09

### 3. 数据集需要下载 LeRobot 格式，还是下载原始版本？

下载的数据格式没有要求，选手可以自行选择。

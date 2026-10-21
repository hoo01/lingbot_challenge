# R3 强视觉增强与 EMA 全量 SFT 训练计划

> 计划日期：2026-10-21
> 状态：`draft_pending_implementation_capacity_resume_and_pace_eval`
> 性质：新的训练阶段，不是原 4090 Full SFT 的精确断点续训

## 0. 给执行同学的结论

这是一份**冻结训练意图的设计文档，不是可直接运行的启动说明**。截至本文日期，仓库内还没有R3的
训练配置、增强/过滤/EMA实现、预检脚本和启动脚本。不要把旧路线命令改几个参数后直接开长训。

| 项目 | 当前结论 |
| --- | --- |
| 训练硬件 | 使用已经验证过的`4×RTX 4090 48GB`；普通24GB 4090不满足本配方 |
| A100 | 不需要；除非实际4090只有24GB，或团队坚持让完整FP32 EMA常驻GPU |
| 父权重 | 固定为第2节的Lab20k Full SFT +6000 FP32权重 |
| 训练性质 | 权重热启动、新建AdamW的新阶段，不是+6000训练状态的精确resume |
| 正式预算 | 4000次optimizer update，global batch 128 |
| 训练启动阻塞项 | R3源码/配置落库、父模型与数据验收、20-step容量门检、完整保存恢复门检 |
| 评测阻塞项 | PACE源码、参数与258/500逐回合回执；它不阻塞训练实现和容量测试 |

职责边界：训练执行者负责实现、门检、长训、恢复、导出和证据；评测负责人负责提供并冻结PACE、
评测键与采用阈值；任何改变父权重、数据范围、global batch、LR、增强概率、过滤范围或EMA开关的
提议，都必须先修改本文并由负责人确认，不能在正式run中临时决定。

## 1. 目标与核心判断

本轮目标不是继续比较 `+6000`、`+7000`、`+9000` 等相邻检查点，而是在当前已成熟的
Lab20k Full SFT 权重上启动一个**面向视觉泛化的新训练阶段**，争取同时保持 Clean 能力并提高
Randomized 成功率。

现有闭环结果表明：

| 阶段 | 结果 | 可支持的判断 |
| --- | ---: | --- |
| Lab r256 8k | 186/500，37.2% | 仍处于明显欠训练阶段 |
| Lab r256 14k | 238/500，47.6% | 相比 8k 出现主要跃升 |
| Lab r256 16k | 243/500，48.6% | 进入平台区间 |
| Lab r256 18k | 237/500，47.4% | 相邻步数并非单调提升 |
| Lab r256 20k | 249/500，49.8% | 本组固定 500 中最高，但仍属平台波动 |
| 原 Lab20k，同协议 2000 回合 | 970/2000，48.5% | Full SFT 的父模型基准 |
| 4090 Full SFT +5000，同协议 2000 回合 | 1075/2000，53.75% | 整套 Full SFT 配方取得 +5.25 个百分点收益 |
| Lab20k Full SFT +6000，固定步长 | 249/500，49.8% | 当前本地已归档的固定 `execute=50` 结果 |
| Lab20k Full SFT +6000，PACE自适应步长 | 258/500，51.6% | 当前500有效回合面板最高观测值；比同权重固定步长多9次成功（+1.8个百分点） |
| 4090 Full SFT +7000，固定步长 | 249/500，49.8% | 与+6000固定步长相同，未显示继续加步的收益 |

主要事实是：原 Lab 在 14k 之后约为 `47.4%–49.8%` 的平台；后续明显收益来自合并 LoRA、
开放完整学生参数、改变优化器/增强/采样等**阶段性配方变化**。同一+6000权重仅改变推理控制协议，
观测值也从49.8%升到51.6%，说明训练权重和闭环执行策略必须分开优化、分开归因。现有证据不支持
“固定配方多训练几千步就会稳定变好”。与此同时，已观察到的 Clean 与 Randomized 成绩仍有约
30–40个百分点差距，因此本轮把外观鲁棒性而不是训练步数本身设为主变量。

PACE的258/500目前来自评测端记录。由于本仓库尚未归档其算法、参数、逐回合结果和不可变源码版本，
本文只把它称为“当前最高观测协议”，不把+1.8个百分点表述为已经证明可复现的稳定收益。相同的
581次尝试和81次初始化失败提示两次评测可能使用了相同回退键，但仍必须用实际
`task/setting/seed`交集和配对输赢表确认。

历史+5000的53.75%来自每组20个有效回合、总计2000回合的另一套评测，数值虽更高，却不能与本面板
500回合结果直接排序。因此“最高”在本文中特指截图所示500有效回合面板，不代表全部历史协议的
绝对最高分。

本轮主假设为：

> 在仅使用官方 Clean demonstration 的前提下，以更强但保持多视角/时序一致性的视觉增强、
> 可审计的数据质量过滤、较小语言学习率以及训练期 EMA，能够提高外观变化下的策略稳定性，
> 并避免只靠延长原 Full SFT 日程造成的晚期漂移。

训练 loss 下降不等于该假设成立。只有冻结模型后的闭环 `check_success()` 结果才能决定是否采用。

## 2. 父权重与实验身份

默认父权重改为当前已固定revision、可完整核验且对应500回合面板最高组合的私有仓库权重：

```text
HumaLog/Lab-QV256
revision: ee6a059ce178641fab3f2d87ee2296accc49e812
experiments/lab20k_full_sft_20261016/checkpoints/global_step_6000/hf_ckpt_fp32/
```

对应当前部署基线为上述+6000权重加评测端PACE自适应步长。训练只继承+6000模型参数；PACE是闭环
推理/执行协议，不进入优化器状态，也不能被写成训练checkpoint的一部分。

它是从未做 RETAIN 插值的 Lab20k 稠密 FP32 权重出发、经过 4090 Full SFT 得到的完整学生模型。
父模型必须绑定 Lab 路线自身资产：

```text
norm SHA256:
b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d

robot config SHA256:
1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b
```

父仓库的不可变HF revision已经由现有交付固定；正式启动仍须按发布清单核对14个权重分片、norm、
robot config和processor/tokenizer，不得改用浮动的`main`。PACE的代码commit、完整参数、状态机
重置时机和逐回合评测回执必须另行归档；仅有“自适应步长”名称不足以复现当前最高结果。

模型构造与部署继续绑定官方上游：

```text
Robbyant/lingbot-vla-v2
upstream revision: be969b8fd117fb70550c5d4bf4bc328211b5b1b6
```

R3训练补丁自身的Git commit当前为`TBD`，这是训练启动阻塞项。实现完成后必须把commit和实际涉及的
训练入口、数据、增强、优化器、EMA、checkpoint文件SHA写入预检回执；不能只记录YAML的SHA。

当前+6000 HF交付只包含推理模型，不包含该阶段的AdamW、scheduler、RNG或DataLoader状态。因此本轮：

- 从父模型的 FP32 参数和持久 buffer 初始化；
- 在FSDP切分前后核对全部1672参数和36个buffer，禁止先载入BF16模块再提升回FP32造成不可逆舍入；
- 新建 AdamW、scheduler 和训练 RNG；
- `stage_step` 从 0 重新计数；
- 不得将本轮 `stage_step_4000` 写成原路线的 `global_step_13000`；
- 运行身份固定使用 `r3_aug_ema_fullsft_from_lab20k_sft6000_<timestamp>`；若EMA门检失败，则固定使用
  `r3_aug_fullsft_from_lab20k_sft6000_<timestamp>`。

父模型可使用现有下载器取得并校验：

```bash
python train/lab20k_full_sft/download.py --output "$PWD/models/lab20k-sft6000"
```

该命令只准备父推理权重和绑定资产，不会生成R3优化器状态，也不是R3启动命令。

若启动前找到其他候选的完整模型、Adam、数据状态与完成回执，也应另立变更记录，不能在不改本文
身份、基线和评测配对设计的情况下替换父权重。

## 3. 合规边界

本轮训练数据严格限制为官方 RoboTwin 2.0 Aloha-AgileX：

```text
50 个任务 × 每任务 50 条 Clean demonstration = 2500 条轨迹
```

允许：

- 对已有 Clean RGB 做视觉增强；
- 使用同任务官方 `seen_instructions` 池中的等价指令；
- 根据 demonstration 自身的 jerk、平滑度、动作能量和夹爪变化做过滤或加权；
- 使用训练期 EMA，最终导出一份权重；
- 仅沿用+6000父路线已经使用的冻结深度、未来深度与视频教师，不新增教师。

禁止：

- Randomized 数据以任何形式进入训练、norm、过滤阈值或样本权重计算；
- 外部机器人数据、额外仿真 rollout 或重新渲染 observation；
- 根据任务开发成功率、任务 ID 或 Randomized 失败情况建立任务特定训练路由；
- 最终推理加载多份权重或按任务切换模型。

Randomized 只在候选和推理协议冻结后用于最终测试，不用于本轮 checkpoint 选择。

## 4. 数据与动作契约

保持既有 Lab 路线定义，不在本轮重新估计 norm：

```text
dataset format: official_raw_hdf5
source manifest SHA256: 911e7ea61baecd505368fe6d5eb70052827b989ba91f93b6d6d13d331df28e78
training seed: 42
```

- 输入为三路当前 RGB、当前 14 维机器人状态和官方指令；
- RGB 边界为原始 `uint8/float [0,255]`，processor 只缩放一次；
- 模型预测 H50，10 个 flow 积分步；
- 12 个机械臂关节标签为 `q[t+h]-q[t]`，使用逐 horizon MeanStd；
- 两路夹爪为绝对位置，使用 MinMax；
- 每个 horizon 的相对关节恢复均加同一个当前 `q[t]`，不得沿时间累加；
- 无效 canonical 维度与越界尾部不计动作损失；
- 冻结深度、未来深度和视频教师继续保持 `eval()` 与 `no_grad()`。

指令增强只允许从该demonstration已有的官方`seen_instructions`数组中随机取一条，不生成新文本、
不调用外部模型改写。Sampler、DataLoader、增强随机流和指令选择的状态都必须进入完整checkpoint；
恢复后下一批样本身份和随机选择必须可验证。

### 4.1 数据质量过滤

正式训练前生成不可变 `data_filter_manifest.json`，至少记录：来源文件 SHA、过滤规则、逐条排除索引、
排除前后有效窗口数以及生成脚本的源码提交。

第一项候选过滤为已经定位的原命令突跳：

```text
task: place_shoe
episode: 45
excluded zero-based starts: 95..144
reason: H50 future target crosses the 144 -> 145 joint-command jump
```

这只排除 50 个窗口，不删除整个任务或整条 episode。`95..144`为两端均包含的整数区间。R3 v1
默认只启用这一项过滤；其它jerk/Z-score、指令冲突或整条episode过滤均为关闭状态。若要增加过滤，
必须先修改本文、补齐原始索引/阈值/SHA/边界样本审计并重新冻结配置；不得在启动机器上临时开启。

R3 v1采样使用过滤后的逐帧均匀分布，不启用动作能量、夹爪切换或任务难度加权，不得沿用基于闭环
弱任务名单生成的任务权重。这里是有意不继承+6000交付中的`source-task_weights.csv`；该CSV只保留作
父路线来源证据，不能传给R3 sampler。所有过滤后eligible start在全局池中等概率，分布式sampler负责
确定性shuffle和各rank不重叠切分。若未来增加加权，应作为另一版实验处理。

## 5. 视觉增强配方

本轮增强在训练 worker 内完成，不写回或重新生成 observation。多相机、当前帧和未来教师帧共享
同一个分支以及同一组brightness/contrast/saturation/gamma/sigma标量，保持视角与时序一致。
高斯噪声场对每张图独立采样，但都使用由该训练样本增强seed派生的可复现局部生成器；不能读取或
修改进程全局NumPy/PyTorch随机状态。

R3 v1稳态分支固定为：

| 分支 | 概率 | 参数 |
| --- | ---: | --- |
| 原图 | 10% | 解码、resize和必要预处理 |
| 色彩增强 | 40% | brightness 0.7–1.3；contrast 0.6–1.4；saturation 0.5–1.5 |
| 色彩 + gamma | 25% | 先使用“色彩增强”同范围参数，再用gamma 0.85–1.15 |
| 色彩 + 弱像素噪声 | 25% | 先使用“色彩增强”同范围参数，再加unit空间高斯噪声sigma 0.003–0.01并clamp |

前 200 次 optimizer update 采用线性增强课程：从旧稳态配方
`25/50/12.5/12.5` 过渡到上述 `10/40/25/25`。增强内部显式 `/255`，返回时显式 `×255`；
禁止根据像素最大值猜测输入是否为 unit 图。

精确定义：以已完成optimizer update数`u`计，`alpha=min(max(u,0)/200,1)`，每个分支概率为
`p(u)=(1-alpha)*p_old+alpha*p_new`。正式第一个batch使用`u=0`的旧概率，完成第200次更新后达到
新概率并保持不变。brightness/contrast/saturation/gamma/sigma均在各自闭区间内均匀采样；所有
概率和参数必须写入冻结配置，禁止依赖库默认值。

算子顺序固定为`uint8/raw255 → unit float → brightness → contrast → saturation → gamma/noise → clamp[0,1]
→ raw255 float`，之后才进入原processor。实现必须用固定输入保存增强前后golden tensor及SHA，避免
torchvision/PIL版本差异造成队友机器上的隐式变化。

本轮不临时加入裁剪、旋转、遮挡、背景替换或几何变换，以免破坏腕部相机与动作标签的空间对应。

## 6. 模型、精度与优化器

未在本文列出的模型、MoE、教师、辅助loss和数值配置，以历史
[`source-config.json`](../evidence/lab20k_full_sft_20261016_sft6000/evidence/source-config.json)
为基线（SHA256 `1531acd8d3b7185b3c472dfa02f0d9bbc68d4721909eb380f03ee4ccd3b84772`）。该文件含旧机器
绝对路径，只能作为值的证据，不能直接执行。

R3冻结配置相对该基线只允许改变：父权重路径、输出/run身份、4000步日程、四组LR、warmup/floor、
过滤后逐帧sampler、本文增强、EMA开关与checkpoint频率。预检必须输出机器可读配置diff；出现白名单
以外变化时拒绝启动。既有深度/未来深度/视频loss系数、MoE/router、H50、图像尺寸、教师结构和
Action FP32等保持不变。

### 6.1 可训练范围

全部学生参数继续训练：

- Language：Q/K/V/O、MLP、embedding 等完整语言学生参数；
- Vision encoder 与 merger；
- 36 层 Action Expert，包括 MoE/router/shared expert；
- state/action/time projection 与 alignment 模块。

预检结果必须等于父模型的`6,375,906,359`个参数、`1672`个参数张量和`36`个持久buffer；否则拒绝
启动。不重新插入LoRA adapter。

### 6.2 分布式与数值配置

| 项目 | 计划值 |
| --- | --- |
| 硬件 | 4×RTX 4090 48GB |
| 并行 | FSDP2 full shard |
| 每卡 microbatch | 16 |
| gradient accumulation | 2 |
| global batch | 128 |
| gradient checkpointing | 开启 |
| FP32 master / 保存 | 开启 |
| Language / Vision 计算 | BF16 mixed precision |
| Action blocks | FP32 |
| gradient reduction | FP32 |
| gradient clipping | global norm 1.0 |
| seed | 42 |
| DataLoader | 每rank 8 workers，共32；`prefetch_factor=4`、`pin_memory=true`、`drop_last=true` |

“4090 48GB”是硬门槛，不是普通零售24GB卡的泛称。启动前保存
`nvidia-smi --query-gpu=index,name,memory.total --format=csv`输出；4张卡必须型号一致且每卡可用显存
满足48GB路线。既有实测为micro16/accum2峰值44.556GiB、均值21.821秒/完整更新；micro32已OOM。

### 6.3 AdamW 与学习率

本轮新建 AdamW：

```text
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0
foreach = false
fused = false
```

全部学生参数按对象身份去重，恰好进入一个参数组。计划 4000 次 optimizer update，warmup 200，
随后 cosine 衰减至各组峰值的 10%。

| 参数组 | 峰值 LR | 第 4000 步目标 LR |
| --- | ---: | ---: |
| Language | 2e-6 | 2e-7 |
| Vision / merger | 5e-6 | 5e-7 |
| Action Expert | 8e-6 | 8e-7 |
| Projection / alignment | 8e-6 | 8e-7 |

较小 Language LR 用于限制已经适配的指令表示漂移；Vision 与动作侧承担本轮主要更新。一次更新内每个
microbatch loss 除以累积次数，scheduler 只前进一步。

精确定义：`stage_step=0`时LR为0；第1至200次成功optimizer update做线性warmup，第200次达到峰值；
第201至4000次按cosine降到表中目标LR，第4000次恰好取目标值。发生overflow而未执行optimizer update
时，`stage_step`、增强课程、EMA和scheduler都不得前进。实现必须记录每个参数组的实际LR。

## 7. EMA 设计与容量门禁

主候选使用训练期学生参数 EMA：

```text
ema_decay = 0.999
update_after_optimizer_step = true
initial_value = online weights at stage_step 0
bias_correction = false
```

EMA 必须覆盖全部浮点学生参数，非浮点 buffer 使用最终在线模型值。不能只平均部分任务或根据评测结果
选择性平均。最终分别导出在线权重和一份 EMA 权重，两者都是单一完整模型。

第`u`次成功optimizer update后执行一次`ema = 0.999 * ema + 0.001 * online`。EMA张量按完整参数名
绑定并与FSDP分片逐一核对；不得通过参数遍历顺序猜测对应关系。EMA状态必须进入训练checkpoint并在
恢复测试中验证下一步更新一致。正式run开始前，门检回执必须把`ema_enabled`冻结为true或false；
长训过程中不得因为速度或loss临时切换。

4090 的既有 `micro16/accum2` 峰值 allocated 约 44.6GiB，不能假设额外 FP32 EMA 常驻 GPU 一定可行。
R3 v1只接受FSDP参数名绑定的CPU FP32分片EMA。其它GPU/CPU复制或低精度EMA实现必须先修订本文，
不能由执行者临场替换。

20-step profile 必须记录显存、主机内存、step time 和 EMA 更新耗时。如果 EMA 使峰值显存越界、
主机内存不足或稳态 step time 增加超过 15%，按预先登记的 fallback **关闭 EMA**，实验改名为
`r3_aug_fullsft`；不得继续使用 `EMA` 名称或伪造平均权重。不能为了容纳 EMA 悄悄降低 global batch。

优先实现CPU FP32分片EMA；禁止在每个rank复制一份完整25.5GB EMA。若这一实现未在截止前通过门检，
默认使用预登记fallback `ema_enabled=false`，继续在4×4090上训练，不为EMA单独迁移A100。

## 8. 启动门禁

正式启动前，仓库中至少应出现并提交以下可执行物；文件名可以调整，但职责不能缺失：

```text
train/r3_aug_ema_full_sft/config.yaml
train/r3_aug_ema_full_sft/launch.sh
train/r3_aug_ema_full_sft/preflight.py
train/r3_aug_ema_full_sft/data_filter.py
train/r3_aug_ema_full_sft/augmentation.py
train/r3_aug_ema_full_sft/ema.py             # 仅当ema_enabled=true
train/r3_aug_ema_full_sft/tests/
```

`launch.sh`必须至少支持只读`--check`和短程`--smoke 20`，正式模式要求显式`--run-id`，并拒绝覆盖已存在
目录。当前这些文件尚不存在，因此本文此刻不能直接交给队友执行长训。

以下训练门禁全部通过后才能正式启动：

1. 父 HF revision、全部模型分片、norm、robot config、processor/tokenizer逐文件验收；
2. 完整模型载入后参数张量数、元素数和36个buffer与父回执一致；
3. 固定输入下保存前后 normalized `[50,55]` 和 physical `[50,14]` 动作通过容差审计；
4. 真实 HDF5、暗图/亮图、raw255单缩放、多相机顺序、动作往返和双 worker 测试通过；
5. 过滤 manifest 及增强分支的确定性重放通过；
6. 20次完整更新无 OOM、NaN/Inf，峰值显存和稳态耗时已记录；
7. 新阶段完整 checkpoint 保存、逐文件读回和新进程恢复后再更新一步通过；
8. SwanLab 项目、run name、source commit、dirty状态、配置SHA和数据身份已写入运行元数据；
9. 输出盘与远端归档满足第10节容量要求。

工程门检产生的更新全部丢弃，正式运行必须重新从冻结父权重启动。

PACE回执不是训练启动门禁：训练实现和4000步长训可以在评测负责人整理PACE时并行进行。但在第11节
的模型比较开始前，PACE实现和评测键必须冻结，否则只能做固定`execute=50`的工程验收，不能宣布
R3优于父模型。

## 9. 正式训练与停止条件

正式预算为 4000 次 optimizer update。增强课程、LR、采样、过滤和 EMA 开关一旦启动不得中途修改。

每1000步保存一个可恢复训练点，正式交付保留2000和4000里程碑：

```text
stage_step_1000  完整训练状态，恢复保护点
stage_step_2000  完整训练状态 + FP32在线模型导出，正式里程碑
stage_step_3000  完整训练状态，恢复保护点
stage_step_4000  完整训练状态 + FP32在线模型导出 + EMA导出（若门禁采用），最终里程碑
```

1000/3000保护点只能在下一完整点通过本地读回、新进程恢复和远端SHA回执后删除。保存不得与下一次
optimizer update并发修改同一状态；所有rank成功并写入完成标记后，该checkpoint才算存在。

每 250 步记录训练标量，但不因短窗口 loss 波动提前改变配方。以下情况立即停止并保留证据：

- loss、梯度或动作出现 NaN/Inf；
- 固定动作审计越界或norm身份漂移；
- 训练参数、buffer或优化器无法从新进程恢复；
- 数据过滤/增强不再符合冻结 manifest；
- 输出空间不足以原子完成下一个完整 checkpoint；
- EMA或异步归档使训练资源超过已验收上限。

没有上述故障时，默认完成4000步。离线 loss 变低或单个验证batch变好均不能提前宣布训练成功。

按既有4090路线21.821秒/完整更新估算，4000步纯训练约24.2小时；EMA允许的15%开销上限对应约
27.9小时。加初始化、四次保存、两次在线导出和一次可选EMA导出，机器预算按26–31小时估计，建议
至少租36小时。该数字是排期预算，不替代20-step实测；若实测推算超过租期，必须在正式启动前汇报。

## 10. 保存、空间与远端备份

一份完整FP32学生约25.5GB。关闭EMA时，含在线模型、Adam一二阶矩、RNG与数据状态的完整训练点
预计约76.5GB（约71.3GiB）；启用EMA时还必须保存约25.5GB EMA状态，完整训练点预计约102GB
（约95GiB）。不能只导出最终EMA而省略恢复状态。实际值以20-step保存/恢复门检为准。

门检后以实际GiB记：完整训练点`D`、一份在线FP32导出`E`、EMA导出`A`（关闭EMA时为0）、与输出盘
同盘的父bundle`P`（父模型在其它盘时为0）。为同时保留2000和4000里程碑，正式启动时输出文件系统
空闲空间必须至少满足：

```text
base = 2*D + 2*E + A + P
required_free_gib = base + max(20, 0.10*base)
```

按当前估值且父bundle同盘，启用EMA约需314GiB空闲；关闭EMA约需236GiB。若实际保存实现需要额外
聚合临时副本，应把该副本加入公式。不得仅因远端“正在上传”就提前计算为可释放空间。

采用串行保全：

1. 本地写完每个1000步训练点，逐文件读回并生成完成标记；
2. 异步上传私有归档并按实际SHA远端读回；
3. 只有新点恢复验收和远端完整回执都通过后，才允许按保留策略清理上一个保护点；
4. 2000和4000里程碑不得作为普通保护点自动删除；
5. 在线FP32和EMA推理导出分别使用不可变目录，禁止相互覆盖。

若无法保证至少一个旧完整点、一个新临时点、最终导出和安全余量同时存在，则不得启动长训。

## 11. 评测与选模

训练开始前冻结一套覆盖全部50任务的Clean配对键。父`+6000`、R3在线权重和R3 EMA权重必须使用：

- 相同 task、instruction、seed 和初始状态；
- 相同 Lab norm、FP32推理、H50、10 flow steps；
- 相同PACE实现、参数与状态重置语义；另保留固定`execute=50`作为诊断对照；
- 相同超时、初始化错误和 `check_success()` 计分规则。

第一阶段使用 Clean 选模，不能根据 Randomized 成绩选择 step、EMA或推理参数。若资源有限，优先顺序为：

1. R3在线4000 + PACE vs 父+6000 + PACE；
2. R3 EMA4000 + PACE vs R3在线4000 + PACE；
3. 对晋级候选补做固定`execute=50`诊断，区分权重收益与PACE收益；
4. 只有候选和PACE协议全部冻结后再运行Clean/Randomized最终测试。

报告必须同时给出总成功数、逐任务变化、配对输赢、初始化/基础设施错误和置信区间。提升若只集中在少数
任务，不得写成普遍泛化提升。最终提交只能使用一份冻结权重和一套不依赖任务身份的推理协议。

“有实际意义的净提升”和“大面积任务退化”的数值阈值必须由评测负责人在首次R3闭环评测前写入
独立评测请求并冻结；训练执行者不得看完结果后自行定义阈值。采用条件：R3候选在同键、同PACE的
Clean评测中通过该预登记阈值；否则
回退到父+6000权重与已冻结PACE协议。PACE本身必须先用归档代码重放出一致结果，否则51.6%只保留为
历史观测，不作为新训练的验收基线。
Randomized 结果只作冻结后的最终报告，不触发再次训练。

## 12. 风险与解释边界

| 风险 | 控制措施 |
| --- | --- |
| 强增强损害Clean精细视觉 | 保留10%原图、200步增强课程、父模型不可覆盖 |
| 全量训练导致语言能力漂移 | Language LR最低；不凭机器人成功率声称通用VLM能力保留 |
| 新Adam丢失原优化器动量 | 明确登记为新阶段；warmup200；不冒充精确resume |
| EMA造成OOM或严重降速 | 20-step容量门禁；失败即按预登记fallback关闭并改实验名 |
| 数据清洗误删有效行为 | 逐索引manifest、边界样本复核、禁止按开发成功率删任务 |
| 多项变化无法单独归因 | 本轮目标是竞赛候选，不把结果拆解为增强、EMA或过滤的单项因果收益 |
| 固定开发键过拟合 | Clean只用于有限选模；Randomized在候选冻结后使用 |

本轮如果取得提升，只能表述为“R3强增强/数据质量/优化与EMA整套配方优于父模型”。若需要解释单项
贡献，赛后另做受控消融。

## 13. 必需交付物

- R3训练源码、冻结配置、启动器、测试及各自SHA；
- 父HF revision和模型/资产清单；
- PACE源码commit、配置、状态机说明及258/500逐回合回执；
- `data_filter_manifest.json` 与采样概率表；
- 源码commit、dirty状态和环境依赖快照；
- 20-step容量、数值、保存/恢复门检回执；
- `stage_step_2000`、`stage_step_4000`完整状态及远端SHA回执；
- 在线FP32与动作审计/加载回执；EMA启用时还须提交EMA FP32导出及对应回执；
- SwanLab曲线和逐步metrics；
- Clean配对选模结果；
- 冻结候选的Clean/Randomized结果JSON；
- 最终README，披露父模型、LoRA merge历史、全量SFT、增强、过滤、EMA与推理协议。

## 14. 开跑前待补清单

以下项目没有默认猜测值。训练负责人应逐项填入实际回执；标为“训练阻塞”的项目未完成时不得开长训。

| 待补项 | 当前状态 | 影响 |
| --- | --- | --- |
| R3源码commit、配置SHA及`launch.sh --check/--smoke 20` | `TBD` | 训练阻塞 |
| Python/PyTorch/CUDA及依赖锁文件或容器镜像digest | `TBD` | 训练阻塞；不能只写环境名称 |
| 4张GPU的型号、48GB显存回执及20-step峰值/耗时 | `TBD` | 训练阻塞 |
| 原始HDF5实际根目录与manifest逐文件验收 | `TBD` | 训练阻塞；绝对路径写回执，不写死到共享配置 |
| MoGe2、MoRGBD、DINO video教师权重/配置的来源与SHA | `TBD` | 训练阻塞；父HF包不包含这些训练资产 |
| `ema_enabled=true/false`容量门检结论 | `TBD` | 训练阻塞；失败采用无EMA预登记fallback |
| 实测`D/E/A/P`与输出盘空闲空间公式 | `TBD` | 训练阻塞 |
| 私有远端checkpoint归档目的地与读回方法 | `TBD` | 训练阻塞；凭据只放环境/密钥系统，不进Git |
| SwanLab项目 | 计划`Elowynnn/lingbot_cha` | 训练阻塞；若更换须写入配置和回执 |
| PACE源码commit、冻结参数、258/500逐回合回执 | `TBD` | 不阻塞训练；阻塞PACE正式比较 |
| Clean评测键和预登记采用阈值 | `TBD` | 不阻塞训练；阻塞最终选模 |

队友收到本文后，正确的下一步是先实现并提交第8节文件，然后运行只读预检和20-step smoke；不是直接
启动4000步。所有`TBD`应由实际机器和评测回执消除，不能在文档中凭经验补造。

## 15. 依据

- [10月12日至20日训练策略与实验复盘](../reports/training-strategies-20261012-20261020.md)
- [Lab q/v-LoRA r256 与 H0 固定500回合报告](../../eval/results/lab-qv256-h0-20261016.md)
- [Lab20k Full SFT +6000交接](../lab20k_full_sft/README.md)
- [比赛规则解读](../../官方文档/初赛仿真必做任务赛题规则解读.md)
- [比赛常见QA](../../官方文档/蚂蚁灵波具身大模型挑战赛_常见QA.md)

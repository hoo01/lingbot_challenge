# LingBot-VLA 评测

提供 LingBot-VLA v2 / RoboTwin 2.0 的命令行评测和网页平台。默认运行在租用 RTX 5090 上，可通过配置连接其他服务器。

## 最新结果与交付范围

截至2026-10-10，E0与基座同seed补测已结束，自动跟进已暂停；查看报告无需启动GPU任务。

|结果|基座|E0|范围|
|---|---:|---:|---|
|[同seed对照](results/baseline-e0-same-seed-20261010.md)|49/500，9.8%|54/500，10.8%|500相同键；基座496完整策略回合+4超时0分|
|[E0完整报告](results/e0-15000-final.md)|—|54/500，10.8%|500完整策略回合，混合专家筛选协议|
|[历史基座](results/baseline.md)|52/500，10.4%|—|原始500键保留，不等于本次对齐集合|

[结果索引](results/README.md)提供逐任务、成功seed、图表、案例和验收JSON。[GitHub交付清单](交付与提交清单.md)按队内任务书A8/B6逐项列出应交付文件、现有证据和未完成项。当前`eval.sh`是开发复现入口，`setup_eval.sh`复用既有Linux运行环境，并非已验证的空机安装器；子集相关性和受控推理扫描也仍待验证。

## 连接服务器

客户端需要 Python 3.10+、系统 OpenSSH，以及评测服务器的 SSH 登录权限。从仓库根目录运行：

```bash
python eval/connect.py show       # 查看当前服务器和目录
python eval/connect.py check      # 检查 SSH 登录和部署配置
python eval/connect.py tunnel     # 建立隧道，保持终端运行
```

建立隧道后打开 [评测平台](http://127.0.0.1:18788/?server=rented-5090)。网页部署在服务器的 `127.0.0.1:8788`，隧道将它映射到自己电脑的 `127.0.0.1:18788`。评测任务在服务器后台运行；隧道断开后重新运行 `tunnel` 即可。

连接信息在 [configs/servers.json](configs/servers.json)。需要使用自己的 SSH 别名、账号或密钥时，复制为 `configs/servers.local.json` 后修改；此文件已被 Git 忽略。也可用 `--config 路径` 指定配置。密码由 SSH 交互输入；首次连接应核对主机指纹。

## 使用网页评测

1. 在“模型与数据”选择已注册模型，或上传自己的完整模型目录。大文件也可通过 SFTP 放到服务器运行目录的 `incoming/`。
2. 新模型需要注册权重目录、训练 CLI YAML 和归一化 JSON，文件要求见下节。
3. 点击“新建评测”，选择模型、任务、Clean/Randomized 和固定 seeds，可勾选“专家筛选”，然后加入队列。
4. 默认使用双卡并行，也可选择单卡。任务组按固定顺序分配到两张 GPU；每张卡同时运行一个模型服务和一个仿真进程，结果汇总为一份报告。
5. 打开报告查看成功/有效回合、初始化失败、运行错误、三相机图片和动作轨迹，下载分任务 CSV。

录像默认关闭；开启后同步保存 **头部相机与全局 `observer_camera`**，均为 **10 fps、仿真时间 1× 原速**。在回放窗口切换视角会保留时间位置，也可以点击“定位末帧”或打开终帧图片。全局相机用于回放，模型输入仍为头部、左腕、右腕三路图像。

回合在任务成功判定触发或动作达到上限时结束，录像包含实际终态。成功条件由对应 RoboTwin 任务定义。记录中的回合耗时是墙钟时间，视频长度按仿真时间计算。每路视频保存帧数、仿真时间和末帧记录；三相机关键帧和动作记录始终保存。取消任务会保留已完成回合。

### 模型文件要求

微调模型应导出完整、已合并的 HF safetensors 权重，连同实际训练配置和数据统计：

```text
incoming/experiment-001/
  lingbotvla_cli.yaml
  norm_stats_clean.json
  hf_ckpt/
    model-00001-of-00006.safetensors
    ...
    model.safetensors.index.json
    config.json
```

提供完整模型目录、训练 YAML 和归一化 JSON。分片名称和数量以实际导出为准。LoRA 模型先与对应基座合并；同时训练的视觉或投影层参数也应包含在导出中。归一化文件使用训练时的统计。

平台在注册时自动记录文件校验信息，执行前再次核对。原始基座使用上游归一化，微调模型使用自身训练统计。注册后的权重通过软链接引用源文件，请保留上传目录中的原文件。

### 训练、导出与加载兼容

训练 YAML 应包含 `model`、`train`、`data`。平台将它原样保存为 `checkpoints/<MODEL_ID>/original_training_cli.yaml`，另生成推理用的 `lingbotvla_cli.yaml`；分词器和归一化路径指向当前服务器上的资源。运行时按[推理配置](#推理配置)设置精度和计算开关，模型结构、输入尺寸和归一化与导出保持一致。

导出端应验证模型重新加载后的动作一致性。部分位置编码等数值在构建模型时生成；BF16 构建后导出的模型，可能需要在 FP32 加载时恢复相同的构建数值。根据导出验证结果选择加载方式：

|加载方式|适用条件|配置与行为|
|---|---|---|
|`native`（默认）|原始基座，或已通过原生加载动作一致性验证的导出|沿用上游构建与加载方式|
|`bf16_then_fp32`|导出端验证确认需要恢复 BF16 构建时的数值|在加载保存的参数**之前**执行 BF16 → FP32 转换，再严格加载权重；推理仍为 FP32|

使用 `bf16_then_fp32` 时，在 `hf_ckpt/` 中增加 `runtime-construction.json`：

```json
{"nonpersistent_buffer_initialization": "bf16_then_fp32"}
```

平台先完成构建转换，再加载保存的 FP32 参数。该兼容模式使用本平台加载器；原生官方加载器的兼容性由导出端单独验证。原始基座使用 `native`。

新导出按以下顺序验证：

1. **导出一致性**：LoRA 合并前后使用相同观测、噪声和推理设置，比较完整 `50×14` 动作；新进程重载后再与导出前的参考动作比较。记录精度、后端、容差和最大误差；本项目在相同推理条件下的重载验收使用 `atol=rtol=1e-6`。
2. **注册**：网页和 `--register-only` 在 CPU 上检查文件、safetensors 头部和配置，并自动记录文件校验值。导出端的动作审计可放在 `hf_ckpt/action-audit.json` 中随模型保存。
3. **功能评测**：先运行少量 Clean/Randomized 回合，确认模型加载、仿真和报告链路可用，再按计划开展模型效果评测。

大模型可先用 SFTP 上传，再在网页填入服务器路径注册。

## 使用脚本评测

在自己电脑通过配置连接服务器：

```bash
# 原始基座：click_bell 的 Clean/Randomized 各一个回合
python eval/connect.py eval -- --base --tasks click_bell --seeds 100010 --title base-smoke

# 已注册模型：提交到网页队列，由 Worker 执行
python eval/connect.py eval -- --checkpoint-id MODEL_ID --tasks click_bell --seeds 100010 --enqueue

# 注册新模型并返回模型 ID；文件放在所选服务器上
python eval/connect.py eval -- --checkpoint incoming/experiment-001/hf_ckpt \
  --cli incoming/experiment-001/lingbotvla_cli.yaml \
  --norm incoming/experiment-001/norm_stats_clean.json --title experiment-001 --register-only
```

也可 SSH 登录服务器，在 `/root/shared-nvme/lingbot-eval-team` 运行：

```bash
# 已注册模型
bash run_eval.sh --checkpoint-id MODEL_ID --tasks click_bell open_microwave \
  --settings demo_clean demo_randomized --seeds 100010 100011 --title model-eval

# 从完整模型目录注册并评测；路径相对服务器运行目录
bash run_eval.sh --checkpoint incoming/experiment-001/hf_ckpt \
  --cli incoming/experiment-001/lingbotvla_cli.yaml \
  --norm incoming/experiment-001/norm_stats_clean.json \
  --tasks click_bell --seeds 100010 --title sft-smoke

# 50任务 × 2环境 × 2个seed，共200次计划尝试
bash run_eval.sh --checkpoint-id MODEL_ID --all-tasks --seeds 100010 100011 --expert-check
```

`eval.sh` 是 `run_eval.sh` 的同功能入口。CLI 默认按服务器配置使用双卡直接运行，`--parallelism 1` 使用单卡，`--gpu 1` 指定单张卡；`--enqueue` 提交到队列，`--register-only` 注册模型并返回 ID。模型路径和 ID 属于所选服务器；使用网页或 SFTP 上传本地权重后，再运行连接脚本。每次评测创建独立任务和输出目录。

### 双卡调度

服务器配置 `"gpus": ["0", "1"]` 启用双卡，改为 `["0"]` 使用单卡。修改后重启网页管理进程。脚本和网页使用同一调度器。

一次评测按任务交错分成两份；只有一个任务时，按 Clean/Randomized 分组。未发生恢复时，每组seed在同一模型进程中执行，分配表随请求保存。显式恢复会形成新的模型进程段，不能声称整个任务始终使用同一噪声流。各卡独立记录动作、录像、日志和 GPU UUID，报告自动汇总；取消任务会停止两份工作，未处置运行错误会停止该任务尚未完成的部分并保留现场。

每份工作启动一个 seed 42 的模型进程，随后沿用上游连续随机数。比较模型时使用相同的并行度、任务顺序和 seed；单卡与双卡的随机数进程分段不同。

### 推理配置

CLI 和网页共用固定的 `official_defaults_v1` 推理配置。LoRA rank 和训练 batch 保留在训练记录中，评测端加载合并后的完整模型。

|项目|评测时使用的配置|与训练配置的关系|
|---|---|---|
|输入图像|`256×256`|`data.img_size=256`|
|动作与采样|预测 chunk 50、执行 50、flow 10 步|`chunk_size=50`、`num_steps=10`|
|精度|FP32，matmul precision `high`|运行 YAML 设置 `enable_fp32=true`，服务显式关闭 BF16 推理|
|编译|开启 compile|运行 YAML 设置 `use_compile=true`|
|梯度检查点|关闭|运行 YAML 设置 `enable_gradient_checkpointing=false`|
|视觉注意力|FlashAttention 2|运行 YAML 设置 `vit_attn_implementation=flash_attention_2`|
|视觉网格预计算|开启|运行 YAML 设置 `precompute_grid_thw=true`|
|MoE 与设备|上游 fused MoE，全模型 CUDA|视觉、词嵌入及动作模型均在 GPU 上运行|
|动作归一化|基座用上游统计；微调模型用其训练统计|注册时保存对应统计文件|
|策略随机数|上游原生噪声，模型进程启动 seed 42|跨回合连续使用进程随机状态；场景 seed 单独设置|

注册记录的 `inference_overrides` 保存上述五个 YAML 覆盖项；每次运行的 `server-runtime.json` 保存实际精度、设备、采样和构建模式等信息。比较结果时核对这些记录、归一化统计和场景 seed。重启模型会开始新的随机数进程段。

本机诊断中，上游 fused MoE 在固定输入与噪声的重复计算中仍有动作数值波动。动作一致性检查应固定计算后端，并记录重复计算本身的误差。

### 官方代码与统计口径

推理核心直接调用固定上游版本 `be969b8f` 的 [`LingbotVLAv2Server`](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/deploy/lingbot_vla_v2_policy.py)。模型注册、网页、队列、超时保护和构建兼容处理由本项目实现。

仿真端沿用官方任务、观测与动作接口，专家正常路径直接调用固定上游的`setup_demo/play_once/close_env`与成功检查，不再替换抓取搜索、规划器或`move`内部方法。专家通过后，同一任务对象重置同seed并生成策略指令。完整上游批处理入口与本地队列并不相同，正式复测还需冻结接口与配额映射。

仅勾选“专家筛选”或添加`--expert-check`，不等于自动获得下列特定恢复协议。实际按任务的`config`、`protocol.json`和显式恢复清单判定：

|协议|结束与统计口径|
|---|---|
|默认固定候选|按计划seed尝试；成功/完整策略回合，初始化失败、专家拒绝、运行错误另列|
|E0后段的五次预算|按用户批准的每个待补策略回合累计专家发起，跨seed/恢复不清零；耗尽仍执行policy。普通专家异常在已启用的外层计数模式中单列并占预算，策略/写盘/清理错误仍停机|
|基座固定88场景的900秒计0继续|仅显式启用`wall900_zero_continue_v1`且具有可核合法策略前缀的整回合超时记0；`scored=valid+timeout_failures`，成功率分母为scored。超时不称完整有效回合，不换seed或重跑|

正常策略失败计入分母。初始化失败和专家失败的补seed行为必须由已冻结的协议允许，不能为了提高分数自动换场景。E0旧385与新115回合、基座旧412与补88的规则差异见各自报告。赛事FAQ对五次发起计数的说明不等同于确认本地“待补回合”映射。

## 查看输出与处理错误

每个任务写入服务器的 `runs/eval-ID/`。顶层保存请求、汇总报告与 CSV；双卡时，每份工作的下列模型、日志和回合文件分别保存在 `shards/0/`、`shards/1/`，单卡时直接保存在任务目录：

```text
request.json, checkpoint.json, official-defaults-manifest.json, server-runtime.json
process-segment.json
report.json, scores.csv, timing.json
worker.log, server.log, evaluation.log
evaluation/episodes.jsonl, summary.json, execution-events.jsonl
evaluation/<task>/<setting>/<seed>/
  episode.json, trajectory.jsonl, predictions.jsonl
  chunk-*.png, final*.png, [episode.mp4]
```

默认单回合上限900秒、连续240秒没有完整动作块进展即停机。仅上述已授权的固定场景超时协议允许900秒记0继续，240秒无进展或程序/证据错误仍停机。先保存日志、动作、视频及专家发起账，再诊断；恢复必须使用绑定原记录SHA、原seed和预算的显式清单，保留全部旧输出，不通过新建任务或改名目录绕过恢复检查。已结算策略失败或超时0分不重跑。模型重启会形成新的RNG段。

网页打不开时，先检查隧道终端和远端网页进程；任务一直等待时，检查 Worker 是否在线及 GPU 是否被其他任务占用。模型相关问题按发生阶段处理：

|阶段与提示|处理方式|
|---|---|
|注册提示 `file not found`|上传 CLI YAML 和归一化 JSON，并填写所选服务器上的文件路径|
|注册提示 adapter-only / factorized weights|从训练端导出完整合并的 safetensors 模型|
|运行提示注册后的权重或配置已改变|检查文件是否被覆盖；需要评测更新后的模型时，使用新的模型目录重新注册|
|注册提示输入尺寸、chunk 或 flow 配置不符|核对训练与导出配置，准备符合上述推理配置的模型|
|加载时报 missing / unexpected keys 或 shape mismatch|核对基座版本、完整分片和模型结构，按匹配的结构重新导出|
|能加载但动作一致性失败|核对归一化、构建模式、精度、后端与噪声条件，由导出端定位并修复|

## 配置其他服务器

在 `servers.json` 的 `servers` 对象中添加一项，例如：

```json
"other-gpu": {
  "label": "评测服务器 B",
  "ssh": {"host": "eval-server"},
  "reuse_runtime": "/srv/lingbot-runtime",
  "runtime_root": "/srv/lingbot-eval",
  "web_port": 8788,
  "local_port": 18789,
  "gpu": "0",
  "gpus": ["0", "1"]
}
```

`ssh.host` 可使用地址或 `~/.ssh/config` 中的别名；还支持 `user`、`port`、`identity_file`、`preferred_authentications`、`address_family`。当前租用配置使用密码认证和 IPv4。

```bash
python eval/connect.py --server other-gpu check
python eval/connect.py --server other-gpu tunnel
python eval/connect.py --server other-gpu eval -- --checkpoint-id MODEL_ID --enqueue
```

目标选择优先级为 `--server` → `LINGBOT_SERVER` → `default_server`；配置文件优先级为 `--config` → `LINGBOT_SERVERS_CONFIG` → `servers.local.json` → `servers.json`。修改网页使用的服务器列表或端口时，同步服务器端配置。建立目标隧道后，在网页上选择服务器并点“打开”；各台服务器独立保存模型和队列。

## 部署与启动服务

已有部署可直接启动服务。部署到新目录时，安装器复用同机已配置的模型和仿真环境，创建独立网页环境，并共享 GPU 锁：

```bash
# 在目标 Linux 服务器、仓库根目录执行
bash eval/setup_eval.sh --server rented-5090 --check  # 只读预检查
bash eval/setup_eval.sh --server rented-5090          # 安装到配置中的新目录

cd /root/shared-nvme/lingbot-eval-team
bash start_web.sh --with-worker                      # 网页和配置中的各 GPU Worker
# 只查看结果、上传或排队时，可运行 bash start_web.sh
```

使用空目录安装，保留用于软链接复用的原资源目录。新服务器先准备好驱动、仿真器、模型及以下环境和资产，再运行安装器。

|组件|运行要求|
|---|---|
|模型|Python 3.12、PyTorch 2.8.0、transformers 4.57.3、FlashAttention 2.8.3|
|仿真|Python 3.10、PyTorch 2.8.0 / torchvision 0.23.0、NumPy 1.26.4、SciPy 1.10.1|
|仿真/规划|SAPIEN 3.0.0b1、mplib 0.2.1、cuRobo v0.7.8、warp-lang 1.12.0|
|GPU/渲染|32 GiB级别显存、CUDA 12.8兼容栈、NVIDIA Vulkan ICD、OIDN 2.3.3；FP32模型与仿真接近32 GiB|
|资产|LingBot原始权重、Qwen3-VL配置/分词器、RoboTwin embodiments/objects/background_texture、ffmpeg|
|网页|独立CPU环境，依赖见[requirements-web.txt](requirements-web.txt)|

复用目录需包含 `envs/train`、`envs/sim`、`third_party/RoboTwin`、`third_party/lingbot-vla-v2`、`third_party/lingbot-vla-v2-official-be969b8f`、`third_party/oidn-2.3.3.x86_64.linux`、`models/lingbot-vla-v2-6b`、`models/Qwen3-VL-4B-Instruct` 和已验证的 `artifacts/oidn-blackwell.json`。固定源码版本及关键文件哈希见 [upstream-lock.json](configs/upstream-lock.json)。

服务仅监听回环地址，通过 SSH 隧道访问。长期运行可使用 tmux 或进程管理服务；每个运行目录只启动一个网页管理进程。GPU 锁和显存检查会阻止与其他任务争用同一张卡。

CPU测试可在安装网页依赖后，从仓库根目录执行 `python -m pytest eval/tests -q`。

## 实测耗时与排期边界

来自已冻结的回合`seconds`字段；包含该候选的初始化、专家路径（启用时）、策略与收尾，不是单次模型推理延迟。

|样本|回合数|平均秒|中位秒|累计回合耗时|
|---|---:|---:|---:|---:|
|E0完整策略回合|500|185.3|154.4|25.73小时|
|基座新补测中的完整回合|84|412.4|350.0|9.62小时|

新88是历史缺失场景的定向补测，不是随机耗时样本；另4个900秒超时单列。两张卡可重叠执行，累计回合耗时不能直接当日历时长。若仅按E0平均值机械外推10,000个完成回合，累计约514.6回合小时，理想双卡均分约257.3小时；这还没有计入专家拒绝、初始化失败、模型启动/编译、负载不均或停机，不能当作完成时间承诺。计时来源和计算见[交付运行摘要](delivery/20261010/e0-runtime-evidence.json)。

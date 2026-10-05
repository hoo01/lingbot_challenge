# LingBot-VLA 评测

提供 LingBot-VLA v2 / RoboTwin 2.0 的命令行评测和网页平台。默认运行在租用 RTX 5090 上，可通过配置连接其他服务器。已完成的基座评测见[基座结果](results/baseline.md)。

## 连接服务器

客户端需要 Python 3.10+、系统 OpenSSH，以及评测服务器的 SSH 登录权限。从仓库根目录运行：

```bash
python eval/connect.py show       # 查看当前服务器和目录
python eval/connect.py check      # 检查 SSH 登录和部署配置，不使用 GPU
python eval/connect.py tunnel     # 建立隧道，保持终端运行
```

建立隧道后打开 [评测平台](http://127.0.0.1:18788/?server=rented-5090)。网页已部署在服务器的 `127.0.0.1:8788`，隧道将它映射到自己电脑的 `127.0.0.1:18788`。关闭浏览器不会停止评测；隧道断开后重新运行 `tunnel` 即可。

连接信息在 [configs/servers.json](configs/servers.json)。需要使用自己的 SSH 别名、账号或密钥时，复制为 `configs/servers.local.json` 后修改；此文件已被 Git 忽略。也可用 `--config 路径` 指定配置。密码由 SSH 交互输入，不写入配置；首次连接应核对主机指纹。

## 使用网页评测

1. 在“模型与数据”选择已注册模型，或上传自己的完整模型目录。大文件也可通过 SFTP 放到服务器运行目录的 `incoming/`。
2. 新模型需要注册权重目录、训练 CLI YAML 和归一化 JSON，文件要求见下节。
3. 点击“新建评测”，选择模型、任务、Clean/Randomized 和固定 seeds，然后加入队列。
4. Worker 按顺序执行任务。同一 GPU 同时运行一个评测；等待中表示尚未领取。
5. 打开报告查看成功/有效回合、初始化失败、运行错误、三相机图片和动作轨迹，下载分任务 CSV。

录像默认关闭；开启后为 **10 fps、仿真时间 1× 原速**。无论是否录像，三相机关键帧和动作记录都会保存。取消任务会保留已完成回合。

### 模型文件要求

微调模型应导出完整、已合并的 HF safetensors 权重，连同实际训练配置和数据统计：

```text
incoming/experiment-001/
  lingbotvla_cli.yaml
  norm_stats_clean.json
  norm_stats_clean.provenance.json
  hf_ckpt/
    model-00001-of-00006.safetensors
    ...
    model.safetensors.index.json
    config.json
    clean_manifest.json
    eval_handoff.json
```

分片名称和数量以实际导出为准。平台不直接加载 adapter-only 或 pickle 权重；LoRA 合并前后的动作一致性由导出端验证。归一化文件必须与训练时一致，provenance 和 clean manifest 由真实数据处理过程生成。

`eval_handoff.json` 绑定导出文件的 SHA-256，结构如下（哈希和步数填入实际值）：

```json
{
  "kind": "clean_sft_export",
  "cli_sha256": "训练CLI的SHA256",
  "norm_sha256": "归一化JSON的SHA256",
  "provenance_sha256": "provenance文件的SHA256",
  "manifest_sha256": "clean_manifest.json的SHA256",
  "training_run": "experiment-001",
  "global_step": 30000
}
```

注册时校验配置和文件绑定，保存权重 SHA；执行前再次核对。原始基座使用上游归一化，微调模型使用自身训练统计。

## 使用脚本评测

在自己电脑通过配置连接服务器：

```bash
# 原始基座：click_bell 的 Clean/Randomized 各一个回合
python eval/connect.py eval -- --base --tasks click_bell --seeds 100010 --title base-smoke

# 已注册模型：提交到网页队列，由 Worker 执行
python eval/connect.py eval -- --checkpoint-id MODEL_ID --tasks click_bell --seeds 100010 --enqueue
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
bash run_eval.sh --checkpoint-id MODEL_ID --all-tasks --seeds 100010 100011
```

`eval.sh` 是 `run_eval.sh` 的同功能入口。CLI 默认直接运行，增加 `--enqueue` 则只排队。模型路径和 ID 属于所选服务器，连接脚本不会自动上传本地权重。每次调用创建独立任务和输出目录。

### 推理与统计口径

CLI 和网页使用同一配置：256×256、FP32/high、chunk/执行50、flow10、compile开启、官方 fused MoE / 视觉 FlashAttention 2、全 CUDA、上游原生噪声。模型服务启动 seed42，回合间不重设模型随机状态。

成功率为“成功回合 / 有效策略回合”。策略失败计入有效分母；初始化失败和运行错误另列。当前入口按固定任务、环境、seed 执行，**不自动补足初始化失败**。这是开发场景评测，尚未接入赛事专家场景筛选与正式复测接口。

## 查看输出与处理错误

每个任务写入服务器的 `runs/eval-ID/`：

```text
request.json, checkpoint.json, official-defaults-manifest.json, server-runtime.json
report.json, scores.csv, timing.json
worker.log, server.log, evaluation.log
evaluation/episodes.jsonl, summary.json, execution-events.jsonl
evaluation/<task>/<setting>/<seed>/
  episode.json, trajectory.jsonl, predictions.jsonl
  chunk-*.png, final*.png, [episode.mp4]
```

单回合上限900秒，连续240秒没有完整动作块进展会停止。出现运行错误后查看报告和日志，保留现场再处理；不要将运行错误记成初始化失败或换 seed。当前没有跨模型进程的自动恢复功能，重新启动模型会形成新的随机数进程段。

常见情况：网页打不开，先检查隧道终端和远端网页进程；任务一直等待，检查 Worker 是否在线及 GPU 是否被其他任务占用；模型注册失败，核对完整分片、CLI、归一化和导出哈希。

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
  "gpu": "0"
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

已有部署无需重新安装。部署到新目录时，安装器复用同机已配置的模型和仿真环境，创建独立网页环境，并共享 GPU 锁：

```bash
# 在目标 Linux 服务器、仓库根目录执行
bash eval/setup_eval.sh --server rented-5090 --check  # 只读预检查
bash eval/setup_eval.sh --server rented-5090          # 安装到配置中的新目录

cd /root/shared-nvme/lingbot-eval-team
bash start_web.sh --with-worker                      # 网页和单GPU Worker
# 只查看结果、上传或排队时，可运行 bash start_web.sh
```

安装目录必须为空；资源通过软链接复用，原资源目录需要保留。新服务器需先准备以下环境和资产，本安装器不负责空白系统的驱动、仿真器和模型下载。

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

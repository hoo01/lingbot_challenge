# Lab20k 后全量 SFT +6000：模型下载与推理实验

本次交付是 **原 Lab q/v-r256 20,000 步模型 → 全量 SFT 新阶段第 6,000 次 optimizer update**。
下载后可直接比较“预测50步、执行50步”和“预测50步、执行25步后重新观测”。这是完整 dense FP32 模型，
不是原 LoRA 训练的 step6000，也不需要再次合并 adapter。
父模型是原20k完整模型，不是RETAIN加权候选。

模型位于私有 [HumaLog/Lab-QV256 的本轮目录](https://huggingface.co/HumaLog/Lab-QV256/tree/main/experiments/lab20k_full_sft_20261016)。
**固定 HF revision 以 [release.json](release.json) 为准**；下载器只拉取该 revision 的本次实验目录，
不会顺带下载原20k的训练恢复状态。队友账号需已有该私有仓库的读取权限。

| 项目 | 交付值 |
| --- | --- |
| 检查点 | `checkpoint_sft_006000`，world4，4份模型/优化器＋4份数据状态保存读回全部相等 |
| 推理目录 | `experiments/lab20k_full_sft_20261016/checkpoints/global_step_6000/hf_ckpt_fp32/` |
| 模型 | 6,375,906,359参数，1672参数张量＋36持久buffer，全部FP32 |
| 父模型 | Lab20k，HF revision `84afa4f2fbef16ad6d41dbb1c3b27967f2d7a8e6` |
| 学生更新范围 | 语言、视觉、Action Expert、投影/对齐模块全量更新；教师冻结 |
| 训练概要 | AdamW，global128；语言/视觉/动作及投影峰值LR分别1e-5/5e-6/2e-5，warmup500，cosine |
| 模型预测 | H50，10个flow积分步，256图像，原生噪声 |
| 动作契约 | 12关节相对MeanStd，夹爪绝对MinMax；本模型自己的norm |

本步仍属于采样切换之前的训练阶段，实际原采样权重表随模型保存在 `evidence/source-task_weights.csv`。
本次只上传完整推理模型、processor/tokenizer、配置和来源证据；**不包含新阶段 AdamW/数据迭代器恢复状态**。
原检查点、训练进程及正在运行的+5000评测均未因此改变。

## 1. 准备代码与环境

以下命令面向队友空闲GPU上的独立 Linux/NVIDIA CUDA 推理环境。使用与官方固定源码相容的 Python3.12、PyTorch2.8、
Transformers4.57.3、FlashAttention2 和官方MoE环境；环境安装以固定源码的 README 为准。
已有可用推理环境可直接复用。包内已含完整 tokenizer/processor 小文件，**不需要再下载基座权重**。

```bash
git clone https://github.com/hoo01/lingbot_challenge.git
git clone https://github.com/Robbyant/lingbot-vla-v2.git
git -C lingbot-vla-v2 checkout be969b8fd117fb70550c5d4bf4bc328211b5b1b6

# 在独立推理环境中登录自己的 HF 账号。
hf auth login

cd lingbot_challenge
python train/lab20k_full_sft/download.py --output "$PWD/models/lab20k-sft6000"
```

下载器会核对配置、norm、映射文件的实际SHA，以及大文件是否齐全；server启动时再对实际加载的
全部1672参数和36持久buffer逐值计算SHA。完整FP32权重约25.5GB；推理还需要激活和编译工作区。
加载器还会先在CPU构造和载入完整模型，主机内存也需留出空间。
容量安排按实际显存实测，不将权重文件大小当作完整推理峰值。

## 2. 启动执行25步的策略服务

```bash
BUNDLE="$PWD/models/lab20k-sft6000/experiments/lab20k_full_sft_20261016"
UPSTREAM="$(cd ../lingbot-vla-v2 && pwd)"

CUDA_VISIBLE_DEVICES=0 python train/lab20k_full_sft/serve.py \
  --bundle "$BUNDLE" --upstream "$UPSTREAM" \
  --execute 25 --port 18025 \
  --runtime-receipt runs/sft6000-execute25-runtime.json
```

出现 `event=ready` 且 `all_saved_values_equal=true` 后，HTTP `/healthz` 可用。
启动后首次真实预测可能包含编译开销；该健康检查不代表已完成真实模型预测。
服务默认仅监听 `127.0.0.1`；按已有 SSH 转发方式连接队友客户端。

对照实验把 `--execute 25` 改为 `--execute 50`，并使用独立端口/记录文件或先退出上一个服务。
两种设置都保持 FP32、官方MoE、vision FA2、compile=true、matmul high、H50与10flow。
`--no-compile` 是另一个需单独记录的推理变量，不应悄悄混进执行长度对照。

## 3. 客户端如何重规划

下图中的“步”是 `env.take_action` 的动作调用，不是底层物理积分子步。

```text
当前RGB、q[t] → 模型预测50步 → 还原50个物理动作 → 执行前25步
     ↑                                                │
     └────────── 取得执行后的新RGB、q[t+25] ────────────┘
```

`serve.py` 使用固定官方接口 `use_length=25, chunk_ret=True`：每次新请求重新预测H50，
在完整反归一化之后返回 `(25,14)`。**不要把模型 `chunk_size` 改成25**，norm的每horizon统计仍是50。
12关节每个horizon的相对位移都加同一次观测的q[t]，不能逐步累加；夹爪保持绝对动作，不再加state。

把以下逻辑接到队友自己的 RoboTwin episode 循环中。它只展示客户端接口，不承担场景初始化、seed管理、
录像、超时及结果归档；这些继续沿用评测端原协议。

```python
import numpy as np
from deploy.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=18025)
metadata = policy.get_server_metadata()
execute = metadata["execute"]
assert metadata["predicted_horizon"] == 50 and execute == 25
policy.reset("robotwin")  # 每个episode一次，不能每个chunk重新置seed。

while env.take_action_cnt < env.step_lim and not env.eval_success:
    obs = env.get_obs()  # 每执行完这一段，重新取得真实观测。
    reply = policy.infer({
        "observation.images.cam_high": obs["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": obs["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": obs["observation"]["right_camera"]["rgb"],
        "observation.state": obs["joint_action"]["vector"],
        "task": env.get_instruction(),
    })
    actions = np.asarray(reply["action"])
    assert actions.shape == (execute, 14) and np.isfinite(actions).all()
    for action in actions:
        env.take_action(action)
        if env.eval_success or env.take_action_cnt >= env.step_lim:
            break
```

客户端需能从固定 upstream 导入 `deploy`（例如设置 `PYTHONPATH="$UPSTREAM"`）。
若评测端硬编码 `expected_execute=50`，需为**新的独立实验**设为25；不能把返回25行的服务接到原固定50协议上。
本次交付没有改动任何当前正式评测配置，也没有自动启动+6000评测。

## 4. 输入、精度与对照记录

- 三路输入必须是**原始 `uint8 HWC RGB`，0..255**。不要转BGR，不要提前除255，不做评测图像增强；
  官方processor只缩放一次。暗图即便最大值很小也不能猜成0..1输入。
- 使用原始有限14关节状态；映射中的夹爪位于索引6、13。server绑定本包的相对动作映射，
  不会退回官方默认 robotwin 配置或H0绝对norm。
- norm SHA：`b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d`。
  robot SHA：`1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b`。
- server在**加载前**将构造出来的BF16子模块提升为FP32，避免先把已保存的FP32权重拷入BF16而丢精度。
  然后核对全部保存值；不将训练混合精度与此FP32官方推理声称为逐位等价。
- 25/50步对照使用相同模型revision、原instruction、task/setting、seed规则、初始状态、动作上限、
  录像和错误处理；记录实际RGB。普通策略失败和真实策略超时仍计分，初始化/仿真错误单列，不能按成败挑seed。
  每个新进程原生seed42不代表跨进程恢复了同一条连续RNG流。
- 建议先单变量比较执行25与50；不要同时改flow步数、精度、norm和执行长度，否则难以解释差异。
  缩短执行长度通常会增加模型请求次数，应同时记录成功率、每episode请求数和耗时，不能预先保证提升。

## 5. 本次验证范围

源检查点完整性、8份保存读回、CPU精确重建和导出逐值读回已通过。
HF固定revision的所有新增文件均按原SHA核对：大权重使用LFS SHA/大小，配置与小证据实际下载读回。
轻量证据见 [`../evidence/lab20k_full_sft_20261016_sft6000/`](../evidence/lab20k_full_sft_20261016_sft6000/)。

七项CPU行为检查覆盖固定发布清单绑定、真实官方 `infer` 的25/50前缀及逐请求重规划、加载前FP32提升、
uint8暗图和错误图像/状态拒绝。**本次未新增+6000的GPU动作或仿真验收，没有该步的正式成绩**。
server中的实际参数校验在队友启动时执行，通过后写入 runtime receipt。

```bash
UPSTREAM_DIR="$UPSTREAM" python -m unittest discover -s train/lab20k_full_sft/tests -v
```

# HumaLog代码材料

本目录提供16000步提交模型的训练实现、实际评测实现、安装与公开资产准备入口。技术方案、参数和loss见`../04_复现与调优说明/reproduction_report.md`。

## 安装与目录

按[INSTALL.md](INSTALL.md)准备训练Python3.12与仿真Python3.10两个环境，按[ASSETS.md](ASSETS.md)准备固定公开数据、教师、检测器、仿真资产及归一化。`environment`包含两个实际环境的165／155项分发包快照及核心依赖约束。

| 路径 | 用途 |
|---|---|
| train.sh、src/train_entry.py | 可配置目录的基座20k训练；另保留内部增强段历史恢复分支 |
| eval.sh、src/eval_entry.py | 新目录的16000模型Clean／Random评测 |
| src/base20k | 原20k数值实现，仅将数据和运行根目录改为环境变量 |
| src/lineage/amd8、cuda8000 | 八卡SFT、8000→10000迁移及保存内存修复实现 |
| src/final_runtime | 与最终训练身份一致的23份实际源码 |
| src/foreground | 任务提示、框检测、补检、保护缓存与RGB绑定 |
| src/eval_runtime | 本次实际16000评测执行层，运行根目录模板化 |
| src/robotwin_overlay | 实际使用的145份任务源码、配置和指令模板 |
| src/prepare_assets.py | 固定公开来源下载、Clean2500索引及文件身份生成 |
| src/merge_results.py、validate_results.py | 完成回执汇总与官方结果结构检查 |
| src/check_package.py | 源码、JSON、norm及可选模型校验 |
| configs、environment、licenses | 固定配置、源码锁、版本与许可 |

## 训练

在本目录、训练环境下执行，WORK为新工作目录：

```bash
bash train.sh --stage base20k --check-only
bash train.sh --stage base20k --assets "$WORK/assets" --runtime-root "$WORK/base20k" --data-only --run reader_check
bash train.sh --stage base20k --assets "$WORK/assets" --runtime-root "$WORK/base20k" --run formal_base20k
```

完整阶段、Adam衔接与前景生产源码见[TRAINING.md](TRAINING.md)。基座20k入口完成目录参数适配；从基座到最终模型的各阶段统一入口仍待完成独立验收。11000步是内部断点，不是评委必须提供的输入；历史恢复分支不等于完整训练复现入口。

## Clean与Random

此入口直接加载最终模型，不需要原服务器路径、中间训练断点或optimizer状态。安装与本机真实渲染检查后，在本目录执行；WORK为运行者自选的新目录：

```bash
bash eval.sh --check-only
COMMON=(--checkpoint-root "../03_模型权重/checkpoint"
  --upstream "$WORK/assets/code/lingbot-vla-v2" --robotwin "$WORK/RoboTwin"
  --oidn-root "$WORK/oidn-2.3.3.x86_64.linux" --render-proof "$WORK/render.json"
  --train-python "$WORK/envs/train/bin/python" --sim-python "$WORK/envs/sim/bin/python"
  --gpu-lock "$WORK/gpu.lock" --episodes 30 --seed 0)
bash eval.sh "${COMMON[@]}" --root "$WORK/eval_clean" --setting clean --gpu 0 --port 18160 --launch
bash eval.sh "${COMMON[@]}" --root "$WORK/eval_random" --setting randomized --gpu 1 --port 18161 --launch
```

两条命令各占用一张空闲卡。多卡分片通过`--tasks`固定互不重叠的任务子集，并使用独立root／port、共享GPU锁。新目录准备时校验14份实际权重SHA；省略`--launch`只准备目录。现有目录不被该入口覆盖。自定义Vulkan及仿真库位置可传`--vulkan-icd`、`--ld-library-path`。

每组固定30计分回合；官方seed0对应环境100000起。FP32、256图像、10flow、H50预测／前25执行、模型进程seed42连续RNG；无专家、训练增强、检测器、教师及录像。有据策略超时计失败，策略前原生初始化错误保留并递增候选；工程错误保留现场。

## 结果与验证

收齐全部100组CPU审计完成回执后：

```bash
python -B src/merge_results.py --receipt-dirs "$WORK"/eval_*/artifacts/completion --output ../01_评测结果/results.json
python -B src/validate_results.py ../01_评测结果/results.json --expected-attempts 30
python -B src/check_package.py --weights
```

汇总器核对回执SHA、模型、协议、3000个唯一计分键及100组覆盖。缺组、损坏、重复或混入其它模型均拒绝输出。

本次材料整理验证源码身份、语法、静态入口和汇总行为。独立GPU模型加载、实际三路预处理、动作反归一化及首次策略动作，以实际启动产生的started.json、first-preprocessing.json和CPU回放结果验收。

## 本候选实测更新（2026-10-25）

3000回合评测及原运行环境六卡验收已完成，Clean72.67%、Random39.27%。正式JSON、逐任务CSV及六分片原始小回执位于../01_评测结果。本轮实际输入、FP32加载与动作往返通过，发行入口在全新安装环境中的独立复现仍单独记录。

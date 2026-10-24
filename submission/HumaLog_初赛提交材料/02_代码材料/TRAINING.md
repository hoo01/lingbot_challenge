# 分阶段训练和状态衔接

## 训练流程

| 阶段 | 正式更新与batch | 实现与状态 |
|---|---|---|
| 指定基座→Lab20k | 20000；4×32×1=128 | src/base20k；72个语言q/v LoRA，r256／alpha512；视觉、动作、投影开放；Muon＋AdamW fallback |
| 原dense Lab20k→8000 | 新AdamW8000；8×16×2=256 | src/lineage/amd8；ROCm适配及有界保存实现 |
| 8000→10000 | 继承Adam2000；4×16×4=256 | src/lineage/cuda8000；reshard.py和checkpoint_extra.py |
| 10000→11000 | 增强段前1000；4×16×3=192 | 300步升温；原前景保护和第一阶段几何配置 |
| 11000→16000 | 增强段后5000；global192 | src/final_runtime实际23文件；45%主视角几何，继承整个6000阶段的调度与状态 |

SFT保持同一55维映射、相对臂／绝对夹爪统计和冻结教师。8000后排除50个异常窗口及已核实冲突指令，过滤配置随包提供。学习率、增强概率、学生强图／教师弱图与辅助监督mask见技术报告2.4—2.6节。

## 基座20k入口

train.sh的base20k分支使用原数值实现，仅对两个根目录做环境变量适配，spawn worker继承相同路径。prepare_assets.py生成新路径的真实输入索引，train.sh --data-only检查实际reader，正式分支启动4-rank torchrun。每次选新的run名，完整保存；身份记录对应发行代码和实际输入。

## 最终边界恢复入口

最终23文件保留严格身份绑定。复现11000→16000边界时，在专用容器的目录空间中提供：

```text
/mnt/raid1/lingbotvla_smoke_20261009/
  code/lingbot-vla-v2/、code/src/、models/、datasets/
/mnt/raid1/lingbotvla_sft_amd8000_to10000_20261020/
  incoming/model8000/、artifacts/、完整10000状态
/mnt/raid1/lingbotvla_sft_aug6000_b192_20261022/
  train_geometryfix11000_requested/  ← src/final_runtime原文件
  artifacts/                       ← 原清单、过滤、来源和实际恢复gate
  foreground/manifest.reviewed.json及逐轨迹缓存
  runs/formal_aug6000_b192/checkpoint_sft_011000/
```

```bash
bash train.sh --stage aug16000 \
  --runtime-root /mnt/raid1/lingbotvla_sft_aug6000_b192_20261022 \
  --resume /mnt/raid1/lingbotvla_sft_aug6000_b192_20261022/runs/formal_aug6000_b192/checkpoint_sft_011000 \
  --run reproduce_aug16000
```

输入为完整4份model／Adam与4份data／RNG分片，complete及readback绑定实际内容。入口检查实际ready-for-formal回执；推理dense分片只包含学生权重和推理资产。

## 跨阶段复现剩余工作

发行材料已提供各阶段数值实现、保存修复、重分片和前景生产源码。基座20k入口完成目录适配；其它历史阶段保留原路径和严格身份检查。将基座训练自动衔接到最终模型，还需统一LoRA dense导出、AMD／CUDA启动配置、10000→11000原配置、前景缓存和完整Adam状态的目录迁移，并做独立GPU验收。

完整训练一键复现仍列为待验收项。已完成的原训练结果、最终源码身份、推理包完整性和新发行入口的独立运行验证分别记录。

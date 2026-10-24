# 数据、模型与公开资产

| 资产 | 固定来源与版本 | 用途 |
|---|---|---|
| LingBot-VLA v2源码 | Robbyant/lingbot-vla-v2；be969b8fd117fb70550c5d4bf4bc328211b5b1b6 | 学生、特征转换、优化器及推理 |
| 指定6B基座 | robbyant/lingbot-vla-v2-6b；11c703bf6a5c1f45b3b69168482da11fdbba53d7 | 唯一训练起点 |
| Clean2500 | TianxingChen/RoboTwin2.0；3dc3b798668feb99ac61cc9086d84cbcc3d79186 | dataset/各任务/aloha-agilex_clean_50.zip |
| MoGe2 | Ruicheng/moge-2-vitb-normal；ca5f0e07ff01d3e5a364c1d954ed12ee1814b368 | 冻结深度模块model.pt |
| MoRGBD | 指定6B仓库同版本，depth/model.pt | 冻结深度教师 |
| DINO-video | 指定6B仓库同版本，dino_video/teacher_step_10000.pth和config.yaml | 冻结视频教师 |
| GroundingDINO Tiny | IDEA-Research/grounding-dino-tiny；a2bb814dd30d776dcf7e30523b00659f4f141c71 | 原Clean图像离线框检测 |
| 推理学生与processor | ../03_模型权重/checkpoint | 同一dense FP32模型及处理配置 |
| 归一化 | configs/assets/norm_stats.json | 相对臂MeanStd、绝对夹爪MinMax |

## 训练资产准备

在代码材料目录、训练环境中执行。INSTALL.md已完成sources步骤后：

```bash
python src/prepare_assets.py --mode models --assets "$WORK/assets"
python src/prepare_assets.py --mode clean-data --assets "$WORK/assets"
python src/prepare_assets.py --mode index --assets "$WORK/assets" --runtime-root "$WORK/base20k"
```

下载器固定revision，对公开LFS校验长度与SHA并保存来源记录。Clean验证50任务各episode0—49，三相机与14维关节帧数一致，抽查解码并逐HDF5核对SHA。产生549787个起始帧，步幅1、2500条全部为train；复制提交模型的冻结统计量。

新路径产生新的绝对路径／mtime清单，字节与历史清单不同；跨阶段恢复应显式迁移新身份。8000后数据过滤项在configs/data-filter-plan.json。

## 前景保护

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("IDEA-Research/grounding-dino-tiny",revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",local_dir="grounding-dino-tiny")'
```

检测器冻结，只对原Clean RGB产生框；训练worker读取缓存。任务提示、补检和打包分别在src/foreground/runtime的task_prompts、cache_worker、cache_policy、correct_foreground、finalize_foreground_worker中。核心RGB绑定和保护规则在aug6000/foreground.py、grounding.py。

已审定标注包可使用CPU绑定入口：

```bash
python src/foreground/build_foreground_cache.py --annotations "$WORK/annotations/bundle.json" \
  --parent-root "$WORK/parent10000" --assets "$WORK/assets" --output "$WORK/foreground_new"
```

正式缓存带原RGB SHA、模型来源、任务策略与质量回执。框不是分割真值，缓存生产完成与前景质量验收分开记录。历史生产路径与最终manifest迁移属于TRAINING.md中的跨阶段适配项。

## 仿真资产

按[RoboTwin安装下载说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)，在新RoboTwin目录中执行：

```bash
cd "$WORK/RoboTwin/assets"
python _download.py
unzip -n background_texture.zip
unzip -n embodiments.zip
unzip -n objects.zip
```

保持Aloha-AgileX、物体、纹理、seen／unseen指令和任务配置；资产版本与本地校验随独立安装记录保存。原Clean示范仍是唯一训练动作数据。

渲染使用[Intel OIDN v2.3.3公开发行包](https://github.com/RenderKit/oidn/releases/tag/v2.3.3)，解压目录为WORK/oidn-2.3.3.x86_64.linux；动态库和真实渲染检查见INSTALL.md。保留各公开资产的原model card、LICENSE与NOTICE。

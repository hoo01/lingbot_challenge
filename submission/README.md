# HumaLog初赛提交资料

本目录发布已整理的技术文档、代码、配置、训练loss和小型证据文件。GitHub版本不含模型权重和processor载荷；完整赛事提交包需另行加入模型。

- [复现与调优报告](HumaLog_初赛提交材料/04_复现与调优说明/reproduction_report.md)
- [评委运行入口说明](HumaLog_初赛提交材料/评委运行入口说明.md)
- [安装、训练与评测入口](HumaLog_初赛提交材料/02_代码材料/README.md)
- [安装环境](HumaLog_初赛提交材料/02_代码材料/INSTALL.md)
- [公开资产与数据](HumaLog_初赛提交材料/02_代码材料/ASSETS.md)
- [分阶段训练与剩余衔接项](HumaLog_初赛提交材料/02_代码材料/TRAINING.md)
- [最后5000步loss原始数据](HumaLog_初赛提交材料/04_复现与调优说明/evidence/loss-scalars-11001-16000.csv)
- [双SO101真机Demo说明](HumaLog_初赛提交材料/05_真机创新可选任务/demo_report.md)
- [原始Demo视频](HumaLog_初赛提交材料/05_真机创新可选任务/demo_video.mp4)
- [小红书活动信息](HumaLog_初赛提交材料/06_小红书创作活动/xiaohongshu_info.md)
- [提交准备清单](提交准备清单.md)

## 模型单独获取

模型来源为[Humalog2/Lab20k-FullSFT-Aug16000](https://huggingface.co/Humalog2/Lab20k-FullSFT-Aug16000)。该HF仓库为私有，需已获读取权限的HF登录；账号凭据不随代码发布。

如需运行README中的模型检查、资产准备或评测，在本目录安装Hugging Face Hub CLI后下载固定版本，保留其目录结构：

```bash
hf download Humalog2/Lab20k-FullSFT-Aug16000 \
  --revision 2b1e8a52668d38dbc91a01bf0d83085ba7a8f8a1 \
  --local-dir "HumaLog_初赛提交材料/03_模型权重/checkpoint"
```

`03_模型权重/model_manifest.json`记录所需权重及处理资产。模型目录由.gitignore排除；Git中的SHA256SUMS描述完整提交材料，因此其中的模型资产需下载后才能逐项检查。

## 当前验收状态

3000回合实测已完成：Clean1090/1500（72.67%）、Random589/1500（39.27%）、综合55.97%。正式results.json、逐任务成绩、六分片回执和实际运行环境验收已补齐。

2026-10-25补入双SO101真机交接Demo原视频与官方结构说明；团队确认使用LingBot-VLA进行真机全参数SFT，视频为模型自主执行。小红书账号Elowyn和笔记链接已填写。评测入口直接使用最终权重，运行者自行指定目录，不需要11000步中间状态或原服务器访问权限。

完整训练的跨阶段衔接及新发行入口独立GPU复现仍待完成，材料继续保留DRAFT_NOT_READY_TO_SUBMIT状态；本次材料整合不冒称完成该项验收。

![最后5000步loss](HumaLog_初赛提交材料/04_复现与调优说明/figures/loss-final5000.png)

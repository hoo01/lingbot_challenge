# 附录：版本与文件校验索引

本表用于复现时定位确切版本，以及核对取得的文件。revision／commit是模型或代码仓库的版本编号；SHA256是文件内容的校验值。技术正文以模型名称、方法和运行配置介绍方案，精确编号集中保存在这里。

## 仓库版本

| 对象 | 来源／用途 | 固定版本 |
|---|---|---|
| 官方6B基座 | robbyant/lingbot-vla-v2-6b，训练起点 | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` |
| 官方代码 | Robbyant/lingbot-vla-v2 | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` |
| 提交模型 | Humalog2/Lab20k-FullSFT-Aug16000 | `2b1e8a52668d38dbc91a01bf0d83085ba7a8f8a1` |
| 训练记录归档 | hoo01/lingbot_challenge；对应完成报告发布 | `e82ef4815fdb3885e3b51a5c07d95779b9bd2673` |
| Clean示范数据 | TianxingChen/RoboTwin2.0 | `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |
| 离线检测器 | IDEA-Research/grounding-dino-tiny | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |

训练记录的发布commit用于定位归档报告。实际训练使用的23个核心文件以[source-lock.json](../../02_代码材料/configs/source-lock.json)追溯，对应的冻结运行身份见[frozen-training-identity.json](frozen-training-identity.json)。

模型仓库目前为私有，最终提交包的checkpoint应包含同版本的实际模型文件。

## 关键资产校验值

| 文件／对象 | SHA256 |
|---|---|
| 数据过滤计划 | `30e7c8c018e1d2110bc255a2f03051034a679afd01c122b68ec2052878f410c5` |
| 归一化统计 | `b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d` |
| 原训练robot配置 | `1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b` |
| 发布包robot配置 | `f4a1de74470ae9d7353bbfe7bdd173305bf391cecd05f4dd7f5276968e426c5e` |
| GroundingDINO Tiny权重 | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |

发布包robot配置将资产路径改为相对路径，因此文件校验值与原训练文件不同，状态／动作字段定义一致。

完整权重及处理资产的文件清单见[model_manifest.json](../../03_模型权重/model_manifest.json)。公开检测器各文件校验值见[groundingdino-public-model.json](groundingdino-public-model.json)。赛事说明及模板的读取记录分别保存在official-source.json、official-template-sha256.json。

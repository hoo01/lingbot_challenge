# 提交模型与推理资产

正式模型目录为checkpoint，保持model_manifest.json中的相对结构。14份FP32分片共25,503,888,740字节，包含1672参数tensor与36持久buffer。

推理资产包括Qwen processor／tokenizer、模型config与index、inference-cli.yaml、relative robot、norm_stats.json及provenance。教师、离线检测器和Adam状态不属于推理文件。

服务器的独立提交目录已经放入全部14份真实权重；本地文档工作副本保存处理配置等小文件，服务器位置与验证方式见model_manifest.json。此次装包核对硬链接文件身份及大小，沿用既有发布SHA；没有重新进行大权重全量审计。

新目录独立评测时，eval.sh按weight_files清单逐文件校验内容SHA，随后显式以FP32参数加载，核对全部1672参数和36buffer及实际输入处理。完整包检查命令为02_代码材料/src/check_package.py --weights，显式重读权重可加--hash-weights。

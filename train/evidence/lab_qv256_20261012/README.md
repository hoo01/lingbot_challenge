# lab_qv256_20261012 轻量证据

原始训练与导出文件经实际读回生成；权重和恢复状态只进入私有 HF。
`redaction-report.json` 区分 source 与 distributed SHA。norm 原字节 SHA 保持 `b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d`。
`checkpoint_inventory.json` 给出 HF 文件身份，最终上传固定 revision 见 `huggingface-archive.json`。
source wrapper 没有启动时 Git commit 记录，以四份冻结源码 SHA 追溯，不能把本次交接提交冒充训练源码提交。

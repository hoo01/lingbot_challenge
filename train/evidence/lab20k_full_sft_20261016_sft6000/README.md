# Lab20k 后全量 SFT +6000 发布证据

这是模型交付证据，不是+6000的正式评测成绩。使用方法见[队友推理交接](../../lab20k_full_sft/README.md)。

- 实验：`lab20k_full_sft_20261016`，新全量SFT阶段第6000次optimizer update；父模型原Lab20k。
- [HF固定revision](https://huggingface.co/HumaLog/Lab-QV256/tree/ee6a059ce178641fab3f2d87ee2296accc49e812/experiments/lab20k_full_sft_20261016/)：`ee6a059ce178641fab3f2d87ee2296accc49e812`，私有。
- 源complete SHA：`56554a1a222c3a83295bd8a5f3733205c496fb1c8d34d4ba13346555cf505cff`。
- 原4份模型/优化器与4份数据状态的保存读回均为true；只CPU重建并上传完整推理权重，未上传训练恢复状态。
- 6375906359参数、1672参数张量、36持久buffer，FP32；导出后全部张量逐值读回一致。
- HF大文件按固定revision LFS SHA/大小核验，小文件实际下载逐字节SHA核对；原有82份模型/证据文件身份保留。
  README仅新增本轮入口；Hub自动向`.gitattributes`追加一条本轮tokenizer的LFS规则，旧规则原字节保留。
- CPU验证：真实官方infer的25/50执行前缀和新请求重规划、加载前FP32提升、图像/关节输入契约。
- 未新增GPU模型预测、仿真回合或+6000成绩；队友启动server后会再核验实际加载的所有保存参数/buffer值。

`evidence/`、`assets/`和`export/`为实际小文件原字节拷贝，未改写路径；其中源机器绝对路径用于追溯，
便携推理入口使用发布包的相对路径和启动时临时重绑定。`export_source.py`是本次实际CPU导出脚本。
`SHA256SUMS.txt`覆盖本目录除清单自身以外的实际字节；不含模型权重、PT恢复状态或凭据。

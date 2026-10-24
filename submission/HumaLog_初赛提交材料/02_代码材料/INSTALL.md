# 安装与运行环境

## 训练与模型服务

实际CUDA环境：Python3.12.3、PyTorch2.8.0/cu128、Transformers4.57.3、Flash Attention2.8.3。以下在Linux新目录执行，CODE指本代码材料目录：

```bash
CODE="$PWD"
WORK="$HOME/humalog_reproduce"
mkdir -p "$WORK/envs"
python3.12 -m venv "$WORK/envs/train"
source "$WORK/envs/train/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$CODE/environment/requirements.txt"
python -m pip install flash-attn==2.8.3 --no-build-isolation
python "$CODE/src/prepare_assets.py" --mode sources --assets "$WORK/assets"
python -m pip install -e "$WORK/assets/code/lingbot-vla-v2" --no-deps
python -m pip install -r "$CODE/environment/requirements-depth-runtime.txt" -c "$CODE/environment/requirements.txt"
python -m pip install --no-deps \
  -e "$WORK/assets/code/lingbot-vla-v2/lingbotvla/models/vla/vision_models/lingbot-depth" \
  -e "$WORK/assets/code/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe"
python -m pip install --no-deps 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
```

Flash Attention编译需要匹配PyTorch、CUDA和编译器。公开深度运行依赖采用实际子集，保留独立的requirements-depth-runtime.txt。AMD段使用ROCm镜像及对应适配；本安装示例用于CUDA四卡与模型服务。

## 仿真

实际仿真环境为Python3.10.22，核心依赖如下；完整分发包及编译扩展名称见environment/sim.installed.txt：

```bash
deactivate
python3.10 -m venv "$WORK/envs/sim"
source "$WORK/envs/sim/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 scipy==1.10.1 sapien==3.0.0b1 mplib==0.2.1 \
  gymnasium==0.29.1 transforms3d==0.4.2 toppra==0.6.3 open3d==0.18.0 \
  imageio==2.34.2 trimesh==4.4.3 opencv-python-headless==4.11.0.86 \
  pyyaml==6.0.3 websockets==15.0.1 msgpack==1.1.1 zarr==2.18.3
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$WORK/RoboTwin"
git -C "$WORK/RoboTwin" checkout --detach 13c3c47ff4312dd62484bcd51be034af55c062d1
cp -a "$CODE/src/robotwin_overlay/." "$WORK/RoboTwin/"
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$WORK/RoboTwin/envs/curobo"
python -m pip install -e "$WORK/RoboTwin/envs/curobo" --no-build-isolation
python -m pip install warp-lang==1.12.0
```

基础Git版本用于取得仓库结构；实际任务源码、配置、指令以随附145文件快照为准，其SHA在configs/robotwin-source-lock.json。上游script/_install.sh还说明SAPIEN UTF-8读取、MPLib与CuRobo安装调整；这些site-packages调整和二进制扩展需要在新环境核对。资产与OIDN见ASSETS.md。

环境快照是已安装分发包清单，包含editable源码扩展，与跨平台求解锁区分。新环境最终应与两个installed.txt比较，并记录pip check／CPU import／GPU加载的实际结果。

## 渲染和实际输入

实际设置为rt、32 samples/pixel、path depth8、OIDN。GPU驱动与NVIDIA Vulkan ICD由主机提供。在空闲设备运行：

```bash
export LD_LIBRARY_PATH="$WORK/oidn-2.3.3.x86_64.linux/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
python "$CODE/src/probe_renderer.py" --gpu 0 --gpu-lock "$WORK/gpu.lock" --output "$WORK/render.json"
```

该入口实际渲染简单场景并核对物理PCI设备；任务／模型验收由eval.sh实际启动完成。5090主机的Vulkan／EGL用户态库须与驱动匹配，使用本机已验证的库路径。

CUDA模型服务只接收模型环境；仿真环境单独指定图形库。推理原图为三路uint8 HWC RGB0..255，processor缩放一次。FeatureTransform使用提交的relative robot配置和norm；每个horizon的臂反归一化都加同一q[t]，夹爪保持绝对值。

```bash
source "$WORK/envs/train/bin/activate"
cd "$CODE"
python -B src/check_package.py
bash train.sh --stage base20k --check-only
bash eval.sh --check-only
```

这三项是CPU材料检查；真实GPU加载和保存恢复以相应运行回执为准。

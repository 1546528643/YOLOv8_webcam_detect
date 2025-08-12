# YOLOv8 摄像头实时目标检测系统

基于 YOLOv8 的实时目标检测系统，支持通用 80 类 COCO 目标检测和自定义标靶检测，并提供一键打包为可执行程序功能。

## 功能特性

- 🎯 **通用目标检测**: 支持 COCO 数据集 80 类物品的实时检测
- 🎯 **自定义标靶检测**: 支持训练和检测自定义标靶（如十环靶），为日后比赛赋能
- 📦 **一键打包**: 支持将程序打包为 exe 文件便于使用
- 🚀 **GPU 加速**: 支持 CUDA 加速推理
- 📹 **实时检测**: 支持摄像头实时检测和视频保存

## 快速开始

1) 创建并激活虚拟环境（建议）

```bash
python3 -m venv .venv && source .venv/bin/activate
```

2) 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 如需 GPU/CUDA 加速，建议先参考 PyTorch 官网安装与你 CUDA 版本匹配的 `torch/torchvision`，再安装其他依赖。

3) 运行摄像头实时检测（通用 80 类）

```bash
python src/webcam_detect.py --model yolov8n.pt --camera 0 --device auto --conf 0.5 --show-fps
```

## 自定义标靶检测

### 1. 生成标靶数据集

使用内置的合成数据生成器创建黑白十环标靶数据集：

```bash
# 生成默认数据集（500 训练 + 100 验证）
python src/generate_synthetic_targets.py --rings 10 --max-rotate 45 --noise

# 自定义参数
python src/generate_synthetic_targets.py --train 1000 --val 200 --imgsz 960 --rings 10 --max-rotate 30 --noise
```

### 2. 训练自定义模型

```bash
# 使用 YOLOv8n 训练
yolo detect train data=datasets/targets/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=-1 device=0

# 使用更大模型（推荐用于小目标）
yolo detect train data=datasets/targets/data.yaml model=yolov8s.pt imgsz=960 epochs=150 batch=-1 device=0
```

### 3. 使用训练好的模型检测

```bash
# 使用训练好的权重进行实时检测
python src/webcam_detect.py --model runs/detect/train/weights/best.pt --device auto --half --show-fps
```

## 程序打包

### 打包为可执行文件

将程序打包为 exe 文件，方便外行人使用：

```bash
# 安装打包工具
pip install -U pyinstaller pyinstaller-hooks-contrib

# 打包程序（包含权重文件）
pyinstaller --noconfirm --clean --onedir --name YOLOv8Webcam `
 --collect-all ultralytics --collect-all torch --collect-all torchvision --collect-all cv2 `
 --add-data "runs/detect/train/weights/best.pt;." `
 src/webcam_detect.py

# 运行打包后的程序
cd dist/YOLOv8Webcam
./YOLOv8Webcam.exe --model ./best.pt --device auto --half --show-fps
```

### 打包注意事项

- 确保权重文件路径正确
- 若用户无 GPU，运行时使用 `--device cpu` 并去掉 `--half`
- 建议使用 `--onedir` 模式，便于分发

## 可选参数

- `--model`: 模型权重路径或模型名（默认 `yolov8n.pt`）
- `--camera`: 摄像头索引（默认 `0`）
- `--device`: `auto`/`cpu`/`cuda:0` 等（默认 `auto`）
- `--imgsz`: 推理分辨率（默认 `640`）
- `--conf`: 置信度阈值（默认 `0.5`）
- `--iou`: IoU 阈值（默认 `0.45`）
- `--half`: 若设备支持，使用 FP16 推理
- `--show-fps`: 在窗口左上角显示 FPS
- `--save path.mp4`: 将可视化结果保存为视频

按 `q` 键退出程序。

## 项目结构

```
YOLOV8_webcam_detect/
├── src/
│   ├── webcam_detect.py          # 主检测程序
│   └── generate_synthetic_targets.py  # 标靶数据集生成器
├── datasets/
│   └── targets/                  # 标靶数据集目录
│       ├── data.yaml            # 数据集配置
│       ├── images/
│       │   ├── train/           # 训练图片
│       │   └── val/             # 验证图片
│       └── labels/
│           ├── train/           # 训练标签
│           └── val/             # 验证标签
├── requirements.txt              # 依赖包
└── README.md                     # 说明文档
```

## 说明

- 默认使用 `ultralytics` YOLOv8 推理接口，支持 `numpy` 的 `BGR` 帧
- 支持通用 80 类 COCO 目标检测和自定义标靶检测
- 内置合成数据生成器，可快速创建标靶训练数据集
- 支持一键打包为 exe 文件，便于分发使用
- 自动设备检测：有 GPU 时自动使用 CUDA，无 GPU 时自动切换到 CPU
- 部分摄像头在 `cv2.CAP_PROP_FPS` 返回 0 时会采用兜底帧率 30 保存视频

## 技术特性

- **模型**: YOLOv8 (ultralytics)
- **推理**: 支持 CPU/GPU 推理，自动设备选择
- **数据增强**: 内置旋转、缩放、噪声等增强
- **实时性能**: 支持 FP16 加速和 FPS 显示
- **易用性**: 一键打包，外行人友好

## 常见问题

### Q: 如何检测自定义目标？
A: 使用 `generate_synthetic_targets.py` 生成数据集，然后训练自定义模型。

### Q: 打包后程序无法运行？
A: 检查权重文件路径，确保使用正确的 `--add-data` 参数。

### Q: 无 GPU 环境如何使用？
A: 使用 `--device cpu` 参数，并去掉 `--half` 参数。

### Q: 如何提高检测精度？
A: 增加训练数据量，使用更大模型（如 yolov8s/m），或提高推理分辨率。

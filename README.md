# 工业仪表智能读数识别系统

## 项目简介

工业仪表智能读数识别系统是一个基于深度学习的自动化仪表检测解决方案，能够实时识别各种指针式仪表的读数。本系统结合了YOLOv5目标检测和U-Net语义分割技术，通过创新的图像处理算法，实现了高精度、高效率的仪表读数识别。
<img width="467" height="255" alt="image" src="https://github.com/user-attachments/assets/435d837d-a17b-4833-96e4-c1af415d6fa7" />


## 主要特性

- 🔍 **高精度检测**：采用改进的YOLOv5模型，表盘检测准确率94%
- 📊 **精确分割**：基于U-Net的指针与刻度分割，IoU达到0.87
- ⚡ **实时处理**：支持25FPS实时视频流处理
- 🎯 **多场景适配**：适应不同光照条件和仪表类型
- 📈 **数据可视化**：实时显示读数变化曲线
- 🔔 **智能告警**：异常状态自动检测与预警

## 环境要求

### 硬件要求
- GPU: NVIDIA GTX 1060 6GB或更高
- 内存: 8GB以上
- 存储: 至少10GB可用空间

### 软件要求
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- PyQt5 5.15+

## 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/meter-reading-system.git
cd meter-reading-system
```

### 2. 创建虚拟环境
```bash
conda create -n meter python=3.8
conda activate meter
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型
将预训练模型放置在 `models/` 目录下：
- `yolov5s_meter.pt` - 仪表检测模型
- `unet_meter_seg.pth` - 分割模型

## 使用方法

### 快速开始

```bash
python main.py
```

### 图形界面操作

1. **图片模式**
   - 点击"选择图片"按钮加载图像
   - 系统自动检测并识别所有仪表
   - 查看详细的处理过程和最终读数

2. **视频模式**
   - 选择视频文件或实时摄像头
   - 系统按秒进行仪表读数识别
   - 实时显示读数变化曲线

3. **参数配置**
   - 分度值设置：根据仪表量程调整（默认0.5）
   - 总刻度数：设置仪表总刻度数（默认16）
   - 摄像头选择：选择视频输入源

### 命令行参数

```bash
python main.py [选项]

选项：
  --mode MODE          运行模式（image/video/camera）
  --source SOURCE      输入源路径
  --conf CONF          置信度阈值（默认0.25）
  --iou IOU            IOU阈值（默认0.45）
  --device DEVICE      运行设备（cpu/cuda:0）
  --save-dir SAVE_DIR  结果保存目录
```

## 功能详解

### 1. 仪表检测模块
- **输入**：原始图像/视频帧
- **输出**：表盘位置坐标（xmin, ymin, width, height）
- **参数**：
  - 输入尺寸：640×640
  - 置信度阈值：0.25
  - NMS阈值：0.45

### 2. 语义分割模块
- **功能**：分割指针、刻度、背景
- **类别**：
  - 0: 背景
  - 1: 指针
  - 2: 刻度
- **输出尺寸**：512×512

### 3. 几何变换模块
```python
# 核心参数配置
METER_SHAPE = [512, 512]          # 处理图像尺寸
CIRCLE_CENTER = [256, 256]         # 表盘中心
CIRCLE_RADIUS = 250               # 表盘半径
RECTANGLE_WIDTH = 1400            # 矩形展开宽度
RECTANGLE_HEIGHT = 140            # 矩形展开高度
```

### 4. 读数计算模块
- **算法**：基于线性插值的精确读数
- **精度**：0.1个分度值
- **输出格式**：浮点数读数

## 文件结构

```
meter-reading-system/
├── models/                 # 模型文件
│   ├── yolov5s_meter.pt
│   └── unet_meter_seg.pth
├── src/                    # 源代码
│   ├── detection/          # 检测模块
│   ├── segmentation/       # 分割模块
│   ├── processing/         # 图像处理
│   └── gui/               # 图形界面
├── configs/               # 配置文件
├── data/                  # 示例数据
├── outputs/              # 输出结果
├── requirements.txt       # 依赖列表
└── README.md             # 说明文档
```

## 性能指标

### 准确率测试
| 模块 | 指标 | 数值 |
|------|------|------|
| 检测 | mAP@0.5 | 0.94 |
| 分割 | 平均IoU | 0.87 |
| 读数 | 平均误差 | 0.23 |

### 速度测试
| 输入类型 | 分辨率 | 处理速度 |
|----------|--------|----------|
| 图片 | 1920×1080 | 0.5秒/张 |
| 视频 | 1280×720 | 25 FPS |
| 摄像头 | 640×480 | 30 FPS |

## 高级配置

### 模型训练
```bash
# 训练检测模型
python train_detector.py --data meter.yaml --cfg yolov5s.yaml --epochs 100

# 训练分割模型
python train_segmenter.py --data meter_seg.yaml --model unet --epochs 150
```

### 参数调优
在 `configs/system_config.yaml` 中修改系统参数：

```yaml
detection:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  input_size: [640, 640]

segmentation:
  num_classes: 3
  input_size: [512, 512]
  model_path: "models/unet_meter_seg.pth"

processing:
  circle_center: [256, 256]
  circle_radius: 250
  rectangle_width: 1400
  rectangle_height: 140
```

## 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 减小批处理大小
   python main.py --batch-size 4
   ```

2. **检测效果差**
   - 检查模型路径是否正确
   - 调整置信度阈值
   - 确认输入图像质量

3. **分割边界模糊**
   - 检查图像预处理参数
   - 调整形态学操作核大小

### 日志查看
系统运行日志保存在 `logs/` 目录下，包含详细的处理信息。

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进这个项目。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

- 邮箱：your-email@example.com
- 项目主页：https://github.com/yourusername/meter-reading-system

## 更新日志

### v1.0.0 (2024-01-20)
- 初始版本发布
- 支持图片、视频、摄像头输入
- 实现完整的仪表读数流程
- 提供图形化操作界面

---

**注意**：使用前请确保已安装所有依赖，并下载相应的预训练模型。如有问题请查看日志文件或提交Issue。

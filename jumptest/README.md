# 跳跃姿态分析系统

这是一个基于计算机视觉和机器学习的跳跃姿态分析系统，能够分析跳跃动作的技术指标、身体姿态、弹跳力和核心力量。

## 功能特性

- **视频处理**: 支持多种视频格式的加载和预处理
- **姿态检测**: 使用MediaPipe进行实时人体姿态估计
- **跳跃分析**: 自动识别跳跃阶段并计算关键指标
- **力量评估**: 评估爆发力、核心力量和协调性
- **可视化报告**: 生成详细的分析图表和报告

## 系统架构

```
jumptest/
├── src/                    # 核心代码模块
│   ├── video_processor.py  # 视频处理模块
│   ├── pose_detector.py    # 姿态检测模块
│   ├── jump_analyzer.py    # 跳跃分析模块
│   └── visualizer.py       # 可视化模块
├── notebooks/              # Jupyter演示文件
│   └── tech_validation.ipynb
├── test_videos/            # 测试视频目录
├── outputs/                # 输出结果目录
├── requirements.txt        # 依赖包清单
└── test_integration.py     # 集成测试脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要依赖

- **OpenCV**: 视频处理和图像操作
- **MediaPipe**: 人体姿态估计
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化
- **SciPy**: 科学计算
- **Jupyter**: 交互式开发环境

## 快速开始

### 1. 运行集成测试

```bash
python test_integration.py
```

### 2. 启动Jupyter演示

```bash
jupyter notebook notebooks/tech_validation.ipynb
```

### 3. 使用真实视频

```python
from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.jump_analyzer import JumpAnalyzer
from src.visualizer import JumpVisualizer

# 1. 加载视频
processor = VideoProcessor("path/to/your/jump_video.mp4")
processor.load_video()
frames = processor.extract_frames()

# 2. 姿态检测
detector = PoseDetector()
pose_results = detector.detect_pose_sequence(frames)

# 3. 跳跃分析
analyzer = JumpAnalyzer(fps=processor.fps)
analysis_result = analyzer.analyze_jump_sequence(pose_results)

# 4. 生成报告
visualizer = JumpVisualizer()
visualizer.visualize_jump_analysis(analysis_result)
```

## 分析指标

### 跳跃指标
- **跳跃高度**: 起跳最低点到最高点的像素差
- **起跳时间**: 从最低点到最高点的时间
- **准备时间**: 准备阶段的持续时间
- **落地时间**: 从最高点到落地的时间

### 力量评估
- **爆发力**: 基于跳跃高度和起跳时间计算
- **核心力量**: 基于关节角度对称性评估
- **协调性**: 基于动作流畅性评估

### 姿态分析
- **稳定性得分**: 各阶段身体中心的稳定性
- **关节角度**: 膝关节和髋关节的角度变化
- **身体对齐**: 肩膀和髋部的对齐情况

## 输出文件

- **分析图表**: `outputs/jump_analysis.png`
- **文本报告**: `outputs/jump_analysis_report.txt`
- **姿态动画**: `outputs/pose_animation.mp4`（可选）

## 技术验证状态

✅ **已验证功能**:
- 视频处理和帧提取
- MediaPipe姿态检测集成
- 跳跃阶段自动识别
- 基础指标计算
- 可视化图表生成

🔧 **待优化**:
- 真实视频数据验证
- 算法精度提升
- 性能优化
- 用户界面开发

## 开发建议

1. **数据收集**: 收集多样化的跳跃视频数据
2. **算法优化**: 基于真实数据调整分析参数
3. **精度验证**: 与专业运动分析设备对比验证
4. **界面开发**: 开发Web或移动端用户界面
5. **性能优化**: 优化处理速度和内存使用

## 使用场景

- **运动训练**: 帮助运动员分析和改进跳跃技术
- **体能测试**: 评估个人的弹跳力和协调性
- **康复医学**: 监测康复训练中的动作质量
- **体育科研**: 提供量化的跳跃动作分析数据

## 注意事项

1. 确保视频中人体清晰可见
2. 建议使用稳定的摄像设备
3. 光照条件良好有助于提高检测精度
4. 首次运行可能需要下载MediaPipe模型

## 贡献指南

欢迎提交Issue和Pull Request来改进系统功能。

## 许可证

本项目仅用于技术验证和学习目的。
#!/usr/bin/env python3
"""
生成跳跃分析对比报告
将M1和M2的分析结果合并到一个对比报告中
"""

import sys
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加src目录到路径
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer
from visualizer import JumpVisualizer


def analyze_video_for_comparison(video_path):
    """分析视频用于对比"""
    print(f"分析视频: {video_path}")
    
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        return None, None
    
    video_info = processor.get_video_info()
    
    # 提取帧
    fps = video_info['fps']
    frame_step = max(1, int(fps // 2))
    total_frames = video_info['total_frames']
    selected_frames = list(range(0, total_frames, frame_step))
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    # 姿态检测
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    # 跳跃分析
    analyzer = JumpAnalyzer(fps=fps / frame_step)
    analysis_result = analyzer.analyze_jump_sequence(pose_results)
    
    processor.release()
    
    return analysis_result, video_info


def create_comparison_charts(analysis1, analysis2, video_names):
    """创建对比图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('跳跃动作对比分析', fontsize=16, fontweight='bold')
    
    # 1. 身体中心轨迹对比
    ax = axes[0, 0]
    for i, (analysis, name, color) in enumerate(zip([analysis1, analysis2], video_names, ['blue', 'red'])):
        body_centers = analysis.get('body_centers', [])
        valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
        
        if valid_centers:
            frames = [i for i, _ in valid_centers]
            y_coords = [center[1] for _, center in valid_centers]
            ax.plot(frames, y_coords, color=color, linewidth=2, marker='o', label=name, alpha=0.7)
    
    ax.set_xlabel('帧数')
    ax.set_ylabel('垂直位置')
    ax.set_title('身体中心轨迹对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 跳跃指标对比
    ax = axes[0, 1]
    metrics = ['jump_height_pixels', 'takeoff_duration', 'total_duration']
    metric_names = ['跳跃高度\n(像素)', '起跳时间\n(秒)', '总时间\n(秒)']
    
    values1 = []
    values2 = []
    
    for metric in metrics:
        val1 = analysis1.get('jump_metrics', {}).get(metric, 0)
        val2 = analysis2.get('jump_metrics', {}).get(metric, 0)
        values1.append(abs(val1) if val1 is not None else 0)  # 使用绝对值处理负值
        values2.append(abs(val2) if val2 is not None else 0)
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values1, width, label=video_names[0], color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, values2, width, label=video_names[1], color='red', alpha=0.7)
    
    ax.set_xlabel('指标')
    ax.set_ylabel('数值')
    ax.set_title('跳跃指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 力量评估雷达图对比
    ax = axes[0, 2]
    categories = ['爆发力', '核心力量', '协调性']
    
    # 获取两个人的力量数据
    strength1 = analysis1.get('strength_assessment', {})
    strength2 = analysis2.get('strength_assessment', {})
    
    values1 = [
        strength1.get('explosive_power', 0),
        strength1.get('core_strength', 0),
        strength1.get('coordination', 0)
    ]
    
    values2 = [
        strength2.get('explosive_power', 0),
        strength2.get('core_strength', 0),
        strength2.get('coordination', 0)
    ]
    
    # 雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values1 += values1[:1]  # 闭合图形
    values2 += values2[:1]
    angles += angles[:1]
    
    # 清除当前轴并创建极坐标图
    axes[0, 2].remove()
    ax = fig.add_subplot(2, 3, 3, projection='polar')
    
    # 绘制雷达图
    ax.plot(angles, values1, 'o-', linewidth=2, color='blue', label=video_names[0])
    ax.fill(angles, values1, alpha=0.25, color='blue')
    ax.plot(angles, values2, 'o-', linewidth=2, color='red', label=video_names[1])
    ax.fill(angles, values2, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('力量评估对比')
    ax.legend()
    ax.grid(True)
    
    # 4. 综合得分对比
    ax = axes[1, 0]
    scores1 = [
        analysis1.get('strength_assessment', {}).get('explosive_power', 0),
        analysis1.get('strength_assessment', {}).get('core_strength', 0),
        analysis1.get('strength_assessment', {}).get('coordination', 0),
        analysis1.get('strength_assessment', {}).get('overall_score', 0)
    ]
    
    scores2 = [
        analysis2.get('strength_assessment', {}).get('explosive_power', 0),
        analysis2.get('strength_assessment', {}).get('core_strength', 0),
        analysis2.get('strength_assessment', {}).get('coordination', 0),
        analysis2.get('strength_assessment', {}).get('overall_score', 0)
    ]
    
    score_names = ['爆发力', '核心力量', '协调性', '综合得分']
    x = np.arange(len(score_names))
    
    bars1 = ax.bar(x - width/2, scores1, width, label=video_names[0], color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, scores2, width, label=video_names[1], color='red', alpha=0.7)
    
    ax.set_xlabel('评估项目')
    ax.set_ylabel('得分')
    ax.set_title('力量评估得分对比')
    ax.set_xticks(x)
    ax.set_xticklabels(score_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. 姿态稳定性对比
    ax = axes[1, 1]
    phases = ['preparation_posture', 'takeoff_posture', 'landing_posture']
    phase_names = ['准备阶段', '起跳阶段', '落地阶段']
    
    stability1 = []
    stability2 = []
    
    for phase in phases:
        posture1 = analysis1.get('posture_analysis', {}).get(phase, {})
        posture2 = analysis2.get('posture_analysis', {}).get(phase, {})
        
        stability1.append(posture1.get('stability_score', 0) or 0)
        stability2.append(posture2.get('stability_score', 0) or 0)
    
    x = np.arange(len(phase_names))
    bars1 = ax.bar(x - width/2, stability1, width, label=video_names[0], color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, stability2, width, label=video_names[1], color='red', alpha=0.7)
    
    ax.set_xlabel('跳跃阶段')
    ax.set_ylabel('稳定性得分')
    ax.set_title('姿态稳定性对比')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. 改进建议对比
    ax = axes[1, 2]
    ax.axis('off')
    
    # 比较分析文本
    overall1 = analysis1.get('strength_assessment', {}).get('overall_score', 0)
    overall2 = analysis2.get('strength_assessment', {}).get('overall_score', 0)
    
    if overall1 > overall2:
        winner = video_names[0]
        difference = overall1 - overall2
    else:
        winner = video_names[1]
        difference = overall2 - overall1
    
    comparison_text = f"""
对比分析结果:

🏆 综合表现更优: {winner}
📊 得分差距: {difference:.3f}

各项对比:
• 爆发力: {video_names[0]} vs {video_names[1]}
  {scores1[0]:.3f} vs {scores2[0]:.3f}

• 核心力量: {video_names[0]} vs {video_names[1]}
  {scores1[1]:.3f} vs {scores2[1]:.3f}

• 协调性: {video_names[0]} vs {video_names[1]}
  {scores1[2]:.3f} vs {scores2[2]:.3f}

训练建议:
🔸 两人可以互相学习对方的优势
🔸 针对弱项进行专项训练
🔸 保持优势项目的训练强度
    """
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    ax.set_title('对比分析总结')
    
    plt.tight_layout()
    
    # 转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64


def generate_comparison_html_report(analysis1, analysis2, video_info1, video_info2, video_names, output_path):
    """生成对比HTML报告"""
    
    # 创建对比图表
    comparison_chart = create_comparison_charts(analysis1, analysis2, video_names)
    
    # 获取视频文件路径（相对路径）
    video_path1 = f"../test_videos/{video_names[0]}"
    video_path2 = f"../test_videos/{video_names[1]}"
    
    # HTML模板
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>跳跃动作对比分析报告</title>
        <style>
            body {{
                font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            .video-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 30px 0;
            }}
            .video-card {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .video-card h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .video-player {{
                width: 100%;
                max-width: 400px;
                height: 300px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .metrics-comparison {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 30px 0;
            }}
            .person-metrics {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
            }}
            .person1 {{
                border-left: 4px solid #3498db;
            }}
            .person2 {{
                border-left: 4px solid #e74c3c;
            }}
            .metric-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #bdc3c7;
            }}
            .metric-label {{
                font-weight: bold;
                color: #34495e;
            }}
            .metric-value {{
                font-size: 18px;
                color: #2c3e50;
            }}
            .score-bar {{
                background-color: #ecf0f1;
                height: 20px;
                border-radius: 10px;
                margin: 10px 0;
                overflow: hidden;
            }}
            .score-fill-person1 {{
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 10px;
                transition: width 0.3s ease;
            }}
            .score-fill-person2 {{
                height: 100%;
                background: linear-gradient(90deg, #e74c3c, #f39c12);
                border-radius: 10px;
                transition: width 0.3s ease;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .winner-badge {{
                background: linear-gradient(45deg, #f39c12, #f1c40f);
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
                display: inline-block;
                margin-left: 10px;
            }}
            .video-info {{
                background: #34495e;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 12px;
            }}
            .comparison-summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                margin: 30px 0;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .summary-item {{
                text-align: center;
                padding: 15px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }}
            .summary-value {{
                font-size: 24px;
                font-weight: bold;
                display: block;
            }}
            .summary-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏃‍♂️ 跳跃动作对比分析报告</h1>
            
            <h2>📹 原始视频对比</h2>
            <div class="video-container">
                <div class="video-card person1">
                    <h3>{video_names[0]}</h3>
                    <video class="video-player" controls>
                        <source src="{video_path1}" type="video/mp4">
                        您的浏览器不支持视频播放。
                    </video>
                    <div class="video-info">
                        📏 分辨率: {video_info1.get('width', 'N/A')} × {video_info1.get('height', 'N/A')}<br>
                        🎬 帧率: {video_info1.get('fps', 0):.1f} FPS<br>
                        ⏱️ 时长: {video_info1.get('duration', 0):.2f} 秒
                    </div>
                </div>
                
                <div class="video-card person2">
                    <h3>{video_names[1]}</h3>
                    <video class="video-player" controls>
                        <source src="{video_path2}" type="video/mp4">
                        您的浏览器不支持视频播放。
                    </video>
                    <div class="video-info">
                        📏 分辨率: {video_info2.get('width', 'N/A')} × {video_info2.get('height', 'N/A')}<br>
                        🎬 帧率: {video_info2.get('fps', 0):.1f} FPS<br>
                        ⏱️ 时长: {video_info2.get('duration', 0):.2f} 秒
                    </div>
                </div>
            </div>
    """
    
    # 获取分析数据
    jump_metrics1 = analysis1.get('jump_metrics', {})
    jump_metrics2 = analysis2.get('jump_metrics', {})
    strength1 = analysis1.get('strength_assessment', {})
    strength2 = analysis2.get('strength_assessment', {})
    
    # 确定获胜者
    overall1 = strength1.get('overall_score', 0)
    overall2 = strength2.get('overall_score', 0)
    
    if overall1 > overall2:
        winner_text1 = '<span class="winner-badge">🏆 综合优胜</span>'
        winner_text2 = ''
    else:
        winner_text1 = ''
        winner_text2 = '<span class="winner-badge">🏆 综合优胜</span>'
    
    # 添加对比摘要
    html_content += f"""
            <div class="comparison-summary">
                <h3 style="margin-top: 0; text-align: center;">📊 对比摘要</h3>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="summary-value">{overall1:.3f}</span>
                        <span class="summary-label">{video_names[0]} 综合得分</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-value">{overall2:.3f}</span>
                        <span class="summary-label">{video_names[1]} 综合得分</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-value">{abs(overall1 - overall2):.3f}</span>
                        <span class="summary-label">得分差距</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-value">{'✓' if 'error' not in jump_metrics1 and 'error' not in jump_metrics2 else '✗'}</span>
                        <span class="summary-label">分析完整性</span>
                    </div>
                </div>
            </div>
            
            <h2>📊 详细指标对比</h2>
            <div class="metrics-comparison">
                <div class="person-metrics person1">
                    <h3>{video_names[0]} {winner_text1}</h3>
    """
    
    # 添加第一个人的指标
    if 'error' not in jump_metrics1:
        html_content += f"""
                    <div class="metric-item">
                        <span class="metric-label">跳跃高度</span>
                        <span class="metric-value">{jump_metrics1.get('jump_height_pixels', 0):.1f} 像素</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">起跳时间</span>
                        <span class="metric-value">{abs(jump_metrics1.get('takeoff_duration', 0)):.3f} 秒</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">总时间</span>
                        <span class="metric-value">{jump_metrics1.get('total_duration', 0):.3f} 秒</span>
                    </div>
        """
    
    if 'error' not in strength1:
        html_content += f"""
                    <div class="metric-item">
                        <span class="metric-label">综合得分</span>
                        <span class="metric-value">{strength1.get('overall_score', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person1" style="width: {strength1.get('overall_score', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">爆发力</span>
                        <span class="metric-value">{strength1.get('explosive_power', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person1" style="width: {strength1.get('explosive_power', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">核心力量</span>
                        <span class="metric-value">{strength1.get('core_strength', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person1" style="width: {strength1.get('core_strength', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">协调性</span>
                        <span class="metric-value">{strength1.get('coordination', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person1" style="width: {strength1.get('coordination', 0) * 100}%"></div>
                    </div>
        """
    
    html_content += f"""
                </div>
                
                <div class="person-metrics person2">
                    <h3>{video_names[1]} {winner_text2}</h3>
    """
    
    # 添加第二个人的指标
    if 'error' not in jump_metrics2:
        html_content += f"""
                    <div class="metric-item">
                        <span class="metric-label">跳跃高度</span>
                        <span class="metric-value">{jump_metrics2.get('jump_height_pixels', 0):.1f} 像素</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">起跳时间</span>
                        <span class="metric-value">{abs(jump_metrics2.get('takeoff_duration', 0)):.3f} 秒</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">总时间</span>
                        <span class="metric-value">{jump_metrics2.get('total_duration', 0):.3f} 秒</span>
                    </div>
        """
    
    if 'error' not in strength2:
        html_content += f"""
                    <div class="metric-item">
                        <span class="metric-label">综合得分</span>
                        <span class="metric-value">{strength2.get('overall_score', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person2" style="width: {strength2.get('overall_score', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">爆发力</span>
                        <span class="metric-value">{strength2.get('explosive_power', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person2" style="width: {strength2.get('explosive_power', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">核心力量</span>
                        <span class="metric-value">{strength2.get('core_strength', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person2" style="width: {strength2.get('core_strength', 0) * 100}%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span class="metric-label">协调性</span>
                        <span class="metric-value">{strength2.get('coordination', 0):.3f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill-person2" style="width: {strength2.get('coordination', 0) * 100}%"></div>
                    </div>
        """
    
    html_content += f"""
                </div>
            </div>
            
            <h2>📈 可视化对比分析</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{comparison_chart}" alt="跳跃对比分析图表">
            </div>
            
            <h2>📝 对比分析结论</h2>
            <div class="comparison-summary">
                <h4>🔍 分析要点：</h4>
                <ul>
    """
    
    # 添加分析结论
    if overall1 > overall2:
        winner = video_names[0]
        loser = video_names[1]
        score_diff = overall1 - overall2
    else:
        winner = video_names[1]
        loser = video_names[0]
        score_diff = overall2 - overall1
    
    html_content += f"""
                    <li><strong>综合表现：</strong>{winner} 在综合评分中领先 {score_diff:.3f} 分</li>
                    <li><strong>优势分析：</strong>两位测试者各有特色，建议相互学习对方的优势技术</li>
                    <li><strong>改进方向：</strong>针对各自的薄弱环节进行专项训练</li>
                    <li><strong>训练建议：</strong>保持现有优势的同时，重点提升协调性和核心力量</li>
                </ul>
                
                <h4>🎯 个性化建议：</h4>
                <p><strong>{video_names[0]}：</strong>
    """
    
    # 个性化建议
    explosive1 = strength1.get('explosive_power', 0)
    core1 = strength1.get('core_strength', 0)
    coord1 = strength1.get('coordination', 0)
    
    suggestions1 = []
    if explosive1 < 0.5:
        suggestions1.append("加强爆发力训练（深蹲跳、蛙跳）")
    if core1 < 0.5:
        suggestions1.append("增强核心力量（平板支撑、俄罗斯转体）")
    if coord1 < 0.5:
        suggestions1.append("提高协调性（单腿平衡、敏捷训练）")
    
    if not suggestions1:
        suggestions1.append("各项指标均衡，继续保持当前训练强度")
    
    html_content += "、".join(suggestions1)
    
    explosive2 = strength2.get('explosive_power', 0)
    core2 = strength2.get('core_strength', 0)
    coord2 = strength2.get('coordination', 0)
    
    suggestions2 = []
    if explosive2 < 0.5:
        suggestions2.append("加强爆发力训练（深蹲跳、蛙跳）")
    if core2 < 0.5:
        suggestions2.append("增强核心力量（平板支撑、俄罗斯转体）")
    if coord2 < 0.5:
        suggestions2.append("提高协调性（单腿平衡、敏捷训练）")
    
    if not suggestions2:
        suggestions2.append("各项指标均衡，继续保持当前训练强度")
    
    html_content += f"</p><p><strong>{video_names[1]}：</strong>" + "、".join(suggestions2)
    
    # 获取当前时间
    import datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content += f"""
                </p>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px;">
                <p>本对比报告由跳跃姿态分析系统自动生成</p>
                <p>分析时间: {current_time}</p>
                <p>💡 建议定期重复测试以跟踪进步情况</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True


def main():
    """主函数"""
    print("=== 生成跳跃分析对比报告 ===\n")
    
    # 视频文件列表
    video_files = ['M1.mp4', 'M2.mp4']
    video_names = ['M1.mp4', 'M2.mp4']
    
    # 分析两个视频
    analyses = []
    video_infos = []
    
    for video_file in video_files:
        video_path = os.path.join('test_videos', video_file)
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return
        
        print(f"分析视频: {video_file}")
        analysis_result, video_info = analyze_video_for_comparison(video_path)
        
        if analysis_result is None:
            print(f"❌ 视频 {video_file} 分析失败")
            return
        
        analyses.append(analysis_result)
        video_infos.append(video_info)
        print(f"✅ {video_file} 分析完成")
    
    # 生成对比报告
    output_path = os.path.join('outputs', 'jump_comparison_report.html')
    print(f"\n生成对比报告: {output_path}")
    
    success = generate_comparison_html_report(
        analyses[0], analyses[1],
        video_infos[0], video_infos[1],
        video_names,
        output_path
    )
    
    if success:
        print("✅ 对比报告生成成功！")
        print(f"📁 报告路径: {output_path}")
        print("🌐 请用浏览器打开HTML文件查看对比分析")
        
        # 显示简要对比结果
        strength1 = analyses[0].get('strength_assessment', {})
        strength2 = analyses[1].get('strength_assessment', {})
        
        if 'error' not in strength1 and 'error' not in strength2:
            overall1 = strength1.get('overall_score', 0)
            overall2 = strength2.get('overall_score', 0)
            
            print(f"\n📊 简要对比结果:")
            print(f"   {video_names[0]} 综合得分: {overall1:.3f}")
            print(f"   {video_names[1]} 综合得分: {overall2:.3f}")
            
            if overall1 > overall2:
                print(f"   🏆 {video_names[0]} 综合表现更优，领先 {overall1 - overall2:.3f} 分")
            else:
                print(f"   🏆 {video_names[1]} 综合表现更优，领先 {overall2 - overall1:.3f} 分")
    else:
        print("❌ 对比报告生成失败")


if __name__ == "__main__":
    import cv2
    import datetime
    
    main()
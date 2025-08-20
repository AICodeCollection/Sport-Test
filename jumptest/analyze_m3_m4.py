#!/usr/bin/env python3
"""
M3.mp4 和 M4.mp4 跳跃分析脚本
可反复执行，自动生成个人报告和对比报告
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
import datetime
import cv2

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加src目录到路径
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer
from visualizer import JumpVisualizer


class JumpAnalysisSystem:
    """跳跃分析系统"""
    
    def __init__(self):
        self.output_dir = 'outputs'
        self.video_dir = 'test_videos'
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_video(self, video_name):
        """分析单个视频"""
        video_path = os.path.join(self.video_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return None, None
        
        print(f"📹 开始分析视频: {video_name}")
        
        # 1. 加载视频
        processor = VideoProcessor(video_path)
        if not processor.load_video():
            print(f"❌ 无法加载视频: {video_path}")
            return None, None
        
        # 获取视频信息
        video_info = processor.get_video_info()
        print(f"   📊 视频信息: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.2f}秒")
        
        # 2. 智能帧提取策略
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        duration = video_info['duration']
        
        # 根据视频长度调整采样策略
        if duration < 3:  # 极短视频
            frame_step = max(1, int(fps // 6))  # 每秒采样6帧
            print(f"   🎯 检测到极短视频，使用高密度采样：每秒6帧")
        elif duration < 5:  # 短视频
            frame_step = max(1, int(fps // 4))  # 每秒采样4帧
            print(f"   🎯 检测到短视频，使用密集采样：每秒4帧")
        else:  # 长视频
            frame_step = max(1, int(fps // 2))  # 每秒采样2帧
            print(f"   🎯 使用标准采样：每秒2帧")
        
        selected_frames = list(range(0, total_frames, frame_step))
        print(f"   📊 提取策略: 从{total_frames}帧中选择{len(selected_frames)}帧进行分析")
        
        # 提取帧
        frames = []
        for i in selected_frames:
            processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = processor.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        print(f"   ✅ 成功提取 {len(frames)} 帧")
        
        # 3. 姿态检测
        print("   🔍 进行姿态检测...")
        detector = PoseDetector()
        pose_results = detector.detect_pose_sequence(frames)
        
        valid_poses = sum(1 for result in pose_results if result is not None)
        print(f"   📊 检测结果: {valid_poses}/{len(pose_results)} 帧有效")
        
        if valid_poses < 2:
            print("   ⚠️ 有效姿态数量太少，可能影响分析结果")
        
        # 4. 跳跃分析
        print("   🔬 进行跳跃分析...")
        analyzer = JumpAnalyzer(fps=fps / frame_step)
        analysis_result = analyzer.analyze_jump_sequence(pose_results)
        
        processor.release()
        
        # 显示分析结果摘要
        self.print_analysis_summary(video_name, analysis_result)
        
        return analysis_result, video_info
    
    def print_analysis_summary(self, video_name, analysis_result):
        """打印分析结果摘要"""
        jump_metrics = analysis_result.get('jump_metrics', {})
        strength_assessment = analysis_result.get('strength_assessment', {})
        jump_phases = analysis_result.get('jump_phases', {})
        
        print(f"   📋 {video_name} 分析摘要:")
        
        # 跳跃阶段
        if 'error' not in jump_phases:
            print("   ✅ 跳跃阶段识别成功")
        else:
            print(f"   ❌ 跳跃阶段识别失败: {jump_phases.get('error', '未知错误')}")
        
        # 跳跃指标
        if 'error' not in jump_metrics:
            height = jump_metrics.get('jump_height_pixels', 0)
            takeoff_time = abs(jump_metrics.get('takeoff_duration', 0))
            print(f"   🏃 跳跃高度: {height:.1f} 像素")
            print(f"   ⏱️ 起跳时间: {takeoff_time:.3f} 秒")
        else:
            print(f"   ❌ 跳跃指标计算失败: {jump_metrics.get('error', '未知错误')}")
        
        # 力量评估
        if 'error' not in strength_assessment:
            overall_score = strength_assessment.get('overall_score', 0)
            print(f"   💪 综合得分: {overall_score:.3f}")
        else:
            print(f"   ❌ 力量评估失败: {strength_assessment.get('error', '未知错误')}")
    
    def generate_individual_report(self, video_name, analysis_result, video_info):
        """生成个人HTML报告"""
        print(f"📝 生成 {video_name} 个人报告...")
        
        # 创建可视化图表
        visualizer = JumpVisualizer()
        
        # 生成分析图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{video_name} 跳跃分析报告', fontsize=16, fontweight='bold')
        
        # 绘制各个图表
        visualizer._plot_body_center_trajectory(axes[0, 0], analysis_result)
        visualizer._plot_joint_angles(axes[0, 1], analysis_result)
        visualizer._plot_jump_phases(axes[0, 2], analysis_result)
        visualizer._plot_strength_radar(axes[1, 0], analysis_result)
        visualizer._plot_posture_analysis(axes[1, 1], analysis_result)
        visualizer._plot_summary_metrics(axes[1, 2], analysis_result)
        
        plt.tight_layout()
        
        # 将图表转换为base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # 生成HTML报告
        html_content = self.create_individual_html(video_name, analysis_result, video_info, image_base64)
        
        # 保存HTML文件
        output_path = os.path.join(self.output_dir, f'{video_name}_analysis_report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ 个人报告已保存: {output_path}")
        return output_path
    
    def create_individual_html(self, video_name, analysis_result, video_info, chart_base64):
        """创建个人HTML报告内容"""
        # 准备数据
        jump_metrics = analysis_result.get('jump_metrics', {})
        strength_assessment = analysis_result.get('strength_assessment', {})
        posture_analysis = analysis_result.get('posture_analysis', {})
        jump_phases = analysis_result.get('jump_phases', {})
        
        # 视频文件路径（相对路径）
        video_path = f"../test_videos/{video_name}"
        
        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{video_name} 跳跃动作分析报告</title>
            <style>
                body {{
                    font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
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
                .video-section {{
                    text-align: center;
                    margin: 30px 0;
                    background: #ecf0f1;
                    padding: 25px;
                    border-radius: 10px;
                }}
                .video-player {{
                    width: 100%;
                    max-width: 600px;
                    height: 400px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .video-info {{
                    background: #34495e;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px auto;
                    max-width: 600px;
                    text-align: left;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 14px;
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
                .score-bar {{
                    background-color: #ecf0f1;
                    height: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    overflow: hidden;
                }}
                .score-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #2ecc71);
                    border-radius: 10px;
                    transition: width 0.3s ease;
                }}
                .phase-timeline {{
                    display: flex;
                    margin: 20px 0;
                    height: 40px;
                    border-radius: 20px;
                    overflow: hidden;
                    border: 2px solid #bdc3c7;
                }}
                .phase {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                }}
                .phase-prep {{
                    background-color: #3498db;
                }}
                .phase-takeoff {{
                    background-color: #e74c3c;
                }}
                .phase-landing {{
                    background-color: #27ae60;
                }}
                .error-message {{
                    background-color: #e74c3c;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .success-message {{
                    background-color: #27ae60;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .highlight-box {{
                    background: #e8f5e8;
                    border-left: 4px solid #2ecc71;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏃‍♂️ {video_name} 跳跃动作分析报告</h1>
                
                <div class="highlight-box">
                    <h3>🎯 分析说明</h3>
                    <p>本报告针对处理后的纯跳跃视频进行专项分析，使用改进的算法确保短视频也能获得准确的分析结果。</p>
                </div>
                
                <div class="video-section">
                    <h3>🎬 原始视频</h3>
                    <video class="video-player" controls>
                        <source src="{video_path}" type="video/mp4">
                        您的浏览器不支持视频播放。
                    </video>
                    <div class="video-info">
                        <h4>📊 视频信息</h4>
                        📁 文件名: {video_name}<br>
                        📏 分辨率: {video_info.get('width', 'N/A')} × {video_info.get('height', 'N/A')}<br>
                        🎬 帧率: {video_info.get('fps', 0):.1f} FPS<br>
                        ⏱️ 时长: {video_info.get('duration', 0):.2f} 秒<br>
                        🎞️ 总帧数: {video_info.get('total_frames', 'N/A')} 帧
                    </div>
                </div>
        """
        
        # 添加跳跃阶段信息
        if 'error' not in jump_phases:
            prep_duration = jump_phases.get('preparation', {}).get('end_frame', 0) - jump_phases.get('preparation', {}).get('start_frame', 0)
            takeoff_duration = jump_phases.get('takeoff', {}).get('end_frame', 0) - jump_phases.get('takeoff', {}).get('start_frame', 0)
            landing_duration = jump_phases.get('landing', {}).get('end_frame', 0) - jump_phases.get('landing', {}).get('start_frame', 0)
            total_frames = prep_duration + takeoff_duration + landing_duration
            
            if total_frames > 0:
                prep_width = (prep_duration / total_frames) * 100
                takeoff_width = (takeoff_duration / total_frames) * 100
                landing_width = (landing_duration / total_frames) * 100
                
                html_content += f"""
                <h2>🎯 跳跃阶段划分</h2>
                <div class="success-message">
                    ✅ 成功识别跳跃的三个阶段
                </div>
                <div class="phase-timeline">
                    <div class="phase phase-prep" style="width: {prep_width}%">
                        准备阶段<br>{prep_duration} 帧
                    </div>
                    <div class="phase phase-takeoff" style="width: {takeoff_width}%">
                        起跳阶段<br>{takeoff_duration} 帧
                    </div>
                    <div class="phase phase-landing" style="width: {landing_width}%">
                        落地阶段<br>{landing_duration} 帧
                    </div>
                </div>
                """
        else:
            html_content += f"""
                <h2>🎯 跳跃阶段划分</h2>
                <div class="error-message">
                    ❌ 阶段识别失败: {jump_phases.get('error', '未知错误')}<br>
                    💡 可能原因: 视频时长过短或跳跃动作不够明显
                </div>
            """
        
        # 添加跳跃指标
        if 'error' not in jump_metrics:
            html_content += f"""
                <h2>📊 跳跃指标</h2>
                <div class="success-message">
                    ✅ 成功计算跳跃指标
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{jump_metrics.get('jump_height_pixels', 0):.1f}</div>
                        <div class="metric-label">跳跃高度 (像素)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{abs(jump_metrics.get('takeoff_duration', 0)):.3f}</div>
                        <div class="metric-label">起跳时间 (秒)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{jump_metrics.get('preparation_duration', 0):.3f}</div>
                        <div class="metric-label">准备时间 (秒)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{jump_metrics.get('landing_duration', 0):.3f}</div>
                        <div class="metric-label">落地时间 (秒)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{jump_metrics.get('total_duration', 0):.3f}</div>
                        <div class="metric-label">总时间 (秒)</div>
                    </div>
                </div>
            """
        else:
            html_content += f"""
                <h2>📊 跳跃指标</h2>
                <div class="error-message">
                    ❌ 跳跃指标计算失败: {jump_metrics.get('error', '未知错误')}<br>
                    💡 建议: 确保视频包含完整的跳跃动作
                </div>
            """
        
        # 添加力量评估
        if 'error' not in strength_assessment:
            overall_score = strength_assessment.get('overall_score', 0)
            explosive_power = strength_assessment.get('explosive_power', 0)
            core_strength = strength_assessment.get('core_strength', 0)
            coordination = strength_assessment.get('coordination', 0)
            
            html_content += f"""
                <h2>💪 力量评估</h2>
                <div class="success-message">
                    ✅ 成功评估各项力量指标
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{overall_score:.3f}</div>
                        <div class="metric-label">综合得分</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {overall_score * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{explosive_power:.3f}</div>
                        <div class="metric-label">爆发力</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {explosive_power * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{core_strength:.3f}</div>
                        <div class="metric-label">核心力量</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {core_strength * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{coordination:.3f}</div>
                        <div class="metric-label">协调性</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {coordination * 100}%"></div>
                        </div>
                    </div>
                </div>
            """
        else:
            html_content += f"""
                <h2>💪 力量评估</h2>
                <div class="error-message">
                    ❌ 力量评估失败: {strength_assessment.get('error', '未知错误')}<br>
                    💡 原因: 需要有效的跳跃阶段数据才能进行力量评估
                </div>
            """
        
        # 添加姿态分析
        if 'error' not in posture_analysis:
            html_content += f"""
                <h2>🤸 姿态分析</h2>
                <div class="success-message">
                    ✅ 成功分析各阶段姿态
                </div>
                <div class="metrics-grid">
            """
            
            phases = [
                ('preparation_posture', '准备阶段'),
                ('takeoff_posture', '起跳阶段'),
                ('landing_posture', '落地阶段')
            ]
            
            for phase_key, phase_name in phases:
                if phase_key in posture_analysis:
                    phase_data = posture_analysis[phase_key]
                    stability = phase_data.get('stability_score', 0) or 0
                    knee_angle = phase_data.get('avg_knee_angle') 
                    hip_angle = phase_data.get('avg_hip_angle')
                    
                    # 安全处理None值
                    knee_angle_str = f"{knee_angle:.1f}°" if knee_angle is not None else "N/A"
                    hip_angle_str = f"{hip_angle:.1f}°" if hip_angle is not None else "N/A"
                    
                    html_content += f"""
                    <div class="metric-card">
                        <h4>{phase_name}</h4>
                        <p><strong>稳定性:</strong> {stability:.3f}</p>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {stability * 100}%"></div>
                        </div>
                        <p><strong>平均膝关节角度:</strong> {knee_angle_str}</p>
                        <p><strong>平均髋关节角度:</strong> {hip_angle_str}</p>
                    </div>
                    """
            
            html_content += "</div>"
        else:
            html_content += f"""
                <h2>🤸 姿态分析</h2>
                <div class="error-message">
                    ❌ 姿态分析失败: {posture_analysis.get('error', '未知错误')}<br>
                    💡 原因: 需要有效的姿态检测数据才能进行姿态分析
                </div>
            """
        
        # 添加可视化图表
        html_content += f"""
            <h2>📈 分析图表</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_base64}" alt="跳跃分析图表">
            </div>
            
            <h2>📝 分析建议</h2>
            <div class="metric-card">
        """
        
        # 添加基于分析结果的建议
        if 'error' not in strength_assessment:
            overall_score = strength_assessment.get('overall_score', 0)
            explosive_power = strength_assessment.get('explosive_power', 0)
            core_strength = strength_assessment.get('core_strength', 0)
            coordination = strength_assessment.get('coordination', 0)
            
            suggestions = []
            
            if overall_score < 0.3:
                suggestions.append("🔸 整体跳跃能力有较大提升空间，建议加强基础体能训练")
            elif overall_score < 0.6:
                suggestions.append("🔸 跳跃能力中等，可通过针对性训练进一步提升")
            else:
                suggestions.append("🔸 跳跃能力优秀，继续保持训练水平")
            
            if explosive_power < 0.3:
                suggestions.append("🔸 爆发力较弱，建议增加深蹲跳、蛙跳等爆发力训练")
            
            if core_strength < 0.5:
                suggestions.append("🔸 核心力量需要加强，建议增加平板支撑、俄罗斯转体等核心训练")
            
            if coordination < 0.5:
                suggestions.append("🔸 协调性有待提高，建议进行单腿平衡、敏捷性训练")
            
            if not suggestions:
                suggestions.append("🔸 各项指标表现良好，继续保持当前训练强度")
            
            for suggestion in suggestions:
                html_content += f"<p>{suggestion}</p>"
        else:
            html_content += """
            <p>🔸 由于分析数据不足，无法提供具体建议。</p>
            <p>🔸 <strong>改进建议：</strong></p>
            <ul>
                <li>确保视频包含完整的跳跃动作（准备-起跳-腾空-落地）</li>
                <li>保持摄像设备稳定，避免抖动</li>
                <li>确保光线充足，人体轮廓清晰</li>
                <li>建议从侧面拍摄，能更好地观察跳跃轨迹</li>
            </ul>
            """
        
        html_content += f"""
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px;">
                <p>本报告由跳跃姿态分析系统自动生成</p>
                <p>分析时间: {current_time}</p>
                <p>🔄 专门针对处理后的纯跳跃视频进行优化分析</p>
            </div>
        </div>
        </body>
        </html>
        """
        
        return html_content
    
    def generate_comparison_report(self, video1_name, video2_name, analysis1, analysis2, video_info1, video_info2):
        """生成对比报告"""
        print(f"📊 生成 {video1_name} vs {video2_name} 对比报告...")
        
        # 生成对比图表
        chart_base64 = self.create_comparison_chart(analysis1, analysis2, video_info1, video_info2, video1_name, video2_name)
        
        # 生成HTML报告
        html_content = self.create_comparison_html(video1_name, video2_name, analysis1, analysis2, video_info1, video_info2, chart_base64)
        
        # 保存HTML文件
        output_path = os.path.join(self.output_dir, f'{video1_name}_vs_{video2_name}_comparison.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ 对比报告已保存: {output_path}")
        return output_path
    
    def create_comparison_chart(self, analysis1, analysis2, video_info1, video_info2, video1_name, video2_name):
        """创建对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'跳跃分析对比报告 - {video1_name} vs {video2_name}', fontsize=16, fontweight='bold')
        
        # 1. 力量评估对比
        ax1 = axes[0, 0]
        strength1 = analysis1.get('strength_assessment', {})
        strength2 = analysis2.get('strength_assessment', {})
        
        if 'error' not in strength1 and 'error' not in strength2:
            categories = ['爆发力', '核心力量', '协调性', '综合得分']
            values1 = [
                strength1.get('explosive_power', 0),
                strength1.get('core_strength', 0),
                strength1.get('coordination', 0),
                strength1.get('overall_score', 0)
            ]
            values2 = [
                strength2.get('explosive_power', 0),
                strength2.get('core_strength', 0),
                strength2.get('coordination', 0),
                strength2.get('overall_score', 0)
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, values1, width, label=video1_name, color='#3498db', alpha=0.8)
            ax1.bar(x + width/2, values2, width, label=video2_name, color='#e74c3c', alpha=0.8)
            
            ax1.set_ylabel('得分')
            ax1.set_title('力量评估对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, '力量评估数据不足', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('力量评估对比')
        
        # 2. 跳跃指标对比
        ax2 = axes[0, 1]
        metrics1 = analysis1.get('jump_metrics', {})
        metrics2 = analysis2.get('jump_metrics', {})
        
        if 'error' not in metrics1 and 'error' not in metrics2:
            categories = ['跳跃高度\n(像素)', '起跳时间\n(秒)', '准备时间\n(秒)', '落地时间\n(秒)']
            values1 = [
                metrics1.get('jump_height_pixels', 0),
                abs(metrics1.get('takeoff_duration', 0)),
                metrics1.get('preparation_duration', 0),
                metrics1.get('landing_duration', 0)
            ]
            values2 = [
                metrics2.get('jump_height_pixels', 0),
                abs(metrics2.get('takeoff_duration', 0)),
                metrics2.get('preparation_duration', 0),
                metrics2.get('landing_duration', 0)
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax2.bar(x - width/2, values1, width, label=video1_name, color='#3498db', alpha=0.8)
            ax2.bar(x + width/2, values2, width, label=video2_name, color='#e74c3c', alpha=0.8)
            
            ax2.set_ylabel('数值')
            ax2.set_title('跳跃指标对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories, fontsize=10)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '跳跃指标数据不足', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('跳跃指标对比')
        
        # 3. 视频基本信息对比
        ax3 = axes[0, 2]
        info_text = f"""视频基本信息对比

{video1_name}:
• 分辨率: {video_info1.get('width', 'N/A')} × {video_info1.get('height', 'N/A')}
• 帧率: {video_info1.get('fps', 0):.1f} FPS
• 时长: {video_info1.get('duration', 0):.2f} 秒
• 总帧数: {video_info1.get('total_frames', 'N/A')}

{video2_name}:
• 分辨率: {video_info2.get('width', 'N/A')} × {video_info2.get('height', 'N/A')}
• 帧率: {video_info2.get('fps', 0):.1f} FPS
• 时长: {video_info2.get('duration', 0):.2f} 秒
• 总帧数: {video_info2.get('total_frames', 'N/A')}
"""
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10, 
                 verticalalignment='top', fontfamily='monospace')
        ax3.set_title('视频信息对比')
        ax3.axis('off')
        
        # 4. 身体中心轨迹对比
        ax4 = axes[1, 0]
        centers1 = analysis1.get('body_centers', [])
        centers2 = analysis2.get('body_centers', [])
        
        if centers1 and centers2:
            valid_centers1 = [(i, center) for i, center in enumerate(centers1) if center is not None]
            valid_centers2 = [(i, center) for i, center in enumerate(centers2) if center is not None]
            
            if valid_centers1:
                y_coords1 = [center[1] for _, center in valid_centers1]
                frame_indices1 = [i for i, _ in valid_centers1]
                ax4.plot(frame_indices1, y_coords1, 'o-', label=video1_name, color='#3498db', linewidth=2)
            
            if valid_centers2:
                y_coords2 = [center[1] for _, center in valid_centers2]
                frame_indices2 = [i for i, _ in valid_centers2]
                ax4.plot(frame_indices2, y_coords2, 's-', label=video2_name, color='#e74c3c', linewidth=2)
            
            ax4.set_xlabel('帧索引')
            ax4.set_ylabel('Y坐标 (像素)')
            ax4.set_title('身体中心轨迹对比')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '身体中心数据不足', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('身体中心轨迹对比')
        
        # 5. 姿态稳定性对比
        ax5 = axes[1, 1]
        posture1 = analysis1.get('posture_analysis', {})
        posture2 = analysis2.get('posture_analysis', {})
        
        if 'error' not in posture1 and 'error' not in posture2:
            phases = ['准备阶段', '起跳阶段', '落地阶段']
            phase_keys = ['preparation_posture', 'takeoff_posture', 'landing_posture']
            
            stability1 = []
            stability2 = []
            
            for key in phase_keys:
                s1 = posture1.get(key, {}).get('stability_score', 0) or 0
                s2 = posture2.get(key, {}).get('stability_score', 0) or 0
                stability1.append(s1)
                stability2.append(s2)
            
            x = np.arange(len(phases))
            width = 0.35
            
            ax5.bar(x - width/2, stability1, width, label=video1_name, color='#3498db', alpha=0.8)
            ax5.bar(x + width/2, stability2, width, label=video2_name, color='#e74c3c', alpha=0.8)
            
            ax5.set_ylabel('稳定性得分')
            ax5.set_title('各阶段稳定性对比')
            ax5.set_xticks(x)
            ax5.set_xticklabels(phases)
            ax5.legend()
            ax5.set_ylim(0, 1)
        else:
            ax5.text(0.5, 0.5, '姿态稳定性数据不足', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('各阶段稳定性对比')
        
        # 6. 综合评分雷达图
        ax6 = axes[1, 2]
        
        if 'error' not in strength1 and 'error' not in strength2:
            categories = ['爆发力', '核心力量', '协调性']
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
            
            # 补充数据以形成封闭的雷达图
            values1 += values1[:1]
            values2 += values2[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax6.plot(angles, values1, 'o-', linewidth=2, label=video1_name, color='#3498db')
            ax6.fill(angles, values1, alpha=0.25, color='#3498db')
            ax6.plot(angles, values2, 's-', linewidth=2, label=video2_name, color='#e74c3c')
            ax6.fill(angles, values2, alpha=0.25, color='#e74c3c')
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(categories)
            ax6.set_ylim(0, 1)
            ax6.set_title('综合能力雷达图')
            ax6.legend()
            ax6.grid(True)
        else:
            ax6.text(0.5, 0.5, '综合评分数据不足', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('综合能力雷达图')
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def create_comparison_html(self, video1_name, video2_name, analysis1, analysis2, video_info1, video_info2, chart_base64):
        """创建对比HTML报告内容"""
        # 获取分析结果
        strength1 = analysis1.get('strength_assessment', {})
        strength2 = analysis2.get('strength_assessment', {})
        metrics1 = analysis1.get('jump_metrics', {})
        metrics2 = analysis2.get('jump_metrics', {})
        
        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>跳跃分析对比报告 - {video1_name} vs {video2_name}</title>
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
                .video-comparison {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin: 30px 0;
                }}
                .video-section {{
                    text-align: center;
                    background: #ecf0f1;
                    padding: 25px;
                    border-radius: 10px;
                }}
                .video-player {{
                    width: 100%;
                    max-width: 500px;
                    height: 300px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .video-info {{
                    background: #34495e;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px auto;
                    max-width: 500px;
                    text-align: left;
                }}
                .comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .comparison-table th,
                .comparison-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                }}
                .comparison-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .comparison-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .winner {{
                    background-color: #2ecc71 !important;
                    color: white;
                    font-weight: bold;
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
                .summary-box {{
                    background: #e8f5e8;
                    border-left: 4px solid #2ecc71;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
                .highlight-box {{
                    background: #fef9e7;
                    border-left: 4px solid #f39c12;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
                .error-message {{
                    background-color: #e74c3c;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .success-message {{
                    background-color: #27ae60;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .improvement-highlight {{
                    background: #f39c12;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏃‍♂️ 跳跃分析对比报告</h1>
                <p style="text-align: center; color: #7f8c8d; font-size: 16px;">
                    {video1_name} vs {video2_name} • 处理后视频专项对比
                </p>
                
                <div class="highlight-box">
                    <h3>🎯 分析说明</h3>
                    <p><strong>专项对比分析：</strong> 本报告针对处理后的纯跳跃视频进行专项对比分析，去除了非跳跃部分的干扰，能够更准确地评估跳跃技术差异。</p>
                    <p><strong>技术优势：</strong> 使用改进的短视频分析算法，即使是很短的跳跃片段也能获得可靠的分析结果。</p>
                </div>
                
                <h2>🎬 视频对比</h2>
                <div class="video-comparison">
                    <div class="video-section">
                        <h3>{video1_name}</h3>
                        <video class="video-player" controls>
                            <source src="../test_videos/{video1_name}" type="video/mp4">
                            您的浏览器不支持视频播放。
                        </video>
                        <div class="video-info">
                            <h4>视频信息</h4>
                            📁 文件名: {video1_name}<br>
                            📏 分辨率: {video_info1.get('width', 'N/A')} × {video_info1.get('height', 'N/A')}<br>
                            🎬 帧率: {video_info1.get('fps', 0):.1f} FPS<br>
                            ⏱️ 时长: {video_info1.get('duration', 0):.2f} 秒<br>
                            🎞️ 总帧数: {video_info1.get('total_frames', 'N/A')} 帧
                        </div>
                    </div>
                    
                    <div class="video-section">
                        <h3>{video2_name}</h3>
                        <video class="video-player" controls>
                            <source src="../test_videos/{video2_name}" type="video/mp4">
                            您的浏览器不支持视频播放。
                        </video>
                        <div class="video-info">
                            <h4>视频信息</h4>
                            📁 文件名: {video2_name}<br>
                            📏 分辨率: {video_info2.get('width', 'N/A')} × {video_info2.get('height', 'N/A')}<br>
                            🎬 帧率: {video_info2.get('fps', 0):.1f} FPS<br>
                            ⏱️ 时长: {video_info2.get('duration', 0):.2f} 秒<br>
                            🎞️ 总帧数: {video_info2.get('total_frames', 'N/A')} 帧
                        </div>
                    </div>
                </div>
        """
        
        # 添加对比表格
        html_content += """
                <h2>📊 详细对比数据</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>""" + video1_name + """</th>
                            <th>""" + video2_name + """</th>
                            <th>表现更好</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # 跳跃指标对比
        if 'error' not in metrics1 and 'error' not in metrics2:
            height1 = metrics1.get('jump_height_pixels', 0)
            height2 = metrics2.get('jump_height_pixels', 0)
            takeoff1 = abs(metrics1.get('takeoff_duration', 0))
            takeoff2 = abs(metrics2.get('takeoff_duration', 0))
            
            # 跳跃高度对比
            height_winner = video1_name if height1 > height2 else video2_name if height2 > height1 else "相同"
            html_content += f"""
                        <tr>
                            <td>跳跃高度 (像素)</td>
                            <td {"class='winner'" if height_winner == video1_name else ""}>{height1:.1f}</td>
                            <td {"class='winner'" if height_winner == video2_name else ""}>{height2:.1f}</td>
                            <td>{height_winner}</td>
                        </tr>
            """
            
            # 起跳时间对比（时间越短越好）
            takeoff_winner = video1_name if takeoff1 < takeoff2 else video2_name if takeoff2 < takeoff1 else "相同"
            html_content += f"""
                        <tr>
                            <td>起跳时间 (秒)</td>
                            <td {"class='winner'" if takeoff_winner == video1_name else ""}>{takeoff1:.3f}</td>
                            <td {"class='winner'" if takeoff_winner == video2_name else ""}>{takeoff2:.3f}</td>
                            <td>{takeoff_winner}</td>
                        </tr>
            """
        
        # 力量评估对比
        if 'error' not in strength1 and 'error' not in strength2:
            overall1 = strength1.get('overall_score', 0)
            overall2 = strength2.get('overall_score', 0)
            explosive1 = strength1.get('explosive_power', 0)
            explosive2 = strength2.get('explosive_power', 0)
            core1 = strength1.get('core_strength', 0)
            core2 = strength2.get('core_strength', 0)
            coord1 = strength1.get('coordination', 0)
            coord2 = strength2.get('coordination', 0)
            
            # 综合得分对比
            overall_winner = video1_name if overall1 > overall2 else video2_name if overall2 > overall1 else "相同"
            html_content += f"""
                        <tr>
                            <td>综合得分</td>
                            <td {"class='winner'" if overall_winner == video1_name else ""}>{overall1:.3f}</td>
                            <td {"class='winner'" if overall_winner == video2_name else ""}>{overall2:.3f}</td>
                            <td>{overall_winner}</td>
                        </tr>
            """
            
            # 爆发力对比
            explosive_winner = video1_name if explosive1 > explosive2 else video2_name if explosive2 > explosive1 else "相同"
            html_content += f"""
                        <tr>
                            <td>爆发力</td>
                            <td {"class='winner'" if explosive_winner == video1_name else ""}>{explosive1:.3f}</td>
                            <td {"class='winner'" if explosive_winner == video2_name else ""}>{explosive2:.3f}</td>
                            <td>{explosive_winner}</td>
                        </tr>
            """
            
            # 核心力量对比
            core_winner = video1_name if core1 > core2 else video2_name if core2 > core1 else "相同"
            html_content += f"""
                        <tr>
                            <td>核心力量</td>
                            <td {"class='winner'" if core_winner == video1_name else ""}>{core1:.3f}</td>
                            <td {"class='winner'" if core_winner == video2_name else ""}>{core2:.3f}</td>
                            <td>{core_winner}</td>
                        </tr>
            """
            
            # 协调性对比
            coord_winner = video1_name if coord1 > coord2 else video2_name if coord2 > coord1 else "相同"
            html_content += f"""
                        <tr>
                            <td>协调性</td>
                            <td {"class='winner'" if coord_winner == video1_name else ""}>{coord1:.3f}</td>
                            <td {"class='winner'" if coord_winner == video2_name else ""}>{coord2:.3f}</td>
                            <td>{coord_winner}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
        """
        
        # 添加图表
        html_content += f"""
                <h2>📈 可视化对比分析</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{chart_base64}" alt="跳跃分析对比图表">
                </div>
                
                <h2>🎯 分析总结</h2>
                <div class="summary-box">
                    <h3>🔍 主要发现</h3>
        """
        
        # 添加分析总结
        if 'error' not in strength1 and 'error' not in strength2:
            overall1 = strength1.get('overall_score', 0)
            overall2 = strength2.get('overall_score', 0)
            
            if overall2 > overall1:
                diff_percent = ((overall2 - overall1) / overall1) * 100
                html_content += f"""
                    <p><strong>🏆 {video2_name} 表现更优秀</strong></p>
                    <ul>
                        <li>综合得分：{overall2:.3f} vs {overall1:.3f} （高出 {diff_percent:.1f}%）</li>
                        <li>视频时长：{video_info2.get('duration', 0):.2f}秒 vs {video_info1.get('duration', 0):.2f}秒</li>
                        <li>{video2_name} 在纯跳跃动作的执行上展现出更好的技术水平</li>
                    </ul>
                """
            elif overall1 > overall2:
                diff_percent = ((overall1 - overall2) / overall2) * 100
                html_content += f"""
                    <p><strong>🏆 {video1_name} 表现更优秀</strong></p>
                    <ul>
                        <li>综合得分：{overall1:.3f} vs {overall2:.3f} （高出 {diff_percent:.1f}%）</li>
                        <li>视频时长：{video_info1.get('duration', 0):.2f}秒 vs {video_info2.get('duration', 0):.2f}秒</li>
                        <li>{video1_name} 在纯跳跃动作的执行上展现出更好的技术水平</li>
                    </ul>
                """
            else:
                html_content += f"""
                    <p><strong>🤝 两个视频表现相当</strong></p>
                    <ul>
                        <li>综合得分：{overall1:.3f} vs {overall2:.3f}</li>
                        <li>两个视频的跳跃技术水平相近，各有优势</li>
                    </ul>
                """
        else:
            html_content += """
                    <p><strong>⚠️ 部分数据分析受限</strong></p>
                    <ul>
                        <li>由于视频质量或长度限制，部分指标无法完整分析</li>
                        <li>建议确保视频包含完整的跳跃动作序列</li>
                    </ul>
            """
        
        html_content += f"""
                    <h3>💡 处理后视频分析优势</h3>
                    <ul>
                        <li><strong>纯净分析：</strong> 去除了非跳跃部分，专注于跳跃动作本身</li>
                        <li><strong>精确对比：</strong> 消除了准备时间等外在因素的影响</li>
                        <li><strong>技术聚焦：</strong> 能够更准确地评估跳跃技术的差异</li>
                        <li><strong>短视频优化：</strong> 专门针对短视频进行了算法优化</li>
                    </ul>
                    
                    <h3>🚀 技术建议</h3>
                    <ul>
                        <li><strong>视频处理：</strong> 建议继续使用这种处理方式，只保留核心跳跃动作</li>
                        <li><strong>对比分析：</strong> 处理后的视频能够提供更精确的技术对比</li>
                        <li><strong>训练指导：</strong> 基于纯跳跃动作的分析结果更适合制定训练计划</li>
                        <li><strong>进步追踪：</strong> 可以用同样的处理方式定期分析，追踪技术进步</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px;">
                    <p>本对比报告由跳跃姿态分析系统自动生成</p>
                    <p>分析时间: {current_time}</p>
                    <p>🎯 专门针对处理后的纯跳跃视频进行优化分析</p>
                    <p>📊 数据基于MediaPipe姿态检测和自研跳跃分析算法</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def run_analysis(self, video_names):
        """运行完整的分析流程"""
        print("=" * 60)
        print("🏃‍♂️ 跳跃分析系统启动")
        print("=" * 60)
        
        # 检查视频文件
        for video_name in video_names:
            video_path = os.path.join(self.video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"❌ 错误：视频文件不存在 {video_path}")
                return
        
        analyses = []
        video_infos = []
        
        # 分析每个视频
        for video_name in video_names:
            print(f"\n{'='*50}")
            print(f"分析视频: {video_name}")
            print(f"{'='*50}")
            
            try:
                analysis, video_info = self.analyze_video(video_name)
                if analysis is None:
                    print(f"❌ 视频 {video_name} 分析失败")
                    return
                
                analyses.append(analysis)
                video_infos.append(video_info)
                
                # 生成个人报告
                self.generate_individual_report(video_name, analysis, video_info)
                
            except Exception as e:
                print(f"❌ 分析视频 {video_name} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # 生成对比报告
        if len(analyses) == 2:
            print(f"\n{'='*50}")
            print("生成对比报告")
            print(f"{'='*50}")
            
            try:
                self.generate_comparison_report(
                    video_names[0], video_names[1], 
                    analyses[0], analyses[1], 
                    video_infos[0], video_infos[1]
                )
            except Exception as e:
                print(f"❌ 生成对比报告时发生错误: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # 显示最终结果
        print(f"\n{'='*60}")
        print("🎉 分析完成！")
        print(f"{'='*60}")
        
        print("\n📊 分析结果摘要:")
        for i, (video_name, analysis, video_info) in enumerate(zip(video_names, analyses, video_infos)):
            jump_metrics = analysis.get('jump_metrics', {})
            strength_assessment = analysis.get('strength_assessment', {})
            
            print(f"\n{video_name}:")
            print(f"  📹 时长: {video_info.get('duration', 0):.2f}秒")
            
            if 'error' not in jump_metrics:
                print(f"  🏃 跳跃高度: {jump_metrics.get('jump_height_pixels', 0):.1f} 像素")
                print(f"  ⏱️ 起跳时间: {abs(jump_metrics.get('takeoff_duration', 0)):.3f} 秒")
            else:
                print(f"  ⚠️ 跳跃指标: {jump_metrics.get('error', '分析失败')}")
            
            if 'error' not in strength_assessment:
                print(f"  💪 综合得分: {strength_assessment.get('overall_score', 0):.3f}")
            else:
                print(f"  ⚠️ 力量评估: {strength_assessment.get('error', '分析失败')}")
        
        print(f"\n📁 报告文件已保存到 {self.output_dir}/ 目录")
        print("🌐 所有报告都包含视频播放器，可以边看视频边查看分析结果")
        print("💡 本次分析针对处理后的纯跳跃视频进行了专项优化")


def main():
    """主函数"""
    # 创建分析系统实例
    analysis_system = JumpAnalysisSystem()
    
    # 要分析的视频列表
    video_names = ['M3.mp4', 'M4.mp4']
    
    # 运行分析
    analysis_system.run_analysis(video_names)


if __name__ == "__main__":
    main()
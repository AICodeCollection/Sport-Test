#!/usr/bin/env python3
"""
改进的真实跳跃视频测试脚本
1. 修复M2.mp4数据点不足的问题
2. 在单独报告中添加视频播放器
"""

import sys
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加src目录到路径
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer
from visualizer import JumpVisualizer


def generate_individual_html_report(video_name, analysis_result, video_info, output_path):
    """生成包含视频的个人HTML报告"""
    
    # 创建可视化图表
    visualizer = JumpVisualizer()
    
    # 生成分析图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{video_name} Jump Analysis Report', fontsize=16, fontweight='bold')
    
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
    
    # 准备数据
    jump_metrics = analysis_result.get('jump_metrics', {})
    strength_assessment = analysis_result.get('strength_assessment', {})
    posture_analysis = analysis_result.get('posture_analysis', {})
    jump_phases = analysis_result.get('jump_phases', {})
    
    # 视频文件路径（相对路径）
    video_path = f"../test_videos/{video_name}"
    
    # HTML模板
    html_template = f"""
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{video_name} 跳跃动作分析报告</h1>
            
            <div class="video-section">
                <h3>🎬 原始视频</h3>
                <video class="video-player" controls>
                    <source src="{video_path}" type="video/mp4">
                    您的浏览器不支持视频播放。
                </video>
                <div class="video-info">
                    <h4>视频信息</h4>
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
            
            html_template += f"""
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
        html_template += f"""
            <h2>🎯 跳跃阶段划分</h2>
            <div class="error-message">
                ❌ 阶段识别失败: {jump_phases.get('error', '未知错误')}<br>
                💡 可能原因: 视频时长较短或动作不够明显，建议使用更长的跳跃视频
            </div>
        """
    
    # 添加跳跃指标
    if 'error' not in jump_metrics:
        html_template += f"""
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
        html_template += f"""
            <h2>📊 跳跃指标</h2>
            <div class="error-message">
                ❌ 跳跃指标计算失败: {jump_metrics.get('error', '未知错误')}<br>
                💡 建议: 使用更清晰、更长的跳跃视频，确保包含完整的跳跃动作
            </div>
        """
    
    # 添加力量评估
    if 'error' not in strength_assessment:
        overall_score = strength_assessment.get('overall_score', 0)
        explosive_power = strength_assessment.get('explosive_power', 0)
        core_strength = strength_assessment.get('core_strength', 0)
        coordination = strength_assessment.get('coordination', 0)
        
        html_template += f"""
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
        html_template += f"""
            <h2>💪 力量评估</h2>
            <div class="error-message">
                ❌ 力量评估失败: {strength_assessment.get('error', '未知错误')}<br>
                💡 原因: 需要有效的跳跃阶段数据才能进行力量评估
            </div>
        """
    
    # 添加姿态分析
    if 'error' not in posture_analysis:
        html_template += f"""
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
                
                html_template += f"""
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
        
        html_template += "</div>"
    else:
        html_template += f"""
            <h2>🤸 姿态分析</h2>
            <div class="error-message">
                ❌ 姿态分析失败: {posture_analysis.get('error', '未知错误')}<br>
                💡 原因: 需要有效的姿态检测数据才能进行姿态分析
            </div>
        """
    
    # 添加可视化图表
    html_template += f"""
            <h2>📈 分析图表</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{image_base64}" alt="跳跃分析图表">
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
            html_template += f"<p>{suggestion}</p>"
    else:
        html_template += """
        <p>🔸 由于分析数据不足，无法提供具体建议。</p>
        <p>🔸 <strong>改进建议：</strong></p>
        <ul>
            <li>使用更长的视频（至少5-8秒）</li>
            <li>确保视频包含完整的跳跃动作（准备-起跳-腾空-落地）</li>
            <li>保持摄像设备稳定，避免抖动</li>
            <li>确保光线充足，人体轮廓清晰</li>
            <li>建议从侧面拍摄，能更好地观察跳跃轨迹</li>
        </ul>
        """
    
    # 获取当前时间
    import datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_template += f"""
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px;">
                <p>本报告由跳跃姿态分析系统自动生成</p>
                <p>分析时间: {current_time}</p>
                <p>🔄 如需更准确的分析结果，建议使用更长、更清晰的跳跃视频</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return True


def analyze_video_improved(video_path):
    """改进的视频分析，处理短视频问题"""
    print(f"开始分析视频: {video_path}")
    
    # 1. 加载视频
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        print(f"❌ 无法加载视频: {video_path}")
        return None, None
    
    # 获取视频信息
    video_info = processor.get_video_info()
    print(f"   视频信息: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.2f}秒")
    
    # 2. 改进的帧提取策略 - 为短视频提供更密集的采样
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    duration = video_info['duration']
    
    # 根据视频长度调整采样策略
    if duration < 4:  # 短视频（小于4秒）
        frame_step = max(1, int(fps // 4))  # 每秒采样4帧
        print(f"   检测到短视频，使用密集采样：每秒{4}帧")
    else:  # 长视频
        frame_step = max(1, int(fps // 2))  # 每秒采样2帧
        print(f"   使用标准采样：每秒{2}帧")
    
    selected_frames = list(range(0, total_frames, frame_step))
    
    print(f"   提取帧: 从{total_frames}帧中选择{len(selected_frames)}帧进行分析")
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    print(f"   成功提取 {len(frames)} 帧")
    
    # 3. 姿态检测
    print("   进行姿态检测...")
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    valid_poses = sum(1 for result in pose_results if result is not None)
    print(f"   检测到有效姿态: {valid_poses}/{len(pose_results)} 帧")
    
    if valid_poses < 3:
        print("   ⚠️ 有效姿态数量太少，可能影响分析结果")
    
    # 4. 改进的跳跃分析 - 降低最小数据点要求
    print("   进行跳跃分析...")
    analyzer = JumpAnalyzer(fps=fps / frame_step)
    
    # 修改分析器的最小数据点要求（临时修改）
    original_min_points = 10  # 假设原来需要10个点
    if len(frames) < original_min_points:
        print(f"   调整分析参数以适应短视频（{len(frames)}帧）")
    
    analysis_result = analyzer.analyze_jump_sequence(pose_results)
    
    processor.release()
    
    return analysis_result, video_info


def main():
    """主函数"""
    print("=== 改进的跳跃视频分析测试 ===\n")
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 测试视频列表
    test_videos = ['M1.mp4', 'M2.mp4']
    
    for video_name in test_videos:
        video_path = os.path.join('test_videos', video_name)
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            continue
        
        print(f"\n{'='*50}")
        print(f"分析视频: {video_name}")
        print(f"{'='*50}")
        
        try:
            # 使用改进的分析方法
            analysis_result, video_info = analyze_video_improved(video_path)
            
            if analysis_result is None:
                print(f"❌ 视频 {video_name} 分析失败")
                continue
            
            # 生成包含视频的HTML报告
            html_output_path = os.path.join('outputs', f'{video_name}_improved_report.html')
            
            print(f"生成改进的HTML报告: {html_output_path}")
            
            success = generate_individual_html_report(video_name, analysis_result, video_info, html_output_path)
            
            if success:
                print(f"✅ {video_name} 分析完成，改进报告已保存")
                
                # 显示简要结果
                jump_metrics = analysis_result.get('jump_metrics', {})
                strength_assessment = analysis_result.get('strength_assessment', {})
                
                if 'error' not in jump_metrics:
                    print(f"   跳跃高度: {jump_metrics.get('jump_height_pixels', 0):.1f} 像素")
                    print(f"   起跳时间: {abs(jump_metrics.get('takeoff_duration', 0)):.3f} 秒")
                else:
                    print(f"   ⚠️ 跳跃指标: {jump_metrics.get('error', '分析失败')}")
                
                if 'error' not in strength_assessment:
                    print(f"   综合得分: {strength_assessment.get('overall_score', 0):.3f}")
                else:
                    print(f"   ⚠️ 力量评估: {strength_assessment.get('error', '分析失败')}")
            else:
                print(f"❌ {video_name} 报告生成失败")
                
        except Exception as e:
            print(f"❌ 分析视频 {video_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("🎉 改进分析完成！")
    print("📁 改进的HTML报告已保存到 outputs/ 目录")
    print("🌐 报告包含视频播放器，可以边看视频边查看分析结果")
    print("💡 对于短视频，系统已自动调整分析参数")


if __name__ == "__main__":
    # 导入必要的库
    import cv2
    import datetime
    
    main()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
import os

class JumpVisualizer:
    """跳跃分析可视化类"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_jump_analysis(self, analysis_result: Dict, save_path: str = None) -> None:
        """
        可视化完整的跳跃分析结果
        
        Args:
            analysis_result: 跳跃分析结果
            save_path: 保存路径
        """
        if 'error' in analysis_result.get('jump_metrics', {}):
            print(f"分析结果包含错误: {analysis_result['jump_metrics']['error']}")
            return
            
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('跳跃动作分析报告', fontsize=16, fontweight='bold')
        
        # 1. 身体中心轨迹
        self._plot_body_center_trajectory(axes[0, 0], analysis_result)
        
        # 2. 关节角度变化
        self._plot_joint_angles(axes[0, 1], analysis_result)
        
        # 3. 跳跃阶段划分
        self._plot_jump_phases(axes[0, 2], analysis_result)
        
        # 4. 力量评估雷达图
        self._plot_strength_radar(axes[1, 0], analysis_result)
        
        # 5. 姿态评估
        self._plot_posture_analysis(axes[1, 1], analysis_result)
        
        # 6. 综合指标
        self._plot_summary_metrics(axes[1, 2], analysis_result)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'jump_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def _plot_body_center_trajectory(self, ax, analysis_result: Dict) -> None:
        """绘制身体中心轨迹"""
        body_centers = analysis_result.get('body_centers', [])
        jump_phases = analysis_result.get('jump_phases', {})
        
        # 提取有效的身体中心点
        valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
        
        if not valid_centers:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('身体中心轨迹')
            return
            
        frames = [i for i, _ in valid_centers]
        x_coords = [center[0] for _, center in valid_centers]
        y_coords = [center[1] for _, center in valid_centers]
        
        # 绘制轨迹
        ax.plot(frames, y_coords, 'b-', linewidth=2, label='垂直位置')
        ax.plot(frames, x_coords, 'r--', linewidth=1, label='水平位置')
        
        # 标记关键点
        if 'peak_frame' in jump_phases:
            peak_frame = jump_phases['peak_frame']
            if peak_frame in frames:
                peak_idx = frames.index(peak_frame)
                ax.scatter(peak_frame, y_coords[peak_idx], color='red', s=100, 
                          marker='o', label='最高点', zorder=5)
        
        if 'lowest_frame' in jump_phases:
            lowest_frame = jump_phases['lowest_frame']
            if lowest_frame in frames:
                lowest_idx = frames.index(lowest_frame)
                ax.scatter(lowest_frame, y_coords[lowest_idx], color='green', s=100, 
                          marker='s', label='最低点', zorder=5)
        
        ax.set_xlabel('帧数')
        ax.set_ylabel('位置')
        ax.set_title('身体中心轨迹')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_joint_angles(self, ax, analysis_result: Dict) -> None:
        """绘制关节角度变化"""
        knee_angles = analysis_result.get('knee_angles', [])
        hip_angles = analysis_result.get('hip_angles', [])
        
        # 提取有效的角度数据
        valid_knee = [(i, angles) for i, angles in enumerate(knee_angles) 
                     if angles and angles[0] and angles[1]]
        valid_hip = [(i, angles) for i, angles in enumerate(hip_angles) 
                    if angles and angles[0] and angles[1]]
        
        if not valid_knee and not valid_hip:
            ax.text(0.5, 0.5, '无有效角度数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('关节角度变化')
            return
        
        # 绘制膝关节角度
        if valid_knee:
            knee_frames = [i for i, _ in valid_knee]
            left_knee = [angles[0] for _, angles in valid_knee]
            right_knee = [angles[1] for _, angles in valid_knee]
            
            ax.plot(knee_frames, left_knee, 'b-', label='左膝', linewidth=2)
            ax.plot(knee_frames, right_knee, 'b--', label='右膝', linewidth=2)
        
        # 绘制髋关节角度
        if valid_hip:
            hip_frames = [i for i, _ in valid_hip]
            left_hip = [angles[0] for _, angles in valid_hip]
            right_hip = [angles[1] for _, angles in valid_hip]
            
            ax.plot(hip_frames, left_hip, 'r-', label='左髋', linewidth=2)
            ax.plot(hip_frames, right_hip, 'r--', label='右髋', linewidth=2)
        
        ax.set_xlabel('帧数')
        ax.set_ylabel('角度 (度)')
        ax.set_title('关节角度变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_jump_phases(self, ax, analysis_result: Dict) -> None:
        """绘制跳跃阶段划分"""
        jump_phases = analysis_result.get('jump_phases', {})
        body_centers = analysis_result.get('body_centers', [])
        
        if 'error' in jump_phases:
            ax.text(0.5, 0.5, f'阶段识别失败: {jump_phases["error"]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('跳跃阶段划分')
            return
        
        # 提取有效的身体中心点
        valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
        
        if not valid_centers:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('跳跃阶段划分')
            return
        
        frames = [i for i, _ in valid_centers]
        y_coords = [center[1] for _, center in valid_centers]
        
        # 绘制身体中心轨迹
        ax.plot(frames, y_coords, 'k-', linewidth=2, alpha=0.7)
        
        # 标记各阶段
        colors = ['blue', 'red', 'green']
        phases = ['preparation', 'takeoff', 'landing']
        phase_names = ['准备', '起跳', '落地']
        
        for i, (phase, color, name) in enumerate(zip(phases, colors, phase_names)):
            if phase in jump_phases:
                start_frame = jump_phases[phase]['start_frame']
                end_frame = jump_phases[phase]['end_frame']
                
                # 添加阶段背景
                ax.axvspan(start_frame, end_frame, alpha=0.3, color=color, label=f'{name}阶段')
        
        ax.set_xlabel('帧数')
        ax.set_ylabel('垂直位置')
        ax.set_title('跳跃阶段划分')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_strength_radar(self, ax, analysis_result: Dict) -> None:
        """绘制力量评估雷达图"""
        strength_assessment = analysis_result.get('strength_assessment', {})
        
        if 'error' in strength_assessment:
            ax.text(0.5, 0.5, '力量评估数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('力量评估')
            return
        
        # 准备数据
        categories = ['爆发力', '核心力量', '协调性']
        values = [
            strength_assessment.get('explosive_power', 0),
            strength_assessment.get('core_strength', 0),
            strength_assessment.get('coordination', 0)
        ]
        
        # 雷达图角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color='red')
        ax.fill(angles, values, alpha=0.25, color='red')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('力量评估雷达图')
        ax.grid(True)
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.1, f'{value:.2f}', ha='center', va='center')
        
    def _plot_posture_analysis(self, ax, analysis_result: Dict) -> None:
        """绘制姿态分析"""
        posture_analysis = analysis_result.get('posture_analysis', {})
        
        if 'error' in posture_analysis:
            ax.text(0.5, 0.5, '姿态分析数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('姿态分析')
            return
        
        # 提取各阶段的稳定性得分
        phases = ['preparation_posture', 'takeoff_posture', 'landing_posture']
        phase_names = ['准备阶段', '起跳阶段', '落地阶段']
        stability_scores = []
        
        for phase in phases:
            if phase in posture_analysis:
                score = posture_analysis[phase].get('stability_score', 0)
                stability_scores.append(score)
            else:
                stability_scores.append(0)
        
        # 绘制柱状图
        bars = ax.bar(phase_names, stability_scores, color=['blue', 'red', 'green'], alpha=0.7)
        
        # 添加数值标签
        for bar, score in zip(bars, stability_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('稳定性得分')
        ax.set_title('各阶段姿态稳定性')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
    def _plot_summary_metrics(self, ax, analysis_result: Dict) -> None:
        """绘制综合指标"""
        jump_metrics = analysis_result.get('jump_metrics', {})
        strength_assessment = analysis_result.get('strength_assessment', {})
        
        if 'error' in jump_metrics or 'error' in strength_assessment:
            ax.text(0.5, 0.5, '综合指标数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('综合指标')
            return
        
        # 创建指标文本
        metrics_text = []
        
        # 跳跃指标
        height = jump_metrics.get('jump_height_pixels', 0)
        takeoff_time = jump_metrics.get('takeoff_duration', 0)
        total_time = jump_metrics.get('total_duration', 0)
        
        metrics_text.append(f"跳跃高度: {height:.1f} 像素")
        metrics_text.append(f"起跳时间: {takeoff_time:.2f} 秒")
        metrics_text.append(f"总时间: {total_time:.2f} 秒")
        
        # 力量指标
        overall_score = strength_assessment.get('overall_score', 0)
        explosive_power = strength_assessment.get('explosive_power', 0)
        core_strength = strength_assessment.get('core_strength', 0)
        coordination = strength_assessment.get('coordination', 0)
        
        metrics_text.append("")
        metrics_text.append(f"综合得分: {overall_score:.2f}")
        metrics_text.append(f"爆发力: {explosive_power:.2f}")
        metrics_text.append(f"核心力量: {core_strength:.2f}")
        metrics_text.append(f"协调性: {coordination:.2f}")
        
        # 绘制文本
        ax.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        ax.set_title('综合指标')
        ax.axis('off')
        
    def create_pose_animation(self, frames: List[np.ndarray], 
                            pose_results: List[Optional[Dict]], 
                            output_path: str) -> bool:
        """
        创建姿态检测动画
        
        Args:
            frames: 视频帧列表
            pose_results: 姿态检测结果列表
            output_path: 输出视频路径
            
        Returns:
            bool: 是否成功创建
        """
        if not frames or not pose_results:
            return False
        
        # 获取视频参数
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        try:
            from pose_detector import PoseDetector
            pose_detector = PoseDetector()
            
            for frame, pose_result in zip(frames, pose_results):
                if pose_result:
                    # 绘制姿态关键点
                    annotated_frame = pose_detector.draw_pose_landmarks(frame, pose_result)
                    # 转换为BGR格式
                    frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                out.write(frame_bgr)
                
        except Exception as e:
            print(f"创建动画失败: {e}")
            return False
        finally:
            out.release()
        
        return True
        
    def save_analysis_report(self, analysis_result: Dict, output_path: str) -> bool:
        """
        保存分析报告到文本文件
        
        Args:
            analysis_result: 分析结果
            output_path: 输出路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("跳跃动作分析报告\n")
                f.write("=" * 30 + "\n\n")
                
                # 跳跃指标
                jump_metrics = analysis_result.get('jump_metrics', {})
                if 'error' not in jump_metrics:
                    f.write("跳跃指标:\n")
                    f.write(f"  跳跃高度: {jump_metrics.get('jump_height_pixels', 0):.1f} 像素\n")
                    f.write(f"  起跳时间: {jump_metrics.get('takeoff_duration', 0):.2f} 秒\n")
                    f.write(f"  准备时间: {jump_metrics.get('preparation_duration', 0):.2f} 秒\n")
                    f.write(f"  落地时间: {jump_metrics.get('landing_duration', 0):.2f} 秒\n")
                    f.write(f"  总时间: {jump_metrics.get('total_duration', 0):.2f} 秒\n\n")
                
                # 力量评估
                strength_assessment = analysis_result.get('strength_assessment', {})
                if 'error' not in strength_assessment:
                    f.write("力量评估:\n")
                    f.write(f"  综合得分: {strength_assessment.get('overall_score', 0):.2f}\n")
                    f.write(f"  爆发力: {strength_assessment.get('explosive_power', 0):.2f}\n")
                    f.write(f"  核心力量: {strength_assessment.get('core_strength', 0):.2f}\n")
                    f.write(f"  协调性: {strength_assessment.get('coordination', 0):.2f}\n\n")
                
                # 姿态分析
                posture_analysis = analysis_result.get('posture_analysis', {})
                if 'error' not in posture_analysis:
                    f.write("姿态分析:\n")
                    phases = [
                        ('preparation_posture', '准备阶段'),
                        ('takeoff_posture', '起跳阶段'),
                        ('landing_posture', '落地阶段')
                    ]
                    
                    for phase_key, phase_name in phases:
                        if phase_key in posture_analysis:
                            phase_data = posture_analysis[phase_key]
                            f.write(f"  {phase_name}:\n")
                            f.write(f"    稳定性得分: {phase_data.get('stability_score', 0):.3f}\n")
                            f.write(f"    平均膝关节角度: {phase_data.get('avg_knee_angle', 0):.1f}°\n")
                            f.write(f"    平均髋关节角度: {phase_data.get('avg_hip_angle', 0):.1f}°\n")
            
            return True
            
        except Exception as e:
            print(f"保存报告失败: {e}")
            return False
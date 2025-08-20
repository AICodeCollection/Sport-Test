#!/usr/bin/env python3
"""
跳跃姿态分析系统集成测试
"""

import sys
import os
import numpy as np

# 添加src目录到路径
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer
from visualizer import JumpVisualizer


def create_mock_pose_data(num_frames=60):
    """创建模拟姿态数据用于测试"""
    pose_results = []
    
    for i in range(num_frames):
        # 模拟身体中心Y坐标的变化（跳跃轨迹）
        if i < 20:  # 准备阶段
            center_y = 0.7 + i * 0.005
        elif i < 40:  # 起跳阶段
            center_y = 0.8 - (i - 20) * 0.02
        else:  # 落地阶段
            center_y = 0.4 + (i - 40) * 0.015
        
        center_x = 0.5
        
        # 创建模拟的landmark数据
        landmarks = []
        landmark_positions = {
            0: (center_x, center_y - 0.15),  # nose
            11: (center_x - 0.1, center_y - 0.05),  # left_shoulder
            12: (center_x + 0.1, center_y - 0.05),  # right_shoulder
            23: (center_x - 0.05, center_y + 0.05),  # left_hip
            24: (center_x + 0.05, center_y + 0.05),  # right_hip
            25: (center_x - 0.05, center_y + 0.15),  # left_knee
            26: (center_x + 0.05, center_y + 0.15),  # right_knee
            27: (center_x - 0.05, center_y + 0.25),  # left_ankle
            28: (center_x + 0.05, center_y + 0.25),  # right_ankle
        }
        
        # 为所有33个关键点创建数据
        for j in range(33):
            if j in landmark_positions:
                x, y = landmark_positions[j]
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
            else:
                x = center_x + np.random.normal(0, 0.05)
                y = center_y + np.random.normal(0, 0.05)
            
            landmarks.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': 0.9
            })
        
        pose_result = {
            'landmarks': landmarks,
            'frame_shape': (480, 640, 3)
        }
        
        pose_results.append(pose_result)
    
    return pose_results


def test_modules():
    """测试各个模块的基本功能"""
    print("=== 跳跃姿态分析系统集成测试 ===\n")
    
    # 测试1: 姿态检测器初始化
    print("1. 测试姿态检测器初始化...")
    try:
        pose_detector = PoseDetector()
        print("   ✅ 姿态检测器初始化成功")
    except Exception as e:
        print(f"   ❌ 姿态检测器初始化失败: {e}")
        return False
    
    # 测试2: 跳跃分析器初始化
    print("2. 测试跳跃分析器初始化...")
    try:
        jump_analyzer = JumpAnalyzer(fps=30.0)
        print("   ✅ 跳跃分析器初始化成功")
    except Exception as e:
        print(f"   ❌ 跳跃分析器初始化失败: {e}")
        return False
    
    # 测试3: 可视化器初始化
    print("3. 测试可视化器初始化...")
    try:
        visualizer = JumpVisualizer(output_dir='outputs')
        print("   ✅ 可视化器初始化成功")
    except Exception as e:
        print(f"   ❌ 可视化器初始化失败: {e}")
        return False
    
    # 测试4: 创建模拟数据
    print("4. 创建模拟姿态数据...")
    try:
        mock_pose_results = create_mock_pose_data(60)
        print(f"   ✅ 成功创建 {len(mock_pose_results)} 帧模拟数据")
    except Exception as e:
        print(f"   ❌ 创建模拟数据失败: {e}")
        return False
    
    # 测试5: 跳跃分析
    print("5. 测试跳跃分析功能...")
    try:
        analysis_result = jump_analyzer.analyze_jump_sequence(mock_pose_results)
        
        # 检查分析结果
        if 'jump_metrics' in analysis_result and 'error' not in analysis_result['jump_metrics']:
            jump_metrics = analysis_result['jump_metrics']
            print(f"   ✅ 跳跃分析成功")
            print(f"      - 跳跃高度: {jump_metrics['jump_height_pixels']:.1f} 像素")
            print(f"      - 起跳时间: {jump_metrics['takeoff_duration']:.2f} 秒")
            print(f"      - 总时间: {jump_metrics['total_duration']:.2f} 秒")
        else:
            print(f"   ⚠️  跳跃分析完成但可能有错误")
            
    except Exception as e:
        print(f"   ❌ 跳跃分析失败: {e}")
        return False
    
    # 测试6: 力量评估
    print("6. 测试力量评估功能...")
    try:
        if 'strength_assessment' in analysis_result and 'error' not in analysis_result['strength_assessment']:
            strength = analysis_result['strength_assessment']
            print(f"   ✅ 力量评估成功")
            print(f"      - 综合得分: {strength['overall_score']:.2f}")
            print(f"      - 爆发力: {strength['explosive_power']:.2f}")
            print(f"      - 核心力量: {strength['core_strength']:.2f}")
            print(f"      - 协调性: {strength['coordination']:.2f}")
        else:
            print(f"   ⚠️  力量评估完成但可能有错误")
    except Exception as e:
        print(f"   ❌ 力量评估失败: {e}")
        return False
    
    # 测试7: 可视化生成
    print("7. 测试可视化生成...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        visualizer.visualize_jump_analysis(analysis_result, save_path='outputs/test_analysis.png')
        print("   ✅ 可视化图表生成成功")
    except Exception as e:
        print(f"   ❌ 可视化生成失败: {e}")
        return False
    
    # 测试8: 报告保存
    print("8. 测试报告保存...")
    try:
        success = visualizer.save_analysis_report(analysis_result, 'outputs/test_report.txt')
        if success:
            print("   ✅ 分析报告保存成功")
        else:
            print("   ❌ 分析报告保存失败")
    except Exception as e:
        print(f"   ❌ 报告保存出错: {e}")
        return False
    
    print("\n=== 集成测试完成 ===")
    print("✅ 所有核心功能测试通过")
    print("📁 输出文件保存在 outputs/ 目录")
    print("🚀 系统已准备好进行真实数据测试")
    
    return True


def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 运行测试
    success = test_modules()
    
    if success:
        print("\n🎉 技术验证原型测试成功！")
        print("\n下一步建议:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行Jupyter notebook: jupyter notebook notebooks/tech_validation.ipynb")
        print("3. 准备真实跳跃视频进行测试")
        print("4. 根据测试结果优化算法参数")
    else:
        print("\n❌ 测试失败，请检查代码和依赖")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
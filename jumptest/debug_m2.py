#!/usr/bin/env python3
"""
调试M2.mp4分析问题
"""

import sys
import cv2
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer

def debug_video_analysis(video_path):
    """详细分析视频问题"""
    print(f"调试视频: {video_path}")
    
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        print("❌ 无法加载视频")
        return
    
    print('\n视频信息:')
    info = processor.get_video_info()
    for k, v in info.items():
        print(f'  {k}: {v}')
    
    # 提取帧
    fps = info['fps']
    frame_step = max(1, int(fps // 2))
    total_frames = info['total_frames']
    selected_frames = list(range(0, total_frames, frame_step))
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    print(f'\n帧处理:')
    print(f'  总帧数: {total_frames}')
    print(f'  采样步长: {frame_step}')
    print(f'  选择帧数: {len(selected_frames)}')
    print(f'  成功提取: {len(frames)}')
    
    # 姿态检测
    print(f'\n姿态检测:')
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    valid_poses = sum(1 for result in pose_results if result is not None)
    print(f'  检测结果: {valid_poses}/{len(pose_results)} 帧有效')
    
    # 分析每帧的姿态检测结果
    for i, result in enumerate(pose_results):
        if result is None:
            print(f'    帧 {i}: 无姿态')
        else:
            landmarks = result.get('landmarks', [])
            print(f'    帧 {i}: {len(landmarks)} 个关键点')
    
    # 跳跃分析
    print(f'\n跳跃分析:')
    analyzer = JumpAnalyzer(fps=fps / frame_step)
    analysis_result = analyzer.analyze_jump_sequence(pose_results)
    
    # 详细分析结果
    jump_phases = analysis_result.get('jump_phases', {})
    jump_metrics = analysis_result.get('jump_metrics', {})
    strength_assessment = analysis_result.get('strength_assessment', {})
    posture_analysis = analysis_result.get('posture_analysis', {})
    
    print('跳跃阶段:')
    if 'error' in jump_phases:
        print(f'  ❌ 错误: {jump_phases["error"]}')
    else:
        print('  ✅ 成功识别阶段')
        for phase_name, phase_info in jump_phases.items():
            if isinstance(phase_info, dict) and 'start_frame' in phase_info:
                print(f'    {phase_name}: {phase_info["start_frame"]} - {phase_info["end_frame"]}')
    
    print('\n跳跃指标:')
    if 'error' in jump_metrics:
        print(f'  ❌ 错误: {jump_metrics["error"]}')
    else:
        print('  ✅ 成功计算指标')
        for k, v in jump_metrics.items():
            print(f'    {k}: {v}')
    
    print('\n力量评估:')
    if 'error' in strength_assessment:
        print(f'  ❌ 错误: {strength_assessment["error"]}')
    else:
        print('  ✅ 成功评估力量')
        for k, v in strength_assessment.items():
            print(f'    {k}: {v}')
    
    print('\n姿态分析:')
    if 'error' in posture_analysis:
        print(f'  ❌ 错误: {posture_analysis["error"]}')
    else:
        print('  ✅ 成功分析姿态')
        for phase_name, phase_data in posture_analysis.items():
            if isinstance(phase_data, dict):
                print(f'    {phase_name}:')
                for k, v in phase_data.items():
                    print(f'      {k}: {v}')
    
    processor.release()
    return analysis_result

if __name__ == "__main__":
    print("=== M1.mp4 分析 ===")
    result1 = debug_video_analysis('test_videos/M1.mp4')
    
    print("\n" + "="*50)
    print("=== M2.mp4 分析 ===")
    result2 = debug_video_analysis('test_videos/M2.mp4')
    
    # 对比结果
    print("\n" + "="*50)
    print("=== 对比分析 ===")
    
    if result1 and result2:
        strength1 = result1.get('strength_assessment', {})
        strength2 = result2.get('strength_assessment', {})
        
        print("M1 力量评估:", 'error' not in strength1)
        print("M2 力量评估:", 'error' not in strength2)
        
        if 'error' not in strength1:
            print(f"M1 综合得分: {strength1.get('overall_score', 0):.3f}")
        if 'error' not in strength2:
            print(f"M2 综合得分: {strength2.get('overall_score', 0):.3f}")
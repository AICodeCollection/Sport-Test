#!/usr/bin/env python3
"""
调试跳跃高度计算问题
分析为什么跳跃高度只有0.2像素
"""

import sys
import cv2
import numpy as np
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector
from jump_analyzer import JumpAnalyzer

def debug_jump_height_calculation(video_path):
    """详细调试跳跃高度计算过程"""
    print(f"🔍 调试视频: {video_path}")
    print("="*60)
    
    # 1. 加载视频
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        print("❌ 无法加载视频")
        return
    
    video_info = processor.get_video_info()
    print(f"📹 视频信息:")
    print(f"   分辨率: {video_info['width']} × {video_info['height']}")
    print(f"   帧率: {video_info['fps']:.1f} FPS")
    print(f"   时长: {video_info['duration']:.2f} 秒")
    print(f"   总帧数: {video_info['total_frames']}")
    
    # 2. 提取帧
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    duration = video_info['duration']
    
    if duration < 3:
        frame_step = max(1, int(fps // 6))
    elif duration < 5:
        frame_step = max(1, int(fps // 4))
    else:
        frame_step = max(1, int(fps // 2))
    
    selected_frames = list(range(0, total_frames, frame_step))
    print(f"\n🎞️ 帧提取:")
    print(f"   采样步长: {frame_step}")
    print(f"   选择帧数: {len(selected_frames)}")
    print(f"   选择的帧索引: {selected_frames}")
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    print(f"   成功提取: {len(frames)} 帧")
    
    # 3. 姿态检测
    print(f"\n🔍 姿态检测:")
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    valid_poses = sum(1 for result in pose_results if result is not None)
    print(f"   有效姿态: {valid_poses}/{len(pose_results)} 帧")
    
    # 4. 详细分析身体中心点
    print(f"\n📊 身体中心点分析:")
    body_centers = []
    
    for i, pose_result in enumerate(pose_results):
        if pose_result:
            center = detector.get_body_center(pose_result)
            body_centers.append(center)
            if center:
                print(f"   帧 {i}: 身体中心 = ({center[0]:.2f}, {center[1]:.2f})")
            else:
                print(f"   帧 {i}: 无法计算身体中心")
        else:
            body_centers.append(None)
            print(f"   帧 {i}: 无姿态检测结果")
    
    # 5. 分析Y坐标变化
    valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
    
    if len(valid_centers) >= 2:
        print(f"\n📈 Y坐标变化分析:")
        y_coords = [center[1] for _, center in valid_centers]
        frame_indices = [i for i, _ in valid_centers]
        
        print(f"   有效帧索引: {frame_indices}")
        print(f"   对应Y坐标: {[f'{y:.2f}' for y in y_coords]}")
        
        min_y = min(y_coords)
        max_y = max(y_coords)
        min_idx = y_coords.index(min_y)
        max_idx = y_coords.index(max_y)
        
        print(f"\n   最高点 (Y坐标最小): 帧{frame_indices[min_idx]}, Y={min_y:.2f}")
        print(f"   最低点 (Y坐标最大): 帧{frame_indices[max_idx]}, Y={max_y:.2f}")
        print(f"   计算的跳跃高度: |{max_y:.2f} - {min_y:.2f}| = {abs(max_y - min_y):.2f} 像素")
        
        # 检查是否Y坐标系颠倒了
        print(f"\n🔍 坐标系分析:")
        print(f"   注意: 在图像坐标系中，Y=0在顶部，Y值越大越靠下")
        print(f"   因此，跳跃最高点的Y坐标应该是最小值")
        print(f"   跳跃最低点的Y坐标应该是最大值")
        
        # 重新计算真正的跳跃高度
        true_jump_height = abs(max_y - min_y)
        print(f"   真实跳跃高度: {true_jump_height:.2f} 像素")
        
        # 转换为实际距离估算
        # 假设一个人的身高约为视频高度的70-80%
        video_height = video_info['height']
        estimated_person_height_pixels = video_height * 0.75  # 假设人占画面75%
        estimated_person_height_cm = 170  # 假设身高170cm
        
        pixels_per_cm = estimated_person_height_pixels / estimated_person_height_cm
        jump_height_cm = true_jump_height / pixels_per_cm
        
        print(f"\n📏 跳跃高度估算:")
        print(f"   视频高度: {video_height} 像素")
        print(f"   估算人体高度: {estimated_person_height_pixels:.0f} 像素 (约170cm)")
        print(f"   像素比例: {pixels_per_cm:.2f} 像素/厘米")
        print(f"   跳跃高度: {true_jump_height:.2f} 像素 ≈ {jump_height_cm:.1f} 厘米")
        
        # 检查数据质量
        y_variation = max(y_coords) - min(y_coords)
        print(f"\n🎯 数据质量分析:")
        print(f"   Y坐标变化范围: {y_variation:.2f} 像素")
        
        if y_variation < 1:
            print("   ⚠️ 警告: Y坐标变化极小，可能的原因:")
            print("     1. 视频中的跳跃幅度很小")
            print("     2. 姿态检测不够准确")
            print("     3. 相机距离太远，跳跃在画面中显得很小")
            print("     4. 视频帧数太少，没有捕捉到完整的跳跃过程")
        elif y_variation < 10:
            print("   📝 注意: Y坐标变化较小，这是合理的，因为:")
            print("     1. 处理后的视频只包含跳跃核心动作")
            print("     2. 相机可能距离较远")
            print("     3. 跳跃高度在视频画面中确实较小")
        else:
            print("   ✅ Y坐标变化正常")
        
    else:
        print("   ❌ 有效身体中心点不足，无法分析")
    
    # 6. 跳跃分析器的计算
    print(f"\n🔬 跳跃分析器计算:")
    analyzer = JumpAnalyzer(fps=fps / frame_step)
    analysis_result = analyzer.analyze_jump_sequence(pose_results)
    
    jump_phases = analysis_result.get('jump_phases', {})
    jump_metrics = analysis_result.get('jump_metrics', {})
    
    if 'error' not in jump_phases:
        print(f"   ✅ 跳跃阶段识别成功")
        print(f"   最低帧: {jump_phases['lowest_frame']}")
        print(f"   最高帧: {jump_phases['peak_frame']}")
    else:
        print(f"   ❌ 跳跃阶段识别失败: {jump_phases['error']}")
    
    if 'error' not in jump_metrics:
        print(f"   ✅ 跳跃指标计算成功")
        print(f"   算法计算的跳跃高度: {jump_metrics['jump_height_pixels']:.2f} 像素")
    else:
        print(f"   ❌ 跳跃指标计算失败: {jump_metrics['error']}")
    
    processor.release()
    
    return analysis_result, body_centers

def main():
    """主函数"""
    print("🔍 跳跃高度计算调试工具")
    print("="*60)
    
    # 测试所有视频
    test_videos = ['M1.mp4', 'M2.mp4', 'M3.mp4', 'M4.mp4']
    
    for video_name in test_videos:
        video_path = f'test_videos/{video_name}'
        
        if not os.path.exists(video_path):
            print(f"⚠️ 跳过不存在的视频: {video_path}")
            continue
        
        print(f"\n{'='*60}")
        result, centers = debug_jump_height_calculation(video_path)
        print(f"{'='*60}")
        
        # 等待用户继续
        input(f"\n按回车键继续分析下一个视频...")

if __name__ == "__main__":
    import os
    main()
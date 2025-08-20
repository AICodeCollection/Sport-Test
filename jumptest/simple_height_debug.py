#!/usr/bin/env python3
"""
简化版跳跃高度调试 - 专门分析M3.mp4
"""

import sys
import cv2
import numpy as np
sys.path.append('src')

from video_processor import VideoProcessor
from pose_detector import PoseDetector

def analyze_m3_jump_height():
    """分析M3.mp4的跳跃高度问题"""
    video_path = 'test_videos/M3.mp4'
    print(f"🔍 分析 M3.mp4 的跳跃高度问题")
    print("="*50)
    
    # 1. 加载视频
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        print("❌ 无法加载视频")
        return
    
    video_info = processor.get_video_info()
    print(f"📹 视频信息:")
    print(f"   分辨率: {video_info['width']} × {video_info['height']}")
    print(f"   时长: {video_info['duration']:.2f} 秒")
    print(f"   总帧数: {video_info['total_frames']}")
    
    # 2. 提取关键帧进行分析
    fps = video_info['fps']
    frame_step = max(1, int(fps // 4))  # 每秒4帧
    selected_frames = list(range(0, video_info['total_frames'], frame_step))
    
    print(f"\n🎞️ 提取 {len(selected_frames)} 帧进行分析")
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    # 3. 姿态检测
    print(f"\n🔍 进行姿态检测...")
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    # 4. 分析身体中心点
    print(f"\n📊 身体中心点Y坐标分析:")
    body_centers = []
    y_coordinates = []
    
    for i, pose_result in enumerate(pose_results):
        if pose_result:
            center = detector.get_body_center(pose_result)
            body_centers.append(center)
            if center:
                y_coordinates.append(center[1])
                print(f"   帧 {i:2d}: Y坐标 = {center[1]:7.2f}")
            else:
                print(f"   帧 {i:2d}: 无法计算身体中心")
        else:
            body_centers.append(None)
            print(f"   帧 {i:2d}: 无姿态检测")
    
    if len(y_coordinates) >= 2:
        print(f"\n📈 Y坐标统计:")
        print(f"   最小Y (最高点): {min(y_coordinates):.2f}")
        print(f"   最大Y (最低点): {max(y_coordinates):.2f}")
        print(f"   Y坐标变化范围: {max(y_coordinates) - min(y_coordinates):.2f} 像素")
        print(f"   Y坐标标准差: {np.std(y_coordinates):.2f}")
        
        # 计算真实跳跃高度
        jump_height_pixels = max(y_coordinates) - min(y_coordinates)
        print(f"\n🏃 跳跃高度分析:")
        print(f"   跳跃高度: {jump_height_pixels:.2f} 像素")
        
        # 估算实际高度
        video_height = video_info['height']
        # 假设人体占画面高度的70%，实际身高170cm
        person_height_pixels = video_height * 0.7
        pixels_per_cm = person_height_pixels / 170
        jump_height_cm = jump_height_pixels / pixels_per_cm
        
        print(f"   估算实际跳跃高度: {jump_height_cm:.1f} 厘米")
        
        # 分析为什么算法得到0.2
        print(f"\n🔍 问题分析:")
        if jump_height_pixels < 1:
            print(f"   ⚠️ 跳跃高度非常小 ({jump_height_pixels:.2f}像素)")
            print(f"   可能原因:")
            print(f"   1. 相机距离很远，跳跃在画面中显得很小")
            print(f"   2. 处理后的视频只包含核心跳跃动作，幅度较小")
            print(f"   3. 姿态检测的身体中心计算可能不够精确")
            print(f"   4. 视频质量或角度影响了检测精度")
        
        # 查看Y坐标的变化趋势
        print(f"\n📊 Y坐标变化趋势:")
        for i, y in enumerate(y_coordinates):
            trend = ""
            if i > 0:
                diff = y - y_coordinates[i-1]
                if diff > 0.1:
                    trend = "↓ 下降"
                elif diff < -0.1:
                    trend = "↑ 上升"
                else:
                    trend = "→ 平稳"
            print(f"   帧 {i}: {y:7.2f} {trend}")
        
    else:
        print("   ❌ 有效Y坐标数据不足")
    
    processor.release()

if __name__ == "__main__":
    analyze_m3_jump_height()
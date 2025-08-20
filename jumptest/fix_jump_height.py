#!/usr/bin/env python3
"""
修复跳跃高度计算问题
MediaPipe返回的是归一化坐标，需要转换为像素坐标
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


class FixedJumpAnalyzer:
    """修复版跳跃分析器 - 正确处理像素坐标"""
    
    def __init__(self, fps: float = 30.0, video_width: int = 720, video_height: int = 1280):
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.pose_detector = PoseDetector()
    
    def convert_normalized_to_pixels(self, normalized_coords, image_width, image_height):
        """将归一化坐标转换为像素坐标"""
        if normalized_coords is None:
            return None
        return (
            normalized_coords[0] * image_width,
            normalized_coords[1] * image_height
        )
    
    def analyze_jump_with_fixed_height(self, pose_results, video_width, video_height):
        """修复版跳跃分析 - 正确计算像素高度"""
        
        # 1. 提取身体中心点并转换为像素坐标
        body_centers_normalized = []
        body_centers_pixels = []
        
        for pose_result in pose_results:
            if pose_result:
                # 获取归一化坐标
                center_norm = self.pose_detector.get_body_center(pose_result)
                body_centers_normalized.append(center_norm)
                
                # 转换为像素坐标
                if center_norm:
                    center_pixels = self.convert_normalized_to_pixels(center_norm, video_width, video_height)
                    body_centers_pixels.append(center_pixels)
                else:
                    body_centers_pixels.append(None)
            else:
                body_centers_normalized.append(None)
                body_centers_pixels.append(None)
        
        # 2. 分析归一化坐标
        valid_normalized = [center for center in body_centers_normalized if center is not None]
        valid_pixels = [center for center in body_centers_pixels if center is not None]
        
        if len(valid_normalized) < 3:
            return {
                'error': '有效数据点不足',
                'normalized_centers': body_centers_normalized,
                'pixel_centers': body_centers_pixels
            }
        
        # 3. 计算跳跃高度（归一化和像素两个版本）
        norm_y_coords = [center[1] for center in valid_normalized]
        pixel_y_coords = [center[1] for center in valid_pixels]
        
        # 归一化版本
        norm_min_y = min(norm_y_coords)  # 最高点
        norm_max_y = max(norm_y_coords)  # 最低点
        norm_jump_height = norm_max_y - norm_min_y
        
        # 像素版本
        pixel_min_y = min(pixel_y_coords)  # 最高点
        pixel_max_y = max(pixel_y_coords)  # 最低点
        pixel_jump_height = pixel_max_y - pixel_min_y
        
        # 4. 估算实际跳跃高度
        # 假设人体占画面高度的70%，实际身高170cm
        person_height_pixels = video_height * 0.7
        pixels_per_cm = person_height_pixels / 170
        jump_height_cm = pixel_jump_height / pixels_per_cm
        
        return {
            'normalized_jump_height': norm_jump_height,
            'pixel_jump_height': pixel_jump_height,
            'estimated_jump_height_cm': jump_height_cm,
            'normalized_y_range': (norm_min_y, norm_max_y),
            'pixel_y_range': (pixel_min_y, pixel_max_y),
            'normalized_centers': body_centers_normalized,
            'pixel_centers': body_centers_pixels,
            'video_dimensions': (video_width, video_height),
            'conversion_factor': pixels_per_cm
        }


def analyze_video_with_fixed_height(video_path):
    """使用修复版算法分析视频"""
    print(f"🔧 使用修复版算法分析: {video_path}")
    
    # 1. 加载视频
    processor = VideoProcessor(video_path)
    if not processor.load_video():
        print(f"❌ 无法加载视频: {video_path}")
        return None
    
    video_info = processor.get_video_info()
    print(f"   📊 视频: {video_info['width']}×{video_info['height']}, {video_info['duration']:.2f}秒")
    
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
    
    frames = []
    for i in selected_frames:
        processor.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = processor.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    print(f"   🎞️ 提取了 {len(frames)} 帧")
    
    # 3. 姿态检测
    detector = PoseDetector()
    pose_results = detector.detect_pose_sequence(frames)
    
    valid_poses = sum(1 for result in pose_results if result is not None)
    print(f"   🔍 检测到 {valid_poses}/{len(pose_results)} 个有效姿态")
    
    # 4. 使用修复版分析器
    analyzer = FixedJumpAnalyzer(
        fps=fps / frame_step,
        video_width=video_info['width'],
        video_height=video_info['height']
    )
    
    result = analyzer.analyze_jump_with_fixed_height(
        pose_results, 
        video_info['width'], 
        video_info['height']
    )
    
    processor.release()
    return result, video_info


def print_comparison_results(video_name, result):
    """打印对比结果"""
    print(f"\n📊 {video_name} 修复前后对比:")
    print("="*50)
    
    if 'error' in result:
        print(f"❌ 分析失败: {result['error']}")
        return
    
    norm_height = result['normalized_jump_height']
    pixel_height = result['pixel_jump_height']
    cm_height = result['estimated_jump_height_cm']
    
    print(f"🔧 修复前 (错误版本):")
    print(f"   跳跃高度: {norm_height:.2f} 像素 (实际是归一化坐标)")
    
    print(f"\n✅ 修复后 (正确版本):")
    print(f"   归一化跳跃高度: {norm_height:.3f}")
    print(f"   真实像素跳跃高度: {pixel_height:.1f} 像素")
    print(f"   估算实际跳跃高度: {cm_height:.1f} 厘米")
    
    print(f"\n📏 详细数据:")
    norm_range = result['normalized_y_range']
    pixel_range = result['pixel_y_range']
    video_dims = result['video_dimensions']
    
    print(f"   视频尺寸: {video_dims[0]} × {video_dims[1]}")
    print(f"   归一化Y范围: {norm_range[0]:.3f} ~ {norm_range[1]:.3f}")
    print(f"   像素Y范围: {pixel_range[0]:.1f} ~ {pixel_range[1]:.1f}")
    print(f"   转换比例: {result['conversion_factor']:.2f} 像素/厘米")


def main():
    """主函数"""
    print("🔧 跳跃高度计算修复工具")
    print("="*60)
    print("说明: MediaPipe返回归一化坐标(0-1)，需要乘以图像尺寸得到像素坐标")
    print("="*60)
    
    # 测试所有视频
    test_videos = ['M1.mp4', 'M2.mp4', 'M3.mp4', 'M4.mp4']
    
    for video_name in test_videos:
        video_path = f'test_videos/{video_name}'
        
        if not os.path.exists(video_path):
            print(f"⚠️ 跳过不存在的视频: {video_path}")
            continue
        
        try:
            result, video_info = analyze_video_with_fixed_height(video_path)
            if result:
                print_comparison_results(video_name, result)
            else:
                print(f"❌ {video_name} 分析失败")
        except Exception as e:
            print(f"❌ 分析 {video_name} 时出错: {e}")
        
        print("\n" + "="*60)
    
    print("\n🎯 结论:")
    print("原来显示的0.2'像素'实际上是0.2的归一化坐标差值")
    print("真实的跳跃高度应该是: 0.2 × 视频高度 ≈ 256像素 ≈ 27厘米")


if __name__ == "__main__":
    main()
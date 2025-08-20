#!/usr/bin/env python3
"""
跑步姿态分析系统主程序
用于命令行直接分析视频文件
"""

import sys
import os
import argparse
from pose_detector import RunningPoseDetector
from gait_analyzer import GaitAnalyzer
import json

def main():
    parser = argparse.ArgumentParser(description='跑步姿态分析系统')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('--output', '-o', help='结果输出文件路径（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"错误：视频文件 {args.video_path} 不存在")
        sys.exit(1)
    
    print("正在初始化姿态检测器...")
    pose_detector = RunningPoseDetector()
    gait_analyzer = GaitAnalyzer()
    
    print("正在分析视频...")
    try:
        # 检测姿态
        frame_data = pose_detector.process_video(args.video_path)
        
        if not frame_data:
            print("错误：无法检测到人体姿态")
            sys.exit(1)
        
        print(f"成功检测到 {len(frame_data)} 帧数据")
        
        # 分析步态
        gait_data = gait_analyzer.analyze_gait_cycle(frame_data)
        form_data = gait_analyzer.analyze_running_form(frame_data)
        
        # 生成建议
        recommendations = gait_analyzer.generate_recommendations(gait_data, form_data)
        
        # 准备结果
        result = {
            'video_path': args.video_path,
            'gait_metrics': {
                'cadence': round(gait_data['cadence'], 1),
                'stride_length': round(gait_data['stride_length'], 3),
                'total_steps': gait_data['total_steps']
            },
            'form_analysis': {
                'avg_knee_angle': round(sum(form_data['knee_angles']) / len(form_data['knee_angles']), 1) if form_data['knee_angles'] else 0,
                'avg_forward_lean': round(sum(form_data['forward_lean']) / len(form_data['forward_lean']), 1) if form_data['forward_lean'] else 0
            },
            'recommendations': recommendations
        }
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {args.output}")
        
        # 控制台输出
        print("\n" + "="*50)
        print("跑步姿态分析结果")
        print("="*50)
        print(f"步频: {result['gait_metrics']['cadence']} 步/分钟")
        print(f"步幅: {result['gait_metrics']['stride_length']} (相对单位)")
        print(f"总步数: {result['gait_metrics']['total_steps']} 步")
        print(f"平均膝关节角度: {result['form_analysis']['avg_knee_angle']} 度")
        print(f"平均身体前倾角度: {result['form_analysis']['avg_forward_lean']} 度")
        
        print("\n改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        if args.verbose:
            print(f"\n详细数据:")
            print(f"检测到的左脚着地次数: {len(gait_data['left_foot_strikes'])}")
            print(f"检测到的右脚着地次数: {len(gait_data['right_foot_strikes'])}")
            print(f"膝关节角度变化范围: {min(form_data['knee_angles']) if form_data['knee_angles'] else 0:.1f} - {max(form_data['knee_angles']) if form_data['knee_angles'] else 0:.1f} 度")
            print(f"身体前倾角度变化范围: {min(form_data['forward_lean']) if form_data['forward_lean'] else 0:.1f} - {max(form_data['forward_lean']) if form_data['forward_lean'] else 0:.1f} 度")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
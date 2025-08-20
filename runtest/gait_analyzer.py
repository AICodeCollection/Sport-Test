import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class GaitAnalyzer:
    def __init__(self):
        self.foot_strike_threshold = 0.02  # 脚部着地检测阈值
        self.min_step_duration = 0.3  # 最小步长持续时间(秒)
        
    def analyze_gait_cycle(self, frame_data: List[Dict]) -> Dict:
        """分析步态周期"""
        timestamps = [frame['timestamp'] for frame in frame_data]
        
        # 提取左右脚的垂直位置
        left_foot_y = []
        right_foot_y = []
        
        for frame in frame_data:
            landmarks = frame['landmarks']
            key_points = self._extract_key_points(landmarks)
            
            left_foot_y.append(key_points.get('left_ankle', {}).get('y', 0))
            right_foot_y.append(key_points.get('right_ankle', {}).get('y', 0))
        
        # 检测脚部着地时刻
        left_strikes = self._detect_foot_strikes(left_foot_y, timestamps)
        right_strikes = self._detect_foot_strikes(right_foot_y, timestamps)
        
        # 计算步频和步幅
        cadence = self._calculate_cadence(left_strikes, right_strikes)
        stride_length = self._calculate_stride_length(frame_data)
        
        return {
            'cadence': cadence,
            'stride_length': stride_length,
            'left_foot_strikes': left_strikes,
            'right_foot_strikes': right_strikes,
            'total_steps': len(left_strikes) + len(right_strikes)
        }
    
    def _extract_key_points(self, landmarks: List[Dict]) -> Dict:
        """提取关键身体部位的坐标"""
        key_points = {}
        
        landmark_indices = {
            'left_ankle': 27,
            'right_ankle': 28,
            'left_knee': 25,
            'right_knee': 26,
            'left_hip': 23,
            'right_hip': 24,
            'left_shoulder': 11,
            'right_shoulder': 12
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                key_points[name] = {
                    'x': landmarks[idx]['x'],
                    'y': landmarks[idx]['y'],
                    'z': landmarks[idx]['z']
                }
        
        return key_points
    
    def _detect_foot_strikes(self, foot_y: List[float], timestamps: List[float]) -> List[float]:
        """检测脚部着地时刻"""
        foot_y = np.array(foot_y)
        
        # 找到局部最大值（脚部最低点，因为y坐标系统）
        peaks, _ = find_peaks(foot_y, distance=int(self.min_step_duration * 30))  # 假设30fps
        
        strike_times = [timestamps[i] for i in peaks]
        return strike_times
    
    def _calculate_cadence(self, left_strikes: List[float], right_strikes: List[float]) -> float:
        """计算步频 (步数/分钟)"""
        if not left_strikes and not right_strikes:
            return 0
        
        all_strikes = sorted(left_strikes + right_strikes)
        if len(all_strikes) < 2:
            return 0
        
        total_time = all_strikes[-1] - all_strikes[0]
        total_steps = len(all_strikes)
        
        # 步频 = 步数/分钟
        cadence = (total_steps / total_time) * 60
        return cadence
    
    def _calculate_stride_length(self, frame_data: List[Dict]) -> float:
        """计算步幅（基于身体移动距离的估算）"""
        if len(frame_data) < 2:
            return 0
        
        # 使用髋部中心点来估算前进距离
        hip_positions = []
        
        for frame in frame_data:
            landmarks = frame['landmarks']
            key_points = self._extract_key_points(landmarks)
            
            left_hip = key_points.get('left_hip', {})
            right_hip = key_points.get('right_hip', {})
            
            if left_hip and right_hip:
                center_x = (left_hip['x'] + right_hip['x']) / 2
                hip_positions.append(center_x)
        
        if len(hip_positions) < 2:
            return 0
        
        # 计算总的水平移动距离
        total_distance = abs(hip_positions[-1] - hip_positions[0])
        return total_distance
    
    def analyze_running_form(self, frame_data: List[Dict]) -> Dict:
        """分析跑步姿态"""
        form_metrics = {
            'knee_angles': [],
            'hip_angles': [],
            'ankle_angles': [],
            'forward_lean': [],
            'arm_swing': []
        }
        
        for frame in frame_data:
            landmarks = frame['landmarks']
            key_points = self._extract_key_points(landmarks)
            
            # 计算膝关节角度
            if all(k in key_points for k in ['left_hip', 'left_knee', 'left_ankle']):
                knee_angle = self._calculate_angle(
                    key_points['left_hip'],
                    key_points['left_knee'],
                    key_points['left_ankle']
                )
                form_metrics['knee_angles'].append(knee_angle)
            
            # 计算身体前倾角度
            if all(k in key_points for k in ['left_shoulder', 'left_hip']):
                lean_angle = self._calculate_forward_lean(
                    key_points['left_shoulder'],
                    key_points['left_hip']
                )
                form_metrics['forward_lean'].append(lean_angle)
        
        return form_metrics
    
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """计算三点之间的角度"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_forward_lean(self, shoulder: Dict, hip: Dict) -> float:
        """计算身体前倾角度"""
        # 计算肩膀到髋部的向量与垂直线的夹角
        vertical_vector = np.array([0, 1])
        body_vector = np.array([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
        
        cos_angle = np.dot(vertical_vector, body_vector) / (np.linalg.norm(vertical_vector) * np.linalg.norm(body_vector))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def generate_recommendations(self, gait_data: Dict, form_data: Dict) -> List[str]:
        """生成跑步建议"""
        recommendations = []
        
        # 步频建议
        cadence = gait_data.get('cadence', 0)
        if cadence < 160:
            recommendations.append("步频较低，建议增加步频到160-180步/分钟，可以减少受伤风险")
        elif cadence > 200:
            recommendations.append("步频过高，建议适当降低步频，保持在160-180步/分钟范围内")
        
        # 膝关节角度建议
        knee_angles = form_data.get('knee_angles', [])
        if knee_angles:
            avg_knee_angle = np.mean(knee_angles)
            if avg_knee_angle < 140:
                recommendations.append("膝关节弯曲过度，建议保持更自然的步态")
            elif avg_knee_angle > 170:
                recommendations.append("膝关节过于僵直，建议适当增加膝关节弯曲")
        
        # 身体前倾建议
        forward_lean = form_data.get('forward_lean', [])
        if forward_lean:
            avg_lean = np.mean(forward_lean)
            if avg_lean > 15:
                recommendations.append("身体前倾过度，建议保持更直立的姿态")
            elif avg_lean < 5:
                recommendations.append("身体过于直立，建议适当前倾以提高跑步效率")
        
        if not recommendations:
            recommendations.append("您的跑步姿态总体良好，继续保持!")
        
        return recommendations
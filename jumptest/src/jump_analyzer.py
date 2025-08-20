import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from scipy.signal import find_peaks, savgol_filter
from pose_detector import PoseDetector

class JumpAnalyzer:
    """跳跃分析类，分析跳跃动作的各项指标"""
    
    def __init__(self, fps: float = 30.0):
        """
        初始化跳跃分析器
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        self.pose_detector = PoseDetector()
        
    def analyze_jump_sequence(self, pose_results: List[Optional[Dict]]) -> Dict:
        """
        分析完整的跳跃序列
        
        Args:
            pose_results: 姿态检测结果列表
            
        Returns:
            Dict: 跳跃分析结果
        """
        # 提取关键指标
        body_centers = self._extract_body_centers(pose_results)
        knee_angles = self._extract_knee_angles(pose_results)
        hip_angles = self._extract_hip_angles(pose_results)
        
        # 识别跳跃阶段
        jump_phases = self._identify_jump_phases(body_centers)
        
        # 计算跳跃参数
        jump_metrics = self._calculate_jump_metrics(body_centers, jump_phases)
        
        # 分析身体姿态
        posture_analysis = self._analyze_posture(pose_results, jump_phases)
        
        # 评估弹跳力和核心力量
        strength_assessment = self._assess_strength(knee_angles, hip_angles, jump_metrics)
        
        return {
            'jump_phases': jump_phases,
            'jump_metrics': jump_metrics,
            'posture_analysis': posture_analysis,
            'strength_assessment': strength_assessment,
            'body_centers': body_centers,
            'knee_angles': knee_angles,
            'hip_angles': hip_angles
        }
        
    def _extract_body_centers(self, pose_results: List[Optional[Dict]]) -> List[Optional[Tuple[float, float]]]:
        """提取身体中心点序列"""
        body_centers = []
        
        for pose_result in pose_results:
            if pose_result:
                center = self.pose_detector.get_body_center(pose_result)
                body_centers.append(center)
            else:
                body_centers.append(None)
                
        return body_centers
        
    def _extract_knee_angles(self, pose_results: List[Optional[Dict]]) -> List[Optional[Tuple[float, float]]]:
        """提取膝关节角度序列"""
        knee_angles = []
        
        for pose_result in pose_results:
            if pose_result:
                # 左膝角度
                left_hip = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_hip')
                left_knee = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_knee')
                left_ankle = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_ankle')
                
                # 右膝角度
                right_hip = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_hip')
                right_knee = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_knee')
                right_ankle = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_ankle')
                
                left_angle = None
                right_angle = None
                
                if left_hip and left_knee and left_ankle:
                    left_angle = self.pose_detector.calculate_angle(left_hip, left_knee, left_ankle)
                    
                if right_hip and right_knee and right_ankle:
                    right_angle = self.pose_detector.calculate_angle(right_hip, right_knee, right_ankle)
                    
                knee_angles.append((left_angle, right_angle))
            else:
                knee_angles.append(None)
                
        return knee_angles
        
    def _extract_hip_angles(self, pose_results: List[Optional[Dict]]) -> List[Optional[Tuple[float, float]]]:
        """提取髋关节角度序列"""
        hip_angles = []
        
        for pose_result in pose_results:
            if pose_result:
                # 左髋角度
                left_shoulder = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_shoulder')
                left_hip = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_hip')
                left_knee = self.pose_detector.get_keypoint_coordinates(pose_result, 'left_knee')
                
                # 右髋角度
                right_shoulder = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_shoulder')
                right_hip = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_hip')
                right_knee = self.pose_detector.get_keypoint_coordinates(pose_result, 'right_knee')
                
                left_angle = None
                right_angle = None
                
                if left_shoulder and left_hip and left_knee:
                    left_angle = self.pose_detector.calculate_angle(left_shoulder, left_hip, left_knee)
                    
                if right_shoulder and right_hip and right_knee:
                    right_angle = self.pose_detector.calculate_angle(right_shoulder, right_hip, right_knee)
                    
                hip_angles.append((left_angle, right_angle))
            else:
                hip_angles.append(None)
                
        return hip_angles
        
    def _identify_jump_phases(self, body_centers: List[Optional[Tuple[float, float]]]) -> Dict:
        """识别跳跃的各个阶段"""
        # 过滤无效数据
        valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
        
        if len(valid_centers) < 3:  # 降低最小要求从10到3
            return {'error': '有效数据点不足，至少需要3个有效帧'}
            
        # 提取Y坐标（垂直位置）
        y_coords = [center[1] for _, center in valid_centers]
        frame_indices = [i for i, _ in valid_centers]
        
        # 平滑处理
        if len(y_coords) > 5:
            y_coords_smooth = savgol_filter(y_coords, 5, 2)
        else:
            y_coords_smooth = y_coords
            
        # 寻找最低点（准备阶段结束）和最高点（腾空最高点）
        min_idx = np.argmin(y_coords_smooth)
        max_idx = np.argmax(y_coords_smooth)
        
        # 确定各阶段
        phases = {
            'preparation': {
                'start_frame': frame_indices[0],
                'end_frame': frame_indices[min_idx],
                'description': '准备阶段'
            },
            'takeoff': {
                'start_frame': frame_indices[min_idx],
                'end_frame': frame_indices[max_idx],
                'description': '起跳阶段'
            },
            'landing': {
                'start_frame': frame_indices[max_idx],
                'end_frame': frame_indices[-1],
                'description': '落地阶段'
            },
            'peak_frame': frame_indices[max_idx],
            'lowest_frame': frame_indices[min_idx]
        }
        
        return phases
        
    def _calculate_jump_metrics(self, body_centers: List[Optional[Tuple[float, float]]], 
                               jump_phases: Dict) -> Dict:
        """计算跳跃指标"""
        if 'error' in jump_phases:
            return {'error': jump_phases['error']}
            
        # 获取有效的身体中心点
        valid_centers = [(i, center) for i, center in enumerate(body_centers) if center is not None]
        
        if len(valid_centers) < 2:  # 降低要求到最低2个点
            return {'error': '有效数据点不足，至少需要2个有效帧'}
            
        # 计算跳跃高度（像素差）
        lowest_y = None
        highest_y = None
        
        for i, center in valid_centers:
            if i == jump_phases['lowest_frame']:
                lowest_y = center[1]
            elif i == jump_phases['peak_frame']:
                highest_y = center[1]
                
        if lowest_y is not None and highest_y is not None:
            jump_height_pixels = abs(lowest_y - highest_y)
        else:
            jump_height_pixels = 0
            
        # 计算腾空时间
        takeoff_duration = (jump_phases['takeoff']['end_frame'] - 
                           jump_phases['takeoff']['start_frame']) / self.fps
        
        # 计算准备时间
        preparation_duration = (jump_phases['preparation']['end_frame'] - 
                               jump_phases['preparation']['start_frame']) / self.fps
        
        # 计算落地时间
        landing_duration = (jump_phases['landing']['end_frame'] - 
                           jump_phases['landing']['start_frame']) / self.fps
        
        return {
            'jump_height_pixels': jump_height_pixels,
            'takeoff_duration': takeoff_duration,
            'preparation_duration': preparation_duration,
            'landing_duration': landing_duration,
            'total_duration': preparation_duration + takeoff_duration + landing_duration
        }
        
    def _analyze_posture(self, pose_results: List[Optional[Dict]], jump_phases: Dict) -> Dict:
        """分析身体姿态"""
        if 'error' in jump_phases:
            return {'error': jump_phases['error']}
            
        analysis = {
            'preparation_posture': self._analyze_phase_posture(pose_results, jump_phases['preparation']),
            'takeoff_posture': self._analyze_phase_posture(pose_results, jump_phases['takeoff']),
            'landing_posture': self._analyze_phase_posture(pose_results, jump_phases['landing'])
        }
        
        return analysis
        
    def _analyze_phase_posture(self, pose_results: List[Optional[Dict]], phase: Dict) -> Dict:
        """分析特定阶段的姿态"""
        start_frame = phase['start_frame']
        end_frame = phase['end_frame']
        
        # 收集该阶段的姿态数据
        phase_poses = pose_results[start_frame:end_frame + 1]
        
        # 计算平均关节角度
        knee_angles = []
        hip_angles = []
        shoulder_alignment = []
        
        for pose in phase_poses:
            if pose:
                # 膝关节角度
                left_hip = self.pose_detector.get_keypoint_coordinates(pose, 'left_hip')
                left_knee = self.pose_detector.get_keypoint_coordinates(pose, 'left_knee')
                left_ankle = self.pose_detector.get_keypoint_coordinates(pose, 'left_ankle')
                
                if left_hip and left_knee and left_ankle:
                    knee_angle = self.pose_detector.calculate_angle(left_hip, left_knee, left_ankle)
                    knee_angles.append(knee_angle)
                
                # 髋关节角度
                left_shoulder = self.pose_detector.get_keypoint_coordinates(pose, 'left_shoulder')
                if left_shoulder and left_hip and left_knee:
                    hip_angle = self.pose_detector.calculate_angle(left_shoulder, left_hip, left_knee)
                    hip_angles.append(hip_angle)
                
                # 肩膀对齐
                left_shoulder = self.pose_detector.get_keypoint_coordinates(pose, 'left_shoulder')
                right_shoulder = self.pose_detector.get_keypoint_coordinates(pose, 'right_shoulder')
                if left_shoulder and right_shoulder:
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                    shoulder_alignment.append(shoulder_diff)
        
        return {
            'avg_knee_angle': np.mean(knee_angles) if knee_angles else None,
            'avg_hip_angle': np.mean(hip_angles) if hip_angles else None,
            'shoulder_alignment': np.mean(shoulder_alignment) if shoulder_alignment else None,
            'stability_score': self._calculate_stability_score(phase_poses)
        }
        
    def _calculate_stability_score(self, poses: List[Optional[Dict]]) -> float:
        """计算稳定性得分"""
        if not poses:
            return 0.0
            
        # 计算身体中心点的变化
        centers = []
        for pose in poses:
            if pose:
                center = self.pose_detector.get_body_center(pose)
                if center:
                    centers.append(center)
        
        if len(centers) < 2:
            return 0.0
            
        # 计算中心点变化的标准差
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # 稳定性得分（变化越小越稳定）
        stability = 1.0 / (1.0 + math.sqrt(x_std**2 + y_std**2))
        
        return stability
        
    def _assess_strength(self, knee_angles: List[Optional[Tuple[float, float]]], 
                        hip_angles: List[Optional[Tuple[float, float]]], 
                        jump_metrics: Dict) -> Dict:
        """评估弹跳力和核心力量"""
        if 'error' in jump_metrics:
            return {'error': jump_metrics['error']}
            
        # 弹跳力评估
        explosive_power = self._assess_explosive_power(jump_metrics)
        
        # 核心力量评估
        core_strength = self._assess_core_strength(knee_angles, hip_angles)
        
        # 协调性评估
        coordination = self._assess_coordination(knee_angles, hip_angles)
        
        return {
            'explosive_power': explosive_power,
            'core_strength': core_strength,
            'coordination': coordination,
            'overall_score': (explosive_power + core_strength + coordination) / 3
        }
        
    def _assess_explosive_power(self, jump_metrics: Dict) -> float:
        """评估爆发力"""
        # 基于跳跃高度和起跳时间
        height = jump_metrics.get('jump_height_pixels', 0)
        duration = jump_metrics.get('takeoff_duration', 1)
        
        # 爆发力 = 高度 / 时间
        if duration > 0:
            power = height / duration
            # 归一化到0-1范围
            return min(1.0, power / 100)
        
        return 0.0
        
    def _assess_core_strength(self, knee_angles: List[Optional[Tuple[float, float]]], 
                            hip_angles: List[Optional[Tuple[float, float]]]) -> float:
        """评估核心力量"""
        # 基于关节角度的稳定性
        valid_knee_angles = [angles for angles in knee_angles if angles and angles[0] and angles[1]]
        valid_hip_angles = [angles for angles in hip_angles if angles and angles[0] and angles[1]]
        
        if not valid_knee_angles or not valid_hip_angles:
            return 0.0
            
        # 计算左右对称性
        knee_symmetry = self._calculate_symmetry(valid_knee_angles)
        hip_symmetry = self._calculate_symmetry(valid_hip_angles)
        
        # 核心力量得分
        core_score = (knee_symmetry + hip_symmetry) / 2
        
        return core_score
        
    def _assess_coordination(self, knee_angles: List[Optional[Tuple[float, float]]], 
                           hip_angles: List[Optional[Tuple[float, float]]]) -> float:
        """评估协调性"""
        # 基于关节角度的变化平滑性
        valid_knee_angles = [angles for angles in knee_angles if angles and angles[0] and angles[1]]
        valid_hip_angles = [angles for angles in hip_angles if angles and angles[0] and angles[1]]
        
        if len(valid_knee_angles) < 3 or len(valid_hip_angles) < 3:
            return 0.0
            
        # 计算角度变化的平滑性
        knee_smoothness = self._calculate_smoothness(valid_knee_angles)
        hip_smoothness = self._calculate_smoothness(valid_hip_angles)
        
        # 协调性得分
        coordination_score = (knee_smoothness + hip_smoothness) / 2
        
        return coordination_score
        
    def _calculate_symmetry(self, angle_pairs: List[Tuple[float, float]]) -> float:
        """计算左右对称性"""
        asymmetries = []
        
        for left, right in angle_pairs:
            if left and right:
                asymmetry = abs(left - right) / max(left, right)
                asymmetries.append(asymmetry)
        
        if not asymmetries:
            return 0.0
            
        # 对称性得分（不对称性越小越好）
        avg_asymmetry = np.mean(asymmetries)
        symmetry_score = 1.0 / (1.0 + avg_asymmetry)
        
        return symmetry_score
        
    def _calculate_smoothness(self, angle_pairs: List[Tuple[float, float]]) -> float:
        """计算角度变化的平滑性"""
        left_angles = [pair[0] for pair in angle_pairs if pair[0]]
        right_angles = [pair[1] for pair in angle_pairs if pair[1]]
        
        if len(left_angles) < 3 or len(right_angles) < 3:
            return 0.0
            
        # 计算角度变化的二阶导数（加速度）
        left_smoothness = self._calculate_angle_smoothness(left_angles)
        right_smoothness = self._calculate_angle_smoothness(right_angles)
        
        return (left_smoothness + right_smoothness) / 2
        
    def _calculate_angle_smoothness(self, angles: List[float]) -> float:
        """计算单个角度序列的平滑性"""
        if len(angles) < 3:
            return 0.0
            
        # 计算二阶差分
        second_diff = np.diff(angles, 2)
        
        # 平滑性得分（二阶差分越小越平滑）
        smoothness = 1.0 / (1.0 + np.mean(np.abs(second_diff)))
        
        return smoothness
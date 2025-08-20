import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import json

class PoseDetector:
    """姿态检测类，使用MediaPipe进行人体姿态估计"""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        初始化姿态检测器
        
        Args:
            model_complexity: 模型复杂度 (0, 1, 2)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 关键点索引映射
        self.pose_landmarks_dict = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'left_index': 18, 'left_thumb': 19,
            'right_pinky': 20, 'right_index': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测单帧中的人体姿态
        
        Args:
            frame: 输入帧 (RGB格式)
            
        Returns:
            Optional[Dict]: 姿态检测结果或None
        """
        # 转换为MediaPipe所需的格式
        results = self.pose.process(frame)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            return {
                'landmarks': landmarks,
                'pose_landmarks': results.pose_landmarks,
                'frame_shape': frame.shape
            }
        
        return None
        
    def detect_pose_sequence(self, frames: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        检测视频序列中的姿态
        
        Args:
            frames: 帧序列
            
        Returns:
            List[Optional[Dict]]: 姿态检测结果列表
        """
        pose_results = []
        
        for i, frame in enumerate(frames):
            result = self.detect_pose(frame)
            pose_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(frames)} 帧")
                
        return pose_results
        
    def get_keypoint_coordinates(self, pose_result: Dict, keypoint_name: str) -> Optional[Tuple[float, float]]:
        """
        获取指定关键点的坐标
        
        Args:
            pose_result: 姿态检测结果
            keypoint_name: 关键点名称
            
        Returns:
            Optional[Tuple[float, float]]: 坐标 (x, y) 或None
        """
        if keypoint_name not in self.pose_landmarks_dict:
            return None
            
        keypoint_index = self.pose_landmarks_dict[keypoint_name]
        
        if pose_result and 'landmarks' in pose_result:
            landmarks = pose_result['landmarks']
            if keypoint_index < len(landmarks):
                landmark = landmarks[keypoint_index]
                return (landmark['x'], landmark['y'])
                
        return None
        
    def get_multiple_keypoints(self, pose_result: Dict, keypoint_names: List[str]) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        获取多个关键点的坐标
        
        Args:
            pose_result: 姿态检测结果
            keypoint_names: 关键点名称列表
            
        Returns:
            Dict[str, Optional[Tuple[float, float]]]: 关键点坐标字典
        """
        keypoints = {}
        for name in keypoint_names:
            keypoints[name] = self.get_keypoint_coordinates(pose_result, name)
        return keypoints
        
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        计算三点组成的角度
        
        Args:
            p1: 第一个点
            p2: 中间点（角度顶点）
            p3: 第三个点
            
        Returns:
            float: 角度（度）
        """
        # 向量计算
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
        
    def get_body_center(self, pose_result: Dict) -> Optional[Tuple[float, float]]:
        """
        获取身体中心点（肩部和髋部的中点）
        
        Args:
            pose_result: 姿态检测结果
            
        Returns:
            Optional[Tuple[float, float]]: 身体中心坐标
        """
        shoulders = self.get_multiple_keypoints(pose_result, ['left_shoulder', 'right_shoulder'])
        hips = self.get_multiple_keypoints(pose_result, ['left_hip', 'right_hip'])
        
        # 计算肩部中心
        left_shoulder = shoulders['left_shoulder']
        right_shoulder = shoulders['right_shoulder']
        if left_shoulder and right_shoulder:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
        else:
            return None
            
        # 计算髋部中心
        left_hip = hips['left_hip']
        right_hip = hips['right_hip']
        if left_hip and right_hip:
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
        else:
            return None
            
        # 身体中心
        body_center = ((shoulder_center[0] + hip_center[0]) / 2,
                      (shoulder_center[1] + hip_center[1]) / 2)
        
        return body_center
        
    def draw_pose_landmarks(self, frame: np.ndarray, pose_result: Dict) -> np.ndarray:
        """
        在帧上绘制姿态关键点
        
        Args:
            frame: 输入帧
            pose_result: 姿态检测结果
            
        Returns:
            np.ndarray: 绘制后的帧
        """
        if pose_result and 'pose_landmarks' in pose_result:
            annotated_frame = frame.copy()
            
            # 绘制姿态关键点
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_result['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            return annotated_frame
        
        return frame
        
    def save_pose_data(self, pose_results: List[Optional[Dict]], output_path: str) -> bool:
        """
        保存姿态数据到JSON文件
        
        Args:
            pose_results: 姿态检测结果列表
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 转换为可序列化的格式
            serializable_data = []
            for result in pose_results:
                if result:
                    serializable_data.append({
                        'landmarks': result['landmarks'],
                        'frame_shape': result['frame_shape']
                    })
                else:
                    serializable_data.append(None)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"保存姿态数据失败: {e}")
            return False
            
    def load_pose_data(self, input_path: str) -> List[Optional[Dict]]:
        """
        从JSON文件加载姿态数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            List[Optional[Dict]]: 姿态检测结果列表
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载姿态数据失败: {e}")
            return []
            
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'pose'):
            self.pose.close()
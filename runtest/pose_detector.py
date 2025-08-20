import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import math

class RunningPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """检测单帧图像中的姿态关键点"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
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
                'pose_landmarks': results.pose_landmarks
            }
        return None
    
    def process_video(self, video_path: str) -> List[Dict]:
        """处理视频文件，提取所有帧的姿态数据"""
        cap = cv2.VideoCapture(video_path)
        frame_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            pose_data = self.detect_pose(frame)
            if pose_data:
                pose_data['frame_number'] = frame_count
                pose_data['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                frame_data.append(pose_data)
            
            frame_count += 1
            
        cap.release()
        return frame_data
    
    def extract_key_points(self, landmarks: List[Dict]) -> Dict:
        """提取关键身体部位的坐标"""
        key_points = {}
        
        # MediaPipe关键点索引
        landmark_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                key_points[name] = {
                    'x': landmarks[idx]['x'],
                    'y': landmarks[idx]['y'],
                    'z': landmarks[idx]['z']
                }
        
        return key_points
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """计算三点之间的角度"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_distance(self, p1: Dict, p2: Dict) -> float:
        """计算两点之间的距离"""
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
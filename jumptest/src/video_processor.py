import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class VideoProcessor:
    """视频处理类，负责视频的加载、预处理和帧提取"""
    
    def __init__(self, video_path: str):
        """
        初始化视频处理器
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.frames = []
        
    def load_video(self) -> bool:
        """
        加载视频文件
        
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(self.video_path):
            print(f"视频文件不存在: {self.video_path}")
            return False
            
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"无法打开视频文件: {self.video_path}")
            return False
            
        # 获取视频基本信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} 帧")
        return True
        
    def extract_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> List[np.ndarray]:
        """
        提取视频帧
        
        Args:
            start_frame: 开始帧
            end_frame: 结束帧（None表示到视频结尾）
            
        Returns:
            List[np.ndarray]: 帧列表
        """
        if self.cap is None:
            print("视频未加载，请先调用load_video()")
            return []
            
        frames = []
        
        if end_frame is None:
            end_frame = self.total_frames
            
        # 设置起始帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 转换为RGB格式（OpenCV默认是BGR）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            current_frame += 1
            
        self.frames = frames
        print(f"提取了 {len(frames)} 帧")
        return frames
        
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        预处理单帧
        
        Args:
            frame: 输入帧
            target_size: 目标尺寸 (width, height)
            
        Returns:
            np.ndarray: 处理后的帧
        """
        # 调整尺寸
        resized = cv2.resize(frame, target_size)
        
        # 可以在这里添加其他预处理步骤
        # 例如：去噪、增强对比度等
        
        return resized
        
    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        获取指定时间点的帧
        
        Args:
            time_seconds: 时间点（秒）
            
        Returns:
            Optional[np.ndarray]: 帧数据或None
        """
        if self.cap is None:
            return None
            
        frame_number = int(time_seconds * self.fps)
        if frame_number >= self.total_frames:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
        
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        保存帧到文件
        
        Args:
            frame: 帧数据
            output_path: 输出路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 转换为BGR格式以便保存
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
            return True
        except Exception as e:
            print(f"保存帧失败: {e}")
            return False
            
    def get_video_info(self) -> dict:
        """
        获取视频信息
        
        Returns:
            dict: 视频信息字典
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.total_frames / self.fps if self.fps > 0 else 0
        }
        
    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def __del__(self):
        """析构函数，确保资源释放"""
        self.release()
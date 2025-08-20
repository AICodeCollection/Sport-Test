# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a jump posture analysis system that uses computer vision and machine learning to analyze jumping movements. The system can assess jump technique, explosive power, core strength, and body posture from uploaded videos.

## Development Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Integration Tests
```bash
python test_integration.py
```

### Start Jupyter Demo
```bash
jupyter notebook notebooks/tech_validation.ipynb
```

## Architecture

The system follows a modular architecture with four main components:

- **VideoProcessor** (`src/video_processor.py`): Handles video loading, frame extraction, and preprocessing
- **PoseDetector** (`src/pose_detector.py`): Uses MediaPipe for real-time human pose estimation
- **JumpAnalyzer** (`src/jump_analyzer.py`): Analyzes jump sequences and calculates metrics
- **JumpVisualizer** (`src/visualizer.py`): Generates analysis reports and visualizations

### Key Technologies
- OpenCV for video processing
- MediaPipe for pose estimation
- NumPy/SciPy for numerical computations
- Matplotlib for visualization

## Common Development Tasks

### Running the Full Analysis Pipeline
```python
from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.jump_analyzer import JumpAnalyzer
from src.visualizer import JumpVisualizer

# Load and process video
processor = VideoProcessor("video_path.mp4")
processor.load_video()
frames = processor.extract_frames()

# Detect poses
detector = PoseDetector()
pose_results = detector.detect_pose_sequence(frames)

# Analyze jump
analyzer = JumpAnalyzer(fps=processor.fps)
analysis_result = analyzer.analyze_jump_sequence(pose_results)

# Generate visualization
visualizer = JumpVisualizer()
visualizer.visualize_jump_analysis(analysis_result)
```

### Testing with Mock Data
Use `test_integration.py` for testing core functionality without real video data.

## Output Files
- Analysis charts: `outputs/jump_analysis.png`
- Text reports: `outputs/jump_analysis_report.txt`
- Pose animations: `outputs/pose_animation.mp4` (optional)
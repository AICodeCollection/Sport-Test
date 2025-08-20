# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a running gait analysis system that uses computer vision to analyze running form from video files. The system uses MediaPipe for pose detection and provides detailed biomechanical analysis with recommendations for improvement.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Web Application
```bash
python app.py
```
The web interface will be available at http://localhost:8888

### Command Line Analysis
```bash
python main.py <video_path> [--output output.json] [--verbose]
```

### Testing
No specific test framework configured yet. Manual testing with sample videos recommended.

## Architecture

### Core Components

1. **pose_detector.py** - MediaPipe-based pose detection
   - `RunningPoseDetector` class handles video processing and landmark extraction
   - Extracts 33 key body landmarks per frame
   - Provides angle and distance calculation utilities

2. **gait_analyzer.py** - Biomechanical analysis engine
   - `GaitAnalyzer` class processes pose data to extract running metrics
   - Calculates cadence, stride length, joint angles
   - Generates personalized recommendations based on form analysis

3. **app.py** - Flask web application
   - RESTful API for video upload and analysis
   - Generates visualization charts using matplotlib
   - Returns JSON results with metrics and recommendations

4. **main.py** - Command-line interface
   - Standalone script for batch processing
   - Supports JSON output and verbose logging

### Key Analysis Metrics

- **Cadence**: Steps per minute calculation
- **Stride Length**: Estimated from hip center movement
- **Joint Angles**: Knee, hip, and ankle angle analysis
- **Forward Lean**: Body posture measurement
- **Foot Strike Pattern**: Ground contact detection

### File Structure
```
├── pose_detector.py      # Core pose detection
├── gait_analyzer.py      # Biomechanical analysis
├── app.py               # Web application
├── main.py              # CLI interface
├── requirements.txt     # Dependencies
├── templates/           # HTML templates
│   └── index.html      # Main web interface
├── uploads/            # Video upload directory
└── results/            # Analysis results storage
```

### Dependencies

- OpenCV for video processing
- MediaPipe for pose detection
- NumPy/SciPy for numerical analysis
- Matplotlib for visualization
- Flask for web interface
- Pandas for data handling
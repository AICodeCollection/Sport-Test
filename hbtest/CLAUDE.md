# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time heart rate detection web application that uses PPG (Photoplethysmography) technology to measure heart rate through webcam video analysis. The application detects subtle color changes in facial blood circulation to calculate BPM in real-time.

## Development Commands

This is a client-side web application that runs directly in the browser:

- **Local Development**: Open `index.html` in a modern browser or use a local server
- **Testing**: Manual testing through browser (requires HTTPS for camera access)
- **Deployment**: Any web server capable of serving static files

## Architecture

### Core Components

1. **HeartRateApp** (`js/app.js`): Main application controller
2. **HeartRateDetector** (`js/heart-rate-detector.js`): Core detection logic
3. **SignalProcessor** (`js/signal-processing.js`): Signal processing algorithms
4. **FFT** (`js/fft.js`): Fast Fourier Transform implementation

### Signal Processing Pipeline

1. **Video Capture**: 30fps webcam stream via WebRTC
2. **Face Detection**: Uses face-api.js for face tracking
3. **ROI Extraction**: Extracts forehead/cheek regions
4. **RGB Signal Extraction**: Analyzes green channel (most sensitive to blood flow)
5. **Signal Processing**: Band-pass filtering (0.8-3Hz), noise reduction
6. **Heart Rate Calculation**: FFT analysis to find dominant frequency → BPM

### Key Technical Details

- **Sampling Rate**: 30fps (sufficient for heart rate detection)
- **Heart Rate Range**: 45-200 BPM
- **Signal Buffer**: 10 seconds of data
- **FFT Window**: 512 points with Hanning window
- **ROI**: Forehead region extracted from face landmarks

### Browser Requirements

- Modern browser with WebRTC support
- Camera permissions
- HTTPS environment (for production)
- Good lighting conditions for accurate detection
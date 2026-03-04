# Drowsiness Detector
# Real-Time Drowsiness Detection

## Overview
Detects drowsiness in real-time using webcam by tracking eye closure and yawning.

## Features
- Eye Aspect Ratio (EAR) for detecting sleepiness
- Mouth Aspect Ratio (MAR) for yawning
- Sound & popup alert
- Logs events to CSV

## Requirements
- Python 3.x
- OpenCV, Mediapipe, Numpy, Playsound, Pandas

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Put `alarm.wav` in `data/`
3. Run: `python src/main.py`
4. Press ESC to exit
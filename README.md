# Real-Time Finger Detection

A real-time computer vision project that detects a hand through a webcam and counts raised fingers using Python, OpenCV, and MediaPipe.

## Features

- Real-time hand detection
- 21 hand landmarks
- Finger counting from 0–5
- Joint-angle based finger detection
- Hand skeleton visualization
- Frame-history smoothing for stable results
- Real-time webcam processing

## Technologies

- Python
- OpenCV
- MediaPipe
- Computer Vision
- Geometry / Trigonometry

## How It Works

Webcam → OpenCV → MediaPipe Hand Landmarker → 21 Hand Landmarks → Joint Angle Analysis → Finger Counting → Frame Smoothing → Live Result

## Finger Detection Logic

The four fingers are detected using angles at the PIP and DIP joints.

A finger is considered open when both joint angles are greater than 160°.

The thumb is evaluated separately using its landmark geometry.

Recent finger-count results are stored and the most common result is displayed to reduce flickering.

## Installation

Install the required packages:

pip install opencv-python mediapipe

Make sure `hand_landmarker.task` is in the same folder as `project1.py`.

## Run

python3 project1.py

Press `Q` to close the application.

## Project Structure

Real-Time-Finger-Detection/

├── project1.py  
└── hand_landmarker.task

## Future Improvements

- Improve hand orientation handling
- Support multiple hands
- Add gesture recognition
- Add custom actions for different gestures
- Improve the user interface
- Add FPS and confidence information

## Author

**Himanshu Pareek**

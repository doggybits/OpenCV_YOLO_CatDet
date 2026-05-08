# OpenCV_YOLO_CatDet 🐱

**Smart Cat Door / Exclusive Cat Box System**

A computer vision system that detects cats and classifies their color (white vs orange) in real-time using **OpenCV + YOLO**. 

The goal is to create an **exclusive cat box** that allows a white cat to enter but denies access to an orange cat (who plays too rough and disturbs the white cat's quiet space).

![Project Concept](https://via.placeholder.com/800x400?text=Cat+Detection+Demo)  
<!-- Replace with actual screenshot/GIF later -->

## 🎯 Project Goal

- Detect when a cat approaches the box
- Classify the cat's primary color (White / Orange)
- Trigger different actions (e.g., open door for white cat, keep closed for orange cat)
- Provide a peaceful hidey box for the calmer white cat

## ✨ Features

- Real-time cat detection using **YOLO**
- Color classification (White vs Orange) based on fur in the bounding box
- Modular design with separate utilities
- Webcam / camera stream support
- Debug and visualization tools

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV** (core computer vision)
- **YOLO** (object detection)
- Custom color classification logic

## 📁 Files Overview

- `main.py` — Main application entry point
- `cat_color_classifier.py` — Core logic for detecting and classifying cat color
- `camera_utils.py` — Camera handling and frame processing utilities
- `face_recognition_module.py` — (Optional) Face/individual cat recognition
- `cat_color_debug.py` — Debugging and visualization tools
- `LICENSE` — GNU GPL v2

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/doggybits/OpenCV_YOLO_CatDet.git
cd OpenCV_YOLO_CatDet

# 2. Install dependencies
pip install opencv-python numpy ultralytics  # or whatever YOLO wrapper you're using

# 3. Run the main program
python main.py

## Planned Implementation
# 1. Integrate with a physical servo-controlled cat door
# 2. Improve color classification robustness (lighting variations)
# 3. Add individual cat recognition (face / pattern)
# 4. Deploy on Raspberry Pi / edge device
# 5. MQTT / Home Assistant integration for notifications

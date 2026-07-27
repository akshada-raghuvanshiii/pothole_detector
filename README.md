# Intelligent Road Pothole Detection and Severity Assessment using YOLOv8

## Overview

This project implements a real-time road pothole detection system using a custom-trained YOLOv8 model. In addition to object detection, the system estimates pothole severity using rule-based logic that considers pothole size, lane position, and temporal smoothing across consecutive frames.

The project demonstrates the use of computer vision techniques for intelligent road condition monitoring and autonomous driving applications.

---

## Features

- Real-time pothole detection
- Custom YOLOv8 model
- Region of Interest (ROI) optimization
- Adaptive confidence thresholds
- Rule-based severity estimation
- Temporal smoothing for stable predictions
- OpenCV visualization

---

## Project Structure

```
Pothole-Detection-Using-YOLOv8/
│
├── detect.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── assets/
└── test_outputs/
```

---

## Installation

```bash
git clone https://github.com/yourusername/Pothole-Detection-Using-YOLOv8.git

cd Pothole-Detection-Using-YOLOv8

pip install -r requirements.txt
```

---

## Usage

Update the paths inside `detect.py` if necessary.

Run

```bash
python detect.py
```

Controls

- Space → Pause
- Q → Quit

---

## Methodology

1. Capture video frames
2. Restrict detection to the road ROI
3. Detect potholes using YOLOv8
4. Estimate severity using:
   - Bounding box size
   - Lane position
   - Temporal averaging
5. Display annotated output

---

## Dataset

The dataset used for training is **not included** in this repository and belongs to its original creators.

---

## Future Improvements

- GPS localization
- Road condition mapping
- ROS2 integration
- Multi-class road defect detection

---

## Author

Akshada Raghuvanshi

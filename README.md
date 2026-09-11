# 🤖 AI Vision Recognition

![Python](https://img.shields.io/badge/Python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![YOLOE](https://img.shields.io/badge/YOLOE-Object%20Detection-red)
[![GitHub Stars](https://img.shields.io/github/stars/Rakeshgowda17/AI-Vision-Recognition?style=flat)](https://github.com/Rakeshgowda17/AI-Vision-Recognition)
[Live Demo](https://rakeshgowda17.github.io/AI-Vision-Recognition/)
A real-time AI computer vision application built with **Python, OpenCV, MediaPipe, and Ultralytics YOLOE**.

The system uses a webcam to detect **hand landmarks, count fingers, recognize hand gestures, and detect objects in real time**.

## Features

- Hand landmark detection
- Finger counting from 0 to 5
- Hand gesture recognition
- Open-vocabulary object detection
- Real-time webcam processing
- Object confidence scores
- Calendar, phone, bottle, book and other object recognition

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Ultralytics YOLOE
- NumPy
- Git & GitHub

## Project Structure

```text
AI-Vision-Recognition/
│
├── venv/
├── hand_landmarker.task
├── main.py
├── object_detection.py
├── final_app.py
├── README.md
└── requirements.txt

How It Works:
1. Hand Detection
MediaPipe detects the hand and identifies 21 hand landmarks.

2. Finger Counting
The application analyzes the hand landmarks and counts raised fingers.
Example:
FINGERS: 3

3. Gesture Recognition
The application recognizes basic hand gestures such as:

FIST
OPEN HAND
PEACE
POINT
THUMBS UP
THUMBS DOWN

4. Object Recognition
YOLOE is used for real-time object detection.

Example objects include:
Person
Cell phone
Calendar
Bottle
Book
Laptop
Keyboard
Mouse
Chair
Watch

Installation:
Create and activate a Python virtual environment:
python -m venv venv

Activate it on Windows:
.\venv\Scripts\Activate.ps1

Install the required packages:
pip install opencv-python mediapipe numpy ultralytics

Run the Application

Activate the virtual environment and run:
python final_app.py

The webcam will open automatically.
Press:
Q
to close the application.

Example Output
The application can display information such as:
GESTURE: PEACE
FINGERS: 2
person 0.89
cell phone 0.82
bottle 0.91
calendar 0.76

## Project Screenshots

### Object Detection
![Object Detection](assets/object-detection.jpg)

### Open Hand Gesture
![Open Hand Gesture](assets/open-hand-gesture.jpg)

### Finger Counting
![Finger Counting](assets/finger-counting.jpg)

### Hand Landmarks
![Hand Landmarks](assets/hand-landmarks.jpg)

Future Improvements:
Improve gesture recognition accuracy
Add more hand gestures
Add voice output
Add custom object detection
Add a graphical user interface
Improve performance on low-end computers
Add support for more objects
Deploy the application as a desktop or web application

Author:
Developed as an AI and Computer Vision project using Python.

License:
This project is intended for educational and portfolio purposes.

Step 2 — Create `requirements.txt`

Right-click your project folder again → **New File** → name it:
```text
requirements.txt

Paste:
opencv-python
mediapipe
numpy
ultralytics


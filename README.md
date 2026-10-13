# AI Traffic Safety & Collision Detection Simulator

An interactive Streamlit application powered by YOLOv8 and OpenCV for real-time traffic video object tracking, proximity hazard analysis, and CSV session logging.

## Features
- **Real-time YOLO Tracking:** Utilizes Ultralytics YOLOv8 and ByteTrack to track vehicles and pedestrians across video streams.
- **Proximity Collision Warning Engine:** Calculates bounding box center distances and flags risk events when objects come too close.
- **Interactive Streamlit Dashboard:** Customize confidence thresholds, tracking models, and proximity limits from a sidebar control panel.
- **Audit Logging & CSV Export:** Records frame-by-frame telemetry and provides instant CSV report downloads.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

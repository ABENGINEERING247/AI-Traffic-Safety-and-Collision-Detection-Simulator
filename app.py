import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Page Layout Setup
st.set_page_config(
    page_title="AI Traffic Safety & Collision Simulator with Logs",
    page_icon="🤖",
    layout="wide",
)

st.title("🚦 AI Traffic Safety & Collision Simulator + Analytics Export")
st.markdown(
    "Monitor real-time object tracking, inspect collision warnings, and export complete session logs as a CSV file."
)

# Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLOv8 Engine", ["yolov8n.pt", "yolov8s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider(
    "Collision Proximity Limit (px)", 30, 250, 80, 5
)

# Load Model with Caching
@st.cache_resource
def get_model(model_name):
  return YOLO(model_name)

model = get_model(selected_model)

# File Management Section
uploaded_video = st.file_uploader(
    "Upload Traffic Video Source", type=["mp4", "mov", "avi"]
)

if uploaded_video:
  temp_path = "temp_uploaded_traffic.mp4"
  with open(temp_path, "wb") as f:
    f.write(uploaded_video.read())

  # Dashboard Layout Columns
  col1, col2 = st.columns([2, 1])

  with col1:
    st.subheader("Live Video Stream Feed")
    video_placeholder = st.empty()

  with col2:
    st.subheader("Real-Time Telemetry")
    metric_box_vehicles = st.metric("Active Objects Tracked", 0)
    metric_box_warnings = st.metric("Collision Warnings", 0, delta="0")
    status_alert = st.empty()

  if st.button("Initialize Simulation & Logging"):
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    simulation_logs = []
    frame_idx = 0
    warning_count = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_idx += 1
      timestamp_sec = round(frame_idx / fps, 2)

      # Run YOLO Tracking
      results = model.track(
          frame,
          conf=conf_threshold,
          persist=True,
          tracker="bytetrack.yaml",
          verbose=False,
      )
      annotated = results[0].plot()

      boxes = results[0].boxes.xyxy.cpu().numpy()
      active_objects = len(boxes)
      current_risk = False

      # Proximity Matrix Checks
      if active_objects >= 2:
        for i in range(active_objects):
          for j in range(i + 1, active_objects):
            b1, b2 = boxes[i], boxes[j]
            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
            c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
            dist = np.linalg.norm(c1 - c2)

            if dist < proximity_limit:
              current_risk = True
              cv2.line(
                  annotated,
                  (int(c1[0]), int(c1[1])),
                  (int(c2[0]), int(c2[1])),
                  (0, 0, 255),
                  3,
              )

      if current_risk:
        warning_count += 1
        cv2.putText(
            annotated,
            "WARNING: HAZARD PROXIMITY DETECTED",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )
        status_alert.error("🚨 Collision risk detected in current frame!")
      else:
        status_alert.success("🟢 Traffic flow stable.")

      # Append frame telemetry to logs
      simulation_logs.append({
          "Frame": frame_idx,
          "Timestamp (s)": timestamp_sec,
          "Active Objects": active_objects,
          "Collision Risk Warning": "YES" if current_risk else "NO",
      })

      # Update Metrics
      metric_box_vehicles.metric("Active Objects Tracked", active_objects)
      metric_box_warnings.metric("Collision Warnings", warning_count)

      # Render Frame
      annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
      video_placeholder.image(
          annotated, channels="RGB", use_container_width=True
      )

    cap.release()
    st.success("Simulation finished successfully!")

    # Convert logs to DataFrame and provide CSV export
    df_logs = pd.DataFrame(simulation_logs)
    st.markdown("### 📋 Simulation Telemetry & Audit Report")
    st.dataframe(df_logs.head(10), use_container_width=True)

    csv_data = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Simulation CSV Report",
        data=csv_data,
        file_name="traffic_simulation_audit_log.csv",
        mime="text/csv",
    )
else:
  st.info(
      "Upload a video file via the sidebar uploader to run the session logging"
      " simulator."
  )

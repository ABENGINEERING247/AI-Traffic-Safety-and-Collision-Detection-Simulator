import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="AI Traffic Safety & Warden Billu Simulator",
    page_icon="👮‍♂️",
    layout="wide",
)

st.title("🚦 AI Traffic Safety Simulator & Warden Billu Assistant")
st.markdown(
    "Monitor traffic safety using **YOLO26**, analyze static images, videos, or live webcams, track vehicle proximity hazards with **pop-up alerts**, and chat with **Warden Billu**."
)

# 2. Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLO Model Engine", ["yolo26n.pt", "yolo26s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider("Collision Proximity Limit (px)", 30, 250, 80, 5)

source_option = st.sidebar.radio(
    "Select Input Source", 
    ["Static Picture Upload", "Upload Video File", "Live Webcam"]
)

# 3. OpenAI & Warden Billu Configuration (Using st.secrets)
st.sidebar.markdown("---")
st.sidebar.subheader("👮‍♂️ Warden Billu (AI Traffic Assistant)")

if "OPENAI_API_KEY" in st.secrets:
  client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
  client = None
  st.sidebar.warning("⚠️ `OPENAI_API_KEY` not found in Streamlit secrets. Please add it to `.streamlit/secrets.toml`.")

# Initialize Warden Billu chat history
if "billu_messages" not in st.session_state:
  st.session_state.billu_messages = [
      {
          "role": "system",
          "content": (
              "You are Warden Billu, an enthusiastic, authoritative yet friendly neighborhood "
              "traffic police officer and road safety assistant. You speak with a helpful demeanor, "
              "often using traffic-related puns or catchphrases (like 'Beep beep!', 'Stay on the safe lane!'). "
              "Your job is to answer questions about traffic rules, signs, fines, and safe driving habits."
          )
      },
      {
          "role": "assistant",
          "content": "Beep beep! Warden Billu on duty! 👮‍♂️ Ask me anything about traffic rules, road signs, or driving safety!"
      }
  ]

# Render chat history in sidebar
for message in st.session_state.billu_messages:
  if message["role"] != "system":
    with st.sidebar.chat_message(message["role"]):
      st.sidebar.markdown(message["content"])

# Chat input for Warden Billu
if billu_prompt := st.sidebar.chat_input("Ask Warden Billu a traffic question..."):
  if not client:
    st.sidebar.error("Please configure your OpenAI API key in Streamlit secrets.")
  else:
    st.session_state.billu_messages.append({"role": "user", "content": billu_prompt})
    with st.sidebar.chat_message("user"):
      st.sidebar.markdown(billu_prompt)

    try:
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=st.session_state.billu_messages,
          temperature=0.7,
      )
      billu_reply = response.choices[0].message.content
      st.session_state.billu_messages.append({"role": "assistant", "content": billu_reply})
      with st.sidebar.chat_message("assistant"):
        st.sidebar.markdown(billu_reply)
    except Exception as e:
      st.sidebar.error(f"OpenAI error: {e}")

# 4. Load YOLO Model with Caching
@st.cache_resource
def get_model(model_name):
  return YOLO(model_name)

model = get_model(selected_model)

# 5. Define Modal Dialog for Car Hazard Pop-up Alerts
@st.dialog("🚨 CAR COLLISION HAZARD POPUP ALERT!", width="small")
def show_hazard_popup(active_objs, distance_val):
  st.error(f"**Proximity Threshold Breached!**")
  st.write(f"Detected **{active_objs} objects** in close proximity (Distance: **{int(distance_val)}px**).")
  st.markdown("---")
  st.info("👮‍♂️ **Warden Billu says:** *'Slow down immediately, maintain safe braking distance, and yield right-of-way!'*")
  if st.button("Acknowledge & Close Alert"):
    st.rerun()

# 6. Dashboard Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("Visual Analysis Feed")
  media_placeholder = st.empty()

with col2:
  st.subheader("Real-Time Telemetry")
  metric_box_objects = st.metric("Active Objects Tracked", 0)
  metric_box_warnings = st.metric("Collision Warnings", 0, delta="0")
  status_alert = st.empty()

simulation_logs = []

# --- A. STATIC PICTURE UPLOAD ---
if source_option == "Static Picture Upload":
  uploaded_image = st.file_uploader("Upload Traffic Image Source", type=["jpg", "jpeg", "png"])
  
  if uploaded_image:
    image = Image.open(uploaded_image)
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if st.button("Run Image Hazard Analysis"):
      results = model(frame_bgr, conf=conf_threshold, verbose=False)
      annotated = results[0].plot()

      boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
      active_objects = len(boxes)
      current_risk = False
      min_dist = float("inf")

      if active_objects >= 2:
        for i in range(active_objects):
          for j in range(i + 1, active_objects):
            b1, b2 = boxes[i], boxes[j]
            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
            c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2] )
            dist = np.linalg.norm(c1 - c2)

            if dist < min_dist:
              min_dist = dist

            if dist < proximity_limit:
              current_risk = True
              cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

      warning_count = 0
      if current_risk:
        warning_count = 1
        cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        status_alert.error("🚨 Collision risk detected in image!")
        # Trigger Modal Pop-up
        show_hazard_popup(active_objects, min_dist)
      else:
        status_alert.success("🟢 Scene stable. No immediate hazards.")

      metric_box_objects.metric("Active Objects Tracked", active_objects)
      metric_box_warnings.metric("Collision Warnings", warning_count)

      annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
      media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

      simulation_logs.append({
          "Frame / Image": 1,
          "Timestamp (s)": 0.0,
          "Active Objects": active_objects,
          "Collision Risk Warning": "YES" if current_risk else "NO",
      })

      df_logs = pd.DataFrame(simulation_logs)
      st.markdown("### 📋 Image Audit Report")
      st.dataframe(df_logs, use_container_width=True)

      csv_data = df_logs.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Image Audit CSV Report",
          data=csv_data,
          file_name="traffic_image_audit_log.csv",
          mime="text/csv",
      )

# --- B. VIDEO FILE UPLOAD ---
elif source_option == "Upload Video File":
  uploaded_video = st.file_uploader("Upload Traffic Video Source", type=["mp4", "mov", "avi"])
  
  if uploaded_video:
    temp_path = "temp_uploaded_traffic.mp4"
    with open(temp_path, "wb") as f:
      f.write(uploaded_video.read())

    if st.button("Initialize Video Simulation & Logging"):
      cap = cv2.VideoCapture(temp_path)
      fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
      frame_idx = 0
      warning_count = 0

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_idx += 1
        timestamp_sec = round(frame_idx / fps, 2)

        results = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        active_objects = len(boxes)
        current_risk = False
        min_dist = float("inf")

        if active_objects >= 2:
          for i in range(active_objects):
            for j in range(i + 1, active_objects):
              b1, b2 = boxes[i], boxes[j]
              c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
              c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
              dist = np.linalg.norm(c1 - c2)

              if dist < min_dist:
                min_dist = dist

              if dist < proximity_limit:
                current_risk = True
                cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

        if current_risk:
          warning_count += 1
          cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
          status_alert.error("🚨 Collision risk detected in frame!")
          # Trigger popup on first warning event in video
          if warning_count == 1:
            show_hazard_popup(active_objects, min_dist)
        else:
          status_alert.success("🟢 Traffic flow stable.")

        simulation_logs.append({
            "Frame": frame_idx,
            "Timestamp (s)": timestamp_sec,
            "Active Objects": active_objects,
            "Collision Risk Warning": "YES" if current_risk else "NO",
        })

        metric_box_objects.metric("Active Objects Tracked", active_objects)
        metric_box_warnings.metric("Collision Warnings", warning_count)

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        media_placeholder.image(annotated, channels="RGB", use_container_width=True)

      cap.release()
      st.success("Video simulation finished successfully!")

      df_logs = pd.DataFrame(simulation_logs)
      st.markdown("### 📋 Video Simulation Audit Report")
      st.dataframe(df_logs.head(10), use_container_width=True)

      csv_data = df_logs.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Video Audit CSV Report",
          data=csv_data,
          file_name="traffic_video_audit_log.csv",
          mime="text/csv",
      )

# --- C. LIVE WEBCAM FEED ---
elif source_option == "Live Webcam":
  st.markdown("ℹ️ *Ensure your webcam is connected. Use the controls below to manage the feed.*")
  
  if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

  col_start, col_stop = st.columns(2)
  with col_start:
    if st.button("🔴 Start Live Webcam Stream"):
      st.session_state.webcam_running = True
  with col_stop:
    if st.button("⏹️ Stop Webcam Stream"):
      st.session_state.webcam_running = False

  if st.session_state.webcam_running:
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    warning_count = 0
    start_time = time.time()

    while st.session_state.webcam_running and cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        st.error("Failed to access webcam stream. Check device connection.")
        break

      frame_idx += 1
      timestamp_sec = round(time.time() - start_time, 2)

      results = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
      annotated = results[0].plot()

      boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
      active_objects = len(boxes)
      current_risk = False
      min_dist = float("inf")

      if active_objects >= 2:
        for i in range(active_objects):
          for j in range(i + 1, active_objects):
            b1, b2 = boxes[i], boxes[j]
            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
            c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
            dist = np.linalg.norm(c1 - c2)

            if dist < min_dist:
              min_dist = dist

            if dist < proximity_limit:
              current_risk = True
              cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

      if current_risk:
        warning_count += 1
        cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        status_alert.error("🚨 Collision risk detected in live stream!")
        if warning_count == 1:
          show_hazard_popup(active_objects, min_dist)
      else:
        status_alert.success("🟢 Webcam feed stable.")

      metric_box_objects.metric("Active Objects Tracked", active_objects)
      metric_box_warnings.metric("Collision Warnings", warning_count)

      annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
      media_placeholder.image(annotated, channels="RGB", use_container_width=True)

    cap.release()

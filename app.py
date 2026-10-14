from datetime import datetime
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from ultralytics import YOLO
from fpdf import FPDF
from pypdf import PdfReader
from docx import Document

# 1. Page Configuration
st.set_page_config(
    page_title="AI Traffic Safety Simulator & Warden Billu",
    page_icon="👮‍♂️",
    layout="wide",
)

# --- INITIALIZE PERSISTENT LOGS IN SESSION STATE ---
if "simulation_logs" not in st.session_state:
    st.session_state.simulation_logs = []

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# --- ANIMATED WELCOME POPUP ---
@st.dialog("🚗 Welcome to AI Traffic Safety Simulator", width="medium")
def show_welcome_popup():
    st.markdown("""
        <style>
        @keyframes headlight-blink {
            0% { opacity: 0.2; transform: scale(0.9); }
            50% { opacity: 1.0; transform: scale(1.1); filter: drop-shadow(0 0 10px #ff3333); }
            100% { opacity: 0.2; transform: scale(0.9); }
        }
        .blinking-light {
            animation: headlight-blink 0.8s infinite ease-in-out;
            font-size: 32px;
            text-align: center;
        }
        .welcome-banner {
            text-align: center;
            font-family: sans-serif;
        }
        </style>
        
        <div class="welcome-banner">
            <h2>🚦 Smart Road Safety & Hazard Detection</h2>
            <p>Powered by <b>YOLO26</b>, OpenCV, and <b>GPT Luna</b> 👮‍♂️</p>
            <div class="blinking-light">🔴 🚗 🔴</div>
            <p style="font-size: 13px; color: gray;">Headlights active. System ready!</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🚀 Enter Simulator Dashboard", type="primary", use_container_width=True):
        st.session_state.welcomed = True
        st.rerun()

if "welcomed" not in st.session_state:
    st.session_state.welcomed = False
    show_welcome_popup()

# --- TRAFFIC HEADER BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4068, #162447); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 32px;">🚥 AI TRAFFIC SAFETY & COLLISION SIMULATOR</h1>
        <p style="margin: 5px 0 0 0; font-size: 15px; color: #e4e4e4;">Multi-Format Hazard Auditing, Entity Detection & GPT Luna Assistant</p>
    </div>
""", unsafe_allow_html=True)

# 2. Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLO Model Engine", ["yolo26n.pt", "yolo26s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider("Collision Proximity Limit (px)", 30, 250, 80, 5)

source_option = st.sidebar.radio(
    "Select Input Source", 
    ["Media / Document Upload (Up to 400MB)", "Live Webcam"]
)

# 3. OpenAI & GPT Luna Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("👮‍♂️ Warden Billu (GPT Luna Assistant)")

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = None
    st.sidebar.warning("⚠️ `OPENAI_API_KEY` not found in Streamlit secrets.")

if "billu_messages" not in st.session_state:
    st.session_state.billu_messages = [
        {
            "role": "system",
            "content": (
                "You are Warden Billu, an enthusiastic, authoritative yet friendly neighborhood "
                "traffic police officer and road safety assistant powered by GPT Luna. You speak with a helpful demeanor, "
                "often using traffic-related puns or catchphrases (like 'Beep beep!', 'Stay on the safe lane!'). "
                "Your job is to answer questions about traffic rules, signs, fines, and safe driving habits."
            )
        },
        {
            "role": "assistant",
            "content": "Beep beep! Warden Billu on duty (powered by GPT Luna)! 👮‍♂️ Ask me anything about traffic rules, vehicle safety, or driving tips!"
        }
    ]

for message in st.session_state.billu_messages:
    if message["role"] != "system":
        with st.sidebar.chat_message(message["role"]):
            st.sidebar.markdown(message["content"])

if billu_prompt := st.sidebar.chat_input("Ask Warden Billu a traffic question..."):
    if not client:
        st.sidebar.error("Please configure your OpenAI API key in Streamlit secrets.")
    else:
        st.session_state.billu_messages.append({"role": "user", "content": billu_prompt})
        with st.sidebar.chat_message("user"):
            st.sidebar.markdown(billu_prompt)

        try:
            response = client.chat.completions.create(
                model="gpt-6-luna",
                messages=st.session_state.billu_messages,
                temperature=1.0,
            )
            billu_reply = response.choices[0].message.content
            st.session_state.billu_messages.append({"role": "assistant", "content": billu_reply})
            with st.sidebar.chat_message("assistant"):
                st.sidebar.markdown(billu_reply)
        except Exception as e:
            st.sidebar.error(f"OpenAI error: {e}")

@st.cache_resource
def get_model(model_name):
    return YOLO(model_name)

model = get_model(selected_model)

@st.dialog("🚨 CAR COLLISION HAZARD POPUP ALERT!", width="small")
def show_hazard_popup(active_objs, distance_val):
    st.error("**Proximity Threshold Breached!**")
    st.write(f"Detected close vehicle/object proximity (Distance: **{int(distance_val)}px**).")
    st.markdown("---")
    st.info("👮‍♂️ **Warden Billu says:** *'Slow down immediately, maintain safe braking distance, and yield right-of-way!'*")
    if st.button("Acknowledge & Close Alert"):
        st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visual Analysis Feed & Content Viewer")
    media_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Telemetry & Breakdown")
    metric_box_objects = st.metric("Active Objects Tracked", 0)
    metric_box_warnings = st.metric("Collision Warnings", 0, delta="0")
    st.markdown("---")
    st.subheader("Entity Breakdown")
    breakdown_placeholder = st.empty()
    status_alert = st.empty()

def update_breakdown_display(v_counts):
    html_table = """
    <style>
    .breakdown-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        font-size: 13px;
        margin-top: 5px;
    }
    .breakdown-table th {
        background-color: #1f4068;
        color: white;
        text-align: left;
        padding: 6px;
        border: 1px solid #ddd;
    }
    .breakdown-table td {
        padding: 5px 6px;
        border: 1px solid #ddd;
    }
    .breakdown-table tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.05);
    }
    </style>
    <table class="breakdown-table">
        <tr>
            <th>Entity Category</th>
            <th style="text-align: center;">QTY</th>
        </tr>
    """
    for category, qty in v_counts.items():
        html_table += f"<tr><td>{category}</td><td style='text-align: center; font-weight: bold;'>{qty}</td></tr>"
    html_table += "</table>"
    breakdown_placeholder.markdown(html_table, unsafe_allow_html=True)

update_breakdown_display({
    "Cars": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0, 
    "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0
})

def generate_pdf_report(df_data):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(277, 8, txt="AI Traffic Safety & Comprehensive Entity Audit Report", ln=True, align="C")
    
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", "I", 9)
    pdf.cell(277, 6, txt=f"Report Generated On: {generation_time} | Total Audit Records: {len(df_data)}", ln=True, align="C")
    pdf.ln(4)
    
    pdf.set_font("Arial", "B", 8)
    pdf.cell(15, 8, "Frm/ID", 1, 0, "C")
    pdf.cell(32, 8, "Date & Time", 1, 0, "C")
    pdf.cell(12, 8, "Tot", 1, 0, "C")
    pdf.cell(12, 8, "Car", 1, 0, "C")
    pdf.cell(12, 8, "Bike", 1, 0, "C")
    pdf.cell(12, 8, "Bus", 1, 0, "C")
    pdf.cell(15, 8, "Amb", 1, 0, "C")
    pdf.cell(15, 8, "Person", 1, 0, "C")
    pdf.cell(18, 8, "Masked", 1, 0, "C")
    pdf.cell(15, 8, "Disabled", 1, 0, "C")
    pdf.cell(15, 8, "Blind", 1, 0, "C")
    pdf.cell(12, 8, "Risk?", 1, 0, "C")
    pdf.cell(69, 8, "Warden Billu Instructions", 1, 1, "C")
    
    pdf.set_font("Arial", "", 7)
    for index, row in df_data.head(50).iterrows():
        pdf.cell(15, 8, str(row.get("Frame / File", row.get("Frame", 1))), 1, 0, "C")
        pdf.cell(32, 8, str(row.get("Date & Time", "")), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Active Objects", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Cars", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Bikes", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Buses", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Ambulances", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Persons", 0)), 1, 0, "C")
        pdf.cell(18, 8, str(row.get("Persons with Mask", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Disabled Persons", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Blind Persons", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Collision Risk Warning", "NO")), 1, 0, "C")
        pdf.cell(69, 8, str(row.get("Warden Billu Instructions", "Maintain steady speed.")), 1, 1, "L")
        
    return pdf.output(dest='S').encode('latin1')

def classify_all_entities(boxes, result_boxes):
    counts = {
        "Cars": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0, 
        "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0
    }
    for box in result_boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, "").lower()
        
        if "ambulance" in cls_name:
            counts["Ambulances"] += 1
        elif "car" in cls_name:
            counts["Cars"] += 1
        elif "motorcycle" in cls_name or "bike" in cls_name or "bicycle" in cls_name:
            counts["Bikes"] += 1
        elif "bus" in cls_name:
            counts["Buses"] += 1
        elif "mask" in cls_name:
            counts["Persons with Mask"] += 1
        elif "disabled" in cls_name or "wheelchair" in cls_name:
            counts["Disabled Persons"] += 1
        elif "blind" in cls_name or "cane" in cls_name:
            counts["Blind Persons"] += 1
        elif "person" in cls_name:
            counts["Persons"] += 1
            
    return counts

def render_audit_report_section():
    st.markdown("---")
    col_hdr, col_clear = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 📋 Cumulative Audit Log & Report Generator")
    with col_clear:
        if st.button("🗑️ Clear Audit History", type="secondary"):
            st.session_state.simulation_logs = []
            st.rerun()

    if st.session_state.simulation_logs:
        df_logs = pd.DataFrame(st.session_state.simulation_logs)
        st.dataframe(df_logs, use_container_width=True)

        col_csv, col_pdf = st.columns(2)
        with col_csv:
            csv_data = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV Report", data=csv_data, file_name="traffic_audit_report.csv", mime="text/csv")
        with col_pdf:
            pdf_bytes = generate_pdf_report(df_logs)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="traffic_audit_report.pdf", mime="application/pdf")
    else:
        st.info("Audit log is currently empty.")

# Helper to open camera with cross-platform fallback logic
def get_camera():
    # Attempt 1: Standard DirectShow (Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()

    # Attempt 2: Standard Auto Index 0
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    cap.release()

    # Attempt 3: Index 1 (External Webcams)
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        return cap
    cap.release()
    
    return None

# --- MULTI-FORMAT FILE UPLOADER ---
if source_option == "Media / Document Upload (Up to 400MB)":
    uploaded_file = st.file_uploader(
        "Upload Traffic Image, Video, or Document (JPG, PNG, MP4, MOV, AVI, PDF, DOCX, TXT - Max 400MB)", 
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "pdf", "docx", "txt"]
    )
    
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame_bgr, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()

            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
            active_objects = len(boxes)
            v_counts = classify_all_entities(boxes, results[0].boxes)
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

            warning_count = 0
            if current_risk:
                warning_count = 1
                instruction_text = "Reduce speed immediately, yield to pedestrians/ambulances, and prepare to brake."
                cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                status_alert.error("🚨 Collision risk detected in uploaded image!")
                show_hazard_popup(active_objects, min_dist)
            else:
                instruction_text = "Traffic flow clear. Maintain safe speed and remain alert."
                status_alert.success("🟢 Scene stable. No immediate hazards.")

            metric_box_objects.metric("Active Objects Tracked", active_objects)
            metric_box_warnings.metric("Collision Warnings", warning_count)
            update_breakdown_display(v_counts)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            st.session_state.simulation_logs.append({
                "Date & Time": current_datetime,
                "Frame / File": f"Img: {uploaded_file.name}",
                "Timestamp (s)": 0.0,
                "Active Objects": active_objects,
                "Cars": v_counts["Cars"],
                "Bikes": v_counts["Bikes"],
                "Buses": v_counts["Buses"],
                "Ambulances": v_counts["Ambulances"],
                "Persons": v_counts["Persons"],
                "Persons with Mask": v_counts["Persons with Mask"],
                "Disabled Persons": v_counts["Disabled Persons"],
                "Blind Persons": v_counts["Blind Persons"],
                "Collision Risk Warning": "YES" if current_risk else "NO",
                "Warden Billu Instructions": instruction_text,
            })

        elif file_extension in ["mp4", "mov", "avi"]:
            temp_path = "temp_uploaded_traffic.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            warning_count = 0
            popup_triggered = False

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
                v_counts = classify_all_entities(boxes, results[0].boxes)
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
                    instruction_text = "Reduce speed immediately, yield right-of-way, and increase following distance."
                    cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    status_alert.error("🚨 Collision risk detected in video frame!")
                    if not popup_triggered:
                        show_hazard_popup(active_objects, min_dist)
                        popup_triggered = True
                else:
                    instruction_text = "Maintain steady cruising speed and stay alert."
                    status_alert.success("🟢 Traffic flow stable.")

                st.session_state.simulation_logs.append({
                    "Date & Time": current_datetime,
                    "Frame / File": f"{uploaded_file.name} (Fr:{frame_idx})",
                    "Timestamp (s)": timestamp_sec,
                    "Active Objects": active_objects,
                    "Cars": v_counts["Cars"],
                    "Bikes": v_counts["Bikes"],
                    "Buses": v_counts["Buses"],
                    "Ambulances": v_counts["Ambulances"],
                    "Persons": v_counts["Persons"],
                    "Persons with Mask": v_counts["Persons with Mask"],
                    "Disabled Persons": v_counts["Disabled Persons"],
                    "Blind Persons": v_counts["Blind Persons"],
                    "Collision Risk Warning": "YES" if current_risk else "NO",
                    "Warden Billu Instructions": instruction_text,
                })

                metric_box_objects.metric("Active Objects Tracked", active_objects)
                metric_box_warnings.metric("Collision Warnings", warning_count)
                update_breakdown_display(v_counts)

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                media_placeholder.image(annotated, channels="RGB", use_container_width=True)

            cap.release()
            st.success("Video processing completed and appended to audit history!")

        elif file_extension in ["pdf", "docx", "txt"]:
            extracted_text = ""
            if file_extension == "pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            elif file_extension == "docx":
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            elif file_extension == "txt":
                extracted_text = uploaded_file.read().decode("utf-8")

            media_placeholder.markdown(f"""
                <div style="background-color: #162447; padding: 15px; border-radius: 8px; color: white; max-height: 400px; overflow-y: auto;">
                    <h4>📄 Document Text Audit ({uploaded_file.name})</h4>
                    <p style="font-size: 13px; color: #dcdcdc; white-space: pre-wrap;">{extracted_text[:3000]}...</p>
                </div>
            """, unsafe_allow_html=True)

            status_alert.success("🟢 Document text successfully audited.")
            instruction_text = "Document reviewed. Ensure policy adherence and driver compliance."
            
            st.session_state.simulation_logs.append({
                "Date & Time": current_datetime,
                "Frame / File": f"Doc: {uploaded_file.name}",
                "Timestamp (s)": 0.0,
                "Active Objects": 0,
                "Cars": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0,
                "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0,
                "Collision Risk Warning": "NO",
                "Warden Billu Instructions": instruction_text,
            })

    render_audit_report_section()

# --- C. LIVE WEBCAM FEED ---
elif source_option == "Live Webcam":
    st.markdown("ℹ️ *Use the buttons below to control your built-in webcam stream.*")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔴 Start Built-in Webcam Stream", type="primary"):
            st.session_state.webcam_active = True
    with col_btn2:
        if st.button("⏹️ Stop Webcam Stream"):
            st.session_state.webcam_active = False

    if st.session_state.webcam_active:
        cap = get_camera()
        
        if cap is None or not cap.isOpened():
            st.error("⚠️ Local hardware webcam unavailable. Falling back to Browser Webcam input:")
            img_file_buffer = st.camera_input("Take a snapshot from Browser Camera")
            
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                results = model(cv2_img, conf=conf_threshold, verbose=False)
                annotated = results[0].plot()
                
                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
                active_objects = len(boxes)
                v_counts = classify_all_entities(boxes, results[0].boxes)
                
                metric_box_objects.metric("Active Objects Tracked", active_objects)
                update_breakdown_display(v_counts)
                
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        else:
            frame_idx = 0
            warning_count = 0
            start_time = time.time()

            # Stream frames continuously
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read stream frame from camera.")
                    break

                frame_idx += 1
                timestamp_sec = round(time.time() - start_time, 2)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                results = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
                annotated = results[0].plot()

                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
                active_objects = len(boxes)
                v_counts = classify_all_entities(boxes, results[0].boxes)
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
                    instruction_text = "Slow down instantly, watch for pedestrians/ambulances, and stay defensive."
                    cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    status_alert.error("🚨 Collision risk detected in live stream!")
                else:
                    instruction_text = "Normal operation. Maintain regular safe speed limits."
                    status_alert.success("🟢 Webcam feed active.")

                st.session_state.simulation_logs.append({
                    "Date & Time": current_datetime,
                    "Frame / File": f"Webcam (Fr:{frame_idx})",
                    "Timestamp (s)": timestamp_sec,
                    "Active Objects": active_objects,
                    "Cars": v_counts["Cars"],
                    "Bikes": v_counts["Bikes"],
                    "Buses": v_counts["Buses"],
                    "Ambulances": v_counts["Ambulances"],
                    "Persons": v_counts["Persons"],
                    "Persons with Mask": v_counts["Persons with Mask"],
                    "Disabled Persons": v_counts["Disabled Persons"],
                    "Blind Persons": v_counts["Blind Persons"],
                    "Collision Risk Warning": "YES" if current_risk else "NO",
                    "Warden Billu Instructions": instruction_text,
                })

                metric_box_objects.metric("Active Objects Tracked", active_objects)
                metric_box_warnings.metric("Collision Warnings", warning_count)
                update_breakdown_display(v_counts)

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                media_placeholder.image(annotated, channels="RGB", use_container_width=True)

            cap.release()
    else:
        st.info("Webcam stream is currently stopped. Click 'Start Built-in Webcam Stream' to begin.")

    render_audit_report_section()

from datetime import datetime
import re
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

# -------------------------------------------------------------------
# HELPER FUNCTION: STRIP UNICODE EMOJIS FOR FPDF COMPATIBILITY
# -------------------------------------------------------------------
def strip_emojis(text: str) -> str:
    """Removes unicode emoji characters that standard FPDF fonts (Helvetica/Arial) cannot encode."""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"  # Extended symbols/emojis
        "\u2600-\u27BF"          # Miscellaneous symbols
        "\u2300-\u23FF"          # Miscellaneous technical
        "\u2B00-\u2BFF"          # Various symbols & arrows
        "\u200D"                 # Zero-width joiner
        "\uFE0F]+",              # Variation selector
        flags=re.UNICODE
    )
    cleaned_text = emoji_pattern.sub("", text)
    return cleaned_text.encode("latin-1", "ignore").decode("latin-1")

# 1. Page Configuration
st.set_page_config(
    page_title="AI Traffic Safety Simulator & Warden Billu",
    page_icon="👮‍♂️",
    layout="wide",
)

# --- INITIALIZE PERSISTENT LOGS IN SESSION STATE ---
if "simulation_logs" not in st.session_state:
    st.session_state.simulation_logs = []

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
        <p style="margin: 5px 0 0 0; font-size: 15px; color: #e4e4e4;">Multi-Format Hazard Auditing, Entity Detection & Road Safety Certification</p>
    </div>
""", unsafe_allow_html=True)

# 2. Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLO Model Engine", ["yolo26n.pt", "yolo26s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider("Collision Proximity Limit (px)", 30, 250, 80, 5)

source_option = st.sidebar.radio(
    "Select Input Source / Mode", 
    ["Media / Document Upload (Up to 400MB)", "Live Webcam", "📚 Road Safety 100-Question Test"]
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

# --- PDF GENERATOR FOR AUDIT LOGS ---
def generate_pdf_report(df_data):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(277, 8, txt="AI Traffic Safety & Comprehensive Entity Audit Report", ln=True, align="C")
    
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(277, 6, txt=f"Report Generated On: {generation_time} | Total Audit Records: {len(df_data)}", ln=True, align="C")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 8)
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
    
    pdf.set_font("Helvetica", "", 7)
    for index, row in df_data.head(50).iterrows():
        pdf.cell(15, 8, strip_emojis(str(row.get("Frame / File", row.get("Frame", 1)))), 1, 0, "C")
        pdf.cell(32, 8, strip_emojis(str(row.get("Date & Time", ""))), 1, 0, "C")
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
        pdf.cell(69, 8, strip_emojis(str(row.get("Warden Billu Instructions", "Maintain steady speed."))), 1, 1, "L")
        
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return bytes(pdf_output)

# --- PDF GENERATOR FOR CERTIFICATE OF COMPETENCY / INCOMPETENCY ---
def generate_exam_certificate(candidate_name, score, is_competent):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    safe_name = strip_emojis(candidate_name)
    
    pdf.set_line_width(2)
    if is_competent:
        pdf.set_draw_color(31, 64, 104)
    else:
        pdf.set_draw_color(180, 40, 40)
    pdf.rect(10, 10, 277, 190)
    
    pdf.set_line_width(0.5)
    pdf.rect(13, 13, 271, 184)
    
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 24)
    if is_competent:
        pdf.set_text_color(31, 64, 104)
        pdf.cell(277, 12, "CERTIFICATE OF ROAD SAFETY COMPETENCY", ln=True, align="C")
    else:
        pdf.set_text_color(180, 40, 40)
        pdf.cell(277, 12, "NOTICE OF ASSESSMENT - NOT YET COMPETENT", ln=True, align="C")
        
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(277, 8, "Issued by AI Traffic Safety Academy & Warden Billu Inspection Bureau", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(277, 8, "This document certifies that candidate:", ln=True, align="C")
    
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(22, 36, 71)
    pdf.cell(277, 10, safe_name.upper(), ln=True, align="C")
    
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(277, 8, "has completed the Official 100-Question Road Safety & Traffic Regulations Assessment.", ln=True, align="C")
    
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 16)
    percentage = (score / 100.0) * 100
    if is_competent:
        pdf.set_text_color(0, 120, 50)
        pdf.cell(277, 8, f"RESULT: PASSED / COMPETENT ({score} / 100 - {percentage:.1f}%)", ln=True, align="C")
    else:
        pdf.set_text_color(180, 40, 40)
        pdf.cell(277, 8, f"RESULT: NOT YET COMPETENT ({score} / 100 - {percentage:.1f}%)", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(60, 60, 60)
    
    if is_competent:
        remarks = "Warden Billu Evaluation: Excellent knowledge of road signs, right-of-way rules, and hazard avoidance. Authorized for safe road navigation."
    else:
        remarks = "Warden Billu Evaluation: Further review of traffic laws, defensive driving concepts, and warning signs is required prior to re-assessment."
    
    pdf.multi_cell(250, 6, remarks, align="C")
    
    pdf.ln(20)
    issue_date = datetime.now().strftime("%B %d, %Y")
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(138, 6, f"Date of Assessment: {issue_date}", ln=0, align="C")
    pdf.cell(138, 6, "Officer Signature: Warden Billu", ln=1, align="C")
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(138, 4, "System Verified", ln=0, align="C")
    pdf.cell(138, 4, "Chief Inspector, GPT Luna Traffic Bureau", ln=1, align="C")
    
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return bytes(pdf_output)

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


# --- MAIN SELECTION CONDITIONAL BLOCK ---
if source_option == "Media / Document Upload (Up to 400MB)":
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

elif source_option == "Live Webcam":
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

    st.markdown("ℹ️ *Use your device camera to capture live traffic frames for YOLO analysis.*")
    camera_image = st.camera_input("Capture Live Traffic Snapshot")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(cv2_img, conf=conf_threshold, verbose=False)
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
            instruction_text = "Slow down instantly, watch for pedestrians/ambulances, and stay defensive."
            cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            status_alert.error("🚨 Collision risk detected in snapshot!")
            if warning_count == 1:
                show_hazard_popup(active_objects, min_dist)
        else:
            instruction_text = "Normal operation. Maintain regular safe speed limits."
            status_alert.success("🟢 Frame stable.")

        st.session_state.simulation_logs.append({
            "Date & Time": current_datetime,
            "Frame / File": f"Webcam Snapshot",
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

        metric_box_objects.metric("Active Objects Tracked", active_objects)
        metric_box_warnings.metric("Collision Warnings", warning_count)
        update_breakdown_display(v_counts)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

    render_audit_report_section()

elif source_option == "📚 Road Safety 100-Question Test":
    st.subheader("📝 Official Road Safety & Traffic Laws Certification Exam (100 Questions)")
    st.markdown("Test your knowledge across traffic regulations, road signs, defensive driving, and hazard protocols. Submit your test to receive an official score and download your **PDF Certificate of Competency / Evaluation**!")

    candidate_name = st.text_input("👤 Enter Full Name for Certificate:", "Driver Candidate")

    if "quiz_questions" not in st.session_state:
        q_pool = []
        categories = [
            ("Traffic Signs & Signals", [
                ("What does a triangular red-and-white yield sign mean?", ["Stop completely", "Give right-of-way to oncoming traffic", "Speed up to merge", "Turn around immediately"], 1),
                ("What color is a standard motorway service sign?", ["Blue with white text", "Green with white text", "Yellow with black text", "Red with white text"], 0),
                ("What should you do when a traffic light turns solid yellow?", ["Accelerate quickly to cross before red", "Stop if safe to do so before entering intersection", "Honk your horn", "Reverse your vehicle"], 1),
                ("A flashing red traffic light indicates what action?", ["Proceed with caution", "Treat it as a stop sign", "Malfunction - ignore", "Speed up"], 1),
                ("What does a circular blue sign with a white arrow indicate?", ["Mandatory direction", "Prohibited turn", "Parking allowed", "Speed limit"], 0),
            ]),
            ("Defensive Driving & Speed", [
                ("What is the standard recommended following distance under normal conditions?", ["1 second", "2 to 3 seconds", "5 seconds", "10 seconds"], 1),
                ("How should you brake on an icy road?", ["Slam brakes hard", "Apply gentle, progressive pressure without locking wheels", "Pull handbrake", "Turn off engine"], 1),
                ("What should you do if your vehicle starts hydroplaning?", ["Steer sharply and brake hard", "Ease off accelerator and steer straight until tires regain grip", "Pull handbrake", "Shift to reverse"], 1),
                ("When driving in heavy fog, which lights should you use?", ["High beam headlights", "Low beam headlights and fog lights", "Parking lights only", "No lights"], 1),
                ("What is the primary cause of urban intersection collisions?", ["Failure to yield right-of-way", "Vehicle mechanical failure", "Animal crossing", "Tire blowout"], 0),
            ]),
            ("Emergency & Pedestrian Protocols", [
                ("What action is mandatory when an emergency vehicle approaches with sirens active?", ["Maintain speed", "Pull over to the right and stop", "Speed up to stay ahead", "Turn on hazard lights and follow closely"], 1),
                ("Who has absolute right-of-way at an unmarked pedestrian crosswalk?", ["Motorists", "Pedestrians", "Cyclists", "Commercial trucks"], 1),
                ("How should you assist a visually impaired pedestrian using a white cane?", ["Grab their arm forcefully", "Ask if they need assistance and guide them safely across", "Honk to warn them", "Drive around quickly"], 1),
                ("What should you do immediately after being involved in a minor fender-bender?", ["Drive away immediately", "Move vehicles to safety if possible, check for injuries, and exchange info", "Leave cars in the middle of traffic", "Argue loudly"], 1),
                ("What does a flashing amber light on a school bus indicate?", ["Bus is stopping; prepare to stop", "Bus is accelerating", "Bus has a flat tire", "Bus is turning off route"], 0),
            ])
        ]
        
        q_idx = 1
        while len(q_pool) < 100:
            cat_name, base_questions = categories[(q_idx - 1) % len(categories)]
            base_q, options, correct = base_questions[(q_idx - 1) % len(base_questions)]
            q_text = f"Q{q_idx}: [{cat_name}] {base_q}" if q_idx > len(base_questions) else f"Q{q_idx}: {base_q}"
            q_pool.append({
                "id": q_idx,
                "question": q_text,
                "options": options,
                "answer": correct
            })
            q_idx += 1
            
        st.session_state.quiz_questions = q_pool

    with st.form("road_safety_exam_form"):
        user_answers = {}
        for q in st.session_state.quiz_questions:
            st.markdown(f"**{q['question']}**")
            user_answers[q['id']] = st.radio(
                f"Select answer for question {q['id']}:", 
                q['options'], 
                key=f"q_{q['id']}", 
                label_visibility="collapsed"
            )
            st.markdown("")

        submitted_exam = st.form_submit_button("📤 Submit 100-Question Exam & Generate Certificate", type="primary", use_container_width=True)

    if submitted_exam:
        score = 0
        incorrect_list = []
        for q in st.session_state.quiz_questions:
            selected_option = user_answers[q['id']]
            correct_option = q['options'][q['answer']]
            if selected_option == correct_option:
                score += 1
            else:
                incorrect_list.append((q['id'], q['question'], selected_option, correct_option))

        percentage = (score / 100.0) * 100
        is_competent = percentage >= 80.0

        st.markdown("---")
        st.subheader("📊 Exam Results & Official Evaluation")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Total Score", f"{score} / 100")
        col_res2.metric("Percentage", f"{percentage:.1f}%")
        
        if is_competent:
            col_res3.metric("Competency Status", "🟢 COMPETENT")
            st.success("👮‍♂️ **Warden Billu says:** *'Outstanding work! You have proven competent in traffic safety and road regulations!'*")
        else:
            col_res3.metric("Competency Status", "🔴 NOT YET COMPETENT")
            st.error("👮‍♂️ **Warden Billu says:** *'Beep beep! You did not meet the 80% passing threshold for competency. Please study the missed questions and re-try.'*")

        cert_pdf_bytes = generate_exam_certificate(candidate_name, score, is_competent)
        clean_file_name = candidate_name.replace(' ', '_')
        cert_filename = f"Road_Safety_Certificate_{clean_file_name}.pdf" if is_competent else f"Road_Safety_Notice_{clean_file_name}.pdf"
        btn_label = "📜 Download Official PDF Certificate of Competency" if is_competent else "📄 Download PDF Assessment Notice (Not Yet Competent)"

        st.download_button(
            label=btn_label,
            data=cert_pdf_bytes,
            file_name=cert_filename,
            mime="application/pdf",
            type="primary"
        )

        if incorrect_list:
            with st.expander(f"Review Missed Questions ({len(incorrect_list)} items)"):
                for q_id, q_text, user_ans, corr_ans in incorrect_list:
                    st.markdown(f"**{q_text}**")
                    st.markdown(f"❌ Your Answer: `{user_ans}`")
                    st.markdown(f"✅ Correct Answer: `{corr_ans}`")
                    st.markdown("---")

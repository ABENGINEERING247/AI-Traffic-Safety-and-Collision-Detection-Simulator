

# 🚥 AISIMULATOR: AI Traffic Safety & Collision Suite

**AISIMULATOR** is an advanced multi-modal road safety, entity tracking, and traffic collision auditing dashboard powered by **YOLO**, OpenCV, Streamlit, and GPT Luna (**Warden Billu**).

The application detects multi-category road entities (cars, trucks, bikes, buses, ambulances, pedestrians, masked persons, disabled persons, blind persons), monitors safety proximity margins, triggers real-time voice announcements and big-screen alerts, and provides a full custom entity training studio with `.pt` model generation.

---

## ✨ Key Features

### 1. 📢 Real-Time Object Talking & Flashing Alert System

* **Voice Announcements:** Automatically speaks the name of any detected object (e.g., `"Warning! car detected ahead!"`) via browser Text-to-Speech (`window.speechSynthesis`).
* **Big-Screen Flash Banner:** Displays an animated high-visibility flashing alert box on screen when entities are detected or proximity thresholds are breached.
* **Persistent Reminders:** Repeats the audio announcement every **10 seconds** until the user clicks the **✖ CLOSE ALERT** button.
* **Universal Availability:** Active across Media Uploads, Live Webcam snapshots, and Custom Object Webcam testing modes.

### 2. 🚘 Multi-Source Entity Tracking & Analysis

* **Media Upload Studio:** Supports images, videos, and document text audits (JPG, PNG, MP4, MOV, AVI, PDF, DOCX, TXT up to 400MB).
* **Live Webcam Mode:** Captures real-time camera snapshots for immediate entity detection and safety telemetry analysis.
* **Proximity Collision Warnings:** Calculates pixel distances between objects; draws hazard bounding vectors and raises alerts if safety buffers are breached.
* **Entity Breakdown Table:** Displays live dynamic counts for cars, trucks, bikes, buses, ambulances, pedestrians, masked individuals, disabled persons, and blind individuals.

### 3. 🧠 Custom Object / Face Training Studio & .pt Generator

* **Register & Train:** Upload sample reference images to train custom labels (e.g., VIP vehicles, inspector faces, custom obstacles).
* **Live Webcam Testing:** Test registered custom entity models live with real-time talking alerts and on-screen label matching.
* **PyTorch `.pt` Export:** Compile custom models and download complete PyTorch `.pt` binary weights along with a `data.yaml` configuration file.

### 4. 📚 100-Question Road Safety Certification Exam

* **Comprehensive Exam:** 100 randomized multiple-choice questions covering traffic signs, signals, defensive driving, and pedestrian emergency protocols.
* **Automated PDF Certificate Generator:** Produces an official **Certificate of Competency** (for scores ≥ 80%) or an **Assessment Evaluation Notice** featuring Warden Billu's evaluation remarks.

### 5. 📋 Comprehensive Audit Logging & PDF Reports

* **Persistent History:** Maintains session logs with frame timestamps, active object counts, hazard warnings, and Warden Billu driving instructions.
* **Export Options:** Download complete audit logs in CSV or beautifully formatted PDF report formats.

---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.9 or higher
* Streamlit
* PyTorch & Ultralytics YOLO
* OpenCV & Pillow
* FPDF2, PyPDF, python-docx
* OpenAI Python SDK (optional, for Warden Billu AI chat)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/aisimulator.git
cd aisimulator

```

### Step 2: Install Dependencies

```bash
pip install streamlit ultralytics opencv-python numpy pandas pillow openai fpdf2 pypdf python-docx torch

```

### Step 3: Configure OpenAI API Key (Optional)

Create a `.streamlit/secrets.toml` file in your root project directory:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"

```

### Step 4: Run the Application

```bash
streamlit run aisimulator.py

```

---

## 🚀 How to Use

1. **Dashboard Entry:** Click **Enter AISIMULATOR Dashboard** on the welcome popup.
2. **Select Input Mode from Sidebar:**
* **Media Upload:** Upload an image or video to run YOLO tracking, generate hazard alerts, and trigger audio announcements.
* **Live Webcam:** Take snapshot captures from your device camera to trigger live object detection and voice alerts.
* **Custom Training Studio:** Register new custom classes, test them via webcam snapshot with speech support, and download `.pt` state dicts.
* **Road Safety Exam:** Take the 100-question exam and download your official PDF safety competency certificate.


3. **Closing Alerts:** Click **✖ CLOSE ALERT** on the flashing modal banner to dismiss active audio reminders and reset the timer.

---

## 📄 File Structure

```text
aisimulator/
│
├── aisimulator.py            # Main application code
├── .streamlit/
│   └── secrets.toml          # OpenAI API secrets (optional)
├── temp_uploaded_traffic.mp4 # Temporary buffer for uploaded video streams
└── README.md                 # Project documentation

```

---

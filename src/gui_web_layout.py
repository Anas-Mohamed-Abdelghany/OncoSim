import sys
import os
import io
import cv2
import re
import base64
import requests
import numpy as np
import json
import random
import pydicom
import nibabel as nib

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QGroupBox, QFormLayout, QLineEdit, QProgressBar, 
                             QFrame, QMessageBox, QScrollArea, QSizePolicy, QGridLayout, QCheckBox, QRadioButton,
                             QSplashScreen, QDoubleSpinBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt
import google.generativeai as genai
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QTextEdit

# --- CONFIGURE API ---
API_KEY = "AIzaSyDEDLOpD4Cd_nEzCc4PEYFQbP8K53m8Pa4"
genai.configure(api_key=API_KEY)

# --- CUSTOM MODULES ---
import ai_core
import heatmap_engine
import tumor_classifier
import report_generator
import segmentation
import tumor_measurement
import laser_physics
import cloud_ai_engine
import tumor_growth_model

# ==========================================
# MODERN WEB-STYLE CSS
# ==========================================
WEB_STYLESHEET = """
QMainWindow { background-color: #121212; }
QScrollArea { border: none; background-color: #121212; }
QWidget#CentralContent { background-color: #121212; }

/* HEADERS */
QLabel#HeaderTitle { font-size: 26px; font-weight: bold; color: #ffffff; }
QLabel#HeaderSub { font-size: 14px; color: #888888; }
QLabel#CardTitle { font-size: 18px; font-weight: bold; color: #00e5ff; margin-bottom: 10px; }

/* CARDS (Sections) */
QFrame.Card { 
    background-color: #1e1e1e; 
    border-radius: 12px; 
    border: 1px solid #333;
}

/* INPUTS */
QLineEdit { 
    background-color: #2c2c2c; 
    color: white; 
    border: 1px solid #444; 
    padding: 10px; 
    border-radius: 6px; 
    font-size: 14px;
}
QLineEdit:focus { border: 1px solid #00e5ff; }

/* BUTTONS */
QPushButton {
    background-color: #2c2c2c;
    color: white;
    border: 1px solid #444;
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover { background-color: #383838; border: 1px solid #666; }

QPushButton#PrimaryBtn { background-color: #007acc; border: none; }
QPushButton#PrimaryBtn:hover { background-color: #0098ff; }

QPushButton#DangerBtn { background-color: #d32f2f; border: none; }
QPushButton#DangerBtn:hover { background-color: #f44336; }

QPushButton#SuccessBtn { background-color: #2e7d32; border: none; }
QPushButton#SuccessBtn:hover { background-color: #43a047; }
"""

class ClassificationWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, image_array):
        super().__init__()
        self.image = image_array
        
    def run(self):
        """This runs in the background"""
        result = tumor_classifier.tumor_predict(self.image)
        # -------------------
        
        # Send the result dictionary back
        self.result_ready.emit(result)

class SegmentationWorker(QThread):
    result_ready = pyqtSignal(dict) # Will emit the dictionary from segmentation.py

    def __init__(self, image_array):
        super().__init__()
        self.image = image_array
        
    def run(self):
        """This runs in the background to prevent lag"""
        # Call the heavy segmentation function
        result = segmentation.detect_tumor(self.image)
        # Send the result back to the main thread when finished
        self.result_ready.emit(result)

class CombinedAIWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, segmented_pil_image):
        super().__init__()
        self.image_data = segmented_pil_image

        self.api_key = "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"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "amazon/nova-2-lite-v1:free" # A fast and capable model

    def run(self):
        try:
            # 1. Prepare Image
            buffer = io.BytesIO()
            self.image_data.save(buffer, format="PNG")
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{base64_string}"
            
            # 2. The Master Prompt
            prompt = """
            Analyze the provided brain MRI scan which has a tumor highlighted.
            Your task is to act as an expert radiologist and medical physicist.
            Provide a detailed analysis by returning ONLY a single, raw JSON object. Do not add any extra text.
            The JSON must have these exact keys:
            
            {
              "classification": "String (e.g., Glioblastoma, Meningioma)",
              "grade": "String (e.g., Grade IV, Grade I/II)",
              "pathology_analysis": "String (A 2-3 sentence description of the tumor's visual characteristics like shape, enhancement, and location).",
              "recommendation": "String (A brief, actionable clinical recommendation, e.g., 'Immediate surgical resection and ablation recommended.').",
              "laser_parameters": {
                "power_W": "Float (e.g., 15.0)",
                "energy_J": "Float (e.g., 1200.0)",
                "duration_s": "Float (e.g., 80.0)",
                "target_temp_C": "Float (e.g., 65.0)"
              }
            }
            """
            
            # 3. Build API Payload
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            api_messages = [{"role": "user", "content": user_content}]
            payload = {"model": self.model, "messages": api_messages, "max_tokens": 1024}
            
            # 4. Make API Call
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # 5. Parse JSON
            raw_response = response.json()['choices'][0]['message']['content']
            # Clean up potential markdown formatting from the AI
            clean_json_str = re.sub(r"```json|```", "", raw_response).strip()
            
            result = json.loads(clean_json_str)
            self.result_ready.emit(result)

        except Exception as e:
            self.result_ready.emit({"error": f"AI analysis failed: {str(e)}"})

class TumorGrowthWorker(QThread):
    frames_ready = pyqtSignal(list, list)
    def __init__(self, mask, brain_mask, params, end_time): 
        super().__init__()
        self.mask = mask
        self.brain_mask = brain_mask
        self.params = params
        self.end_time = end_time
    def run(self):
        try:
            frames, metrics = tumor_growth_model.simulate_tumor_growth_fast(self.mask, self.brain_mask, self.params, self.end_time)
            self.frames_ready.emit(frames, metrics)
        except Exception as e:
            print(f"❌ Tumor Growth Simulation Failed: {e}")
            self.frames_ready.emit([], [])

class WebStyleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OncoSim")
        self.resize(1300, 900)
        self.setStyleSheet(WEB_STYLESHEET)

        # --- LOGIC VARIABLES ---
        self.image_path = None
        self.raw_image = None
        self.brain_mask = None
        self.temperature_map = None
        self.current_temp = 37.0
        self.is_running = False
        self.tumor_size = 0.0
        self.tumor_centroid = (0, 0)
        self.tumor_type = "Unknown"
        self.ai_engine = ai_core.SurgicalAI()
        self.chat_history = []
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

        self.init_ui()

    def init_ui(self):
        # 1. Main Scroll Area (The "Browser Window")
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # 2. The Long Content Widget (The "Webpage")
        self.content_widget = QWidget()
        self.content_widget.setObjectName("CentralContent")
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setSpacing(30)
        self.main_layout.setContentsMargins(40, 40, 40, 40)

        # --- BUILD SECTIONS ---
        self.create_header()
        self.create_section_upload() 
        self.create_section_segmentation()
        self.create_section_detailed_diagnosis()
        self.create_section_visuals()
        self.create_section_planning()
        self.create_section_execution()
        self.create_section_growth_visuals()
        self.create_section_growth_controls()
        self.create_section_chatbot()
        self.create_section_report()


        # Finish Setup
        self.main_layout.addStretch() # Push everything up
        self.scroll_area.setWidget(self.content_widget)
        self.setCentralWidget(self.scroll_area)

    # ==========================================
    #  UI SECTIONS (CARDS)
    # ==========================================

    def create_header(self):
        header_layout = QVBoxLayout()
        header_layout.setSpacing(15)
        
        logo_label = QLabel()
        pixmap = QPixmap("logo_trans.png")
        logo_label.setPixmap(pixmap.scaledToWidth(1200, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Intelligent Tumor Ablation & Guidance System")
        title.setObjectName("HeaderTitle")
        title.setStyleSheet("font-size: 50px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignCenter)
        
        sub = QLabel("An integrated platform for the entire ablation workflow, from diagnosis to real-time guidance and reporting.")
        sub.setObjectName("HeaderSub")
        sub.setStyleSheet("font-size: 30px; color: #888888;")
        sub.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(logo_label)
        header_layout.addWidget(title)
        header_layout.addWidget(sub)
        
        header_layout.addSpacing(30) 
        
        self.main_layout.addLayout(header_layout)

    def create_section_upload(self):
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QHBoxLayout(card)

        # Col 1: Upload
        col1 = QVBoxLayout()
        lbl_title = QLabel("Patient Data Ingestion")
        lbl_title.setObjectName("CardTitle")
        
        self.btn_upload = QPushButton("📂 Upload Scan (DCM/NII/JPG)")
        self.btn_upload.setObjectName("PrimaryBtn")
        self.btn_upload.clicked.connect(self.upload_image)
        
        self.lbl_file_info = QLabel("No file loaded.")
        self.lbl_file_info.setStyleSheet("color: #888; margin-top: 5px;")
        
        col1.addWidget(lbl_title)
        col1.addWidget(self.btn_upload)
        col1.addWidget(self.lbl_file_info)
        col1.addStretch()

        layout.addLayout(col1)
        # Diagnosis part removed from here
        
        self.main_layout.addWidget(card)
    
    def create_section_segmentation(self):
        """Card 2: Segmentation Workflow (Visuals Only)"""
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # --- ROW 1: HEADER & BUTTON ---
        head_layout = QHBoxLayout()
        lbl_title = QLabel("Tumor Segmentation")
        lbl_title.setObjectName("CardTitle")
        
        self.btn_run_seg = QPushButton("🔍 Start Segmentation")
        self.btn_run_seg.setStyleSheet("background-color: #00bcd4; color: black; font-weight: bold;")
        self.btn_run_seg.setFixedWidth(200)
        self.btn_run_seg.clicked.connect(self.run_segmentation_process)
        self.btn_run_seg.setEnabled(False)
        
        head_layout.addWidget(lbl_title)
        head_layout.addStretch()
        head_layout.addWidget(self.btn_run_seg)

        # --- ROW 2: SPLIT IMAGES ---
        split_layout = QHBoxLayout()
        split_layout.setSpacing(20)
        
        # Left: Raw
        left_box = QVBoxLayout()
        left_box.setSpacing(5); left_box.setContentsMargins(0, 0, 0, 0); left_box.setAlignment(Qt.AlignTop)
        
        lbl_sub1 = QLabel("Original Input")
        lbl_sub1.setStyleSheet("color: #aaa; font-size: 12px; font-weight: bold;")
        lbl_sub1.setAlignment(Qt.AlignCenter)
        lbl_sub1.setFixedHeight(20)
        
        self.lbl_seg_raw = QLabel("Load Image...")
        self.lbl_seg_raw.setAlignment(Qt.AlignCenter)
        self.lbl_seg_raw.setFixedHeight(350)
        self.lbl_seg_raw.setStyleSheet("background-color: black; border-radius: 8px; border: 1px solid #444;")
        self.lbl_seg_raw.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        left_box.addWidget(lbl_sub1); left_box.addWidget(self.lbl_seg_raw)

        # Right: Result
        right_box = QVBoxLayout()
        right_box.setSpacing(5); right_box.setContentsMargins(0, 0, 0, 0); right_box.setAlignment(Qt.AlignTop)
        
        lbl_sub2 = QLabel("PDE Segmentation Mask")
        lbl_sub2.setStyleSheet("color: #00bcd4; font-size: 12px; font-weight: bold;")
        lbl_sub2.setAlignment(Qt.AlignCenter)
        lbl_sub2.setFixedHeight(20)
        
        self.lbl_seg_result = QLabel("Waiting for Segmentation...")
        self.lbl_seg_result.setAlignment(Qt.AlignCenter)
        self.lbl_seg_result.setFixedHeight(350)
        self.lbl_seg_result.setStyleSheet("background-color: black; border-radius: 8px; border: 1px solid #00bcd4;")
        self.lbl_seg_result.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        right_box.addWidget(lbl_sub2); right_box.addWidget(self.lbl_seg_result)
        
        split_layout.addLayout(left_box, 1)
        split_layout.addLayout(right_box, 1)

        layout.addLayout(head_layout)
        layout.addLayout(split_layout)
        self.main_layout.addWidget(card)

    def create_section_detailed_diagnosis(self):

        self.diagnosis_card = QFrame()
        self.diagnosis_card.setProperty("class", "Card")
        self.diagnosis_card.setVisible(True) 
        
        layout = QVBoxLayout(self.diagnosis_card)
        layout.setSpacing(20)
        
        title = QLabel("Clinical Diagnosis & Deep Report")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00e5ff;")
        
        # --- CLINICAL SUMMARY GRID ---
        summary_layout = QGridLayout()
        summary_layout.setSpacing(20)
        
        lbl_style = "color: #888; font-size: 12px; font-weight: bold; text-transform: uppercase;"
        val_style = "color: #dcc9b6; font-size: 20px; font-weight: bold;"
        
        # Row 0
        summary_layout.addWidget(QLabel("Classification", styleSheet=lbl_style), 0, 0)
        self.lbl_type = QLabel("-"); self.lbl_type.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_type, 1, 0)

        summary_layout.addWidget(QLabel("Estimated Depth", styleSheet=lbl_style), 0, 1)
        self.lbl_depth = QLabel("-"); self.lbl_depth.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_depth, 1, 1)

        summary_layout.addWidget(QLabel("Estimated Grade", styleSheet=lbl_style), 0, 2)
        self.lbl_grade = QLabel("-"); self.lbl_grade.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_grade, 1, 2)

        summary_layout.addWidget(QLabel("Location", styleSheet=lbl_style), 0, 3)
        self.lbl_location_ai = QLabel("-"); self.lbl_location_ai.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_location_ai, 1, 3)

        summary_layout.addWidget(QLabel("Shape", styleSheet=lbl_style), 0, 4)
        self.lbl_shape = QLabel("-"); self.lbl_shape.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_shape, 1, 4)

        summary_layout.addWidget(QLabel("Centroid", styleSheet=lbl_style), 2, 0)
        self.lbl_loc = QLabel("-"); self.lbl_loc.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_loc, 3, 0)

        summary_layout.addWidget(QLabel("Area", styleSheet=lbl_style), 2, 1)
        self.lbl_area = QLabel("-"); self.lbl_area.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_area, 3, 1)

        summary_layout.addWidget(QLabel("Max Diameter", styleSheet=lbl_style), 2, 2)
        self.lbl_size = QLabel("-"); self.lbl_size.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_size, 3, 2)

        summary_layout.addWidget(QLabel("Bounding Box", styleSheet=lbl_style), 2, 3)
        self.lbl_dims = QLabel("-"); self.lbl_dims.setStyleSheet(val_style)
        summary_layout.addWidget(self.lbl_dims, 3, 3)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #333;")
        
        # --- AI STATS ---
        grid = QGridLayout()
        grid.setSpacing(15)

        row_lbl_style = "color: #aaa; font-size: 14px;"
        conf_style = "color: #00e5ff; font-size: 18px; font-weight: bold;"
        text_style = "color: #fff; font-size: 15px; line-height: 1.4;"
        action_style = "color: #ff6d00; font-weight: bold; border-left: 4px solid #ff6d00; padding-left: 10px; font-size: 14px;"

        grid.addWidget(QLabel("Model Confidence:", styleSheet=row_lbl_style), 0, 0)
        self.rpt_conf = QLabel("-%"); self.rpt_conf.setStyleSheet(conf_style)
        grid.addWidget(self.rpt_conf, 0, 1)

        grid.addWidget(QLabel("Pathology Analysis:", styleSheet=row_lbl_style), 1, 0)
        self.rpt_desc = QLabel("Waiting..."); self.rpt_desc.setStyleSheet(text_style); self.rpt_desc.setWordWrap(True)
        grid.addWidget(self.rpt_desc, 1, 1)

        grid.addWidget(QLabel("Recommendation:", styleSheet=row_lbl_style), 2, 0)
        self.rpt_action = QLabel("-"); self.rpt_action.setStyleSheet(action_style); self.rpt_action.setWordWrap(True)
        grid.addWidget(self.rpt_action, 2, 1)
        grid.setColumnStretch(1, 1)

        layout.addWidget(title)
        layout.addLayout(summary_layout)
        layout.addWidget(line)
        layout.addLayout(grid)
        self.main_layout.addWidget(self.diagnosis_card)

    def create_section_visuals(self):
        card = QFrame()
        card.setProperty("class", "Card")
        
        # --- FIX: Use a layout that aligns to the top ---
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10) # Reduce space between title and image
        layout.setAlignment(Qt.AlignTop) # This forces everything up
        # -----------------------------------------------
        
        # Main Title
        lbl_title = QLabel("Real-Time Thermal Visualization")
        lbl_title.setObjectName("CardTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFixedHeight(30) # Keep title short
        
        # --- SINGLE IMAGE VIEW ---
        self.lbl_live_image = QLabel("Run Segmentation to Enable...")
        self.lbl_live_image.setAlignment(Qt.AlignCenter)
        self.lbl_live_image.setMinimumHeight(500)
        self.lbl_live_image.setStyleSheet("background-color: #111; border-radius: 8px; border: 1px dashed #333; color: #444;")
        self.lbl_live_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        layout.addWidget(lbl_title)
        layout.addWidget(self.lbl_live_image)
        
        self.main_layout.addWidget(card)

    def create_section_planning(self):
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        lbl_title = QLabel("Treatment Planning & Physics")
        lbl_title.setObjectName("CardTitle")
        
        h_layout = QHBoxLayout()
        
        # =========================================
        # COL 1: INPUTS & MODES
        # =========================================
        left_container = QVBoxLayout()
        self.form_layout = QFormLayout()
        
        # --- NEW INPUTS ---
        self.in_power = QLineEdit("0.0")
        self.in_energy = QLineEdit("0.0")
        self.in_target = QLineEdit("0.0")    # Now represents DELTA
        self.in_total_dur = QLineEdit("0.0")
        self.in_pulse_dur = QLineEdit("0.5")
        self.lbl_pulse_dur = QLabel("Pulse Width (ms):") 
        
        # Styles
        label_style = "color: #ff6d00; font-weight: bold; font-size: 13px;"
        
        lbl_p = QLabel("Power (W):"); lbl_p.setStyleSheet(label_style)
        lbl_e = QLabel("Total Energy (J):"); lbl_e.setStyleSheet(label_style)
        lbl_t = QLabel("Target Rise (ΔT):"); lbl_t.setStyleSheet(label_style) # Renamed
        lbl_tot = QLabel("Total Duration (s):"); lbl_tot.setStyleSheet(label_style)
        self.lbl_pulse_dur.setStyleSheet("color: #ffcc00; font-weight: bold; font-size: 13px;")
        
        self.form_layout.addRow(lbl_p, self.in_power)
        self.form_layout.addRow(lbl_e, self.in_energy)
        self.form_layout.addRow(lbl_t, self.in_target)
        self.form_layout.addRow(lbl_tot, self.in_total_dur)
        self.form_layout.addRow(self.lbl_pulse_dur, self.in_pulse_dur)
        
        self.lbl_pulse_dur.setVisible(False)
        self.in_pulse_dur.setVisible(False)
        
        # 2. Modes
        mode_grp = QGroupBox("Waveform Modulation")
        mode_grp.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; color: #aaa; font-size: 11px; } QGroupBox::title { top: -8px; left: 10px; }")
        mode_layout = QHBoxLayout(mode_grp)
        
        self.chk_continuous = QCheckBox("Continuous")
        self.chk_continuous.setChecked(True)
        self.chk_continuous.setStyleSheet("color: #00e5ff; font-weight: bold;")
        self.chk_continuous.toggled.connect(self.toggle_laser_mode)
        
        self.chk_pulsed = QCheckBox("Pulsed")
        self.chk_pulsed.setStyleSheet("color: #ffcc00; font-weight: bold;")
        self.chk_pulsed.toggled.connect(self.toggle_laser_mode)
        
        mode_layout.addWidget(self.chk_continuous)
        mode_layout.addWidget(self.chk_pulsed)
        
        left_container.addLayout(self.form_layout)
        left_container.addWidget(mode_grp)

        # =========================================
        # COL 2: RADIATION MATERIAL
        # =========================================
        middle_container = QVBoxLayout()
        rad_grp = QGroupBox("Laser Source / Wavelength")
        rad_grp.setStyleSheet("""
            QGroupBox { border: 1px solid #444; border-radius: 4px; color: #aaa; font-size: 16px; padding-top: 20px; } 
            QGroupBox::title { top: -4px; left: 10px; }
            QRadioButton { color: #ffffff; font-size: 13px; spacing: 8px; }
            QRadioButton::indicator { width: 14px; height: 14px; }
            QLabel { color: #e0e0e0; font-weight: bold; }
        """)
        rad_layout = QVBoxLayout(rad_grp)
        rad_layout.setSpacing(10)
        
        self.rad_ndyag = QRadioButton("Nd:YAG"); self.rad_ndyag.toggled.connect(self.update_wavelength_logic)
        self.rad_diode = QRadioButton("Diode"); self.rad_diode.toggled.connect(self.update_wavelength_logic)
        self.rad_co2 = QRadioButton("CO2"); self.rad_co2.toggled.connect(self.update_wavelength_logic)
        self.rad_custom = QRadioButton("Custom"); self.rad_custom.toggled.connect(self.update_wavelength_logic)
        
        wave_input_layout = QHBoxLayout()
        self.in_wavelength = QLineEdit("1064"); self.in_wavelength.setPlaceholderText("Value")
        self.in_wavelength.setStyleSheet("color: #00e5ff; font-weight: bold;")
        self.lbl_unit = QLabel("nm"); self.lbl_unit.setStyleSheet("color: #888;")
        wave_input_layout.addWidget(QLabel("Wave:")); wave_input_layout.addWidget(self.in_wavelength); wave_input_layout.addWidget(self.lbl_unit)
        
        rad_layout.addWidget(self.rad_ndyag); rad_layout.addWidget(self.rad_diode)
        rad_layout.addWidget(self.rad_co2); rad_layout.addWidget(self.rad_custom)
        rad_layout.addSpacing(100); rad_layout.addLayout(wave_input_layout); rad_layout.addStretch()
        self.rad_ndyag.setChecked(True); self.in_wavelength.setEnabled(False)
        middle_container.addWidget(rad_grp)
        
        # =========================================
        # COL 3: BUTTONS
        # =========================================
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.btn_auto_cal = QPushButton("⚡ Auto-Calibrate Parameters")
        self.btn_auto_cal.clicked.connect(self.action_suggest_params)
        self.btn_ai_params = QPushButton("🤖 AI Parameters")
        self.btn_ai_params.clicked.connect(self.ai_suggest_params)
        self.btn_strategy = QPushButton("📋 Get Surgical Strategy")
        self.btn_strategy.clicked.connect(self.action_suggest_strategy)
        btn_layout.addWidget(self.btn_auto_cal)
        btn_layout.addSpacing(15)
        btn_layout.addWidget(self.btn_ai_params)
        btn_layout.addSpacing(15)
        btn_layout.addWidget(self.btn_strategy)
        btn_layout.addStretch()

        h_layout.addLayout(left_container, 2)
        h_layout.addSpacing(20)
        h_layout.addLayout(middle_container, 2)
        h_layout.addSpacing(20)
        h_layout.addLayout(btn_layout, 1)
        
        layout.addWidget(lbl_title)
        layout.addLayout(h_layout)
        self.main_layout.addWidget(card)

    def create_section_execution(self):
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        
        lbl_title = QLabel("Active Treatment & AI Monitoring")
        lbl_title.setObjectName("CardTitle")
        
        # --- ROW 1: CONTROLS ---
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        
        # 1. Start Button (Green, Wide)
        self.btn_start = QPushButton("INITIALIZE ABLATION")
        self.btn_start.setObjectName("SuccessBtn")
        self.btn_start.setFixedHeight(70)
        self.btn_start.clicked.connect(self.toggle_ablation)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; 
                color: white; 
                font-size: 20px; 
                font-weight: bold; 
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #43a047; }
            QPushButton:disabled { background-color: #1b5e20; color: #888; }
        """)
        
        # 2. Reset Button (Grey, Narrow) <--- NEW
        self.btn_reset = QPushButton("Reset System")
        self.btn_reset.setToolTip("Reset System")
        self.btn_reset.setFixedHeight(70)
        self.btn_reset.clicked.connect(self.reset_simulation)
        self.btn_reset.setEnabled(False) # Enabled only when image loaded
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #c1121f; 
                color: white; 
                font-size: 20px; 
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #bc4749; }
            QPushButton:disabled { background-color: #780000; color: #888; }
        """)
        
        self.lbl_temp = QDoubleSpinBox()
        self.lbl_temp.setRange(0.0, 200.0)
        self.lbl_temp.setValue(0.0)
        self.lbl_temp.setSuffix("°C")
        self.lbl_temp.setSingleStep(0.1)
        self.lbl_temp.setAlignment(Qt.AlignCenter)
        self.lbl_temp.setFixedHeight(70)
        self.lbl_temp.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        self.lbl_temp.valueChanged.connect(self.update_baseline_manually)
        
        self.lbl_temp.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #00e5ff;
                font-size: 40px;
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 6px;
                padding-left: 10px;
            }
            
            /* The Buttons Area (Right Side) */
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 50px;
                background-color: #333;
                border-left: 2px solid white; /* Vertical White Line */
            }
            
            /* Specific Border for Up Button to create horizontal split */
            QDoubleSpinBox::up-button {
                border-bottom: 1px solid white; /* Horizontal White Line */
                border-top-right-radius: 6px;
            }
            
            QDoubleSpinBox::down-button {
                border-bottom-right-radius: 6px;
            }

            /* Hover Effects */
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #555;
            }
            
            /* The Arrow Icons */
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
                width: 8px;
                height: 10px;
                color: white; /* Ensures visibility */
            }
        """)
        
        # 4. Status Box
        self.lbl_ai_status = QLabel("STANDBY")
        self.lbl_ai_status.setAlignment(Qt.AlignCenter)
        self.lbl_ai_status.setFixedHeight(70)
        self.lbl_ai_status.setStyleSheet("""
            background-color: #333; 
            color: #888; 
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 6px;
            border: 1px solid #444;
        """)
        
        top_row.addWidget(self.btn_start, 4)
        top_row.addWidget(self.btn_reset, 1)
        top_row.addWidget(self.lbl_temp, 1)
        top_row.addWidget(self.lbl_ai_status, 1)
        
        # --- ROW 2: LOG TEXT ---
        self.lbl_ai_log = QLabel("System Ready.")
        self.lbl_ai_log.setStyleSheet("color: #aaa; font-weight: bold; font-size: 15px; margin-top: 10px;")
        self.lbl_ai_log.setAlignment(Qt.AlignCenter)
        self.lbl_ai_log.setWordWrap(True)

        layout.addWidget(lbl_title)
        layout.addLayout(top_row)
        layout.addWidget(self.lbl_ai_log)
        
        self.main_layout.addWidget(card)

    def create_section_growth_visuals(self):
        """Card: Tumor Growth Visualization with corrected vertical alignment."""
        card = QFrame()
        card.setProperty("class", "Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)  # This is important

        # --- HEADER ROW ---
        header = QHBoxLayout()
        
        title = QLabel("Predicted Tumor Growth Over Time")
        title.setObjectName("CardTitle")
        title.setFixedHeight(30)

        # --- Use a simple QVBoxLayout for the data labels ---
        # Data Display (Right Side)
        data_container = QVBoxLayout()
        data_container.setAlignment(Qt.AlignTop)
        data_container.setSpacing(5)

        self.lbl_growth_time = QLabel("Time: 0.0 Hours")
        self.lbl_growth_time.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 13px;")

        self.lbl_growth_delta = QLabel("Growth: +0.00 mm")
        self.lbl_growth_delta.setStyleSheet("color: #ffcc00; font-weight: bold; font-size: 13px;")

        data_container.addWidget(self.lbl_growth_time)
        data_container.addWidget(self.lbl_growth_delta)
        # ---------------------------------------------------

        header.addWidget(title)
        header.addStretch()
        header.addLayout(data_container)

        # --- IMAGE BOX ---
        self.lbl_growth_sim = QLabel("Run Segmentation to Enable...")
        self.lbl_growth_sim.setAlignment(Qt.AlignCenter)
        self.lbl_growth_sim.setMinimumHeight(500)
        self.lbl_growth_sim.setStyleSheet(
            "background-color: black; border-radius: 8px; border: 1px solid #444; color: #666;"
        )
        self.lbl_growth_sim.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_growth_sim.setScaledContents(False)

        # --- Set stretch factor to 0 for the header so image box takes remaining space ---
        layout.addLayout(header, 0)
        layout.addWidget(self.lbl_growth_sim, 1)
        # ---------------------------------------------------------------------------------

        self.main_layout.addWidget(card)

    def create_section_growth_controls(self):
        """Card: Tumor Growth Simulation Parameters with visible arrows."""

        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)

        title = QLabel("Tumor Growth Simulation")
        title.setObjectName("CardTitle")

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        label_style = "color: #aaa; font-size: 11px; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;"

        # --- Time Scale Toggle (needed for button height reference) ---
        time_scale_container = QVBoxLayout()
        time_label = QLabel("TIME SCALE"); time_label.setStyleSheet(label_style)
        toggle_layout = QHBoxLayout()
        self.btn_hours = QPushButton("Hours"); self.btn_days = QPushButton("Days")
        self.active_toggle_style = "background-color: #007acc; color: white; border: none;"
        self.inactive_toggle_style = "background-color: #333; color: #888; border: 1px solid #444;"
        self.btn_hours.setStyleSheet(self.active_toggle_style); self.btn_days.setStyleSheet(self.inactive_toggle_style)
        self.btn_hours.setCheckable(True); self.btn_days.setCheckable(True); self.btn_hours.setChecked(True)
        self.btn_hours.clicked.connect(lambda: self.update_time_scale_toggle(self.btn_hours))
        self.btn_days.clicked.connect(lambda: self.update_time_scale_toggle(self.btn_days))
        toggle_layout.addWidget(self.btn_hours); toggle_layout.addWidget(self.btn_days)
        time_scale_container.addWidget(time_label); time_scale_container.addLayout(toggle_layout); time_scale_container.addStretch()
        controls_layout.addLayout(time_scale_container)

        # --- Record the toggle button height ---
        # The spinbox buttons will match this height
        # We'll use btn_hours (since both should be the same)
        # This only works after .show() or after layout, so we'll assume a fallback
        TOGGLE_BTN_HEIGHT = self.btn_hours.sizeHint().height() if self.btn_hours.sizeHint().height() > 0 else 32

        # --- FIX: ADDED ARROW STYLES + Match stepper height to toggle button ---
        SPINBOX_STYLE = f"""
            QDoubleSpinBox {{
                background-color: #2c2c2c;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 5px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 25px;
                height: {TOGGLE_BTN_HEIGHT}px;
                min-height: {TOGGLE_BTN_HEIGHT}px;
                max-height: {TOGGLE_BTN_HEIGHT}px;
                background-color: #383838;
            }}
            QDoubleSpinBox::up-button {{ border-top-right-radius: 6px; }}
            QDoubleSpinBox::down-button {{ border-bottom-right-radius: 6px; }}
            
            /* CSS Trick to draw triangles */
            QDoubleSpinBox::up-arrow {{
                width: 0px; height: 0px;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-bottom: 6px solid white;
            }}
            QDoubleSpinBox::down-arrow {{
                width: 0px; height: 0px;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid white;
            }}
        """
        # -------------------------------

        def create_parameter_box(label_text, min_val, max_val, default_val, decimals, step):
            box_layout = QVBoxLayout()
            label = QLabel(label_text); label.setStyleSheet(label_style)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val); spinbox.setValue(default_val)
            spinbox.setDecimals(decimals); spinbox.setSingleStep(step)
            spinbox.setStyleSheet(SPINBOX_STYLE)
            spinbox.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
            box_layout.addWidget(label); box_layout.addWidget(spinbox); box_layout.addStretch()
            return box_layout, spinbox

        # --- Parameters (SpinBoxes) ---
        diff_layout, self.spin_D = create_parameter_box("DIFFUSION (D)", 0.1, 2.0, 0.8, 2, 0.1)
        pro_layout, self.spin_rho = create_parameter_box("PROLIFERATION (ρ)", 0.1, 1.0, 0.5, 2, 0.1)
        nec_layout, self.spin_beta = create_parameter_box("NECROSIS (β)", 0.01, 0.5, 0.1, 2, 0.01)
        dur_layout, self.spin_duration = create_parameter_box("SIM DURATION", 10, 500, 100, 0, 10)

        controls_layout.addLayout(diff_layout)
        controls_layout.addLayout(pro_layout)
        controls_layout.addLayout(nec_layout)
        controls_layout.addLayout(dur_layout)

        self.btn_run_growth_sim = QPushButton("▶️ Start Growth Simulation")
        self.btn_run_growth_sim.setStyleSheet("background-color: #9c27b0; color: white; margin-top: 15px;")
        self.btn_run_growth_sim.setEnabled(True)
        self.btn_run_growth_sim.clicked.connect(self.run_growth_simulation)

        layout.addWidget(title)
        layout.addLayout(controls_layout)
        layout.addWidget(self.btn_run_growth_sim)
        self.main_layout.addWidget(card)
    
    def create_section_chatbot(self):
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        
        lbl_title = QLabel("AI Surgical Assistant")
        lbl_title.setObjectName("CardTitle")
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(200)
        self.chat_display.setStyleSheet("""
            QTextEdit { 
                background-color: #222; 
                color: #ddd; 
                border: 1px solid #444; 
                border-radius: 6px; 
                padding: 10px;
                font-size: 14px;
            }
        """)
        self.chat_display.setText("<i>System: Complete Segmentation & Analysis to enable chat...</i>")

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a follow-up question...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        self.chat_input.setEnabled(False)
        self.btn_send_chat = QPushButton("➤")
        self.btn_send_chat.setFixedWidth(50)
        self.btn_send_chat.clicked.connect(self.send_chat_message)
        self.btn_send_chat.setEnabled(False)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.btn_send_chat)

        # This button is now hidden and will be removed in a future step if not needed
        self.btn_init_chat = QPushButton("💬 Start Initial Analysis")
        self.btn_init_chat.setVisible(False) 

        # Make the New Chat Session button full-width and larger
        self.btn_restart_chat = QPushButton("↺ New Chat Session")
        self.btn_restart_chat.setStyleSheet("""
            background-color: #555; 
            color: white;
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 6px;
            padding: 16px 0px;
        """)
        self.btn_restart_chat.clicked.connect(self.restart_chat_session) # <-- CORRECTED
        self.btn_restart_chat.setEnabled(False) # Start disabled
        self.btn_restart_chat.setMinimumHeight(48)
        self.btn_restart_chat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Instead of a QHBoxLayout, use a zero-spacing QVBoxLayout to take the full width
        layout.addWidget(lbl_title)
        layout.addWidget(self.chat_display)
        layout.addLayout(input_layout)
        layout.addSpacing(10)
        layout.addWidget(self.btn_restart_chat)  # Full-width button
        self.main_layout.addWidget(card)

    def create_section_report(self):
        """Card 7: Final Report Generation"""
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        
        lbl_title = QLabel("Post-Operative Reporting")
        lbl_title.setObjectName("CardTitle")
        
        # --- PATIENT NAME ---
        name_layout = QHBoxLayout()
        lbl_name = QLabel("Patient Name:")
        lbl_name.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        self.in_patient_name = QLineEdit()
        self.in_patient_name.setPlaceholderText("Enter Patient ID or Name...")
        name_layout.addWidget(lbl_name)
        name_layout.addWidget(self.in_patient_name)
        
        # --- CHECKBOXES FOR DATA ---
        from PyQt5.QtWidgets import QGridLayout  # If not already imported above
        chk_layout = QGridLayout()  # Use a grid for better spacing
        
        self.chk_images = QCheckBox("Visual Overview"); self.chk_images.setChecked(True)
        self.chk_diagnosis = QCheckBox("Diagnosis"); self.chk_diagnosis.setChecked(True)
        self.chk_physics = QCheckBox("Ablation Physics"); self.chk_physics.setChecked(True)
        
        # NEW
        self.chk_growth_sim = QCheckBox("Growth Simulation")
        self.chk_growth_sim.setChecked(True)
        
        self.chk_chat = QCheckBox("AI Chat Log"); self.chk_chat.setChecked(True)
        
        # Style checkboxes
        for chk in [self.chk_images, self.chk_diagnosis, self.chk_physics, self.chk_growth_sim, self.chk_chat]:
            chk.setStyleSheet("color: #ccc;")
        
        chk_layout.addWidget(self.chk_images, 0, 0)
        chk_layout.addWidget(self.chk_diagnosis, 0, 1)
        chk_layout.addWidget(self.chk_physics, 0, 2)
        chk_layout.addWidget(self.chk_growth_sim, 0, 3)
        chk_layout.addWidget(self.chk_chat, 0, 4)
        
        # Button and final layout
        self.btn_export = QPushButton("📄 Export Clinical Report")
        self.btn_export.setFixedHeight(50)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #ff9800; color: black; 
                font-size: 16px; font-weight: bold; border: none;
            }
            QPushButton:hover { background-color: #ffb74d; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_export.clicked.connect(self.action_export_report)
        self.btn_export.setEnabled(False)
        
        layout.addWidget(lbl_title)
        layout.addLayout(name_layout)
        layout.addSpacing(10)
        layout.addLayout(chk_layout)
        layout.addSpacing(10)
        layout.addWidget(self.btn_export)
        
        self.main_layout.addWidget(card)

    # ==========================================
    #  LOGIC FUNCTIONS (Copied & Adapted)
    # ==========================================

    def reset_simulation(self):
        """Resets the simulation to initial state."""
        if self.raw_image is None: return

        # 1. Stop if running
        if self.is_running:
            self.toggle_ablation()  # This will stop timer and reset the button style

        # --- FIX: Explicitly reset the Start button here as well ---
        self.btn_start.setText("INITIALIZE ABLATION")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white; font-size: 20px; font-weight: bold; border-radius: 6px;
            }
            QPushButton:hover { background-color: #43a047; }
        """)
        # -----------------------------------------------------------

        # 2. Unlock SpinBox and Reset Style
        self.lbl_temp.setReadOnly(False)
        self.lbl_temp.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #00e5ff;
                font-size: 40px;
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 6px;
                padding-left: 10px;
            }
            
            /* The Buttons Area (Right Side) */
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 50px;
                background-color: #333;
                border-left: 2px solid white; /* Vertical White Line */
            }
            
            /* Specific Border for Up Button to create horizontal split */
            QDoubleSpinBox::up-button {
                border-bottom: 1px solid white; /* Horizontal White Line */
                border-top-right-radius: 6px;
            }
            
            QDoubleSpinBox::down-button {
                border-bottom-right-radius: 6px;
            }

            /* Hover Effects */
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #555;
            }
            
            /* The Arrow Icons */
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
                width: 8px;
                height: 10px;
                color: white; /* Ensures visibility */
            }
        """)

        # 3. Reset Physics and UI
        self.current_temp = self.lbl_temp.value()
        self.start_temp = self.current_temp
        self.ai_engine.reset()

        self.lbl_ai_status.setText("STANDBY")
        self.lbl_ai_status.setStyleSheet("background-color: #333; color: #888; font-size: 20px; font-weight: bold; border-radius: 6px; border: 1px solid #444;")
        self.lbl_ai_log.setText("System Reset. Adjust temp and restart.")

        # 4. Reset Heatmap Image
        self.display_image(self.segmented_overlay, self.lbl_live_image)
        self.btn_reset.setEnabled(True)

    
    def upload_image(self):
        filters = "Medical Files (*.dcm *.nii *.nii.gz *.jpg *.png);;All Files (*)"
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Scan', '', filters)
        if fname:
            processed_img, info_text = self.load_medical_file(fname)
            if processed_img is not None:
                self.raw_image = processed_img
                self.segmented_image = None
                
                self.lbl_file_info.setText(f"Loaded: {os.path.basename(fname)}")
                
                # Show in Segmentation Left Box
                self.display_image(self.raw_image, self.lbl_seg_raw)
                
                # --- RESET UI STATE FOR NEW UPLOAD ---
                
                # Segmentation views
                self.lbl_seg_result.setText("Ready to Segment")
                self.lbl_seg_result.clear()
                
                self.lbl_live_image.setText("Waiting for Ablation Start...")
                self.lbl_live_image.setStyleSheet("background-color: #111; border-radius: 8px; border: 1px dashed #333; color: #444;")
                
                # Diagnosis fields
                self.diagnosis_card.setVisible(True)
                self.lbl_type.setText("-")
                self.lbl_grade.setText("-")
                self.lbl_size.setText("-")
                self.lbl_loc.setText("-")
                self.lbl_area.setText("-")
                self.lbl_dims.setText("-")
                self.lbl_shape.setText("-")
                self.lbl_depth.setText("-")
                self.lbl_location_ai.setText("-")
                self.rpt_conf.setText("-%")
                self.rpt_desc.setText("Waiting for analysis...")
                self.rpt_action.setText("-")
                
                # Chat fields
                self.chat_display.setText("<i>System: Complete analysis to enable chat...</i>")
                
                # --- SET BUTTON STATES (FIXED) ---
                # Only "Start Segmentation" should be active.
                self.btn_run_seg.setEnabled(True)
                
                # Disable all other buttons.
                self.btn_start.setEnabled(False)
                self.btn_reset.setEnabled(False)
                self.btn_init_chat.setEnabled(False)
                self.btn_send_chat.setEnabled(False)
                self.chat_input.setEnabled(False)
                if hasattr(self, 'btn_export'):
                    self.btn_export.setEnabled(False)

    def run_segmentation_process(self):
        """
        Starts the segmentation AI in a background thread to prevent GUI lag.
        """
        if self.raw_image is None: return
        
        # 1. Give immediate feedback to the user
        self.lbl_seg_result.setText("Running Advanced Segmentation...")
        self.btn_run_seg.setEnabled(False) # Prevent clicking again
        QApplication.processEvents() # Force the UI to update now
        
        # 2. Create and start the worker
        self.seg_worker = SegmentationWorker(self.raw_image)
        self.seg_worker.result_ready.connect(self.on_segmentation_done) # Link to the "done" function
        self.seg_worker.start()

    def on_segmentation_done(self, seg_data):
        self.brain_mask = seg_data.get("brain_mask")
        if seg_data.get("found"):
            self.tumor_mask = seg_data["mask"]

            # 1. Create the two image versions
            green_layer = np.zeros_like(self.raw_image)
            green_layer[:] = [0, 255, 0]
            green_layer = cv2.bitwise_and(green_layer, green_layer, mask=self.tumor_mask)
            self.segmented_overlay = cv2.addWeighted(self.raw_image, 1, green_layer, 0.4, 0)

            # Create a separate version with metrics for the segmentation view only
            image_with_metrics = self.segmented_overlay.copy()
            

            # 2. Measurement and Drawing
            stats = tumor_measurement.measure_tumor_advanced(self.tumor_mask)

            if stats:
                x1, x2, y1, y2 = stats['bbox']
                cv2.rectangle(image_with_metrics, (x1, y1), (x2, y2), (0, 0, 255), 2)
                center = stats['center']
                cv2.circle(image_with_metrics, center, 5, (0, 255, 255), -1)

                self.tumor_centroid = center 

                # Update Geometry Labels (keep all label updates)
                self.tumor_size = stats['equivalent_diameter_mm']
                self.lbl_area.setText(f"{stats['area_mm2']} mm²")
                self.lbl_loc.setText(str(center))
                self.lbl_size.setText(f"{self.tumor_size} mm")
                self.lbl_dims.setText(f"{stats['width_mm']} x {stats['height_mm']} mm")
                ecc = stats.get('eccentricity', 0)
                shape_desc = "Circular" if ecc < 0.5 else "Elongated"
                self.lbl_shape.setText(f"{ecc} ({shape_desc})")
            else:
                self.tumor_size = 0.0
                h, w, _ = self.raw_image.shape
                self.tumor_centroid = (w//2, h//2)

        else:
            # Handle No Tumor Found
            # ... (keep this logic) ...
            QMessageBox.information(self, "Result", "No tumor detected by segmentation.")
            self.btn_run_seg.setEnabled(True)
            return

        # --- FIX IS HERE ---
        # Segmentation Section:
        self.display_image(self.raw_image, self.lbl_seg_raw)
        self.display_image(image_with_metrics, self.lbl_seg_result)

        # Thermal Section:
        self.lbl_live_image.setStyleSheet("background-color: black; border-radius: 8px; border: 1px solid #444;")
        self.display_image(self.segmented_overlay, self.lbl_live_image)  # <-- Shows green overlay ONLY here

        if hasattr(self, 'lbl_growth_sim'):
            self.display_image(self.segmented_overlay, self.lbl_growth_sim)

        # Restore scroll position after segmentation is done and new widgets are rendered
        # --- 2. START COMBINED AI ANALYSIS ---
        self.diagnosis_card.setVisible(True)
        self.lbl_type.setText("Cloud AI...")
        self.lbl_grade.setText("...")
        self.rpt_desc.setText("Sending segmented image for deep analysis...")

        # Convert segmented image to PIL for the AI
        rgb_img = cv2.cvtColor(self.segmented_overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        self.ai_worker = cloud_ai_engine.CombinedAIWorker(pil_img)
        self.ai_worker.result_ready.connect(self.on_ai_analysis_done)
        self.ai_worker.start()
    def load_medical_file(self, filepath):
        """Standard + Medical Loader"""
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == '.dcm':
                ds = pydicom.dcmread(filepath)
                img = ds.pixel_array
                info = "DICOM"
            elif ext in ['.nii', '.gz']:
                nii = nib.load(filepath)
                d = nii.get_fdata()
                img = np.rot90(d[:, :, d.shape[2]//2]) if len(d.shape)==3 else d
                info = "NIfTI"
            else:
                img = cv2.imread(filepath)
                info = "Standard"
                return img, info

            # Normalize
            img = img.astype(float)
            img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
            img = img.astype(np.uint8)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, info
        except:
            return None, "Error"

    def display_image(self, cv_img, target_label):
        """Scales image to fit inside a specific label"""
        if cv_img is None: return

        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        
        # Scale to the specific target label
        scaled_pix = pix.scaled(
            target_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        target_label.setPixmap(scaled_pix)

    def toggle_laser_mode(self):
        sender = self.sender()
        if sender == self.chk_continuous and self.chk_continuous.isChecked():
            self.chk_pulsed.setChecked(False)
            # Hide ONLY Pulse Duration
            self.lbl_pulse_dur.setVisible(False)
            self.in_pulse_dur.setVisible(False)
            
        elif sender == self.chk_pulsed and self.chk_pulsed.isChecked():
            self.chk_continuous.setChecked(False)
            # Show Pulse Duration
            self.lbl_pulse_dur.setVisible(True)
            self.in_pulse_dur.setVisible(True)
            
        # Ensure at least one is checked
        if not self.chk_continuous.isChecked() and not self.chk_pulsed.isChecked():
            self.chk_continuous.setChecked(True)
            self.lbl_pulse_dur.setVisible(False)
            self.in_pulse_dur.setVisible(False)

    def update_wavelength_logic(self):
        """Updates the wavelength input based on radio selection"""
        if self.rad_ndyag.isChecked():
            self.in_wavelength.setText("1064")
            self.lbl_unit.setText("nm")
            self.in_wavelength.setEnabled(False) # Read-only for preset
            self.in_wavelength.setStyleSheet("color: #aaa; font-weight: bold; background: #222;")
            
        elif self.rad_diode.isChecked():
            self.in_wavelength.setText("980")
            self.lbl_unit.setText("nm")
            self.in_wavelength.setEnabled(False)
            self.in_wavelength.setStyleSheet("color: #aaa; font-weight: bold; background: #222;")
            
        elif self.rad_co2.isChecked():
            self.in_wavelength.setText("10600")
            self.lbl_unit.setText("nm")
            self.in_wavelength.setEnabled(False)
            self.in_wavelength.setStyleSheet("color: #aaa; font-weight: bold; background: #222;")
            
        elif self.rad_custom.isChecked():
            self.in_wavelength.clear()
            self.in_wavelength.setEnabled(True) # Allow typing
            self.in_wavelength.setFocus()
            self.in_wavelength.setStyleSheet("color: #00e5ff; font-weight: bold; border: 1px solid #00e5ff;")
    
    def ai_suggest_params(self):
        """
        Fills the inputs with the parameters suggested by the Cloud AI.
        """
        if not hasattr(self, 'ai_suggested_params') or not self.ai_suggested_params:
            QMessageBox.warning(self, "No AI Data", "Please run segmentation and analysis first.")
            return

        params = self.ai_suggested_params

        # Fill inputs from AI suggestions
        self.in_power.setText(str(params.get('power_W', 0.0)))
        self.in_energy.setText(str(params.get('energy_J', 0.0)))
        self.in_total_dur.setText(str(params.get('duration_s', 0.0)))

        baseline = self.lbl_temp.value()
        target_abs = params.get('target_temp_C', 60.0)
        delta = target_abs - baseline
        if delta < 0:
            delta = 0
        self.in_target.setText(f"{delta:.1f}")

        # --- Optional: If you want to re-calculate using the depth, do this ---
        # depth_mm = getattr(self, 'estimated_depth', 15.0)
        # new_params = laser_physics.calculate_laser_params(self.tumor_size, depth_mm, int(self.in_wavelength.text()))
        # ... then update the boxes with new_params ...

        self.lbl_ai_log.setText("AI: Suggested laser parameters have been loaded.")
    
    def action_suggest_params(self):
        """
        Calls the new physics engine with Wavelength data.
        """
        if self.tumor_size == 0: 
            QMessageBox.warning(self, "No Data", "Please segment the tumor first to get the size.")
            return

        # --- 1. GET INPUTS FOR PHYSICS FORMULA ---
        # A. Depth (Still assumed for 2D)
        depth_mm = 5.0 

        # B. Wavelength (From the UI)
        try:
            wavelength = int(self.in_wavelength.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid numeric wavelength.")
            return

        # --- 2. CALL THE NEW FUNCTION ---
        try:
            params = laser_physics.calculate_laser_params(self.tumor_size, depth_mm, wavelength)
            
            # 3. Update UI
            self.in_power.setText(str(params['power_W']))
            self.in_energy.setText(str(params['energy_J']))
            self.in_total_dur.setText(str(params['duration_s']))
            
            baseline = self.lbl_temp.value()
            target_abs = params['target_temperature_C']
            delta = target_abs - baseline
            if delta < 0: delta = 0
            self.in_target.setText(f"{delta:.1f}")
            
            self.lbl_ai_log.setText(f"AI: Physics calculated for {wavelength}nm at {depth_mm}mm depth.")
            
        except Exception as e:
            QMessageBox.critical(self, "Physics Error", f"Calculation failed: {str(e)}")

    def action_suggest_strategy(self):
        if self.tumor_size == 0: return
        _, tips = ai_core.generate_treatment_plan(self.tumor_size, self.tumor_type)
        QMessageBox.information(self, "Strategy", "\n\n".join(tips))

    def update_baseline_manually(self):
        """Called when user changes SpinBox value manually"""
        if not self.is_running:
            self.current_temp = self.lbl_temp.value()
    
    def toggle_ablation(self):

        if not self.is_running:

            # STARTING

            self.is_running = True

            # --- FIX: Create the initial temperature map ---
            if self.raw_image is not None:
                h, w, _ = self.raw_image.shape
                self.start_temp = self.lbl_temp.value()
                # Create a map filled with the starting temperature
                self.temperature_map = np.full((h, w), self.start_temp, dtype=np.float32)
            else:
                # Fallback if no image is loaded
                self.temperature_map = np.full((512, 512), 37.0, dtype=np.float32)
            # ---------------------------------------------

            # Activate Right Screen
            self.lbl_live_image.setStyleSheet("background-color: black; border-radius: 8px; border: 1px solid #00e5ff;")

            if hasattr(self, 'segmented_overlay'):
                self.display_image(self.segmented_overlay, self.lbl_live_image)

            self.current_temp = self.start_temp
            self.ai_engine.reset()

            self.btn_start.setText("STOP PROCEDURE")

            # Apply RED style with 20px font
            self.btn_start.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f; color: white; font-size: 20px; font-weight: bold; border-radius: 6px;
                }
                QPushButton:hover { background-color: #f44336; }
            """)

            # Lock UI while running
            self.lbl_temp.setReadOnly(True)
            self.lbl_temp.setStyleSheet("QDoubleSpinBox { background-color: #1e1e1e; color: #ffcc00; font-size: 40px; font-weight: bold; border: none; }") # Active style

            self.timer.start(100)

        else:
            # --- FIX: STOPPING MANUALLY ---
            self.is_running = False
            self.timer.stop()
            self.btn_reset.setEnabled(True)

            # Reset button to its initial state
            self.btn_start.setText("INITIALIZE ABLATION")
            self.btn_start.setStyleSheet("""
                QPushButton {
                    background-color: #2e7d32; color: white; font-size: 20px; font-weight: bold; border-radius: 6px;
                }
                QPushButton:hover { background-color: #43a047; }
                QPushButton:disabled { background-color: #1b5e20; color: #888; }
            """)

            self.lbl_ai_status.setText("STOPPED")
            # -----------------------------
            
    def update_simulation(self):

        # 1. Get Parameters
        try: 
            power = float(self.in_power.text())
            delta = float(self.in_target.text())
        except (ValueError, TypeError): 
            power = 10.0
            delta = 23.0

        baseline = getattr(self, 'start_temp', 37.0)
        absolute_target = baseline + delta

        # 2. Call the simple PDE state function
        mask_area = 1000
        if hasattr(self, 'tumor_mask') and self.tumor_mask is not None:
            mask_area = cv2.countNonZero(self.tumor_mask)

        new_temp, margin_temp, is_destroyed = laser_physics.calculate_pde_state(
            self.current_temp, 
            absolute_target, 
            power, 
            mask_area
        )

        # Pulsed Mode logic
        if not self.chk_continuous.isChecked():
            import time
            cycle = int(time.time() * 10) % 10 
            if cycle >= 6:
                new_temp -= 0.3 

        self.current_temp = new_temp

        # 3. Update UI Display
        self.lbl_temp.setValue(self.current_temp)

        # 4. Visualize Heatmap
        base_img = getattr(self, 'segmented_overlay', self.raw_image)
        mask = getattr(self, 'tumor_mask', None)

        heatmap_img = heatmap_engine.generate_heatmap(
            base_img, 
            self.current_temp, 
            absolute_target, 
            baseline, 
            mask
        )

        if heatmap_img is not None:
            self.display_image(heatmap_img, self.lbl_live_image)
            self.last_heatmap_image = heatmap_img

        # 5. AI Safety Checks
        imp = 400 + np.random.randint(-10, 10)

        # --- FIX IS HERE: Call with the correct number of arguments ---
        act, col, msg = self.ai_engine.analyze_telemetry(
            self.current_temp, 
            absolute_target, 
            imp, 
            margin_temp
        )
        # -----------------------------------------------------------

        self.lbl_ai_status.setText(act)
        self.lbl_ai_status.setStyleSheet(f"background-color: {col}; color: black; font-weight: bold; font-size: 20px; padding: 10px; border-radius: 6px;")
        self.lbl_ai_log.setText(msg)

        if act == "STOP": 
            self.toggle_ablation()
            QMessageBox.critical(self, "AI SAFETY", msg)
        elif is_destroyed:
            self.toggle_ablation()
            QMessageBox.information(self, "Success", "AI predicts tumor tissue has been successfully ablated.")
    
    def run_growth_simulation(self):
        if self.tumor_mask is None or self.brain_mask is None:
            QMessageBox.warning(self, "Data Required", "Please run a successful segmentation first.")
            return

        self.set_growth_controls_enabled(False)
        self.lbl_growth_sim.setText("Calculating Frames...")
        QApplication.processEvents()

        # --- Read from new SpinBoxes and Toggle ---
        params = {
            'D': self.spin_D.value(),
            'rho': self.spin_rho.value(),
            'beta': self.spin_beta.value(),
            'time_scale': 'days' if self.btn_days.isChecked() else 'hours'
        }
        end_time = int(self.spin_duration.value())

        self.growth_worker = TumorGrowthWorker(self.tumor_mask, self.brain_mask, params, end_time)
        self.growth_worker.frames_ready.connect(self.on_growth_frames_ready)
        self.growth_worker.start()

    def set_growth_controls_enabled(self, enabled):
        """Locks/unlocks the new growth simulation controls."""

        self.btn_hours.setEnabled(enabled)
        self.btn_days.setEnabled(enabled)
        self.spin_D.setEnabled(enabled)
        self.spin_rho.setEnabled(enabled)
        self.spin_beta.setEnabled(enabled)
        self.spin_duration.setEnabled(enabled)
        self.btn_run_growth_sim.setEnabled(enabled)

        if not enabled:
            self.btn_run_growth_sim.setText("⏳ Simulating...")
        else:
            self.btn_run_growth_sim.setText("▶️ Start Growth Simulation")

    def on_growth_frames_ready(self, frames, metrics):
        if not frames:
            QMessageBox.warning(self, "Simulation Failed", "The growth model could not generate frames.")
            self.set_growth_controls_enabled(True)  # Unlock on failure
            return

        self.growth_frames = frames
        self.growth_metrics = metrics  # <--- Store metrics
        self.current_frame_idx = 0

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_growth_animation)
        self.anim_timer.start(100)

    def update_growth_animation(self):
        if hasattr(self, 'current_frame_idx') and self.current_frame_idx < len(self.growth_frames):
            frame = self.growth_frames[self.current_frame_idx]
            metrics = self.growth_metrics[self.current_frame_idx]

            # --- FIX IS HERE: Check the correct button name ---
            time_scale = "Days" if self.btn_days.isChecked() else "Hours"
            # -------------------------------------------------

            time_val = metrics.get('current_time', 0)

            self.lbl_growth_time.setText(f"Time: {time_val:.1f} {time_scale}")
            self.lbl_growth_delta.setText(f"Growth: +{metrics.get('growth_delta_mm', 0):.2f} mm")

            # Optional: Update the main diagnosis card if grade changes
            if metrics.get('grade_status') == "Increased":
                self.lbl_grade.setText("IV (Increased)")
                self.lbl_grade.setStyleSheet("color: red; font-size: 20px; font-weight: bold;")

            # Blend and display image
            density_map = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
            blended = cv2.addWeighted(self.raw_image, 0.6, density_map, 0.4, 0)
            self.display_image(blended, self.lbl_growth_sim)

            self.current_frame_idx += 1
        else:
            if hasattr(self, 'anim_timer'):
                self.anim_timer.stop()
            self.set_growth_controls_enabled(True)

    def update_time_scale_toggle(self, clicked_button):
        """Handles the logic for the time scale toggle buttons."""
        if clicked_button == self.btn_hours:
            self.btn_hours.setChecked(True)
            self.btn_days.setChecked(False)
            self.btn_hours.setStyleSheet(self.active_toggle_style)
            self.btn_days.setStyleSheet(self.inactive_toggle_style)
        else: # Days button was clicked
            self.btn_days.setChecked(True)
            self.btn_hours.setChecked(False)
            self.btn_days.setStyleSheet(self.active_toggle_style)
            self.btn_hours.setStyleSheet(self.inactive_toggle_style)
    
    def start_initial_analysis(self):
        """Called by 'Start' or 'Restart'. Runs the heavy analysis."""
        if self.segmented_overlay is None:
            QMessageBox.warning(self, "Image Required", "Please run segmentation first.")
            return

        self.chat_display.clear()
        self.chat_history = []
        
        self.chat_display.append("<b>System:</b> Connecting to Cloud AI for deep analysis...")
        self.btn_init_chat.setVisible(False)
        self.btn_restart_chat.setVisible(False)
        self.btn_send_chat.setEnabled(False)
        self.chat_input.setEnabled(False)
        
        rgb_img = cv2.cvtColor(self.segmented_overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        self.ai_worker = cloud_ai_engine.CombinedAIWorker(pil_img)
        self.ai_worker.result_ready.connect(self.on_ai_analysis_done)
        self.ai_worker.start()

    def restart_chat_session(self):
        """Clears the chat window and history for a new conversation."""
        self.chat_display.clear()
        self.chat_history = []
        self.chat_display.setText("<i>System: New chat session started. You can now ask a new question.</i>")
        
        # Re-enable controls for a fresh start
        self.chat_input.setEnabled(True)
        self.btn_send_chat.setEnabled(True)
        self.btn_restart_chat.setEnabled(True) # Keep it enabled
        self.chat_input.setFocus()
    
    def on_ai_analysis_done(self, result):
        """
        Called when the CombinedAIWorker is finished.
        This version correctly displays the full AI report in the chat.
        """
        if "error" in result:
            QMessageBox.critical(self, "Cloud AI Error", result["error"])
            self.lbl_type.setText("Error")
            if hasattr(self, 'btn_restart_chat'):
                self.btn_restart_chat.setEnabled(True)
            return

        # 1. Store the AI's results from the JSON part
        self.ai_suggested_params = result.get('laser_parameters', {})
        self.tumor_type = result.get('classification', 'Unknown')
        self.estimated_depth = result.get('estimated_depth_mm', 15.0)
        
        # 2. Populate the Diagnosis UI card with the JSON summary data
        self.lbl_type.setText(result.get('classification', 'N/A'))
        self.lbl_grade.setText(result.get('grade', 'N/A'))
        self.lbl_depth.setText(f"{self.estimated_depth} mm")

        location = result.get('location', 'N/A')
        
        if location == 'N/A' or not location:
            pathology_text = result.get('pathology_analysis', '')
            location_match = re.search(r"(frontal|parietal|temporal|occipital)\s+lobe", pathology_text, re.IGNORECASE)
            if location_match:
                location = location_match.group(0).title()
            else:
                location = "Cortical" # Final fallback
                
        self.lbl_location_ai.setText(location)

        self.rpt_desc.setText(result.get('pathology_analysis', 'N/A'))
        self.rpt_action.setText(f"RECOMMENDATION: {result.get('recommendation', 'N/A')}")
        self.rpt_conf.setText("Cloud Analyzed")

        # --- 3. DISPLAY THE FULL REPORT IN THE CHAT ---
        self.chat_display.clear()
        
        # Get the full report from the AI's response
        markdown_report = result.get('markdown_report', "AI did not generate a detailed report. You may now ask questions.")
        
        # The history now starts with the AI's full, detailed report for context
        self.chat_history = [{"role": "assistant", "content": markdown_report}]
        
        # Display the full report using the HTML parser
        self.chat_display.append("<span style='color: #dcc9b6; font-size: 16px; font-weight: bold;'>Dr. AI:</span>")
        self.chat_display.insertHtml(self.format_ai_response_as_html(markdown_report))
        
        # --- 4. ENABLE CONTROLS ---
        self.chat_input.setEnabled(True)
        self.btn_send_chat.setEnabled(True)
        if hasattr(self, 'btn_restart_chat'):
            self.btn_restart_chat.setEnabled(True)
        self.chat_input.setFocus()
        
        # --- 5. ENABLE OTHER DOWNSTREAM BUTTONS ---
        self.btn_start.setEnabled(True)
        self.btn_reset.setEnabled(True)
        if hasattr(self, 'btn_export'): self.btn_export.setEnabled(True)

    
    def start_chat_session(self):
        # Determine if this is a "Restart" or an initial "Start"
        is_restart = "Restart" in self.btn_init_chat.text()

        # Check if an image is required
        if self.raw_image is None and self.chk_send_img.isChecked(): 
            QMessageBox.warning(self, "Image Required", "Please load a scan to analyze it, or uncheck 'Analyze Scan'.")
            return

        # Clear the screen on restart
        if is_restart:
            self.chat_display.clear()

        # 1. Update UI and reset history
        self.chat_display.append("<b>System:</b> Connecting to AI Doctor...")
        self.btn_init_chat.setEnabled(False)
        self.btn_send_chat.setEnabled(False)
        self.chat_input.setEnabled(False)
        self.chat_history = []

        # 2. Construct the first user message
        user_content = []
        if self.chk_send_img.isChecked() and self.raw_image is not None:
            rgb_img = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            import io, base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{base64_string}"
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            user_content.append({"type": "text", "text": "Analyze this scan and provide a detailed radiology report."})
            self.chat_display.append("<i>(Sending scan for analysis...)</i>")
        else:
            user_content.append({"type": "text", "text": "Hello Dr. AI. I am ready to discuss a case."})
            self.chat_display.append("<i>(Starting text-only chat...)</i>")

        # 3. Add the first message to history
        self.chat_history.append({"role": "user", "content": user_content})

        # 4. Start the Worker, passing the history
        self.chat_worker = cloud_ai_engine.FollowUpChatWorker(self.chat_history)
        self.chat_worker.response_received.connect(self.on_chat_response)
        self.chat_worker.error_occurred.connect(self.on_chat_error)
        self.chat_worker.start()


    def send_chat_message(self):
        text = self.chat_input.text().strip()
        if not text: return
        
        self.chat_display.append(f"<br><span style='color: #00e5ff;'><b>You:</b> {text}</span>")
        self.chat_input.clear()
        
        self.chat_input.setEnabled(False)
        self.btn_send_chat.setEnabled(False)
        self.btn_init_chat.setEnabled(False)
        
        # --- FIX: Use the correct variable name ---
        self.chat_history.append({"role": "user", "content": text})
        
        # Pass the correct variable to the worker
        self.follow_up_worker = cloud_ai_engine.FollowUpChatWorker(self.chat_history)
        # ----------------------------------------
        
        self.follow_up_worker.response_received.connect(self.on_chat_response)
        self.follow_up_worker.error_occurred.connect(self.on_chat_error)
        self.follow_up_worker.start()

    def on_chat_response(self, text):
        """Handle AI Reply and update button state."""
        self.chat_display.append("<br>")
        self.chat_display.append("<span style='color: #dcc9b6; font-size: 18px; font-weight: bold;'>Dr. AI:</span>")
        formatted_html = self.format_ai_response_as_html(text)
        self.chat_display.insertHtml(formatted_html)
        # Add AI reply to history
        self.chat_history.append({"role": "assistant", "content": text})
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

        # Re-enable controls
        self.chat_input.setEnabled(True)
        self.btn_send_chat.setEnabled(True)
        self.btn_init_chat.setText("↺ Restart Consultation")
        self.chat_input.setFocus()

    def on_chat_error(self, err):
        self.chat_display.append(f"<span style='color: red;'><b>Error:</b> {err}</span>")
        self.btn_init_chat.setVisible(False)
        self.btn_restart_chat.setEnabled(True)  # Re-enable restart on error
        self.btn_restart_chat.setVisible(True)
        self.chat_input.setEnabled(True)
        self.btn_send_chat.setEnabled(True)
        self.chat_input.setFocus()

    def format_ai_response_as_html(self, text):
        """
        Parses AI text and converts it to styled HTML with larger fonts.
        """
        text = text.replace("Dr. AI:", "").strip()
        text = text.replace("System:", "").strip()
        text = text.replace("Bot:", "").strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        html = ""
        in_list = False

        for line in lines:
            if line.startswith("#") or (line.startswith("**") and line.endswith("**")):
                content = line.strip("#* ").upper()
                # 16px -> 19px
                html += f"<h3 style='color: #00e5ff; font-size: 19px; margin-top: 10px; border-bottom: 1px solid #333; padding-bottom: 4px;'>{content}</h3>"
                in_list = False
            
            elif line.startswith("####"):
                content = line.strip("# ").strip()
                # 14px -> 17px
                html += f"<h4 style='color: #dcc9b6; font-size: 17px; margin-top: 8px;'>{content}</h4>"
                in_list = False

            if re.match(r"^(•|1\.|2\.|3\.|4\.|5\.|\-|\*|🌬️|💧|🧠|Try:)\s*", line):
                if not in_list:
                    html += "<ul style='margin-left: 20px;'>"
                    in_list = True
                content = re.sub(r"^(•|1\.|2\.|3\.|4\.|5\.|\-|\*|🌬️|💧|🧠|Try:)\s*", "", line)
                content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
                html += f"<li style='font-size: 17px; margin-bottom: 5px;'>{content}</li>"
            
            elif '|' in line and '---' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                html += "<tr>"
                for part in parts:
                    # 13px -> 16px
                    html += f"<td style='padding: 5px; border: 1px solid #444; font-size: 16px;'>{part}</td>"
                html += "</tr>"
            
            elif line.startswith("---"):
                html += "<hr style='border: 1px solid #333;'>"
                in_list = False
                
            else:
                if in_list:
                    html += "</ul>"
                    in_list = False
                if line.startswith("⚠️"):
                    # 14px -> 17px
                    html += f"<p style='font-size: 17px; color: #ffcc00;'><strong>{line}</strong></p>"
                else:
                    # 14px -> 17px
                    html += f"<p style='font-size: 17px;'>{line}</p>"
        
        if in_list:
            html += "</ul>"
            
        return html

    def on_classification_done(self, result):
        """Called when AI finishes analyzing the image"""
        if result:
            # 1. Update Logic Variables
            self.tumor_type = result['class']
            
            # 2. Update The Summary Header (Top of Card)
            if hasattr(self, 'lbl_type'):
                self.lbl_type.setText(result['class'])
            if hasattr(self, 'lbl_grade'):
                self.lbl_grade.setText(result['grade'])
            
            # 3. Update The Detailed Report (Bottom of Card)
            if hasattr(self, 'rpt_conf'):
                self.rpt_conf.setText(f"{result['confidence']:.1f}%")
                
            if hasattr(self, 'rpt_desc'):
                self.rpt_desc.setText(result['desc'])
                
            if hasattr(self, 'rpt_action'):
                self.rpt_action.setText(f"RECOMMENDATION: {result['action']}")

    def action_export_report(self):
        # 1. Ask save location
        default_name = f"Report_{self.in_patient_name.text().strip() or 'Unknown'}.pdf"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Report", default_name, "PDF Files (*.pdf)")
        if not filename:
            return

        # 2. Prepare Images
        raw_img_path = "temp_raw.png"
        seg_img_path = "temp_seg.png"
        heat_img_path = "temp_heat.png"

        if self.raw_image is not None:
            cv2.imwrite(raw_img_path, self.raw_image)

            if hasattr(self, 'segmented_overlay') and self.segmented_overlay is not None:
                cv2.imwrite(seg_img_path, self.segmented_overlay)
            else:
                cv2.imwrite(seg_img_path, self.raw_image)

            # --- Save the last known heatmap image directly ---
            if hasattr(self, 'last_heatmap_image') and self.last_heatmap_image is not None:
                cv2.imwrite(heat_img_path, self.last_heatmap_image)
            else:
                # Fallback if simulation hasn't run yet
                cv2.imwrite(heat_img_path, self.raw_image)
            # ----------------------------------------------------

        # --- NEW: Capture Growth Sim Image & Data ---
        growth_sim_path = "temp_growth.png"
        growth_delta_text = ""

        # Check if the simulation has run and has a final state
        if hasattr(self, 'growth_frames') and self.growth_frames:
            # Create the final blended image for the report
            final_density_map = cv2.applyColorMap(self.growth_frames[-1], cv2.COLORMAP_VIRIDIS)
            final_blended_img = cv2.addWeighted(self.raw_image, 0.6, final_density_map, 0.4, 0)
            cv2.imwrite(growth_sim_path, final_blended_img)

            # Get the final growth delta from the metrics
            if hasattr(self, 'growth_metrics') and self.growth_metrics:
                final_metric = self.growth_metrics[-1]
                growth_delta_text = f"+{final_metric.get('growth_delta_mm', 0):.2f} mm"
        # ----------------------------------------

        # 3. Bundle Data
        report_data = {
            'patient_name': self.in_patient_name.text().strip() or "Anonymous",
        }

        if self.chk_images.isChecked():
            report_data['raw_image'] = raw_img_path
            report_data['seg_image'] = seg_img_path
            report_data['heat_image'] = heat_img_path

        if self.chk_diagnosis.isChecked():
            rec_text = self.rpt_action.text().replace("RECOMMENDATION: ", "")
            report_data.update({
                'tumor_type': self.tumor_type,
                'grade': self.lbl_grade.text(),
                'size': self.tumor_size,
                'location': self.lbl_loc.text(),
                'location_ai': self.lbl_location_ai.text(),
                'area': self.lbl_area.text(),
                'dims': self.lbl_dims.text(),
                'shape': self.lbl_shape.text(),
                'depth': self.lbl_depth.text(),
                'pathology': self.rpt_desc.text(),
                'recommendation': rec_text
            })

        if self.chk_physics.isChecked():
            mode = "Continuous Wave" if self.chk_continuous.isChecked() else "Pulsed"

            # Get Wavelength / Material Logic
            if hasattr(self, 'rad_ndyag') and self.rad_ndyag.isChecked():
                material = "Nd:YAG (1064 nm)"
            elif hasattr(self, 'rad_diode') and self.rad_diode.isChecked():
                material = "Diode (980 nm)"
            elif hasattr(self, 'rad_co2') and self.rad_co2.isChecked():
                material = "CO2 (10.6 µm)"
            elif hasattr(self, 'rad_custom') and self.rad_custom.isChecked():
                material = f"Custom ({self.in_wavelength.text()} nm)"
            else:
                material = "Standard"

            report_data.update({
                'power': self.in_power.text(),
                'energy': self.in_energy.text(),
                'target_temp': self.in_target.text(),
                'total_duration': self.in_total_dur.text(),
                'mode': mode,
                'material': material,
                'pulse_width': self.in_pulse_dur.text() if "Pulsed" in mode else None
            })

        # --- NEW: Bundle Growth Sim Data ---
        if self.chk_growth_sim.isChecked() and hasattr(self, 'growth_frames'):
            report_data['growth_sim_image'] = growth_sim_path

            # Round the parameters for clean printing
            gp = {
                'D': round(self.spin_D.value(), 3),
                'rho': round(self.spin_rho.value(), 3),
                'beta': round(self.spin_beta.value(), 3),
                'duration': self.spin_duration.value(),
                'time_scale': 'Days' if self.btn_days.isChecked() else 'Hours',
                'growth_delta': growth_delta_text  # Add the final growth
            }
            report_data['growth_params'] = gp
        # -----------------------------------

        if self.chk_chat.isChecked():
            report_data['chat_history'] = getattr(self, 'chat_history', [])

        # 4. Generate and show success/error
        try:
            report_generator.generate_pdf_report(filename, report_data)
            QMessageBox.information(self, "Success", "Report exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"PDF Error: {str(e)}")

        # 5. Cleanup
        for f in [raw_img_path, seg_img_path, heat_img_path, growth_sim_path]:
            if os.path.exists(f):
                os.remove(f)
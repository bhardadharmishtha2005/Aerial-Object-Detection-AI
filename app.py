import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SkyGuard Aerial AI", layout="wide")

# --- CUSTOM PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .status-card {
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .safe-bg { background: linear-gradient(135deg, #28a745, #1e7e34); }
    .danger-bg { background: linear-gradient(135deg, #dc3545, #a71d2a); }
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_with_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Ensure these filenames match your GitHub exactly!
    cnn = load_model('best_model_custom_cnn.keras')
    transfer = load_model('best_model_transfer_learning.keras')
    yolo_model = YOLO('best.pt')
    return cnn, transfer, yolo_model

cnn_model, transfer_model, yolo_model = load_models()

# --- HEADER SECTION ---
st.title("🛡️ SkyGuard: Advanced Aerial Surveillance")
st.markdown("🔍 *Enterprise-grade Bird vs. Drone Detection System*")
st.divider()

# --- INPUT SELECTION ---
col_input, col_info = st.columns([2, 1])

with col_input:
    st.subheader("📥 Data Input")
    input_mode = st.tabs(["Upload Image", "Labmentix Dataset Samples"])
    
    img = None
    
    with input_mode[0]:
        uploaded_file = st.file_uploader("Drop surveillance frame here...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            
    with input_mode[1]:
        # NOTE: Create a folder named 'samples' on GitHub and put images there!
        sample_path = "samples/"
        if os.path.exists(sample_path):
            sample_files = os.listdir(sample_path)
            selected_sample = st.selectbox("Select a benchmark image:", sample_files)
            if selected_sample:
                img = Image.open(os.path.join(sample_path, selected_sample))
        else:
            st.warning("Sample folder not found. Upload 'samples' folder to GitHub to enable this feature.")

with col_info:
    st.subheader("📋 System Status")
    st.info("Network: Connected")
    st.info("Models: Optimized (CNN + YOLOv8)")
    st.info(f"Engineer: Bharda Dharmishtha")

# --- ANALYSIS ENGINE ---
if img is not None:
    st.divider()
    
    # 1. Processing for Classification
    img_resized = img.resize((224, 224))
    img_arr = image.img_to_array(img_resized)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0
    
    # 2. Get Results
    pred_cnn = cnn_model.predict(img_arr)[0]
    pred_transfer = transfer_model.predict(img_arr)[0]
    results_yolo = yolo_model(img)
    
    classes = ['Bird', 'Drone']
    final_idx = np.argmax(pred_transfer)
    label = classes[final_idx]
    conf = np.max(pred_transfer) * 100

    # --- RESULTS DASHBOARD ---
    # Top Row: Threat Alert
    if label == "Drone":
        st.markdown(f'<div class="status-card danger-bg"><h1>⚠️ THREAT DETECTED: {label.upper()}</h1><p>Confidence: {conf:.2f}%</p></div>', unsafe_with_html=True)
    else:
        st.markdown(f'<div class="status-card safe-bg"><h1>✅ CLEAR: {label.upper()}</h1><p>Confidence: {conf:.2f}%</p></div>', unsafe_with_html=True)

    # Bottom Row: Visuals
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='info-card'><h3>Localization (YOLOv8)</h3></div>", unsafe_with_html=True)
        res_plotted = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)
        st.write(f"Objects Detected: {len(results_yolo[0].boxes)}")

    with c2:
        st.markdown("<div class='info-card'><h3>Multi-Model Logic Check</h3></div>", unsafe_with_html=True)
        # Compare both models in a dataframe
        data = {
            "Model Architecture": ["Custom CNN", "Transfer Learning (MobileNet)"],
            "Bird Probability": [f"{pred_cnn[0]*100:.2f}%", f"{pred_transfer[0]*100:.2f}%"],
            "Drone Probability": [f"{pred_cnn[1]*100:.2f}%", f"{pred_transfer[1]*100:.2f}%"]
        }
        st.table(pd.DataFrame(data))
        
        # Add a bar chart for visual proof
        chart_data = pd.DataFrame({
            "Confidence": [pred_transfer[0], pred_transfer[1]],
            "Class": ["Bird", "Drone"]
        }).set_index("Class")
        st.bar_chart(chart_data)

else:
    st.write("---")
    st.markdown("### 🛰️ System Ready. Awaiting Airspace Data.")

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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aerial Object Detection", 
    page_icon="🛡️", 
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    try:
        cnn = load_model('best_model_custom_cnn.keras')
        transfer = load_model('best_model_transfer_learning.keras')
        yolo_m = YOLO('best.pt')
        return cnn, transfer, yolo_m
    except Exception:
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("🛡️ SkyGuard Ops")
    st.write("---")
    
    st.markdown("### 📥 Input Selection")
    # Renamed to Sample Dataset as requested
    input_choice = st.radio("Source:", ["Manual Upload", "samples"])
    
    st.write("---")
    st.markdown("### 🎯 Detection Sensitivity")
    conf_threshold = st.slider("Min Confidence %", 0.0, 1.0, 0.5)
    
    st.write("---")
    st.caption("SkyGuard | Developed by Bharda Dharmishtha")

# --- MAIN INTERFACE ---
# Renamed from "Aerial Object Detection" to SkyGuard
st.title("🛡️ SkyGuard: Aerial Intelligence")
st.markdown("##### Enterprise-Grade Drone & Bird Monitoring System")
st.write("---")

# --- DATA HANDLING ---
img = None

if input_choice == "Manual Upload":
    uploaded_file = st.file_uploader("Drop surveillance frame here...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
else:
    # Improved Sample Dataset Logic
    sample_path = "samples"
    if os.path.exists(sample_path):
        sample_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected = st.selectbox("Select Benchmark Image:", sample_files)
            img = Image.open(os.path.join(sample_path, selected))
        else:
            st.info("The 'samples' folder is currently empty.")
    else:
        # Replaced the big red error with a helpful tip
        st.info("💡 To use Sample Datasets, create a folder named 'samples' in your repository and upload images.")

# --- ANALYTICS ENGINE ---
if img is not None:
    # Pre-processing
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    with st.spinner('Analyzing...'):
        res_cnn = cnn.predict(img_array, verbose=0)[0]
        res_transfer = transfer.predict(img_array, verbose=0)[0]
        
        # Output handling
        if len(res_transfer) == 1:
            drone_p = float(res_transfer[0])
            bird_p = 1.0 - drone_p
            scores = [bird_p, drone_p]
        else:
            scores = res_transfer
            
        yolo_results = yolo(img, conf=conf_threshold, verbose=False)

    classes = ['Bird', 'Drone']
    final_idx = np.argmax(scores)
    final_label = classes[final_idx]
    final_conf = np.max(scores) * 100

    # --- TOP METRICS ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Target Identified", final_label)
    with m2:
        st.metric("System Confidence", f"{final_conf:.2f}%")
    with m3:
        if final_label == "Drone" and final_conf > 70:
            st.error("🚨 THREAT DETECTED")
        else:
            st.success("✅ CLEAR")

    st.write(" ")

    # --- VISUALIZATION ---
    col_vis, col_data = st.columns([1.5, 1])

    with col_vis:
        st.subheader("📍 YOLOv8 Localization")
        res_img_bgr = yolo_results[0].plot()
        res_img_rgb = cv2.cvtColor(res_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(res_img_rgb, use_container_width=True)

    with col_data:
        st.subheader("📊 Probability Analysis")
        chart_df = pd.DataFrame({
            "Confidence": [float(scores[0]), float(scores[1])],
            "Object": ["Bird", "Drone"]
        }).set_index("Object")
        st.bar_chart(chart_df)
        
        with st.expander("Technical Comparison"):
            st.table(pd.DataFrame({
                "Model": ["Custom CNN", "Transfer Learning"],
                "Bird %": [f"{res_cnn[0]*100:.1f}%", f"{scores[0]*100:.1f}%"],
                "Drone %": [f"{res_cnn[1]*100:.1f}%", f"{scores[1]*100:.1f}%"]
            }))
else:
    st.info("Awaiting input. Please upload a surveillance frame or select a sample from the sidebar.")

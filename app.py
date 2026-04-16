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
    page_title="SkyGuard: Aerial Intelligence", 
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
        # Safety: Check if files exist before loading
        cnn = load_model('best_model_custom_cnn.keras') if os.path.exists('best_model_custom_cnn.keras') else None
        transfer = load_model('best_model_transfer_learning.keras') if os.path.exists('best_model_transfer_learning.keras') else None
        yolo_m = YOLO('best.pt') if os.path.exists('best.pt') else None
        return cnn, transfer, yolo_m
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ SkyGuard Ops")
    st.write("---")
    st.markdown("### 📥 Input Selection")
    input_choice = st.radio("Source:", ["Manual Upload", "Sample Dataset"])
    
    st.write("---")
    st.markdown("### 🎯 Detection Sensitivity")
    conf_threshold = st.slider("Min Confidence %", 0.0, 1.0, 0.5)
    
    st.write("---")

# --- MAIN INTERFACE ---
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
    sample_path = "samples"
    if os.path.exists(sample_path):
        sample_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected = st.selectbox("Select Benchmark Image:", sample_files)
            img = Image.open(os.path.join(sample_path, selected))
        else:
            st.info("The 'samples' folder is empty. Please upload images to GitHub.")
    else:
        st.info("💡 To use Sample Datasets, create a folder named 'samples' in your GitHub repo.")

# --- ANALYTICS ENGINE ---
if img is not None:
    # Pre-processing
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing...'):
        # Safety Check for Models
        if cnn and transfer and yolo:
            res_cnn = cnn.predict(img_array, verbose=0)[0]
            res_transfer = transfer.predict(img_array, verbose=0)[0]
            
            # Fix for Shape Issues
            scores = res_transfer if len(res_transfer) > 1 else [1.0 - res_transfer[0], res_transfer[0]]
            yolo_results = yolo(img, conf=conf_threshold, verbose=False)

            classes = ['Bird', 'Drone']
            final_idx = np.argmax(scores)
            final_label = classes[final_idx]
            final_conf = scores[final_idx] * 100

            # --- DISPLAY METRICS ---
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Target", final_label)
            with m2: st.metric("Confidence", f"{final_conf:.2f}%")
            with m3:
                if final_label == "Drone" and final_conf > 70:
                    st.error("🚨 THREAT DETECTED")
                else:
                    st.success("✅ CLEAR")

            # --- VISUALS ---
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("📍 YOLOv8 Localization")
                res_plotted = yolo_results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)
            with c2:
                st.subheader("📊 Analysis")
                chart_data = pd.DataFrame({"Confidence": [float(scores[0]), float(scores[1])], "Object": ["Bird", "Drone"]}).set_index("Object")
                st.bar_chart(chart_data)
        else:
            st.error("Models not found. Ensure .keras and .pt files are in your GitHub root.")
else:
    st.info("Please upload an image or select a sample to begin.")

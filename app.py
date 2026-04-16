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
st.set_page_config(page_title="SkyGuard: Aerial Intelligence", page_icon="🛡️", layout="wide")

# --- CUSTOM UI ---
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

# --- MODEL LOADING (WITH SAFETY) ---
@st.cache_resource
def load_all_models():
    try:
        # Check if files exist to prevent crashes
        cnn_path = 'best_model_custom_cnn.keras'
        transfer_path = 'best_model_transfer_learning.keras'
        yolo_path = 'best.pt'
        
        cnn = load_model(cnn_path) if os.path.exists(cnn_path) else None
        transfer = load_model(transfer_path) if os.path.exists(transfer_path) else None
        yolo_m = YOLO(yolo_path) if os.path.exists(yolo_path) else None
        
        return cnn, transfer, yolo_m
    except Exception as e:
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ SkyGuard Ops")
    st.write("---")
    input_choice = st.radio("Source:", ["Manual Upload", "Sample Dataset"])
    conf_threshold = st.slider("Min Confidence %", 0.0, 1.0, 0.5)
    st.write("---")

# --- MAIN INTERFACE ---
st.title("🛡️ SkyGuard: Aerial Intelligence")
st.markdown("##### Enterprise-Grade Drone & Bird Monitoring System")
st.write("---")

# --- IMAGE SELECTION ---
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
            st.info("The 'samples' folder is empty. Upload images to GitHub to see them here.")
    else:
        st.info("💡 Create a folder named 'samples' in your GitHub to use this feature.")

# --- PROCESSING ---
if img is not None:
    if cnn and transfer and yolo:
        # Pre-processing for CNNs
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing...'):
            res_cnn = cnn.predict(img_array, verbose=0)[0]
            res_transfer = transfer.predict(img_array, verbose=0)[0]
            yolo_results = yolo(img, conf=conf_threshold, verbose=False)

            # Standardize output scores
            scores = res_transfer if len(res_transfer) > 1 else [1.0 - res_transfer[0], res_transfer[0]]
            final_idx = np.argmax(scores)
            label = "Bird" if final_idx == 0 else "Drone"
            conf_val = scores[final_idx] * 100

        # DISPLAY RESULTS
        m1, m2, m3 = st.columns(3)
        m1.metric("Target", label)
        m2.metric("Confidence", f"{conf_val:.2f}%")
        with m3:
            if label == "Drone" and conf_val > 70: st.error("🚨 THREAT DETECTED")
            else: st.success("✅ CLEAR")

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("📍 YOLOv8 Localization")
            # This uses the 'headless' compatible plotting
            res_plotted = yolo_results[0].plot()
            st.image(res_plotted, channels="BGR", use_container_width=True)
        with col2:
            st.subheader("📊 Probability Analysis")
            chart_df = pd.DataFrame({"Score": [float(scores[0]), float(scores[1])], "Object": ["Bird", "Drone"]}).set_index("Object")
            st.bar_chart(chart_df)
    else:
        st.error("Model files (.keras or .pt) missing from root directory.")
else:
    st.info("Please upload an image or choose a sample to begin.")

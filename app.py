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
st.set_page_config(page_title="SkyGuard AI", layout="wide")

# --- FIXED CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #004a99; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
</style>
""", unsafe_with_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    try:
        cnn = load_model('best_model_custom_cnn.keras')
        transfer = load_model('best_model_transfer_learning.keras')
        yolo_m = YOLO('best.pt')
        return cnn, transfer, yolo_m
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- HEADER ---
st.title("🛡️ SkyGuard: Aerial Intelligence")
st.markdown("##### Enterprise-Grade Drone & Bird Monitoring | Developer: Bharda Dharmishtha")
st.divider()

# --- INPUT SECTION ---
col_in, col_st = st.columns([2, 1])

with col_in:
    tab1, tab2 = st.tabs(["📤 Upload Image", "🎯 Labmentix Samples"])
    img = None
    
    with tab1:
        uploaded_file = st.file_uploader("Upload surveillance frame", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            
    with tab2:
        # NOTE: Make sure you have a folder named 'samples' in your GitHub
        if os.path.exists("samples"):
            sample_files = os.listdir("samples")
            if sample_files:
                selected = st.selectbox("Choose a test case:", sample_files)
                img = Image.open(f"samples/{selected}")
            else:
                st.write("Upload images to 'samples/' folder on GitHub to see them here.")
        else:
            st.info("No 'samples' folder found on GitHub yet.")

with col_st:
    st.write("### ⚙️ System Status")
    if cnn and yolo:
        st.success("AI Core: Online")
    else:
        st.error("AI Core: Offline")

# --- ANALYSIS ENGINE ---
if img is not None:
    st.divider()
    
    # Preprocess
    img_res = img.resize((224, 224))
    img_arr = image.img_to_array(img_res) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict
    res_transfer = transfer.predict(img_arr)[0]
    res_cnn = cnn.predict(img_arr)[0]
    results_yolo = yolo(img)
    
    classes = ['Bird', 'Drone']
    final_class = classes[np.argmax(res_transfer)]
    conf = np.max(res_transfer) * 100

    # --- TOP METRICS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Target Identified", final_class)
    m2.metric("System Confidence", f"{conf:.2f}%")
    m3.metric("Threat Status", "HIGH ALERT" if final_class == "Drone" else "NO THREAT")

    st.write(" ")

    # --- VISUALIZATION ---
    v_left, v_right = st.columns(2)
    
    with v_left:
        st.subheader("📍 YOLOv8 Object Localization")
        res_plot = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)

    with v_right:
        st.subheader("📊 Cross-Model Probability")
        chart_data = pd.DataFrame({
            "Confidence": [res_transfer[0], res_transfer[1]],
            "Label": ["Bird", "Drone"]
        }).set_index("Label")
        st.bar_chart(chart_data)
        
        st.info(f"YOLO detected {len(results_yolo[0].boxes)} object(s) in frame.")
else:
    st.write("---")
    st.info("System Ready. Please provide an aerial image for analysis.")

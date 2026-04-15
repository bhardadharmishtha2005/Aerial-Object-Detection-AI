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
st.set_page_config(page_title="Aerial Object Detection", layout="wide")

# --- FORCE WHITE BACKGROUND & CLEAN UI ---
st.markdown("""
    <style>
    /* Force White Background on everything */
    .stApp, .main, .block-container {
        background-color: #FFFFFF !important;
        color: #1F1F1F !important;
    }
    
    /* Change all text to dark for visibility on white */
    h1, h2, h3, h5, p, span, .stMetric label {
        color: #1F1F1F !important;
    }

    /* Professional Metric Boxes */
    [data-testid="stMetricValue"] {
        color: #004a99 !important;
        font-weight: bold;
    }
    
    .stMetric {
        background-color: #f8f9fa !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid #eeeeee !important;
    }

    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- HEADER (CLEAN) ---
st.title("🛡️ Aerial Object Detection")
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
        if os.path.exists("samples"):
            sample_files = [f for f in os.listdir("samples") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_files:
                selected = st.selectbox("Choose a test case:", sample_files)
                img = Image.open(f"samples/{selected}")
            else:
                st.warning("No images found in 'samples/' folder.")
        else:
            st.info("Sample folder not found on GitHub.")

with col_st:
    st.subheader("⚙️ System Status")
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
    
    # Robust Check for Transfer Learning Output
    if len(res_transfer) == 1:
        drone_prob = float(res_transfer[0])
        bird_prob = 1.0 - drone_prob
        transfer_scores = [bird_prob, drone_prob]
    else:
        transfer_scores = res_transfer

    final_class = classes[np.argmax(transfer_scores)]
    conf = np.max(transfer_scores) * 100

    # --- TOP METRICS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Identification", final_class)
    m2.metric("Confidence", f"{conf:.2f}%")
    m3.metric("Threat Level", "HIGH" if final_class == "Drone" else "LOW")

    st.write(" ")

    # --- VISUALIZATION ---
    v_left, v_right = st.columns(2)
    
    with v_left:
        st.subheader("📍 YOLOv8 Localization")
        res_plot = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)

    with v_right:
        st.subheader("📊 Probability Analysis")
        chart_data = pd.DataFrame({
            "Confidence": [float(transfer_scores[0]), float(transfer_scores[1])],
            "Label": ["Bird", "Drone"]
        }).set_index("Label")
        st.bar_chart(chart_data)
        
        # Details Table for technical marks
        st.table(pd.DataFrame({
            "Model": ["Custom CNN", "Transfer Learning"],
            "Bird %": [f"{res_cnn[0]*100:.2f}%", f"{transfer_scores[0]*100:.2f}%"],
            "Drone %": [f"{res_cnn[1]*100:.2f}%", f"{transfer_scores[1]*100:.2f}%"]
        }))
else:
    st.write("---")
    st.info("System Ready. Please provide an aerial image for analysis.")

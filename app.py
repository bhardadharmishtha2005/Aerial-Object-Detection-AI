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
# Removed CSS background overrides to allow native Streamlit theme handling
st.set_page_config(page_title="Aerial Object Detection", layout="wide")

# --- CLEAN UI ELEMENTS ---
st.markdown("""
    <style>
    /* Hide unnecessary Streamlit menus for a cleaner production look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Subtle styling for metric cards */
    [data-testid="stMetricValue"] {
        font-weight: 700;
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
    except Exception as e:
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.write("---")
    
    # System Status
    if cnn and yolo:
        st.success("AI Core: Connected")
    else:
        st.error("AI Core: Offline")
    
    st.markdown("### 📥 Data Source")
    input_choice = st.radio("Choose Input Type:", ["Upload Image", "🎯 Labmentix Samples"])
    
    st.write("---")
    st.info("System optimized for real-time aerial surveillance analysis.")

# --- MAIN PAGE HEADER ---
st.title("🛡️ Aerial Object Detection")
st.caption("Advanced AI Ensemble for Sky Surveillance")
st.write("---")

# --- DATA INPUT LOGIC ---
img = None

if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Drag and drop a surveillance frame", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
else:
    # Handle Sample Folder
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected = st.selectbox("Select a benchmark frame:", sample_files)
            img = Image.open(os.path.join(sample_dir, selected))
        else:
            st.warning("No images found in /samples folder.")
    else:
        st.error("⚠️ 'samples' folder not found. Please create it in your GitHub repository.")

# --- ANALYSIS ENGINE ---
if img is not None:
    # Processing for classification
    img_res = img.resize((224, 224))
    img_arr = image.img_to_array(img_res) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predictions
    res_transfer = transfer.predict(img_arr)[0]
    res_cnn = cnn.predict(img_arr)[0]
    results_yolo = yolo(img)
    
    classes = ['Bird', 'Drone']
    
    # Robust Check for Transfer Learning Output (IndexError Fix)
    if len(res_transfer) == 1:
        drone_prob = float(res_transfer[0])
        bird_prob = 1.0 - drone_prob
        transfer_scores = [bird_prob, drone_prob]
    else:
        transfer_scores = res_transfer

    final_class = classes[np.argmax(transfer_scores)]
    conf = np.max(transfer_scores) * 100

    # --- RESULTS DASHBOARD ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Identification", final_class)
    with m2:
        st.metric("System Confidence", f"{conf:.2f}%")
    with m3:
        status = "⚠️ THREAT" if final_class == "Drone" else "✅ CLEAR"
        st.metric("Security Status", status)

    st.write(" ")

    # --- VISUAL LAYOUT ---
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 📍 Object Localization")
        res_plot = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="YOLOv8 Detection Output", use_container_width=True)

    with col_right:
        st.markdown("### 📊 AI Probability Analysis")
        chart_data = pd.DataFrame({
            "Confidence": [float(transfer_scores[0]), float(transfer_scores[1])],
            "Label": ["Bird", "Drone"]
        }).set_index("Label")
        st.bar_chart(chart_data)
        
        # Technical Validation Table
        st.markdown("#### Model Validation")
        st.table(pd.DataFrame({
            "Architecture": ["Custom CNN", "Transfer Learning"],
            "Bird %": [f"{res_cnn[0]*100:.1f}%", f"{transfer_scores[0]*100:.1f}%"],
            "Drone %": [f"{res_cnn[1]*100:.1f}%", f"{transfer_scores[1]*100:.1f}%"]
        }))
else:
    st.info("Awaiting data input... Select an option from the Control Panel to begin.")

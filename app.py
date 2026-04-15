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

# --- CUSTOM CSS FOR WHITE THEME & SPACING ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    h1 { color: #1E3A8A !important; font-weight: 700; }
    .stMetric { 
        background-color: #F8FAFC !important; 
        border: 1px solid #E2E8F0 !important; 
        border-radius: 12px !important;
    }
    /* Hide specific Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

# --- SIDEBAR (Clean up the Main Page) ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    # System Status in Sidebar
    if cnn and yolo:
        st.success("AI Core: Connected")
    else:
        st.error("AI Core: Offline")
    
    st.markdown("### 📥 Data Source")
    input_choice = st.radio("Choose Input Type:", ["Upload Image", "🎯 Labmentix Samples"])
    
    st.markdown("---")
    st.info("System optimized for real-time aerial surveillance analysis.")

# --- MAIN PAGE HEADER ---
st.title("🛡️ Aerial Object Detection")
st.write("---")

# --- INPUT LOGIC ---
img = None

if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Drop surveillance frame here", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
else:
    # Fix for Sample Folder
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected = st.selectbox("Select a benchmark frame:", sample_files)
            img = Image.open(os.path.join(sample_dir, selected))
        else:
            st.warning("No images found in /samples folder.")
    else:
        st.error("⚠️ 'samples' folder not found on GitHub. Create a folder named 'samples' and upload images to use this feature.")

# --- ANALYSIS ENGINE ---
if img is not None:
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

    # --- TOP METRIC TILES ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Identification", final_class)
    with m2:
        st.metric("System Confidence", f"{conf:.2f}%")
    with m3:
        status = "🔴 THREAT" if final_class == "Drone" else "🟢 CLEAR"
        st.metric("Security Status", status)

    st.write(" ")

    # --- TWO-COLUMN VISUAL LAYOUT ---
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 📍 Object Localization")
        res_plot = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="YOLOv8 Real-Time Detection", use_container_width=True)

    with col_right:
        st.markdown("### 📊 AI Probability")
        chart_data = pd.DataFrame({
            "Confidence": [float(transfer_scores[0]), float(transfer_scores[1])],
            "Label": ["Bird", "Drone"]
        }).set_index("Label")
        st.bar_chart(chart_data)
        
        # Tech Specs Table
        st.markdown("#### Technical Comparison")
        st.table(pd.DataFrame({
            "Model": ["Custom CNN", "Transfer Learning"],
            "Bird Score": [f"{res_cnn[0]*100:.1f}%", f"{transfer_scores[0]*100:.1f}%"],
            "Drone Score": [f"{res_cnn[1]*100:.1f}%", f"{transfer_scores[1]*100:.1f}%"]
        }))
else:
    st.info("Awaiting aerial data... Use the sidebar to upload an image or select a sample.")

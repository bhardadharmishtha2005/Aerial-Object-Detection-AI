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

# --- CUSTOM CSS FOR PROFESSIONAL DASHBOARD ---
st.markdown("""
<style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #004a99; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        # Ensure these filenames match your GitHub repository exactly
        cnn = load_model('best_model_custom_cnn.keras')
        transfer = load_model('best_model_transfer_learning.keras')
        yolo_model = YOLO('best.pt')
        return cnn, transfer, yolo_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

cnn_model, transfer_model, yolo_model = load_models()

# --- HEADER SECTION ---
st.title("🛡️ SkyGuard: Advanced Aerial Surveillance")
st.markdown("##### Professional Drone & Bird Monitoring System | Developed by Bharda Dharmishtha")
st.divider()

# --- INPUT SECTION ---
col_input, col_status = st.columns([2, 1])

with col_input:
    st.subheader("📥 Data Input")
    tab1, tab2 = st.tabs(["Upload Image", "Labmentix Dataset Samples"])
    
    img = None
    
    with tab1:
        uploaded_file = st.file_uploader("Drop surveillance frame here...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            
    with tab2:
        # NOTE: Create a folder named 'samples' on GitHub and upload test images there
        if os.path.exists("samples"):
            sample_files = [f for f in os.listdir("samples") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_files:
                selected_sample = st.selectbox("Select a benchmark image:", sample_files)
                img = Image.open(os.path.join("samples", selected_sample))
            else:
                st.info("Upload images to the 'samples/' folder on GitHub to see them here.")
        else:
            st.warning("Sample folder not found on GitHub. Using manual upload only.")

with col_status:
    st.subheader("📋 System Status")
    if cnn_model and yolo_model:
        st.success("AI Core: Online")
    else:
        st.error("AI Core: Offline (Check Model Files)")
    st.info("Environment: Python 3.12")
    st.info("Architecture: CNN + YOLOv8 Ensemble")

# --- ANALYSIS ENGINE ---
if img is not None:
    st.divider()
    
    # 1. Processing for Classification
    img_resized = img.resize((224, 224))
    img_arr = image.img_to_array(img_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # 2. Prediction Logic
    res_cnn = cnn_model.predict(img_arr)[0]
    res_transfer = transfer_model.predict(img_arr)[0]
    results_yolo = yolo_model(img)
    
    classes = ['Bird', 'Drone']
    
    # Robust Check for Transfer Learning Output (Fixes IndexError)
    if len(res_transfer) == 1:
        drone_prob = float(res_transfer[0])
        bird_prob = 1.0 - drone_prob
        transfer_scores = [bird_prob, drone_prob]
    else:
        transfer_scores = res_transfer

    final_idx = np.argmax(transfer_scores)
    label = classes[final_idx]
    conf = transfer_scores[final_idx] * 100

    # --- RESULTS DASHBOARD ---
    # Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Target Identified", label)
    m2.metric("System Confidence", f"{conf:.2f}%")
    m3.metric("Threat Status", "⚠️ HIGH ALERT" if label == "Drone" else "✅ CLEAR")

    st.write(" ")

    # Bottom Row: Visuals
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='info-card'><h3>Localization (YOLOv8)</h3></div>", unsafe_allow_html=True)
        res_plotted = results_yolo[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)
        st.write(f"Detected Objects in Frame: {len(results_yolo[0].boxes)}")

    with c2:
        st.markdown("<div class='info-card'><h3>Multi-Model Logic Check</h3></div>", unsafe_allow_html=True)
        # Compare both models in a dataframe
        comparison_data = pd.DataFrame({
            "Confidence": [float(transfer_scores[0]), float(transfer_scores[1])],
            "Object Class": ["Bird", "Drone"]
        }).set_index("Object Class")
        
        st.bar_chart(comparison_data)
        
        # Details Table
        st.table(pd.DataFrame({
            "Model Architecture": ["Custom CNN", "Transfer Learning"],
            "Bird Score": [f"{res_cnn[0]*100:.2f}%", f"{transfer_scores[0]*100:.2f}%"],
            "Drone Score": [f"{res_cnn[1]*100:.2f}%", f"{transfer_scores[1]*100:.2f}%"]
        }))

else:
    st.write("---")
    st.markdown("### 🛰️ System Ready. Awaiting Airspace Data.")

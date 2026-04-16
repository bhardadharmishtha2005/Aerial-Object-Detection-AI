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
st.set_page_config(page_title="Aerial Object Detection Intelligence", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    try:
        cnn = load_model('best_model_custom_cnn.keras') if os.path.exists('best_model_custom_cnn.keras') else None
        transfer = load_model('best_model_transfer_learning.keras') if os.path.exists('best_model_transfer_learning.keras') else None
        yolo_m = YOLO('best.pt') if os.path.exists('best.pt') else None
        return cnn, transfer, yolo_m
    except Exception as e:
        return None, None, None

cnn, transfer, yolo = load_all_models()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ SkyGuard Ops")
    st.write("---")
    input_choice = st.radio("Source:", ["Manual Upload", "Sample Dataset"])
    # This slider helps manually tune how strict the AI is
    conf_threshold = st.slider("Min Confidence %", 0.0, 1.0, 0.5)
    st.write("---")

# --- MAIN INTERFACE ---
st.title("🛡️ Aerial Object detection Intelligence")
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
        st.info("💡 Create a folder named 'samples' in your GitHub to use this feature.")

# --- ANALYTICS ENGINE ---
if img is not None:
    if cnn and transfer and yolo:
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing Surveillance Data...'):
            res_transfer = transfer.predict(img_array, verbose=0)[0]
            yolo_results = yolo(img, conf=conf_threshold, verbose=False)

            # --- CONFIDENCE GUARD LOGIC ---
            CERTAINTY_LIMIT = 0.65 # Require at least 65% confidence
            classes = ['Bird', 'Drone']
            
            # Standardize score index
            scores = res_transfer if len(res_transfer) > 1 else [1.0 - res_transfer[0], res_transfer[0]]
            final_idx = np.argmax(scores)
            raw_label = classes[final_idx]
            final_conf = float(scores[final_idx])

            # Check if YOLO found anything
            yolo_detected = len(yolo_results[0].boxes) > 0

            # OVERRIDE: If confidence is too low AND YOLO sees nothing, it's Unwanted
            if final_conf < CERTAINTY_LIMIT and not yolo_detected:
                final_label = "Unwanted / Background"
                display_conf = final_conf 
            else:
                final_label = raw_label
                display_conf = final_conf

            # --- DISPLAY METRICS ---
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Target Status", final_label)
            with m2: st.metric("System Confidence", f"{display_conf*100:.2f}%")
            with m3:
                if final_label == "Invalid image":
                    st.info("⏸️ NO THREAT")
                elif final_label == "Drone" and display_conf > 0.70:
                    st.error("🚨 THREAT DETECTED")
                else:
                    st.success("✅ AIRSPACE CLEAR")

           # --- VISUALS ---
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("📍 YOLOv8 Localization")
                res_plotted = yolo_results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)
          
            with c2:
                st.subheader("📊 Probability Distribution")
                
                # Logic to clear the chart if the result is unwanted
                if final_label == "Unwanted / Background":
                    chart_scores = [0.0, 0.0]
                else:
                    chart_scores = [float(scores[0]), float(scores[1])]
                    
                chart_data = pd.DataFrame({
                    "Confidence": chart_scores, 
                    "Object": ["Bird", "Drone"]
                }).set_index("Object")
                
                st.bar_chart(chart_data)
    else:
        st.error("Critical System Failure: Model files not found.")
else:
    st.info("System Ready. Please provide input to begin monitoring.")

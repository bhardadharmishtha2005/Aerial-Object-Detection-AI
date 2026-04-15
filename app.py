import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="Aerial Object Detection", layout="wide")
st.title("🚁 Aerial Object Detection & Classification")
st.markdown("### Bird vs. Drone Detection Dashboard - Labmentix Internship")

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models():
    # Load Classification Models
    cnn_model = load_model('best_model_custom_cnn.keras')
    transfer_model = load_model('best_model_transfer_learning.keras')
    # Load YOLOv8 Detection Model
    yolo_model = YOLO('best.pt')
    return cnn_model, transfer_model, yolo_model

try:
    cnn, transfer, yolo = load_all_models()
    st.sidebar.success("✅ All models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")

# --- SIDEBAR & UPLOAD ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original Image
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selected Image")
        st.image(img, use_container_width=True)

    # --- CLASSIFICATION LOGIC ---
    # Preprocess for CNN/Transfer Learning
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predictions
    pred_cnn = cnn.predict(img_array)
    pred_transfer = transfer.predict(img_array)
    
    classes = ['Bird', 'Drone']

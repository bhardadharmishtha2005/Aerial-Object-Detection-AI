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
st.markdown("### Bird vs. Drone Detection Dashboard")

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
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")

# --- SIDEBAR & UPLOAD ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to PIL and Display
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Classification Results")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess for Keras Models
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Run Predictions
        res_cnn = cnn.predict(img_array)
        res_transfer = transfer.predict(img_array)
        
        classes = ['Bird', 'Drone']
        
        st.write(f"**Custom CNN Prediction:** {classes[np.argmax(res_cnn)]}")
        st.write(f"**Transfer Learning Prediction:** {classes[np.argmax(res_transfer)]}")

    with col2:
        st.subheader("2. YOLOv8 Real-Time Detection")
        # Run YOLOv8 inference
        results = yolo(img)
        
        # Plot the results on the image
        res_plotted = results[0].plot()
        # Convert BGR to RGB for Streamlit
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        st.image(res_plotted_rgb, caption="YOLOv8 Detection", use_container_width=True)
        
        # Show detection count
        st.write(f"Objects detected: {len(results[0].boxes)}")

else:
    st.info("Please upload an image to start the detection.")

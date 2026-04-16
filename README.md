# Aerial Object Detection Intelligence 🛰️

##  Overview
**Aerial Object Detection Intelligence** is a multi-modal AI dashboard designed for airspace security. It differentiates between authorized aerial activity and potential threats using a layered detection approach.

##  Key Features
- **Localization:** Powered by YOLOv8 for precise real-time bounding box detection.
- **Classification:** Dual-model architecture (Custom CNN & Transfer Learning) for robust object identification.
- **Intelligent Filtering:** Custom post-processing logic to identify and suppress "Unwanted/Background" noise, ensuring only relevant targets are flagged.
- **Live Analytics:** Real-time probability distribution charts and system status indicators.

##  Tech Stack
- **AI/ML:** Ultralytics (YOLO), TensorFlow/Keras
- **UI:** Streamlit
- **Backend:** Python
- **Environment:** Headless OpenCV for Cloud Deployment

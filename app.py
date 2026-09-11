import os

# ============================================================
# STREAMLIT CLOUD / HEADLESS SETTINGS
# ============================================================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Aerial Object Detection Intelligence",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #777;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# YOLO MODEL LOADING
# ============================================================
@st.cache_resource
def load_yolo_model():

    model_path = "best.pt"

    if not os.path.exists(model_path):
        return None

    try:
        model = YOLO(model_path)
        return model

    except Exception as e:
        st.error(f"Unable to load YOLO model: {e}")
        return None


yolo = load_yolo_model()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.title("🛡️ SkyGuard Ops")

    st.write("---")

    st.subheader("Input Source")

    input_choice = st.radio(
        "Select source:",
        ["Manual Upload", "Sample Dataset"]
    )

    st.write("---")

    st.subheader("Detection Settings")

    conf_threshold = st.slider(
        "Minimum Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    st.write("---")

    st.caption("YOLO Aerial Object Detection System")
    st.caption("CPU optimized for Streamlit Cloud")


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🛡️ Aerial Object Detection Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise-Grade Drone & Bird Monitoring System</div>',
    unsafe_allow_html=True
)

st.write("---")


# ============================================================
# MODEL STATUS
# ============================================================
if yolo is None:

    st.error(
        "🚨 YOLO model not found. "
        "Please make sure 'best.pt' is present in your GitHub repository."
    )

    st.stop()


# ============================================================
# IMAGE INPUT
# ============================================================
img = None
image_source_name = None


if input_choice == "Manual Upload":

    uploaded_file = st.file_uploader(
        "📤 Upload surveillance frame",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            img = Image.open(uploaded_file).convert("RGB")
            image_source_name = uploaded_file.name

        except Exception as e:

            st.error(f"Unable to open image: {e}")
            st.stop()


else:

    sample_path = "samples"

    if os.path.exists(sample_path):

        sample_files = [
            f
            for f in os.listdir(sample_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        if sample_files:

            selected = st.selectbox(
                "🖼️ Select Benchmark Image",
                sample_files
            )

            try:

                img = Image.open(
                    os.path.join(sample_path, selected)
                ).convert("RGB")

                image_source_name = selected

            except Exception as e:

                st.error(f"Unable to open sample image: {e}")
                st.stop()

        else:

            st.info(
                "💡 The 'samples' folder exists but contains no images."
            )

    else:

        st.info(
            "💡 Create a folder named 'samples' in your GitHub "
            "repository to use the Sample Dataset option."
        )


# ============================================================
# DETECTION ENGINE
# ============================================================
if img is not None:

    st.write("---")

    st.subheader("📷 Input Surveillance Frame")

    st.image(
        img,
        caption=f"Input: {image_source_name}",
        use_container_width=True
    )

    # --------------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------------
    with st.spinner("🔍 Analyzing surveillance data..."):

        try:

            results = yolo.predict(
                source=img,
                conf=conf_threshold,
                verbose=False,
                device="cpu"
            )

        except Exception as e:

            st.error(
                f"❌ YOLO detection failed: {e}"
            )

            st.stop()


    # ========================================================
    # RESULT PROCESSING
    # ========================================================
    result = results[0]

    boxes = result.boxes

    detected_count = len(boxes)

    detected_objects = []

    if detected_count > 0:

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names.get(
                class_id,
                f"Class {class_id}"
            )

            detected_objects.append(
                {
                    "Object": class_name,
                    "Confidence": confidence
                }
            )


    # ========================================================
    # DETECTION SUMMARY
    # ========================================================
    st.write("---")

    st.subheader("📊 Detection Summary")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Objects Detected",
            detected_count
        )

    with m2:

        if detected_count > 0:

            highest_confidence = max(
                item["Confidence"]
                for item in detected_objects
            )

            st.metric(
                "Highest Confidence",
                f"{highest_confidence * 100:.2f}%"
            )

        else:

            st.metric(
                "Highest Confidence",
                "0%"
            )

    with m3:

        if detected_count > 0:

            object_names = [
                item["Object"]
                for item in detected_objects
            ]

            unique_objects = len(
                set(object_names)
            )

            st.metric(
                "Object Types",
                unique_objects
            )

        else:

            st.metric(
                "Object Types",
                0
            )


    # ========================================================
    # THREAT STATUS
    # ========================================================
    if detected_count == 0:

        st.success(
            "✅ AIRSPACE CLEAR — No objects detected."
        )

    else:

        object_names = [
            item["Object"].lower()
            for item in detected_objects
        ]

        drone_detected = any(
            "drone" in name
            for name in object_names
        )

        bird_detected = any(
            "bird" in name
            for name in object_names
        )

        if drone_detected:

            st.error(
                "🚨 DRONE DETECTED — Potential airspace threat."
            )

        elif bird_detected:

            st.warning(
                "🦅 BIRD DETECTED — Object identified as bird."
            )

        else:

            st.info(
                "🔎 OBJECT DETECTED — Review localization results."
            )


    # ========================================================
    # VISUALIZATION
    # ========================================================
    st.write("---")

    col1, col2 = st.columns([1.6, 1])

    # --------------------------------------------------------
    # YOLO LOCALIZATION
    # --------------------------------------------------------
    with col1:

        st.subheader("📍 YOLO Localization")

        plotted_image = result.plot()

        st.image(
            plotted_image,
            channels="BGR",
            caption="Detected Objects",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PROBABILITY / CONFIDENCE
    # --------------------------------------------------------
    with col2:

        st.subheader("📈 Detection Confidence")

        if detected_count > 0:

            chart_data = pd.DataFrame(
                detected_objects
            )

            chart_data["Confidence (%)"] = (
                chart_data["Confidence"] * 100
            ).round(2)

            chart_data = chart_data[
                ["Object", "Confidence (%)"]
            ]

            chart_data = chart_data.set_index(
                "Object"
            )

            st.bar_chart(
                chart_data
            )

        else:

            st.info(
                "No detection confidence data available."
            )


    # ========================================================
    # DETECTION TABLE
    # ========================================================
    if detected_count > 0:

        st.write("---")

        st.subheader("📋 Detection Details")

        table_data = pd.DataFrame(
            detected_objects
        )

        table_data["Confidence"] = (
            table_data["Confidence"] * 100
        ).round(2).astype(str) + "%"

        table_data.insert(
            0,
            "No.",
            range(1, len(table_data) + 1)
        )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )


else:

    st.info(
        "🛰️ System Ready — "
        "Upload an aerial image to begin monitoring."
    )


# ============================================================
# FOOTER
# ============================================================
st.write("---")

st.caption(
    "🛡️ SkyGuard Ops | Aerial Object Detection Intelligence"
)

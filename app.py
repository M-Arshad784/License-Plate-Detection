import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="License Plate Detection AI",
    page_icon="🚘",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

model = YOLO(model_path)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #080d19;
}

.block-container {
    max-width: 1150px;
    padding-top: 40px;
    padding-bottom: 50px;
}


/* Main title */

.main-title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 35px;
}


/* Badge */

.badge-container {
    text-align: center;
    margin-bottom: 15px;
}

.badge {
    color: #7dd3fc;
    font-size: 14px;
    font-weight: 600;
}


/* Section */

.section-title {
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}


/* Result */

.result-box {
    background-color: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-title {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-name {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 10px 0;
}

.result-confidence {
    color: #7dd3fc;
    font-size: 18px;
    font-weight: 700;
}


/* Detection information */

.detection-box {
    background-color: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="badge-container">'
    '<div class="badge">🧠 Deep Learning • Computer Vision • Object Detection</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🚘 License Plate Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered License Plate Detection using YOLO'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# TWO COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1.1, 0.9],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-title">📤 Upload Vehicle Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Upload a vehicle image and let the trained YOLO model '
        'detect the license plate automatically.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    uploaded_image = st.file_uploader(
        "Choose Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption(
        "Maximum file size: 200 MB • JPG, JPEG, PNG"
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Vehicle Image",
            use_container_width=True
        )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.markdown(
        '<div class="section-title">🧠 AI Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'The trained YOLO object detection model will analyze '
        'the image and identify the license plate with a '
        'confidence score.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if uploaded_image is None:

        st.info(
            "📤 Upload a vehicle image to start AI detection."
        )

    else:

        if st.button(
            "🔍 Detect License Plate",
            use_container_width=True
        ):

            with st.spinner("Detecting license plate..."):

                # Run YOLO prediction
                results = model(image)

                # Get first result
                result = results[0]

                # Get image with detections
                detected_image = result.plot()

                # Get number of detections
                detection_count = len(result.boxes)

            # ====================================================
            # DISPLAY DETECTED IMAGE
            # ====================================================

            st.write("")

            st.subheader("Detection Result")

            st.image(
                detected_image,
                caption="License Plate Detection",
                use_container_width=True
            )

            # ====================================================
            # DISPLAY DETECTION INFORMATION
            # ====================================================

            if detection_count > 0:

                st.markdown(
                    '<div class="detection-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="result-title">'
                    'LICENSE PLATE DETECTED'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-name">'
                    f'🚘 License Plate'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Get confidence of the first detection
                confidence = result.boxes.conf[0].item()

                st.markdown(
                    f'<div class="result-confidence">'
                    f'Confidence: {confidence * 100:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    confidence,
                    text=f"Detection Confidence: {confidence * 100:.2f}%"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    "❌ No license plate detected in this image."
                )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.info(
    "ℹ️ This application uses a YOLO-based object detection "
    "model trained to identify license plates in vehicle images."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Built with ❤️ using Python • Streamlit • Ultralytics YOLO'
    '</div>',
    unsafe_allow_html=True
)

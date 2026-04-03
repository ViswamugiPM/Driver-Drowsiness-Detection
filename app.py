import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("models/mobilenet_model.h5")

# ---------------- CUSTOM CSS (ANIMATION + STYLE) ----------------
st.markdown("""
    <style>

    /* Background gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Title glow */
    h1 {
        text-align: center;
        color: #00ffcc;
        text-shadow: 0px 0px 20px #00ffcc;
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from {text-shadow: 0 0 10px #00ffcc;}
        to {text-shadow: 0 0 25px #00ffaa;}
    }

    /* Upload box */
    .stFileUploader {
        background-color: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🚗 Driver Drowsiness Detection System")
st.subheader("AI-powered road safety assistant")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 System Info")
st.sidebar.write("✔ CNN + MobileNet Model")
st.sidebar.write("✔ Real-time Image Analysis")
st.sidebar.write("✔ Drowsiness Detection Alert System")

st.sidebar.markdown("---")
st.sidebar.info("Upload a driver face image to test the system")

# ---------------- MAIN SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📤 Upload Driver Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

with col2:
    st.markdown("## 🚦 Live System Status")

    status_placeholder = st.empty()

    for i in range(3):
        status_placeholder.markdown("⏳ Analyzing driver state...")
        time.sleep(0.5)
        status_placeholder.markdown("🔍 Processing eye patterns...")
        time.sleep(0.5)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Driver Image", use_container_width=True)

    # preprocess
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)

    st.markdown("---")
    st.markdown("## 🎯 Result")

    result_box = st.empty()

    if prediction[0][0] > 0.5:

        result_box.error("😴 DROWSY DRIVER DETECTED!")

        st.markdown("""
        <div style="
            padding:20px;
            background-color:#ff4b4b;
            border-radius:15px;
            text-align:center;
            font-size:20px;
            font-weight:bold;
        ">
        🚨 ALERT! PLEASE TAKE A BREAK 🚨
        </div>
        """, unsafe_allow_html=True)

    else:

        result_box.success("😐 DRIVER IS ALERT")

        st.markdown("""
        <div style="
            padding:20px;
            background-color:#00c853;
            border-radius:15px;
            text-align:center;
            font-size:20px;
            font-weight:bold;
        ">
        ✔ SAFE DRIVING STATUS
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🚗 AI Driver Safety System | Built with Streamlit & Deep Learning</center>",
    unsafe_allow_html=True
)
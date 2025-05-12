import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time

# Load pre-trained model
model = load_model(r"C:\Users\FINRISE\Desktop\mask_detection_app\model\mask_detector.h5")
labels = ['No Mask', 'Mask']

# Streamlit UI
st.title("😷 Mask Detection App")
st.write("Real-time mask detection using an IP camera stream.")

# Placeholder for video frames
frame_window = st.empty()

# Initialize session state
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False

def capture_video():
    cap = cv2.VideoCapture("http://192.168.68.53:8080/video")

    if not cap.isOpened():
        st.error("❌ Failed to open video stream. Please check your IP and network connection.")
        return
    else:
        st.success("✅ Video stream opened successfully.")

    frame_count = 0

    while not st.session_state.stop_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame from the stream.")
            break

        # Preprocessing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_input = np.expand_dims(image_resized, axis=0) / 255.0

        # Prediction
        prediction = model.predict(image_input)
        label = labels[np.argmax(prediction)]

        # Display label on frame
        cv2.putText(image, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame in Streamlit
        frame_window.image(image, channels="RGB", use_column_width=True)

        time.sleep(0.05)  # Small delay to reduce CPU usage

        frame_count += 1

    cap.release()

# Start and Stop buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Mask Detection"):
        st.session_state.stop_detection = False
        capture_video()

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.stop_detection = True

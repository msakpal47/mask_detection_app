# detect_mask_video.py

import cv2
print(cv2.__version__)

import numpy as np
from tensorflow.keras.models import load_model

# Load trained mask detection model
model = load_model(r"C:\Users\FINRISE\Desktop\mask_detection_app\model\mask_detector.h5")

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict
        pred = model.predict(face_input)[0][0]
        label = "Mask 😷" if pred < 0.5 else "No Mask 😡"
        color = (0, 255, 0) if label == "Mask 😷" else (0, 0, 255)

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

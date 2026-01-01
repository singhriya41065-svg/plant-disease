import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["YOLO_VERBOSE"] = "False"

import streamlit as st
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import tempfile

# ================= LOAD MODELS =================
cnn_model = tf.keras.models.load_model("best_model.h5")
yolo = YOLO("yolov8n.pt")

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# ================= LEAF COLOR DETECTION =================
def is_leaf_by_color(image, threshold=0.18):
    import cv2  # ✅ lazy import

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    yellow_lower = np.array([20, 40, 40])
    yellow_upper = np.array([35, 255, 255])

    brown_lower = np.array([10, 50, 20])
    brown_upper = np.array([20, 255, 200])

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    leaf_mask = green_mask | yellow_mask | brown_mask
    leaf_ratio = np.sum(leaf_mask > 0) / (image.shape[0] * image.shape[1])

    return leaf_ratio > threshold

# ================= YOLO OBJECT DETECTION =================
def detect_object_yolo(image):
    results = yolo(image)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return "Unknown", None

    box = results.boxes[0]
    cls_id = int(box.cls[0])
    label = yolo.names[cls_id]

    return label, box.xyxy[0]

# ================= CNN PREDICTION =================
def cnn_predict(image):
    import cv2  # ✅ lazy import

    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)
    idx = np.argmax(pred)
    conf = np.max(pred)

    plant, disease = class_names[idx].split("___")
    return plant, disease, conf

# ================= STREAMLIT UI =================
st.set_page_config("Plant Leaf Detection", "🌿")
st.title("🌿 Plant Disease Prediction")

mode = st.selectbox("Select Input Type", ["Image", "Video", "Camera"])

# ================= IMAGE =================
if mode == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        frame = np.array(image)

        if is_leaf_by_color(frame):
            plant, disease, conf = cnn_predict(frame)
            st.success("✅ Leaf Detected")
            st.write(f"🌱 Plant: {plant}")
            st.write(f"🦠 Disease: {disease}")
            st.write(f"📊 Confidence: {conf*100:.2f}%")
        else:
            label, box = detect_object_yolo(frame)
            st.error(f"❌ Not a leaf. Detected: {label.upper()}")

        st.image(frame, width=700)

# ================= VIDEO =================
elif mode == "Video":
    import cv2  # ✅ only loads when needed

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video_file.read())

        cap = cv2.VideoCapture(temp.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if is_leaf_by_color(frame_rgb):
                plant, disease, _ = cnn_predict(frame_rgb)
                text = f"LEAF | {plant} | {disease}"
                color = (0, 255, 0)
            else:
                label, box = detect_object_yolo(frame_rgb)
                text = f"NOT LEAF: {label}"
                color = (255, 0, 0)

            cv2.putText(frame_rgb, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stframe.image(frame_rgb, width=700)

        cap.release()

# ================= CAMERA =================
elif mode == "Camera":
    import cv2  # ✅ local only

    st.warning("⚠ Camera works only on local system, not Streamlit Cloud.")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if is_leaf_by_color(frame_rgb):
            plant, disease, _ = cnn_predict(frame_rgb)
            text = f"LEAF | {plant} | {disease}"
            color = (0, 255, 0)
        else:
            label, _ = detect_object_yolo(frame_rgb)
            text = f"NOT LEAF: {label}"
            color = (255, 0, 0)

        cv2.putText(frame_rgb, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        stframe.image(frame_rgb, width=700)

    cap.release()

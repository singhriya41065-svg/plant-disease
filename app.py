import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------- LOAD MODELS ---------------- #
cnn_model = tf.keras.models.load_model("best_model.h5")
yolo = YOLO("yolov8n.pt")  # COCO model

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- FUNCTIONS ---------------- #
def cnn_predict(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    idx = np.argmax(pred)
    conf = np.max(pred)

    plant, disease = class_names[idx].split("___")
    return plant, disease, conf


def coco_detect(image):
    results = yolo(image)[0]

    if len(results.boxes) == 0:
        return None  # assume leaf

    box = results.boxes[0]
    cls_id = int(box.cls[0])
    label = yolo.names[cls_id]
    return label, box.xyxy[0]


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config("Plant Disease Detection", "🌿")
st.title("🌿 Plant Leaf Disease Detection")

option = st.selectbox(
    "Select Input Type",
    ["Image", "Video", "Camera"]
)

# ================= IMAGE ================= #
if option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        frame = np.array(image)

        detection = coco_detect(frame)

        if detection is not None:
            label, box = detection
            st.error(f"❌ Not a leaf. Detected object: **{label.upper()}**")

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            st.image(frame, width=700)

        else:
            plant, disease, conf = cnn_predict(frame)

            st.success(f"🌱 Plant: {plant}")
            st.warning(f"🦠 Disease: {disease}")
            st.info(f"📊 Confidence: {conf*100:.2f}%")
            st.image(frame, width=700)

# ================= VIDEO ================= #
elif option == "Video":
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

            detection = coco_detect(frame)

            if detection is not None:
                label, box = detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(
                    frame,
                    f"NOT LEAF: {label}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )
            else:
                plant, disease, conf = cnn_predict(frame)
                cv2.putText(
                    frame,
                    f"{plant} | {disease}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            stframe.image(frame, channels="BGR", width=700)

        cap.release()

# ================= CAMERA ================= #
elif option == "Camera":
    st.warning("Press CTRL+C in terminal to stop camera")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection = coco_detect(frame)

        if detection is not None:
            label, box = detection
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(
                frame,
                f"NOT LEAF: {label}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )
        else:
            plant, disease, conf = cnn_predict(frame)
            cv2.putText(
                frame,
                f"{plant} | {disease}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        stframe.image(frame, channels="BGR", width=700)

    cap.release()

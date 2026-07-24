import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mask_detector.keras")

_frame_cache = {"faces": [], "boxes": [], "counter": 0}


def load_models():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Mask detector model not found at {MODEL_PATH}. Run train.py first."
        )

    print("[*] Loading mask classifier model...")
    mask_model = tf.keras.models.load_model(MODEL_PATH)

    print("[*] Loading face detector (Haar Cascade)...")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade not found at {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    return mask_model, face_cascade


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def process_frame(frame, mask_model, face_cascade, skip=2):
    _frame_cache["counter"] += 1
    should_detect = _frame_cache["counter"] % (skip + 1) == 0

    if should_detect:
        faces = detect_faces(frame, face_cascade)
        boxes = []
        for box in faces:
            label, conf = predict_mask(frame, box, mask_model)
            boxes.append((box, label, conf))
        _frame_cache["faces"] = faces
        _frame_cache["boxes"] = boxes
    else:
        boxes = _frame_cache["boxes"]

    for box, label, conf in boxes:
        draw_results(frame, box, label, conf)


def predict_mask(frame, face_box, mask_model):
    x1, y1, x2, y2 = face_box
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return "unknown", 0.0

    face_roi = cv2.resize(face_roi, (224, 224))
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = np.expand_dims(face_roi, axis=0)

    pred = mask_model.predict_on_batch(face_roi)[0]
    label_idx = np.argmax(pred)
    confidence = pred[label_idx]
    label = "Mask" if label_idx == 0 else "No Mask"
    return label, float(confidence)


def draw_results(frame, face_box, label, confidence):
    x1, y1, x2, y2 = face_box
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}: {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

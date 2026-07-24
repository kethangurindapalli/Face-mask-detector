import os
import cv2
from flask import Flask, render_template, Response
from utils import load_models, process_frame

app = Flask(__name__)

mask_model = None
face_net = None
camera = None


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)
    return camera


def generate_frames():
    global mask_model, face_net
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        process_frame(frame, mask_model, face_net, skip=2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def main():
    global mask_model, face_net
    print("[*] Loading models...")
    mask_model, face_net = load_models()
    print("[*] Starting Flask server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()

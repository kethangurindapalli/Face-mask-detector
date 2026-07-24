import cv2
from utils import load_models, detect_faces, predict_mask, draw_results


def main():
    print("[*] Loading models...")
    mask_model, face_net = load_models()

    print("[*] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot access webcam")
        return

    print("[*] Detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame, face_net)

        for box in faces:
            label, conf = predict_mask(frame, box, mask_model)
            draw_results(frame, box, label, conf)

        cv2.imshow("Face Mask Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()

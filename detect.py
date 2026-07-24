import cv2
from utils import load_models, process_frame


def main():
    print("[*] Loading models...")
    mask_model, face_net = load_models()

    print("[*] Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[!] Cannot access webcam")
        return

    print("[*] Detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        process_frame(frame, mask_model, face_net, skip=2)

        cv2.imshow("Face Mask Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()

import os
import sys
import urllib.request
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FACE_DETECTOR_DIR = os.path.join(BASE_DIR, "face_detector")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def download_file(url, dest, desc="Downloading"):
    if os.path.exists(dest):
        print(f"[OK] {desc} - already exists, skipping")
        return
    print(f"[*] {desc}...")
    urllib.request.urlretrieve(url, dest)
    print(f"[OK] {desc} - done")


def setup_face_detector():
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    prototxt_path = os.path.join(FACE_DETECTOR_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    os.makedirs(FACE_DETECTOR_DIR, exist_ok=True)
    download_file(prototxt_url, prototxt_path, "Downloading deploy.prototxt")
    download_file(caffemodel_url, caffemodel_path, "Downloading face detector model (10 MB)")


def setup_dataset_via_kagglehub():
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub not installed. Run: pip install kagglehub")
        return False

    os.makedirs(DATASET_DIR, exist_ok=True)

    dataset_path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
    print(f"[*] Dataset downloaded to: {dataset_path}")

    for item in os.listdir(dataset_path):
        src = os.path.join(dataset_path, item)
        dst = os.path.join(DATASET_DIR, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"[OK] Dataset organized in: {DATASET_DIR}")
    return True


def setup_dataset_fallback():
    print("[*] Fallback: preparing manual download instructions...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    for label in ["with_mask", "without_mask"]:
        os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

    print("[!] Auto-download unavailable without kagglehub.")
    print("[!] Option A: pip install kagglehub, then re-run setup_data.py")
    print("[!] Option B: Download manually from:")
    print("    https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
    print(f"    Then extract into: {DATASET_DIR}")
    print("    Structure: dataset/with_mask/*.jpg  and  dataset/without_mask/*.jpg")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 50)
    print("  Face Mask Detector - Setup")
    print("=" * 50)

    print("\n[1/2] Setting up face detector...")
    setup_face_detector()

    print("\n[2/2] Downloading dataset...")
    success = setup_dataset_via_kagglehub()
    if not success:
        print("[!] Trying fallback method...")
        setup_dataset_fallback()

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("  Next: run train.py to train the model")
    print("=" * 50)


if __name__ == "__main__":
    main()

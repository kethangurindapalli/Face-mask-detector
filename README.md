# Face Mask Detector

Real-time face mask detection using MobileNetV2, OpenCV, and Flask.

![Python](https://img.shields.io/badge/python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.21-orange)
![OpenCV](https://img.shields.io/badge/opencv-5.0-green)

## Features

- **MobileNetV2** classifier trained on 12K face mask images (~96.6% val accuracy)
- **Haar Cascade** face detection (built into OpenCV, no extra downloads)
- **Two modes**: CLI webcam window or Flask web app (browser-based)
- **Real-time** inference with bounding boxes and confidence scores

## Pipeline

```
Webcam -> Face Detection (Haar Cascade) -> ROI Extraction -> MobileNetV2 -> Mask / No Mask
```

## Setup

```bash
pip install -r requirements.txt
python setup_data.py     # downloads dataset
python train.py          # trains model (takes ~15-30 min)
```

## Usage

**Flask web app** (browser at http://127.0.0.1:5000):
```bash
python app.py
```

**CLI webcam** (press `q` to quit):
```bash
python detect.py
```

**Windows** (if using Python 3.14+):
```bash
py -3.13 app.py
# or
run.bat app
```

## Notes

- Python 3.13 recommended (TensorFlow does not support 3.14+)
- OpenCV 5.0+ removed Caffe DNN support; uses Haar Cascade instead
- Model is trained on the [Face Mask 12K Images dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

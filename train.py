import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mask_detector.keras")
PLOT_PATH = os.path.join(BASE_DIR, "training_plot.png")

INIT_LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 2000


def load_dataset():
    data, labels = [], []

    for category in ["with_mask", "without_mask"]:
        path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(path):
            print(f"[!] Dataset folder not found: {path}")
            print("[!] Run setup_data.py first")
            sys.exit(1)

        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = files[:MAX_IMAGES_PER_CLASS]

        for fname in files:
            fpath = os.path.join(path, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    fpath, target_size=IMG_SIZE
                )
                img = tf.keras.preprocessing.image.img_to_array(img)
                data.append(img)
                labels.append(category)
            except Exception as e:
                print(f"[!] Skipping {fpath}: {e}")

    return np.array(data), np.array(labels)


def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    x = base.output
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model


def plot_training(H, path):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(H.history["loss"], label="train_loss")
    axes[0].plot(H.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(H.history["accuracy"], label="train_acc")
    axes[1].plot(H.history["val_accuracy"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    print(f"[✓] Training plot saved to {path}")
    plt.close()


def main():
    print("[*] Loading dataset...")
    data, labels = load_dataset()
    print(f"[*] Loaded {len(data)} images")

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)

    data = data / 255.0

    trainX, testX, trainY, testY = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    print("[*] Building MobileNetV2 model...")
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print("[*] Training...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("[*] Evaluating...")
    preds = model.predict(testX, batch_size=BATCH_SIZE)
    report = classification_report(
        testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_
    )
    print(report)

    plot_training(H, PLOT_PATH)
    print(f"[✓] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

"""
Load `artifacts/best_model.keras`, export to Keras HDF5 and TFLite, run a sample inference.
Run:
    python export_best_model.py
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

ARTIFACTS = Path("artifacts")
BEST = ARTIFACTS / "best_model.keras"
H5_OUT = ARTIFACTS / "model.h5"
TFLITE_OUT = ARTIFACTS / "model.tflite"
LABEL_MAP = ARTIFACTS / "label_map.json"
DATA_DIR = Path("data")

IMG_SAMPLE = None  # set to a path string to test a particular image

if not BEST.exists():
    raise SystemExit(f"Best model not found: {BEST}. Aborting.")

print(f"Loading best model from: {BEST}")
# Load without compiling to avoid optimizer state issues
model = tf.keras.models.load_model(str(BEST), compile=False)
print("Model loaded.")

# Save Keras HDF5 (.h5)
print(f"Saving Keras HDF5 to: {H5_OUT}")
model.save(str(H5_OUT), include_optimizer=False, save_format="h5")
print("Saved HDF5.")

# TFLite conversion with fallback to SavedModel
print(f"Converting to TFLite: {TFLITE_OUT}")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    TFLITE_OUT.write_bytes(tflite_model)
    print("TFLite conversion (from Keras) succeeded.")
except Exception as e:
    print(f"Direct Keras->TFLite failed: {e}\nFalling back to SavedModel then converting...")
    saved_dir = ARTIFACTS / "saved_model_for_tflite"
    if saved_dir.exists():
        tf.io.gfile.rmtree(str(saved_dir))
    tf.saved_model.save(model, str(saved_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    TFLITE_OUT.write_bytes(tflite_model)
    print("TFLite conversion (from SavedModel) succeeded.")

# Load labels if available
if LABEL_MAP.exists():
    with LABEL_MAP.open("r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = None

# Build preprocessing function consistent with training pipeline
def preprocess_pil_image(img: Image.Image, image_size: int):
    img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # Training used: Rescaling(1/127.5, offset=-1.0) -> (img/127.5)-1.0
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Select a sample image (first available) if none provided
sample_path = None
if IMG_SAMPLE:
    sample_path = Path(IMG_SAMPLE)
else:
    # search data folder for any image
    for cls in sorted(DATA_DIR.iterdir()):
        if cls.is_dir():
            for f in sorted(cls.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    sample_path = f
                    break
        if sample_path:
            break

# Prepare sample input
if sample_path and sample_path.exists():
    print(f"Using sample image: {sample_path}")
    img = Image.open(sample_path)
    input_tensor = preprocess_pil_image(img, model.input_shape[1])
else:
    print("No sample image found in `data/` — running inference on a random tensor.")
    input_tensor = np.random.uniform(-1.0, 1.0, size=(1, model.input_shape[1], model.input_shape[2], 3)).astype(np.float32)

# Run inference
print("Running inference on the sample...")
logits = model.predict(input_tensor)
probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
pred_idx = int(np.argmax(probs))
pred_prob = float(probs[pred_idx])
if labels:
    pred_label = labels[pred_idx]
    print(f"Predicted: {pred_label} (index {pred_idx}) with probability {pred_prob:.4f}")
else:
    print(f"Predicted index: {pred_idx} with probability {pred_prob:.4f}")

print("Done.")

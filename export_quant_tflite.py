"""Export TFLite models with dynamic-range and (optional) full-integer quantization.

Usage:
    python export_quant_tflite.py --keras artifacts/sign_language_cnn.keras --data_dir data --output artifacts

This script will write:
- artifacts/sign_language_cnn.tflite           (dynamic-range quantized)
- artifacts/sign_language_cnn_int8.tflite      (full-integer quantized using representative dataset)

If representative data is insufficient the int8 export will be skipped.
"""
from pathlib import Path
import argparse
import tensorflow as tf
import numpy as np
import cv2


def representative_dataset_generator(data_dir: Path, image_size: int, max_samples: int = 100):
    paths = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png'))
    if not paths:
        return
    count = 0
    for p in paths:
        if count >= max_samples:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32)
        arr = np.expand_dims(arr, axis=0)
        yield [arr]
        count += 1


def export_dynamic_range(keras_model: tf.keras.Model, out_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print('Wrote dynamic-range TFLite to', out_path)


def export_int8(keras_model: tf.keras.Model, out_path: Path, rep_gen, image_size: int):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # set input/output to int8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print('Wrote int8 TFLite to', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--keras', default='artifacts/sign_language_cnn.keras')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output', default='artifacts')
    p.add_argument('--image_size', type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    keras_path = Path(args.keras)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not keras_path.exists():
        print('Keras model not found:', keras_path)
        return

    keras_model = tf.keras.models.load_model(str(keras_path), compile=False)

    dyn_out = out_dir / 'sign_language_cnn.tflite'
    export_dynamic_range(keras_model, dyn_out)

    # try full integer
    rep_gen = representative_dataset_generator(Path(args.data_dir), args.image_size, max_samples=200)
    try:
        int8_out = out_dir / 'sign_language_cnn_int8.tflite'
        export_int8(keras_model, int8_out, rep_gen, args.image_size)
    except Exception as e:
        print('Skipping int8 export:', e)


if __name__ == '__main__':
    main()

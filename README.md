# Sign Language Recognition

Production-oriented sign language recognition using a lightweight transfer-learning CNN.

## Why this architecture

MobileNetV2 gives a strong accuracy/speed balance for webcam-based gesture recognition. It is lightweight, transfers well to a custom sign-language dataset, and exports cleanly to TFLite for low-latency inference.

MediaPipe hand cropping is used in inference to reduce background noise and improve stability in real time.

## Project structure

- `app.py`: Flask + SocketIO webcam streaming app.
- `train_cnn.py`: Training script.
- `inference_cnn.py`: Real-time webcam inference and reusable predictor.
- `requirements.txt`: Dependencies.
- `Dockerfile`: Deployment container.

## Training

Dataset format:

- `data/A/`
- `data/B/`
- `data/C/`
- one folder per class

Run training:

```bash
python train_cnn.py --data_dir data --output_dir artifacts --use_horizontal_flip
```

Outputs saved in `artifacts/`:

- `sign_language_cnn.keras`
- `sign_language_cnn.tflite`
- `label_map.json`
- `classification_report.json`
- `weak_classes.json`
- `confusion_matrix.png`

## Inference

Standalone webcam demo:

```bash
python inference_cnn.py --model_path artifacts/sign_language_cnn.tflite --labels_path artifacts/label_map.json
```

Press `q` to quit.

## Flask app

Start the web app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

The app streams the webcam feed and emits stable predictions over SocketIO.

## Docker

Build:

```bash
docker build -t sign-language-recognition .
```

Run:

```bash
docker run --rm -p 5000:5000 sign-language-recognition
```

## Performance tips

- Prefer the exported TFLite model for deployment.
- Keep MediaPipe hand cropping enabled for noisy backgrounds.
- If accuracy stalls, fine-tune with more images from weak classes.
- Use a GPU for training if available.
- Collect varied lighting and angle examples for each class.

## Notes

- Horizontal flip augmentation is useful for mirrored webcam data, but disable it if any classes are asymmetrical in a way that changes meaning.
- If you need even lower latency, reduce the smoothing window in `inference_cnn.py`.








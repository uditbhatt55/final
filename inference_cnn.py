from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


class SignLanguageCNNPredictor:
    def __init__(
        self,
        model_path: str = "artifacts/sign_language_cnn.tflite",
        labels_path: str = "artifacts/label_map.json",
        image_size: int = 224,
        smoothing_window: int = 7,
        min_confidence: float = 0.55,
        use_hand_crop: bool = True,
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.use_hand_crop = use_hand_crop
        self.label_names = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.prediction_history: deque[str] = deque(maxlen=smoothing_window)
        self.confidence_history: deque[float] = deque(maxlen=smoothing_window)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if model_path.endswith(".tflite"):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.keras_model = None
        else:
            self.keras_model = tf.keras.models.load_model(model_path)
            self.interpreter = None
            self.input_details = None
            self.output_details = None

    def _crop_hand(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if not self.use_hand_crop:
            return frame_bgr

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        landmarks = results.multi_hand_landmarks[0].landmark
        xs = np.array([landmark.x for landmark in landmarks])
        ys = np.array([landmark.y for landmark in landmarks])

        x_min = max(int(xs.min() * width) - 24, 0)
        y_min = max(int(ys.min() * height) - 24, 0)
        x_max = min(int(xs.max() * width) + 24, width)
        y_max = min(int(ys.max() * height) + 24, height)

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame_bgr[y_min:y_max, x_min:x_max]

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        crop = self._crop_hand(frame_bgr)
        if crop is None:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.float32)[None, ...]

    def _predict_raw(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        input_tensor = self._preprocess(frame_bgr)
        if input_tensor is None:
            return None

        if self.interpreter is not None:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        return self.keras_model.predict(input_tensor, verbose=0)[0]

    def predict(self, frame_bgr: np.ndarray) -> tuple[str | None, float, np.ndarray | None]:
        probabilities = self._predict_raw(frame_bgr)
        if probabilities is None:
            return None, 0.0, None

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = self.label_names[index]

        self.prediction_history.append(label)
        self.confidence_history.append(confidence)

        counts = {name: self.prediction_history.count(name) for name in set(self.prediction_history)}
        best_label = max(counts, key=counts.get)
        stability = counts[best_label] / len(self.prediction_history)
        avg_confidence = float(
            np.mean([conf for name, conf in zip(self.prediction_history, self.confidence_history) if name == best_label])
        )
        smoothed_confidence = avg_confidence * stability

        if smoothed_confidence < self.min_confidence:
            return None, smoothed_confidence, probabilities

        return best_label, smoothed_confidence, probabilities

    def annotate(self, frame_bgr: np.ndarray, label: str | None, confidence: float) -> np.ndarray:
        annotated = frame_bgr.copy()
        if label is None:
            message = "No confident prediction"
            color = (0, 165, 255)
        else:
            message = f"{label} ({confidence * 100:.1f}%)"
            color = (0, 200, 0)

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 54), (18, 18, 18), -1)
        cv2.putText(
            annotated,
            message,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        label, confidence, _ = self.predict(frame_bgr)
        annotated = self.annotate(frame_bgr, label, confidence)
        return annotated, label, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time sign language inference using CNN or TFLite.")
    parser.add_argument("--model_path", default="artifacts/sign_language_cnn.tflite", help="Path to .tflite or .keras model.")
    parser.add_argument("--labels_path", default="artifacts/label_map.json", help="Path to the class mapping JSON file.")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--no_hand_crop", action="store_true", help="Disable MediaPipe hand cropping.")
    return parser.parse_args()


def run_webcam() -> None:
    args = parse_args()
    predictor = SignLanguageCNNPredictor(
        model_path=args.model_path,
        labels_path=args.labels_path,
        use_hand_crop=not args.no_hand_crop,
    )

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            annotated, label, _ = predictor.process_frame(frame)

            if label is not None:
                cv2.putText(
                    annotated,
                    f"Stable: {label}",
                    (18, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Sign Language CNN", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def create_frame_generator(
    camera_index: int = 0,
    model_path: str = "artifacts/sign_language_cnn.tflite",
    labels_path: str = "artifacts/label_map.json",
):
    predictor = SignLanguageCNNPredictor(model_path=model_path, labels_path=labels_path)
    capture = cv2.VideoCapture(camera_index)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            annotated, _, _ = predictor.process_frame(frame)
            success, buffer = cv2.imencode(".jpg", annotated)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        capture.release()


if __name__ == "__main__":
    run_webcam()

import json
import logging
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sign-ai")


# ============================
# Config
# ============================
MODEL_PATH = Path("artifacts") / "best_model.keras"
LABEL_MAP_PATH = Path("artifacts") / "label_map.json"
IMAGE_SIZE = 224
SMOOTH_WINDOW = 5
PRED_THRESHOLD = 0.6

EMIT_MIN_INTERVAL_SEC = 0.08
EMIT_MAX_SILENCE_SEC = 0.25
EMIT_CONF_DELTA = 2.0

GEMINI_MODEL_NAME = "gemini-pro"
# Require GEMINI_API_KEY from environment. Do NOT hardcode keys in source.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ============================
# Gemini Chatbot Setup
# Set GEMINI_API_KEY in environment.
# Example (Linux/macOS): export GEMINI_API_KEY="your_key"
# ============================
gemini_model = None
if genai is None:
    logger.warning("google-generativeai is not installed. /chat will return 503.")
elif GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info("Gemini configured successfully")
    except Exception:
        logger.exception("Failed to initialize Gemini model")
else:
    logger.warning("GEMINI_API_KEY is not set. /chat will return 503 until configured.")


# ===== Predictor integrating best_model.keras + MediaPipe =====
# IMPORTANT CHANGE: Replaced previous landmark-based RandomForest pipeline with
# a CNN-based predictor. Removed old landmark feature extraction, predict_proba,
# and the old label dictionary. This file now crops the hand bbox and feeds the
# crop to the Keras CNN (`artifacts/best_model.keras`).
class RealTimePredictor:
    def __init__(self, model_path: str | Path, label_map_path: str | Path, image_size: int = 224, window: int = 5, threshold: float = 0.6):
        self.image_size = int(image_size)
        self.window = int(window)
        self.threshold = float(threshold)

        # Load model once (no compilation required for inference)
        self.model = tf.keras.models.load_model(str(model_path), compile=False)

        # Load labels
        lm = Path(label_map_path)
        if lm.exists():
            with lm.open('r', encoding='utf-8') as fh:
                self.labels = json.load(fh)
        else:
            self.labels = None

        # MediaPipe hands (fast real-time settings)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # smoothing history deque of tuples (label_or_None, confidence)
        self.history = deque(maxlen=self.window)
        self.last_emitted = None

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB, resize, normalize to (-1,1)
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32)
        arr = (arr / 127.5) - 1.0
        return np.expand_dims(arr, axis=0)

    def _get_smoothed(self):
        # Compute majority label from history (ignore None entries)
        labels = [item[0] for item in self.history if item[0] is not None]
        if not labels:
            return None, 0.0
        most = Counter(labels).most_common(1)[0][0]
        # average confidence for entries with that label
        confs = [c for (l, c) in self.history if l == most]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return most, avg_conf

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        # frame: BGR image from OpenCV
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)
        out_label = None
        out_conf = 0.0

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            x1 = int(max(0, min(xs) * w))
            x2 = int(min(w, max(xs) * w))
            y1 = int(max(0, min(ys) * h))
            y2 = int(min(h, max(ys) * h))

            # add padding
            pad_x = int(0.2 * (x2 - x1))
            pad_y = int(0.2 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                # fallback to whole frame center crop
                crop = frame

            inp = self._preprocess(crop)

            # fast inference: use model call directly
            try:
                logits = self.model(inp, training=False).numpy()
            except Exception:
                logits = self.model.predict(inp, verbose=0)

            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = self.labels[idx] if self.labels else str(idx)

            # if below threshold, treat as None to avoid noise
            if conf < self.threshold:
                label = None
                conf = 0.0

            # update history and smoothing
            self.history.append((label, conf))
            sm_label, sm_conf = self._get_smoothed()

            # determine if we should emit (stable and above threshold)
            if sm_label is not None and sm_label != self.last_emitted:
                # new stable label
                out_label = sm_label
                out_conf = sm_conf
                self.last_emitted = sm_label
            elif sm_label is not None and sm_label == self.last_emitted:
                # same as last emitted, still return it for UI
                out_label = sm_label
                out_conf = sm_conf
            else:
                out_label = None
                out_conf = 0.0

            # annotate (show label and confidence percent)
            display_text = f"{label if label is not None else '-'} {conf*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # no hand detected
            self.history.append((None, 0.0))
            sm_label, sm_conf = self._get_smoothed()
            if sm_label is None:
                out_label = None
                out_conf = 0.0
            else:
                out_label = sm_label
                out_conf = sm_conf

        return frame, out_label, out_conf


predictor = RealTimePredictor(
    model_path=MODEL_PATH,
    label_map_path=LABEL_MAP_PATH,
    image_size=IMAGE_SIZE,
    window=SMOOTH_WINDOW,
    threshold=PRED_THRESHOLD,
)


@app.route('/')
def index():
    return render_template("index.html")


# ============================
# Gemini Chat Route
# ============================
@app.route('/chat', methods=['POST'])
def chat():
    if gemini_model is None:
        return jsonify({
            "reply": "Chatbot is not configured. Set GEMINI_API_KEY in the environment."
        }), 503

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "Please provide a message."}), 400

    try:
        response = gemini_model.generate_content(user_message)
        reply_text = (getattr(response, "text", "") or "").strip()
        if not reply_text:
            reply_text = "I could not generate a response right now."
        return jsonify({"reply": reply_text})
    except Exception:
        logger.exception("Gemini request failed")
        return jsonify({"reply": "Sorry, I ran into an internal error. Please try again."}), 500


# ===== VIDEO STREAM =====
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    last_emit_ts = 0.0
    last_emit_label = None
    last_emit_conf = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            try:
                annotated, label, confidence = predictor.process_frame(frame)
            except Exception:
                logger.exception("Prediction loop error")
                annotated, label, confidence = frame, None, 0.0

            now = time.monotonic()
            conf_pct = float(confidence * 100.0)

            # Socket optimization: limit event flood while keeping UI responsive.
            should_emit = False
            if label is not None:
                label_changed = label != last_emit_label
                conf_changed = abs(conf_pct - last_emit_conf) >= EMIT_CONF_DELTA
                elapsed = now - last_emit_ts

                if label_changed:
                    should_emit = True
                elif elapsed >= EMIT_MIN_INTERVAL_SEC and conf_changed:
                    should_emit = True
                elif elapsed >= EMIT_MAX_SILENCE_SEC:
                    should_emit = True

            if should_emit:
                socketio.emit(
                    "prediction",
                    {"text": label, "confidence": round(conf_pct, 2)},
                )
                last_emit_ts = now
                last_emit_label = label
                last_emit_conf = conf_pct

            ok, buffer = cv2.imencode('.jpg', annotated)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def connect():
    logger.info("Client connected")


@socketio.on('disconnect')
def disconnect():
    logger.info("Client disconnected")


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
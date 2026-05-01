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
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None


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

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
# Require GROQ_API_KEY from environment. Do NOT hardcode keys in source.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.after_request
def apply_security_headers(response):
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response


# ============================
# Groq Chatbot Setup
# Set GROQ_API_KEY in environment.
# Example (Linux/macOS): export GROQ_API_KEY="your_key"
# ============================
groq_client = None
if Groq is None:
    logger.warning("groq is not installed. /chat will return 503.")
elif GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq configured successfully with model: %s", GROQ_MODEL_NAME)
    except Exception:
        logger.exception("Failed to initialize Groq client")
else:
    logger.warning("GROQ_API_KEY is not set. /chat will return 503 until configured.")


# ===== Predictor integrating best_model.keras + MediaPipe =====
# IMPORTANT CHANGE: Replaced previous landmark-based RandomForest pipeline with
# a CNN-based predictor. Removed old landmark feature extraction, predict_proba,
# and the old label dictionary. This file now crops the hand bbox and feeds the
# crop to the Keras CNN (`artifacts/best_model.keras`).
class RealTimePredictor:
    def __init__(self, model_path: str | Path, label_map_path: str | Path, image_size: int = 224, window: int = 5, threshold: float = 0.6, tta_enabled: bool = False):
        self.image_size = int(image_size)
        self.window = int(window)
        self.threshold = float(threshold)
        self.tta_enabled = bool(tta_enabled)

        # Load model once (no compilation required for inference)
        self.model = tf.keras.models.load_model(str(model_path), compile=False)

        # Load labels
        lm = Path(label_map_path)
        if lm.exists():
            with lm.open('r', encoding='utf-8') as fh:
                self.labels = json.load(fh)
        else:
            self.labels = None

        # MediaPipe hands (fast real-time settings). Fall back if API is unavailable.
        self.mp_hands = None
        self.hands = None
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            logger.warning("MediaPipe hand tracker unavailable; using full-frame inference fallback.")

        # smoothing history deque of tuples (label_or_None, confidence)
        self.history = deque(maxlen=self.window)
        self.last_emitted = None

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB and resize. Model handles backbone-specific preprocessing internally.
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32)
        return np.expand_dims(arr, axis=0)

    def _predict_probs(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return raw probability vector for a single frame (no smoothing)."""
        inp = self._preprocess(frame_bgr)
        try:
            outputs = self.model(inp, training=False).numpy()
        except Exception:
            outputs = self.model.predict(inp, verbose=0)
        return outputs[0]

    def _predict_probs_tta(self, frame_bgr: np.ndarray, folds: int = 5) -> np.ndarray:
        """Simple deterministic TTA: original, hflip, small translations and center-scale.

        Returns the averaged probability vector.
        """
        variations = []
        h, w = frame_bgr.shape[:2]

        # original
        variations.append(frame_bgr)

        # horizontal flip
        variations.append(cv2.flip(frame_bgr, 1))

        # center scale slightly (zoom in)
        scale = 1.06
        cx, cy = w // 2, h // 2
        nw, nh = int(w / scale), int(h / scale)
        x1 = max(0, cx - nw // 2)
        y1 = max(0, cy - nh // 2)
        x2 = min(w, x1 + nw)
        y2 = min(h, y1 + nh)
        cropped = frame_bgr[y1:y2, x1:x2]
        if cropped.size:
            variations.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))

        # small translations (left/right/up/down)
        shift = int(min(w, h) * 0.06)
        M_left = np.float32([[1, 0, -shift], [0, 1, 0]])
        M_right = np.float32([[1, 0, shift], [0, 1, 0]])
        M_up = np.float32([[1, 0, 0], [0, 1, -shift]])
        variations.append(cv2.warpAffine(frame_bgr, M_left, (w, h), borderMode=cv2.BORDER_REPLICATE))
        variations.append(cv2.warpAffine(frame_bgr, M_right, (w, h), borderMode=cv2.BORDER_REPLICATE))
        variations.append(cv2.warpAffine(frame_bgr, M_up, (w, h), borderMode=cv2.BORDER_REPLICATE))

        # limit number of variations
        variations = variations[:folds]

        probs_accum = None
        for var in variations:
            p = self._predict_probs(var)
            if probs_accum is None:
                probs_accum = p.copy()
            else:
                probs_accum += p

        probs_accum = probs_accum / float(len(variations))
        return probs_accum

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

    def predict_frame(self, frame_bgr: np.ndarray) -> tuple[str | None, float]:
        # use optional TTA averaging when enabled via instance or via caller
        if self.tta_enabled:
            probs = self._predict_probs_tta(frame_bgr, folds=6)
        else:
            probs = self._predict_probs(frame_bgr)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if self.labels else str(idx)

        if conf < self.threshold:
            return None, 0.0
        return label, conf

    def predict_topk(self, frame_bgr: np.ndarray, k: int = 5, tta: bool = False) -> list[dict]:
        """Return top-k label/prob pairs for debugging.

        Each item: {"label": str, "prob": float, "index": int}
        """
        if tta or self.tta_enabled:
            probs = self._predict_probs_tta(frame_bgr, folds=max(3, k))
        else:
            probs = self._predict_probs(frame_bgr)
        topk_idx = list(np.argsort(probs)[::-1][:k])
        result = []
        for i in topk_idx:
            lbl = self.labels[i] if self.labels else str(i)
            result.append({"index": int(i), "label": lbl, "prob": float(probs[i])})
        return result

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        # frame: BGR image from OpenCV
        h, w, _ = frame.shape
        results = None
        if self.hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
        out_label = None
        out_conf = 0.0

        if results is not None and results.multi_hand_landmarks:
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
                outputs = self.model(inp, training=False).numpy()
            except Exception:
                outputs = self.model.predict(inp, verbose=0)

            probs = outputs[0]
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
            # no hand detected or no hand tracker available; run full-frame fallback inference
            inp = self._preprocess(frame)
            try:
                outputs = self.model(inp, training=False).numpy()
            except Exception:
                outputs = self.model.predict(inp, verbose=0)

            probs = outputs[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = self.labels[idx] if self.labels else str(idx)

            if conf < self.threshold:
                label = None
                conf = 0.0

            self.history.append((label, conf))
            sm_label, sm_conf = self._get_smoothed()
            out_label = sm_label
            out_conf = sm_conf

            display_text = f"{label if label is not None else '-'} {conf*100:.0f}%"
            cv2.putText(frame, display_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame, out_label, out_conf


try:
    predictor = RealTimePredictor(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        image_size=IMAGE_SIZE,
        window=SMOOTH_WINDOW,
        threshold=PRED_THRESHOLD,
        tta_enabled=False,
    )
except Exception:
    logger.exception("Failed to initialize real-time predictor. UI/chat will still run.")
    predictor = None


@app.route('/')
def index():
    return render_template("index.html")


# ============================
# Groq Chat Route
# ============================
@app.route('/chat', methods=['POST'])
def chat():
    if groq_client is None:
        return jsonify({
            "reply": "Chatbot is not configured. Set GROQ_API_KEY in the environment."
        }), 503

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "Please provide a message."}), 400

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.2,
        )
        reply_text = (completion.choices[0].message.content or "").strip()
        if not reply_text:
            reply_text = "I could not generate a response right now."
        return jsonify({"reply": reply_text})
    except Exception:
        logger.exception("Groq request failed for model %s", GROQ_MODEL_NAME)
        return jsonify({"reply": "Sorry, Groq is currently unavailable for this key/model. Try setting GROQ_MODEL or using another API key."}), 502


# ===== VIDEO STREAM =====
def _placeholder_frame(message: str) -> bytes:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (26, 35, 30)
    cv2.putText(canvas, "Sign Language AI", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (220, 240, 230), 3)
    cv2.putText(canvas, message, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (190, 220, 205), 2)
    ok, buffer = cv2.imencode('.jpg', canvas)
    return buffer.tobytes() if ok else b""


def generate_frames():
    if predictor is None:
        msg = "Predictor unavailable in this environment. Chat/UI still active."
        while True:
            frame_bytes = _placeholder_frame(msg)
            if frame_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            time.sleep(0.1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam. Streaming placeholder frame.")
        while True:
            frame_bytes = _placeholder_frame("Webcam not available in this runtime.")
            if frame_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            time.sleep(0.1)

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


@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Predictor unavailable"}), 503

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "Missing image file"}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        # support debug mode to return top-k predictions for troubleshooting
        debug_mode = str(request.args.get("debug", "0")).lower() in ("1", "true", "yes")
        tta_req = str(request.args.get("tta", "0")).lower() in ("1", "true", "yes")
        if debug_mode:
            topk = predictor.predict_topk(frame, k=6, tta=tta_req)
            # also include the highest as primary label/conf (respecting threshold)
            label, confidence = predictor.predict_frame(frame) if not tta_req else (topk[0]["label"], float(topk[0]["prob"]))
        else:
            # allow callers to request TTA for improved single-shot accuracy
            if tta_req:
                probs = predictor._predict_probs_tta(frame, folds=6)
                idx = int(probs.argmax())
                label = predictor.labels[idx] if predictor.labels else str(idx)
                confidence = float(probs[idx])
                if confidence < predictor.threshold:
                    label, confidence = None, 0.0
            else:
                label, confidence = predictor.predict_frame(frame)
    except Exception:
        logger.exception("Predict endpoint failed")
        return jsonify({"error": "Prediction failed"}), 500

    response = {
        "text": label,
        "confidence": round(confidence * 100.0, 2),
    }
    if debug_mode:
        response["topk"] = topk

    return jsonify(response)


@app.route('/collect_misclassified', methods=['POST'])
def collect_misclassified():
    """Save an image the user flags as misclassified for later retraining.

    Expects multipart `image` and optional `correct_label` form field.
    """
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "Missing image file"}), 400

    correct_label = request.form.get("correct_label", "unknown")
    save_dir = Path("data") / "collected"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    filename = f"mis_{correct_label}_{ts}.jpg"
    out_path = save_dir / filename
    data = image_file.read()
    if not data:
        return jsonify({"error": "Empty image file"}), 400

    out_path.write_bytes(data)
    return jsonify({"saved": str(out_path.relative_to(Path.cwd()))}), 201


@socketio.on('connect')
def connect():
    logger.info("Client connected")


@socketio.on('disconnect')
def disconnect():
    logger.info("Client disconnected")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    socketio.run(app, host='0.0.0.0', port=port)
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json

# ===== CONFIG (FAST MODE) =====
WINDOW = 5
CONF_TH = 0.6
HOLD_FRAMES = 3

pred_q = deque(maxlen=WINDOW)
conf_q = deque(maxlen=WINDOW)

last_output = None
stable_count = 0
frame_count = 0
run_seq = 0


def dbg_log(hypothesis_id, location, message, data):
    payload = {
        "sessionId": "99a580",
        "runId": f"pre-fix-{run_seq}",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(__import__("time").time() * 1000),
    }
    try:
        with open("debug-99a580.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #region agent log
dbg_log("H0", "app.py:module_load", "app_module_loaded", {"cwd": __import__("os").getcwd()})
# #endregion

def smooth_pred(p, c):
    pred_q.append(p)
    conf_q.append(c)

    counts = {}
    for x in pred_q:
        counts[x] = counts.get(x, 0) + 1

    best = max(counts, key=counts.get)
    consistency = counts[best] / len(pred_q)

    avg_conf = np.mean([
        conf_q[i] for i, x in enumerate(pred_q) if x == best
    ])

    return best, float(avg_conf * consistency)


# ===== APP =====
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, cors_allowed_origins='*')

# ===== MODEL =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']
classes = model_dict['classes']

# ===== MEDIAPIPE (FAST) =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.route('/')
def index():
    # #region agent log
    dbg_log("H0", "app.py:index", "index_route_hit", {"method": "GET"})
    # #endregion
    return render_template('index.html')


# ===== VIDEO STREAM =====
def generate_frames():
    global last_output, stable_count, frame_count, run_seq

    cap = cv2.VideoCapture(0)
    run_seq += 1
    # #region agent log
    dbg_log("H1", "app.py:generate_frames:start", "camera_stream_started", {"runSeq": run_seq, "cameraOpened": bool(cap.isOpened())})
    # #endregion

    while True:
        success, frame = cap.read()
        if not success:
            # #region agent log
            dbg_log("H1", "app.py:generate_frames:read", "camera_read_failed", {"runSeq": run_seq, "frameCount": frame_count})
            # #endregion
            break

        frame_count += 1

        # 🔥 SKIP FRAMES (FAST)
        if frame_count % 2 != 0:
            continue

        # 🔥 LOW RES = FAST
        frame = cv2.resize(frame, (480, 360))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            try:
                # #region agent log
                dbg_log("H2", "app.py:generate_frames:features", "landmarks_extracted", {"runSeq": run_seq, "featureLen": len(data_aux)})
                # #endregion
                X_input = scaler.transform([np.asarray(data_aux)])
                # #region agent log
                dbg_log("H3", "app.py:generate_frames:scaler", "scaler_transform_ok", {"runSeq": run_seq, "inputShape": list(np.asarray(X_input).shape)})
                # #endregion

                # 🔥 RUN MODEL LESS OFTEN
                if frame_count % 3 != 0:
                    continue

                proba = model.predict_proba(X_input)[0]
                raw_conf = float(np.max(proba))

                if raw_conf < CONF_TH:
                    continue

                pred_idx = int(np.argmax(proba))
                pred_label = classes[pred_idx]
                # #region agent log
                dbg_log("H4", "app.py:generate_frames:predict", "prediction_generated", {"runSeq": run_seq, "predIdx": pred_idx, "predLabel": str(pred_label), "rawConf": raw_conf})
                # #endregion

                sm_label, sm_conf = smooth_pred(pred_label, raw_conf)

                # 🔥 HOLD LOGIC
                if sm_label == last_output:
                    stable_count += 1
                else:
                    stable_count = 0

                last_output = sm_label

                if stable_count < HOLD_FRAMES:
                    continue

                confidence = sm_conf * 100
                character = sm_label

                socketio.emit('prediction', {
                    'text': character,
                    'confidence': confidence
                })
                # #region agent log
                dbg_log("H5", "app.py:generate_frames:emit", "prediction_emitted", {"runSeq": run_seq, "character": str(character), "confidence": float(confidence), "stableCount": stable_count})
                # #endregion

                cv2.putText(
                    frame,
                    f"{character} ({confidence:.1f}%)",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            except Exception as e:
                print("Prediction error:", e)
                # #region agent log
                dbg_log("H2", "app.py:generate_frames:except", "prediction_exception", {"runSeq": run_seq, "error": str(e)})
                # #endregion
        else:
            # #region agent log
            dbg_log("H1", "app.py:generate_frames:no_hand", "no_hand_landmarks", {"runSeq": run_seq, "frameCount": frame_count})
            # #endregion

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def connect():
    print("Client connected")


if __name__ == "__main__":
    socketio.run(app, debug=True)
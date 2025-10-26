import time
import math
import threading
from typing import Dict, List, Any
from collections import deque
from enum import Enum
import os
import io
from functools import wraps

import numpy as np
import cv2
from flask import Flask, render_template, Response, send_file, make_response
from flask_socketio import SocketIO
import mediapipe as mp
from ultralytics import YOLO
import torch

# Optional: speech detection
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# ------------- Flask + SocketIO -------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'aiforsecurity'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

CAMERA_SOURCE = 0  # change if needed

# ------------- Performance knobs -------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

# ------------- Config -------------
# Pose thresholds (degrees)
YAW_THRESH = 26
PITCH_DOWN_THRESH = 65
PITCH_UP_THRESH = -30

# EMA smoothing
EMA_POSE = 0.40
EMA_GAZE = 0.45  # slightly faster gaze response

# Gaze thresholds (normalized -1..1 after calibration)
GAZE_H_ALERT = 1.06  # slightly more sensitive horizontally
GAZE_V_ALERT = 1.5

# Dwell / timing
DISTRACT_DWELL_S = 0.9
AWAY_DWELL_S = 1.2

# Eyes closed alert
EAR_THRESH = 0.18
EYES_CLOSED_ALERT_S = 4.0
EAR_SMOOTH = 0.6
BLINK_SKIP_MS = 180

# Calibration gating
CALIB_HOLD_S = 0.6
CALIB_SAMPLE_STEP_MS = 120
# New: more forgiving calibration stability
CALIB_STABLE_YAW = 14.0
CALIB_STABLE_PITCH = 16.0
CALIB_RELAX_AFTER_S = 10.0     # after this, relax stability
CALIB_RELAX_HOLD_S = 0.35      # shorter hold after relax
CALIB_DYN_YAW_RANGE = 6.0      # if head is steady (low motion), accept
CALIB_DYN_PITCH_RANGE = 8.0

# YOLO detection (threaded)
YOLO_IMGSZ = 416
YOLO_INTERVAL_MS = 350

DEVICE_CLASSES = {  # COCO
    67: "cell phone",
    73: "book",
    63: "laptop",
    65: "remote",
    76: "keyboard",
    64: "mouse",
}

CLASS_THRESH = {
    "cell phone": 0.55,
    "book": 0.50,
    "laptop": 0.50,
    "remote": 0.55,
    "keyboard": 0.55,
    "mouse": 0.62,
}

HOLD_DEVICE_S = 2.0
IGNORE_CLASSES_IN_ALERT = {"keyboard", "mouse", "laptop"}

# Eyes left/right inversion (kept for normalization math)
INVERT_EYES_LR = False

# Optional custom model for earbuds/airpods (put weights next to this script)
CUSTOM_AIDS_WEIGHTS = "cheat_aids.pt"

# Microphone/speech detection
ENABLE_SPEECH_DETECTION = True
SPEECH_HOLD_S = 2.0
SPEECH_ALERT_COOLDOWN_S = 8.0

# Score
SCORE_RECOVER = 0.6
SCORE_MINOR = 1.0
SCORE_MAJOR = 2.2

# Liveliness Check Config
LIVENESS_INTERVAL_S = 180.0   # Every 3 minutes
LIVENESS_WINDOW_S = 30.0      # Response window is 30 seconds
LIVENESS_BLINK_THRESH = 0.17  # easier blink detection
LIVENESS_FIRST_CHECK_S = 45.0 # Time for first check

# ------------- Camera Grabber -------------
class CameraGrabber:
    def __init__(self, src, w=1280, h=720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.q = deque(maxlen=1)
        self.lock = threading.Lock()
        self.stopped = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.q.clear()
                    self.q.append(frame)
            else:
                time.sleep(0.02)

    def read(self):
        with self.lock:
            if len(self.q) == 0:
                return None
            return self.q[-1].copy()

    def release(self):
        self.stopped = True
        try:
            self.t.join(timeout=1)
        except:
            pass
        self.cap.release()

def resize_keep_aspect(img, target_w=640):
    if img is None:
        return None, 0, 0
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return None, 0, 0
    if w <= target_w:
        return img, w, h
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(round(h * scale))
    try:
        out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return out, new_w, new_h
    except:
        return None, 0, 0

# ------------- MediaPipe -------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.45, min_tracking_confidence=0.55
)

mp_fd = mp.solutions.face_detection
face_det = mp_fd.FaceDetection(min_detection_confidence=0.45, model_selection=1)

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.50,  # forgiving to detect raised hand
    min_tracking_confidence=0.45
)

# Landmarks
FACE_LMS = [1, 152, 33, 263, 61, 291]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = {'outer': 33, 'inner': 133, 'top': 159, 'bottom': 145, 'top2': 158, 'bot2': 144}
RIGHT_EYE = {'inner': 362, 'outer': 263, 'top': 386, 'bottom': 374, 'top2': 387, 'bot2': 373}

face_3d_model = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)

def rot_to_euler(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    else:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    return np.degrees([z, y, x])

def get_head_pose(landmarks, w, h, camK, dist):
    pts2d = []
    for idx in FACE_LMS:
        if idx >= len(landmarks): return None, None, None, None
        lm = landmarks[idx]
        pts2d.append([lm.x * w, lm.y * h])
    pts2d = np.array(pts2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(face_3d_model, pts2d, camK, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, None, None, None
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, _ = rot_to_euler(R)
    return yaw, pitch, rvec, tvec

def iris_center(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        if i < len(landmarks):
            lm = landmarks[i]
            pts.append([lm.x*w, lm.y*h])
    if len(pts) < 4:
        return None
    pts = np.array(pts)
    return np.mean(pts, axis=0)

def get_gaze_raw(landmarks, w, h):
    try:
        l_c = iris_center(landmarks, LEFT_IRIS, w, h)
        r_c = iris_center(landmarks, RIGHT_IRIS, w, h)
        if l_c is None or r_c is None:
            return None, None

        l_out = landmarks[LEFT_EYE['outer']]; l_in = landmarks[LEFT_EYE['inner']]
        r_in = landmarks[RIGHT_EYE['inner']]; r_out = landmarks[RIGHT_EYE['outer']]
        l_top = landmarks[LEFT_EYE['top']]; l_bot = landmarks[LEFT_EYE['bottom']]
        r_top = landmarks[RIGHT_EYE['top']]; r_bot = landmarks[RIGHT_EYE['bottom']]

        lw = abs(l_in.x - l_out.x) * w
        rw = abs(r_out.x - r_in.x) * w
        lh = (l_c[0] - l_out.x * w) / (lw + 1e-6) if lw > 4 else 0.5
        rh = (r_c[0] - r_in.x * w) / (rw + 1e-6) if rw > 4 else 0.5

        lhgt = abs(l_bot.y - l_top.y) * h
        rhgt = abs(r_bot.y - r_top.y) * h
        lv = (l_c[1] - l_top.y * h) / (lhgt + 1e-6) if lhgt > 3 else 0.5
        rv = (r_c[1] - r_top.y * h) / (rhgt + 1e-6) if rhgt > 3 else 0.5

        gx = float(np.clip((lh+rh)/2, 0, 1))
        gy = float(np.clip((lv+rv)/2, 0, 1))
        return gx, gy
    except:
        return None, None

def eye_aspect_ratio(landmarks, eye_def, w, h):
    try:
        p1 = np.array([landmarks[eye_def['outer']].x*w, landmarks[eye_def['outer']].y*h])
        p4 = np.array([landmarks[eye_def['inner']].x*w, landmarks[eye_def['inner']].y*h])
        p2 = np.array([landmarks[eye_def['top']].x*w, landmarks[eye_def['top']].y*h])
        p6 = np.array([landmarks[eye_def['bottom']].x*w, landmarks[eye_def['bottom']].y*h])
        p3 = np.array([landmarks[eye_def['top2']].x*w, landmarks[eye_def['top2']].y*h])
        p5 = np.array([landmarks[eye_def['bot2']].x*w, landmarks[eye_def['bot2']].y*h])
        vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horiz = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
        return float(vert / horiz)
    except:
        return 0.3

def draw_midline_dots(frame, landmarks, w, h, color=(0,255,255), count=6, radius=2):
    try:
        idx_forehead = 10
        idx_nose = 1
        if idx_forehead >= len(landmarks) or idx_nose >= len(landmarks):
            return
        fh = landmarks[idx_forehead]; ns = landmarks[idx_nose]
        x1, y1 = int(fh.x * w), int(fh.y * h)
        x2, y2 = int(ns.x * w), int(ns.y * h)
        for t in np.linspace(0.0, 1.0, count):
            x = int(x1*(1-t) + x2*t)
            y = int(y1*(1-t) + y2*t)
            cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    except:
        pass

# ------------- YOLO Detector Thread -------------
class YoloWorker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base = None
        self.custom = None
        try:
            self.base = YOLO('yolov8m.pt')
            print("Loaded YOLO: yolov8m.pt")
        except:
            try:
                self.base = YOLO('yolov8s.pt')
                print("Loaded YOLO: yolov8s.pt")
            except:
                self.base = YOLO('yolov8n.pt')
                print("Loaded YOLO: yolov8n.pt")
        try:
            self.base.fuse()
        except:
            pass

        if os.path.exists(CUSTOM_AIDS_WEIGHTS):
            try:
                self.custom = YOLO(CUSTOM_AIDS_WEIGHTS)
                self.custom.fuse()
                print(f"Loaded custom aids model: {CUSTOM_AIDS_WEIGHTS}")
            except:
                self.custom = None

        self.frame_q = deque(maxlen=1)
        self.last = []
        self.last_time = 0
        self.lock = threading.Lock()
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def update(self, frame):
        with self.lock:
            self.frame_q.clear()
            self.frame_q.append(frame.copy())

    def _valid_phone(self, box, conf, shape):
        return conf >= CLASS_THRESH["cell phone"]

    def _valid_book(self, box, conf, shape):
        return conf >= CLASS_THRESH["book"]

    def _post_filter(self, name, conf, box, frame):
        if name == "cell phone":
            if not self._valid_phone(box, conf, frame.shape):
                return False
        elif name == "book":
            if not self._valid_book(box, conf, frame.shape):
                return False
        return True

    def _merge_results(self, base_res, custom_res):
        dets = []
        for src in [base_res, custom_res]:
            if src is None:
                continue
            if src.boxes is not None:
                for b in src.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    if cls_id in DEVICE_CLASSES:
                        name = DEVICE_CLASSES[cls_id]
                    else:
                        name = src.names.get(cls_id, str(cls_id))
                    dets.append({"name": name, "conf": conf, "box": (x1, y1, x2, y2)})
        return dets

    def _run(self):
        while not self.stop:
            now = time.time()
            if (now - self.last_time) * 1000 < YOLO_INTERVAL_MS:
                time.sleep(0.01); continue
            frame = None
            with self.lock:
                if len(self.frame_q) > 0:
                    frame = self.frame_q[-1]
            if frame is None:
                time.sleep(0.01); continue

            try:
                with torch.inference_mode():
                    base_res = self.base(
                        frame, imgsz=YOLO_IMGSZ, conf=0.40,
                        classes=list(DEVICE_CLASSES.keys()),
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        verbose=False, half=True if torch.cuda.is_available() else False
                    )[0]
                    custom_res = None
                    if self.custom is not None:
                        custom_res = self.custom(
                            frame, imgsz=YOLO_IMGSZ, conf=0.45,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            verbose=False, half=True if torch.cuda.is_available() else False
                        )[0]

                    dets = self._merge_results(base_res, custom_res)
                    filtered = []
                    for d in dets:
                        nm, conf, box = d["name"], d["conf"], d["box"]
                        if nm in CLASS_THRESH and conf < CLASS_THRESH[nm]:
                            continue
                        if nm in ("cell phone", "book"):
                            if not self._post_filter(nm, conf, box, frame):
                                continue
                        filtered.append({"name": nm, "conf": conf, "box": box})

                    self.last = filtered
                    self.last_time = time.time()
            except Exception:
                time.sleep(0.02)

    def get(self):
        return self.last, self.last_time

    def shutdown(self):
        self.stop = True
        try: self.t.join(timeout=1)
        except: pass

yolo_worker = YoloWorker()

# ------------- Speech detector (optional) -------------
class SpeechDetector:
    def __init__(self, enable=True):
        self.enable = enable and SR_AVAILABLE
        self.r = None
        self.m = None
        self.stopper = None
        self.last_speech_ts = 0.0
        self.last_alert_ts = 0.0
        if not self.enable:
            print("Speech detection disabled (speech_recognition unavailable or disabled).")

    def _callback(self, recog, audio):
        self.last_speech_ts = time.time()
        if not self.enable:
            return
        try:
            text = self.r.recognize_google(audio, language="en-US", show_all=False)
            if text:
                try:
                    ts = self.last_speech_ts
                    state["spoken_words"].append({"ts": ts, "text": text})
                    socketio.emit('speech_transcript', {"ts": ts, "text": text})
                except:
                    pass
        except Exception:
            pass

    def start(self):
        if not self.enable:
            return
        try:
            self.r = sr.Recognizer()
            self.r.dynamic_energy_threshold = True
            self.m = sr.Microphone()
            with self.m as source:
                self.r.adjust_for_ambient_noise(source, duration=1.0)
            self.stopper = self.r.listen_in_background(self.m, self._callback, phrase_time_limit=2)
            print("Speech detector started.")
        except Exception as e:
            print("Speech detector could not start:", e)
            self.enable = False

    def recent_speech(self):
        return (time.time() - self.last_speech_ts) <= SPEECH_HOLD_S

    def can_alert(self):
        return (time.time() - self.last_alert_ts) > SPEECH_ALERT_COOLDOWN_S

    def mark_alert(self):
        self.last_alert_ts = time.time()

    def stop(self):
        try:
            if self.stopper:
                self.stopper(wait_for_stop=False)
        except:
            pass

speech = SpeechDetector(ENABLE_SPEECH_DETECTION)
speech.start()

# ------------- Calibration -------------
class CalibPhase(Enum):
    CENTER = "Look STRAIGHT at screen center"
    EYES_LEFT = "Move EYES LEFT (keep head still)"
    EYES_RIGHT = "Move EYES RIGHT (keep head still)"
    HEAD_LEFT = "Turn HEAD LEFT ~15° (keep eyes on center)"
    HEAD_RIGHT = "Turn HEAD RIGHT ~15° (keep eyes on center)"
    EYES_UP = "Move EYES UP slightly"
    EYES_DOWN = "Move EYES DOWN slightly"
    COMPLETE = "Calibration complete!"

PHASE_ORDER = [
    CalibPhase.CENTER, CalibPhase.EYES_LEFT, CalibPhase.EYES_RIGHT,
    CalibPhase.HEAD_LEFT, CalibPhase.HEAD_RIGHT,
    CalibPhase.EYES_UP, CalibPhase.EYES_DOWN
]
SAMPLES_PER_PHASE = 6

state = {
    "calibrated": False,
    "phase": CalibPhase.CENTER,
    "samples": {p: {"yaw": [], "pitch": [], "gx": [], "gy": []} for p in PHASE_ORDER},
    "yaw0": 0.0, "pitch0": 0.0,
    "gx_center": 0.5, "gy_center": 0.5,
    "gx_left": 0.4, "gx_right": 0.6,
    "gy_up": 0.45, "gy_down": 0.55,
    "h_sign": 1.0,
    "yaw2gaze_slope": 0.0,
    "yaw": 0.0, "pitch": 0.0, "gx": 0.5, "gy": 0.5,
    "ear": 0.25, "ear_raw": 0.25, "ear_open_baseline": 0.26,
    "last_blink_ts": 0.0,
    "eyes_closed_since": None,
    "distract_ts": None, "away_ts": None,
    "score": 100.0, "last_tick": time.time(),
    "device_tracks": {},
    "tab_hidden": False, "tab_changed_ts": 0.0,
    "last_eyes_emit_ts": 0.0,
    "last_speech_ui": None, "last_speech_enabled": None,
    "calib_hold_until": 0.0, "calib_last_sample_ms": 0,
    "calib_phase_start_ts": time.time(),
    "session": {"start_ts": time.time()},
    "metrics": {
        "distraction_log": [],
        "multi_people_log": [],
        "eyes_closed_log": [],
        "device_log": [],
        "tab_out_log": [],
        "liveness_failure_log": [],
        "speech_log": []
    },
    "spoken_words": [],
    "flags": {
        "in_multi_people": False,
        "in_eyes_closed_alert": False,
        "in_distraction": False,
        "in_device_alert": False,
    },
    "focus_sum": 0.0,
    "focus_count": 0,
    "streaming_active": True,
    "liveness_last_check_ts": time.time(),
    "liveness_first_check_done": False,
    "liveness_active": False,
    "liveness_start_ts": 0.0,
    "liveness_blink_detected": False,
    "liveness_hand_detected": False,
    "liveness_fail_until_ts": 0.0,
    "liveness_pass_until_ts": 0.0,
    # Frozen gates for each liveness window
    "liveness_baseline": 0.26,
    "liveness_open_gate": 0.22,
    "liveness_close_gate": 0.17,
}

# --- Decorator for stream control ---
def require_streaming(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not state["streaming_active"]:
            print("Stream paused, skipping frame generation.")
            pass
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def index():
    return render_template("PolySpectra.html")

@socketio.on('connect')
def on_connect():
    socketio.emit('speech_status', {
        "enabled": getattr(speech, "enable", False),
        "active": speech.recent_speech() if getattr(speech, "enable", False) else False
    })
    emit_calib_progress()
    socketio.emit('stream_status', {"active": state["streaming_active"]})
    status = "not_calibrated" if not state["calibrated"] else "calibrated"
    socketio.emit('calibration_status_update', {"status": status})


@socketio.on('recalibrate')
def on_recalib():
    global state
    state["calibrated"] = False
    state["phase"] = CalibPhase.CENTER
    state["samples"] = {p: {"yaw": [], "pitch": [], "gx": [], "gy": []} for p in PHASE_ORDER}
    state["calib_hold_until"] = 0.0
    state["calib_last_sample_ms"] = 0
    state["calib_phase_start_ts"] = time.time()
    state["liveness_first_check_done"] = False
    state["liveness_last_check_ts"] = time.time()
    socketio.emit('proctor_alert', {"message": f"CALIBRATION: {state['phase'].value}", "status": "calib"})
    socketio.emit('focus_percentage_update', {"percentage": "---"})
    socketio.emit('score_update', {"score": "100.0", "reason": "RECALIBRATING"})
    emit_calib_progress()
    socketio.emit('calibration_status_update', {"status": "calibrating"})


@socketio.on('tab_state')
def on_tab_state(data):
    is_hidden = bool(data.get('hidden', False))
    if is_hidden and not state["tab_hidden"]:
        state["metrics"]["tab_out_log"].append({"ts": time.time(), "action": "hidden"})
    elif not is_hidden and state["tab_hidden"]:
        state["metrics"]["tab_out_log"].append({"ts": time.time(), "action": "visible"})

    state["tab_hidden"] = is_hidden
    state["tab_changed_ts"] = time.time()


@socketio.on('toggle_speech')
def on_toggle_speech(data):
    enable = bool(data.get('enable', True))
    if hasattr(speech, "enable"):
        if enable and not speech.enable:
            speech.enable = True
            try: speech.start()
            except: speech.enable = False
        elif not enable and speech.enable:
            speech.enable = False
            try: speech.stop()
            except: pass
        socketio.emit('speech_status', {
            "enabled": speech.enable,
            "active": speech.recent_speech() if speech.enable else False
        })

@socketio.on('toggle_stream')
def on_toggle_stream(data):
    global state
    activate = bool(data.get('active', True))
    state["streaming_active"] = activate
    print(f"Stream {'activated' if activate else 'paused'}")
    socketio.emit('stream_status', {"active": state["streaming_active"]})


def emit_calib_progress():
    total_needed = len(PHASE_ORDER) * SAMPLES_PER_PHASE
    done = 0
    for p in PHASE_ORDER:
        done += min(len(state["samples"][p]["gx"]), SAMPLES_PER_PHASE)
    overall = 100.0 * done / max(1, total_needed)
    cur = state["phase"]
    collected = len(state["samples"][cur]["gx"])
    phase_pct = 100.0 * collected / SAMPLES_PER_PHASE
    socketio.emit('calibration_progress', {
        "phase": cur.name,
        "phase_label": cur.value,
        "collected": collected,
        "needed": SAMPLES_PER_PHASE,
        "phase_percent": round(phase_pct, 1),
        "overall_percent": round(overall, 1)
    })

def normalized_gaze(gx_raw, gy_raw):
    gx_comp = gx_raw - state["yaw2gaze_slope"] * state["yaw"]
    denom_h = (state["gx_right"] - state["gx_left"]) / 2.0
    if denom_h < 1e-4: denom_h = 0.15
    gx_norm = state["h_sign"] * (gx_comp - state["gx_center"]) / denom_h
    gx_norm = float(np.clip(gx_norm, -2.0, 2.0))

    denom_v = (state["gy_down"] - state["gy_up"]) / 2.0
    if denom_v < 1e-4: denom_v = 0.18
    gy_norm = (gy_raw - state["gy_center"]) / denom_v
    gy_norm = float(np.clip(gy_norm, -2.0, 2.0))
    return gx_norm, gy_norm

def phase_complete(phase):
    return len(state["samples"][phase]["gx"]) >= SAMPLES_PER_PHASE

def try_advance_phase():
    if not phase_complete(state["phase"]):
        return
    i = PHASE_ORDER.index(state["phase"])
    if i + 1 < len(PHASE_ORDER):
        state["phase"] = PHASE_ORDER[i+1]
        state["calib_phase_start_ts"] = time.time()
        state["calib_hold_until"] = 0.0
        socketio.emit('proctor_alert', {"message": f"CALIBRATION: {state['phase'].value}", "status": "calib"})
        emit_calib_progress()

def finalize_calibration():
    center = state["samples"][CalibPhase.CENTER]
    left_e = state["samples"][CalibPhase.EYES_LEFT]
    right_e = state["samples"][CalibPhase.EYES_RIGHT]
    left_h = state["samples"][CalibPhase.HEAD_LEFT]
    right_h = state["samples"][CalibPhase.HEAD_RIGHT]
    up_e = state["samples"][CalibPhase.EYES_UP]
    down_e = state["samples"][CalibPhase.EYES_DOWN]

    state["yaw0"] = float(np.median(center["yaw"])) if center["yaw"] else 0.0
    state["pitch0"] = float(np.median(center["pitch"])) if center["pitch"] else 0.0

    state["gx_center"] = float(np.median(center["gx"])) if center["gx"] else 0.5
    state["gy_center"] = float(np.median(center["gy"])) if center["gy"] else 0.5

    if left_e["gx"]:
        state["gx_left"] = float(np.median(left_e["gx"]))
    else:
        state["gx_left"] = state["gx_center"] - 0.10

    if right_e["gx"]:
        gx_right_raw = float(np.median(right_e["gx"]))
        if abs(gx_right_raw - state["gx_center"]) < 0.06 or abs(gx_right_raw - state["gx_center"]) > 0.60:
            state["gx_right"] = state["gx_center"] + abs(state["gx_center"] - state["gx_left"])
        else:
            state["gx_right"] = gx_right_raw
    else:
        state["gx_right"] = state["gx_center"] + abs(state["gx_center"] - state["gx_left"])

    if up_e["gy"]:
        state["gy_up"] = float(np.median(up_e["gy"]))
    else:
        state["gy_up"] = state["gy_center"] - 0.08

    if down_e["gy"]:
        gy_down_raw = float(np.median(down_e["gy"]))
        if abs(gy_down_raw - state["gy_center"]) < 0.05 or abs(gy_down_raw - state["gy_center"]) > 0.65:
            state["gy_down"] = state["gy_center"] + abs(state["gy_center"] - state["gy_up"])
        else:
            state["gy_down"] = gy_down_raw
    else:
        state["gy_down"] = state["gy_center"] + abs(state["gy_center"] - state["gy_up"])

    state["h_sign"] = -1.0 if (state["gx_left"] - state["gx_center"]) > 0 else 1.0

    if left_h["yaw"] or right_h["yaw"]:
        xs = np.array([(np.median(left_h["yaw"]) - state["yaw0"]) if left_h["yaw"] else -15.0,
                       (np.median(right_h["yaw"]) - state["yaw0"]) if right_h["yaw"] else 15.0], dtype=np.float32)
        ys = np.array([(np.median(left_h["gx"]) - state["gx_center"]) if left_h["gx"] else -0.01,
                       (np.median(right_h["gx"]) - state["gx_center"]) if right_h["gx"] else 0.01], dtype=np.float32)
        denom = float(np.sum((xs - xs.mean())**2) + 1e-6)
        slope = float(np.sum((xs - xs.mean())*(ys - ys.mean())) / denom)
    else:
        slope = 0.0
    state["yaw2gaze_slope"] = slope

    state["calibrated"] = True
    socketio.emit('proctor_alert', {"message": "CALIBRATION COMPLETE!", "status": "ok"})
    socketio.emit('calibration_complete')
    emit_calib_progress()
    socketio.emit('calibration_status_update', {"status": "calibrated"})


def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)

# ------------- Report -------------
def format_timestamp(ts):
    return time.strftime('%H:%M:%S', time.localtime(ts))

def build_report_bytes():
    start = state["session"].get("start_ts", time.time())
    end = time.time()
    duration = end - start
    avg_focus = (state["focus_sum"] / state["focus_count"]) if state["focus_count"] > 0 else state["score"]
    words = state["spoken_words"]
    metrics = state["metrics"]

    title = "AI Proctoring Session Report"
    lines = [
        "=========================",
        title,
        "=========================",
        f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}",
        f"Session Start:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}",
        f"Session Duration: {int(duration//3600)}h {int((duration%3600)//60)}m {int(duration%60)}s",
        "",
        "--- Scores ---",
        f"Final Focus Score: {state['score']:.1f}",
        f"Average Focus:     {avg_focus:.1f}",
        "",
    ]

    distraction_count = len(metrics["distraction_log"])
    multi_people_count = len(metrics["multi_people_log"])
    eyes_closed_count = len(metrics["eyes_closed_log"])
    device_count = len(metrics["device_log"])
    tab_out_count = sum(1 for event in metrics["tab_out_log"] if event.get("action") == "hidden")
    liveness_fail_count = len(metrics["liveness_failure_log"])
    speech_count = len(metrics.get("speech_log", []))

    lines.extend([
        "--- Summary of Events ---",
        f"Distractions (Head/Gaze): {distraction_count}",
        f"Multiple People Detections: {multi_people_count}",
        f"Eyes Closed Alerts:         {eyes_closed_count}",
        f"Prohibited Device Alerts:   {device_count}",
        f"Tab Switched Out Count:     {tab_out_count}",
        f"Speech Detections:          {speech_count}",
        f"Liveness Check Failures:    {liveness_fail_count}",
        "",
        "--- Detailed Event Log ---",
    ])

    all_events = []
    for event in metrics["distraction_log"]:
        all_events.append({"ts": event["ts"], "type": "Distraction", "details": event["reason"]})
    for event in metrics["multi_people_log"]:
        all_events.append({"ts": event["ts"], "type": "Multiple People", "details": f"{event['count']} detected"})
    for event in metrics["eyes_closed_log"]:
        all_events.append({"ts": event["ts"], "type": "Eyes Closed Alert"})
    for event in metrics["device_log"]:
        all_events.append({"ts": event["ts"], "type": "Device Alert", "details": event["name"]})
    for event in metrics["tab_out_log"]:
        all_events.append({"ts": event["ts"], "type": "Tab Focus", "details": "Hidden" if event["action"] == "hidden" else "Visible"})
    for event in metrics["liveness_failure_log"]:
        all_events.append({"ts": event["ts"], "type": "Liveness Check", "details": "Failed"})
    for event in metrics.get("speech_log", []):
        all_events.append({"ts": event["ts"], "type": "Speech Detected"})

    all_events.sort(key=lambda x: x["ts"])

    if not all_events:
        lines.append("No major events detected during the session.")
    else:
        for event in all_events:
            ts_str = format_timestamp(event["ts"])
            details = f" ({event['details']})" if "details" in event else ""
            lines.append(f"[{ts_str}] {event['type']}{details}")

    lines.append("")
    lines.append("--- Spoken Words Transcript (Last 200) ---")
    if not words:
        lines.append("No speech detected or transcribed.")
    else:
        for w in words[-200:]:
            ts_str = format_timestamp(w["ts"])
            lines.append(f"[{ts_str}] {w['text']}")

    data = ("\n".join(lines)).encode("utf-8", errors="ignore")
    return io.BytesIO(data), "text/plain"


@app.route("/report.txt")
def report_download():
    bio, mimetype = build_report_bytes()
    resp = make_response(send_file(bio, mimetype=mimetype, as_attachment=True, download_name="proctor_report.txt"))
    return resp

# ------------- Stream loop -------------
def generate_frames():
    global state
    cam = CameraGrabber(CAMERA_SOURCE, w=1280, h=720)
    base_w, base_h = 640, 480
    focal = base_w
    camK = np.array([[focal,0,base_w/2],[0,focal,base_h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))

    last_alert = ""
    last_status = ""
    gaze_hist_h = deque(maxlen=5)
    gaze_hist_v = deque(maxlen=5)
    ear_history = deque(maxlen=7)  # was 5 -> small bump to better capture blink pattern

    hud_h = 60
    hud_w = 320
    alpha = 0.4

    try:
        while True:
            if not state["streaming_active"]:
                _ = cam.read()
                time.sleep(0.1)
                continue

            frame_raw = cam.read()
            if frame_raw is None:
                time.sleep(0.02)
                continue

            frame, W, H = resize_keep_aspect(frame_raw, target_w=640)
            if frame is None:
                continue

            overlay = np.zeros_like(frame)

            yolo_worker.update(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            fd_res = face_det.process(rgb)
            rgb.flags.writeable = True

            now = time.time()
            dt = now - state["last_tick"]
            state["last_tick"] = now

            alert_msg = "STATUS: FOCUSED"
            alert_status = "ok"

            num_faces_mesh = 0
            faces_landmarks = []
            if res and res.multi_face_landmarks:
                num_faces_mesh = len(res.multi_face_landmarks)
                faces_landmarks = [f.landmark for f in res.multi_face_landmarks]

            num_faces_fd = len(fd_res.detections) if (fd_res and fd_res.detections) else 0
            num_faces = max(num_faces_mesh, num_faces_fd)

            if num_faces > 1:
                if not state["flags"]["in_multi_people"]:
                    state["metrics"]["multi_people_log"].append({"ts": now, "count": num_faces})
                    state["flags"]["in_multi_people"] = True
                for lmk in faces_landmarks:
                    draw_midline_dots(frame, lmk, W, H, color=(0,255,255), count=6, radius=2)
                alert_msg = f"ALERT: MULTIPLE PEOPLE ({num_faces})"
                alert_status = "alert"
                state["score"] -= SCORE_MAJOR * dt
            else:
                state["flags"]["in_multi_people"] = False

            primary_landmarks = faces_landmarks[0] if len(faces_landmarks) >= 1 else None

            if primary_landmarks is None and num_faces <= 1:
                if state["calibrated"]:
                    if state["away_ts"] is None:
                        state["away_ts"] = now
                    if now - state["away_ts"] > AWAY_DWELL_S:
                        alert_msg = "ALERT: USER AWAY"
                        alert_status = "alert"
                        state["score"] -= SCORE_MAJOR * dt
                else:
                    alert_msg = f"CALIBRATION: {state['phase'].value}"
                    alert_status = "calib"
                    if state["phase"] != CalibPhase.COMPLETE:
                        socketio.emit('calibration_status_update', {"status": "calibrating"})
                    emit_calib_progress()
            else:
                state["away_ts"] = None

            gx_med, gy_med = 0.0, 0.0

            if primary_landmarks is not None and num_faces <= 1:
                draw_midline_dots(frame, primary_landmarks, W, H, color=(0,255,255), count=6, radius=2)

                yaw_raw, pitch_raw, rvec, tvec = get_head_pose(primary_landmarks, W, H, camK, dist)
                if yaw_raw is not None:
                    yaw_rel = yaw_raw - state["yaw0"]
                    pitch_rel = pitch_raw - state["pitch0"]
                    state["yaw"] = (1-EMA_POSE)*state["yaw"] + EMA_POSE*yaw_rel
                    state["pitch"] = (1-EMA_POSE)*state["pitch"] + EMA_POSE*pitch_rel

                # EAR calculation every frame (not gated by gaze)
                ear_l = eye_aspect_ratio(primary_landmarks, LEFT_EYE, W, H)
                ear_r = eye_aspect_ratio(primary_landmarks, RIGHT_EYE, W, H)
                ear = max(0.05, (ear_l + ear_r) / 2.0)
                state["ear_raw"] = ear
                state["ear"] = EAR_SMOOTH * state["ear"] + (1 - EAR_SMOOTH) * ear
                ear_history.append(ear)

                # Slowly update open-eye baseline when clearly open
                if state["ear"] > (EAR_THRESH + 0.02):
                    state["ear_open_baseline"] = 0.98 * state["ear_open_baseline"] + 0.02 * ear

                # Blink timestamp gating
                if state["ear"] < EAR_THRESH and (now - state["last_blink_ts"])*1000 > BLINK_SKIP_MS:
                    state["last_blink_ts"] = now

                # Gaze update
                gx_raw, gy_raw = get_gaze_raw(primary_landmarks, W, H)
                if gx_raw is not None:
                    # Only skip during blink hold
                    if (now - state["last_blink_ts"]) * 1000 > BLINK_SKIP_MS:
                        state["gx"] = (1-EMA_GAZE) * state["gx"] + EMA_GAZE * gx_raw
                        state["gy"] = (1-EMA_GAZE) * state["gy"] + EMA_GAZE * gy_raw

                # -------- Calibration collection (more forgiving + auto-relax) --------
                if not state["calibrated"]:
                    ph = state["phase"]
                    data = state["samples"][ph]
                    if gx_raw is not None and yaw_raw is not None and pitch_raw is not None:
                        now_ms = int(now * 1000)

                        # Determine stability (absolute and dynamic)
                        phase_elapsed = now - state.get("calib_phase_start_ts", now)
                        stable_abs = (abs(state["yaw"]) < CALIB_STABLE_YAW and abs(state["pitch"]) < CALIB_STABLE_PITCH)

                        stable_dyn = False
                        # not using the dyn buffers here, but you kept them available if needed

                        stable_pose = stable_abs or stable_dyn
                        # Auto-relax after a while
                        hold_needed = CALIB_HOLD_S
                        if phase_elapsed > CALIB_RELAX_AFTER_S:
                            hold_needed = CALIB_RELAX_HOLD_S
                            stable_pose = True

                        def try_sample():
                            if state["calib_hold_until"] == 0.0:
                                state["calib_hold_until"] = now + hold_needed
                            if now >= state["calib_hold_until"] and (now_ms - state["calib_last_sample_ms"] >= CALIB_SAMPLE_STEP_MS):
                                data["yaw"].append(yaw_raw); data["pitch"].append(pitch_raw)
                                data["gx"].append(gx_raw);   data["gy"].append(gy_raw)
                                state["calib_last_sample_ms"] = now_ms
                                emit_calib_progress()

                        if ph in (CalibPhase.CENTER, CalibPhase.EYES_LEFT, CalibPhase.EYES_RIGHT,
                                  CalibPhase.EYES_UP, CalibPhase.EYES_DOWN):
                            if stable_pose:
                                try_sample()
                            else:
                                state["calib_hold_until"] = 0.0

                        elif ph == CalibPhase.HEAD_LEFT:
                            yaw_ok = state["yaw"] < (-6.0 if phase_elapsed <= CALIB_RELAX_AFTER_S else -4.0)
                            if yaw_ok:
                                try_sample()
                            else:
                                state["calib_hold_until"] = 0.0

                        elif ph == CalibPhase.HEAD_RIGHT:
                            yaw_ok = state["yaw"] > (6.0 if phase_elapsed <= CALIB_RELAX_AFTER_S else 4.0)
                            if yaw_ok:
                                try_sample()
                            else:
                                state["calib_hold_until"] = 0.0

                    alert_msg = f"CALIBRATION: {ph.value} ({len(data['gx'])}/{SAMPLES_PER_PHASE})"
                    alert_status = "calib"
                    if phase_complete(ph):
                        try_advance_phase()
                    if all(phase_complete(p) for p in PHASE_ORDER if p != CalibPhase.COMPLETE):
                        finalize_calibration()

                # -------- Normal (post-calibration) processing --------
                if state["calibrated"]:
                    gx_n, gy_n = normalized_gaze(state["gx"], state["gy"])
                    gaze_hist_h.append(gx_n)
                    gaze_hist_v.append(gy_n)
                    gx_med = float(np.median(gaze_hist_h))
                    gy_med = float(np.median(gaze_hist_v))

                    eyes_closed_alert = False
                    if state["ear"] < EAR_THRESH:
                        if state["eyes_closed_since"] is None:
                            state["eyes_closed_since"] = now
                        elif (now - state["eyes_closed_since"]) >= EYES_CLOSED_ALERT_S:
                            eyes_closed_alert = True
                            if not state["flags"]["in_eyes_closed_alert"]:
                                state["metrics"]["eyes_closed_log"].append({"ts": now})
                                state["flags"]["in_eyes_closed_alert"] = True
                    else:
                        state["eyes_closed_since"] = None
                        state["flags"]["in_eyes_closed_alert"] = False

                    if now - state["last_eyes_emit_ts"] > 0.5:
                        if state["eyes_closed_since"] is not None:
                            secs = now - state["eyes_closed_since"]
                            socketio.emit('eyes_status', {"closed": True, "seconds": secs})
                        else:
                            socketio.emit('eyes_status', {"closed": False, "seconds": 0})
                        state["last_eyes_emit_ts"] = now

                    # Unified distraction logic: Head or Gaze
                    HEAD_PRIOR_YAW = 12.0
                    HEAD_PRIOR_PITCH = 18.0

                    head_distraction = (
                        abs(state["yaw"]) > YAW_THRESH or
                        state["pitch"] > PITCH_DOWN_THRESH or
                        state["pitch"] < PITCH_UP_THRESH
                    )

                    gaze_distraction = False
                    if not head_distraction:
                        if abs(gx_med) > GAZE_H_ALERT or abs(gy_med) > GAZE_V_ALERT:
                            if abs(state["yaw"]) > HEAD_PRIOR_YAW or abs(state["pitch"]) > HEAD_PRIOR_PITCH:
                                head_distraction = True
                            else:
                                gaze_distraction = True

                    distracted = head_distraction or gaze_distraction
                    reason = "Head" if head_distraction else ("Gaze" if gaze_distraction else "")

                    if eyes_closed_alert:
                        alert_msg = "ALERT: EYES CLOSED"
                        alert_status = "alert"
                        state["score"] -= SCORE_MINOR * dt
                        state["flags"]["in_distraction"] = False
                    else:
                        if distracted:
                            if state["distract_ts"] is None:
                                state["distract_ts"] = now
                            elif (now - state["distract_ts"]) > DISTRACT_DWELL_S:
                                if not state["flags"]["in_distraction"]:
                                    state["metrics"]["distraction_log"].append({"ts": now, "reason": reason})
                                    state["flags"]["in_distraction"] = True
                                alert_msg = f"ALERT: DISTRACTED ({reason})"
                                alert_status = "alert"
                                state["score"] -= SCORE_MINOR * dt
                        else:
                            state["distract_ts"] = None
                            state["flags"]["in_distraction"] = False
                            state["score"] += SCORE_RECOVER * dt

            # --- Liveliness Check Logic ---
            time_since_last = now - state["liveness_last_check_ts"]
            is_ready_for_check = False
            if state["calibrated"]:
                if not state["liveness_first_check_done"]:
                    if time_since_last > LIVENESS_FIRST_CHECK_S:
                        is_ready_for_check = True
                        state["liveness_first_check_done"] = True
                else:
                    if time_since_last > LIVENESS_INTERVAL_S:
                        is_ready_for_check = True

            if state["calibrated"] and not state["liveness_active"] and is_ready_for_check:
                state["liveness_active"] = True
                state["liveness_start_ts"] = now
                state["liveness_blink_detected"] = False
                state["liveness_hand_detected"] = False
                state["liveness_last_check_ts"] = now

                # Freeze blink baseline and gates for this window
                base = float(np.clip(0.5 * state["ear_open_baseline"] + 0.5 * state["ear"], 0.20, 0.36))
                state["liveness_baseline"] = base
                state["liveness_open_gate"] = max(EAR_THRESH + 0.015, base * 0.75)
                state["liveness_close_gate"] = max(0.12, min(0.21, base - 0.055))

                # Clear recent EAR samples so we only use fresh frames during this window
                try:
                    ear_history.clear()
                except Exception:
                    pass

            if state["liveness_active"]:
                # Prioritize liveliness message
                alert_msg = "LIVENESS CHECK: Blink and Raise Hand"
                alert_status = "alert"

                if now - state["liveness_start_ts"] <= LIVENESS_WINDOW_S:
                    # Blink detection using frozen gates for this window
                    if not state["liveness_blink_detected"]:
                        if len(ear_history) >= 4:
                            open_gate = state["liveness_open_gate"]
                            close_gate = state["liveness_close_gate"]

                            was_open = ear_history[-4] > open_gate
                            mid_min = min(ear_history[-3], ear_history[-2])
                            closed_mid = (mid_min < close_gate) or (mid_min < LIVENESS_BLINK_THRESH)
                            open_again = ear_history[-1] > open_gate

                            if was_open and closed_mid and open_again:
                                state["liveness_blink_detected"] = True
                                print("Liveliness: Blink detected (frozen gates)")

                        # Fallback: accept a recent blink using last_blink_ts within the window
                        if not state["liveness_blink_detected"]:
                            if (state["last_blink_ts"] > state["liveness_start_ts"]) and ((now - state["last_blink_ts"]) <= 1.4):
                                state["liveness_blink_detected"] = True
                                print("Liveliness: Blink detected (fallback last_blink_ts)")

                    # Hand detection (upper frame or near face ROI)
                    if not state["liveness_hand_detected"] and hands_model:
                        try:
                            rgb.flags.writeable = False
                            hands_results = hands_model.process(rgb)
                            rgb.flags.writeable = True

                            if hands_results.multi_hand_landmarks:
                                face_roi = None
                                try:
                                    if fd_res and fd_res.detections:
                                        rel = fd_res.detections[0].location_data.relative_bounding_box
                                        fx1 = int(rel.xmin * W); fy1 = int(rel.ymin * H)
                                        fx2 = int((rel.xmin + rel.width) * W); fy2 = int((rel.ymin + rel.height) * H)
                                        pad = 40
                                        face_roi = (max(0, fx1 - pad), max(0, fy1 - pad), min(W, fx2 + pad), min(H, fy2 + pad))
                                except:
                                    face_roi = None

                                hand_raised = False
                                for hl in hands_results.multi_hand_landmarks:
                                    ys = [lm.y for lm in hl.landmark]
                                    xs = [lm.x for lm in hl.landmark]
                                    miny = min(ys)
                                    # Slightly more forgiving upper zone: 72% of the frame
                                    if miny < 0.72:
                                        hand_raised = True
                                    # Or inside/near the face ROI
                                    if face_roi is not None:
                                        x_pixels = [int(x * W) for x in xs]
                                        y_pixels = [int(y * H) for y in ys]
                                        for px, py in zip(x_pixels, y_pixels):
                                            if face_roi[0] <= px <= face_roi[2] and face_roi[1] <= py <= face_roi[3]:
                                                hand_raised = True
                                                break
                                    if hand_raised:
                                        break

                                if hand_raised:
                                    state["liveness_hand_detected"] = True
                                    print("Liveliness: Hand detected")
                        except Exception as e:
                            print(f"Hand detection error during liveliness: {e}")

                    # Success condition
                    if state["liveness_blink_detected"] and state["liveness_hand_detected"]:
                        print("Liveliness: Success!")
                        state["liveness_active"] = False
                        state["liveness_pass_until_ts"] = now + 2.5  # show pass message briefly
                        alert_status = "ok"

                else:  # Timeout
                    print("Liveliness: Failed (Timeout)")
                    state["liveness_active"] = False
                    state["metrics"]["liveness_failure_log"].append({"ts": now})
                    state["liveness_fail_until_ts"] = now + 5.0
                    alert_msg = "ALERT: Liveliness Check Failed!"
                    alert_status = "alert"
                    state["score"] -= SCORE_MAJOR * dt

            # Keep failure message displayed briefly if needed
            if now < state["liveness_fail_until_ts"]:
                if alert_status != 'alert':
                    alert_msg = "ALERT: Liveliness Check Failed!"
                    alert_status = "alert"

            # Show pass message briefly (without overriding alerts)
            if now < state["liveness_pass_until_ts"]:
                if alert_status != 'alert':
                    alert_msg = "LIVENESS CHECK PASSED"
                    alert_status = "ok"

            # --- YOLO Section ---
            dets, _ = yolo_worker.get()
            tracks = state["device_tracks"]
            now_t = time.time()
            for d in dets:
                nm, conf, box = d["name"], d["conf"], d["box"]
                matched_key = None
                for key, t in tracks.items():
                    if t["name"] == nm and iou(box, t["box"]) > 0.5:
                        matched_key = key; break
                if matched_key is None:
                    tracks[id(d)] = {"name": nm, "box": box, "conf": conf, "count": 1, "first": now_t, "last": now_t}
                else:
                    tracks[matched_key]["box"] = box
                    tracks[matched_key]["conf"] = max(tracks[matched_key]["conf"], conf)
                    tracks[matched_key]["count"] += 1
                    tracks[matched_key]["last"] = now_t
            to_del = [k for k, t in tracks.items() if now_t - t["last"] > HOLD_DEVICE_S]
            for k in to_del: del tracks[k]

            for t in tracks.values():
                x1,y1,x2,y2 = t["box"]
                color = (0,0,255) if t["name"] == "cell phone" else (0,255,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{t['name']} {t['conf']:.2f}x{t['count']}",
                            (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            best_device = None
            if (primary_landmarks is not None) and num_faces <= 1:
                best_pri = -1
                for t in tracks.values():
                    if t["count"] < 2: continue
                    nm = t["name"]
                    pri = 0
                    if nm == "cell phone": pri = 3
                    elif nm == "book": pri = 2
                    elif nm == "remote": pri = 1
                    else: pri = 0
                    if pri > best_pri and nm not in IGNORE_CLASSES_IN_ALERT:
                        best_pri = pri; best_device = t

            # Apply device alert, ensuring it doesn't override liveliness prompt/failure
            if best_device is not None and num_faces <= 1 and not state["liveness_active"] and now >= state["liveness_fail_until_ts"] and alert_status != 'alert':
                if not state["flags"]["in_device_alert"]:
                    state["metrics"]["device_log"].append({"ts": now, "name": best_device['name'], "conf": best_device['conf']})
                    state["flags"]["in_device_alert"] = True
                alert_msg = f"ALERT: {best_device['name'].upper()} DETECTED!"
                alert_status = "alert"
                state["score"] -= SCORE_MAJOR * dt
            elif best_device is None:
                state["flags"]["in_device_alert"] = False

            # --- Other Alerts (Tab, Speech) ---
            if state["tab_hidden"] and (now - state["tab_changed_ts"] > 0.7) and not state["liveness_active"] and now >= state["liveness_fail_until_ts"] and alert_status == "ok":
                alert_msg = "ALERT: TAB OUT OF FOCUS"
                alert_status = "alert"
                state["score"] -= SCORE_MINOR * dt

            if ENABLE_SPEECH_DETECTION and speech.enable and state["calibrated"] and num_faces <= 1:
                if speech.recent_speech() and speech.can_alert() and not state["liveness_active"] and now >= state["liveness_fail_until_ts"] and alert_status == "ok":
                    alert_msg = "ALERT: SPEECH DETECTED"
                    alert_status = "alert"
                    state["score"] -= SCORE_MINOR * dt
                    speech.mark_alert()
                    state["metrics"]["speech_log"].append({"ts": now})

            if hasattr(speech, "enable"):
                active = speech.recent_speech() if speech.enable else False
                if (state["last_speech_ui"] != active) or (state["last_speech_enabled"] != speech.enable):
                    socketio.emit('speech_status', {"enabled": speech.enable, "active": active})
                    state["last_speech_ui"] = active
                    state["last_speech_enabled"] = speech.enable

            # --- Final Processing ---
            state["score"] = float(np.clip(state["score"], 0, 100))

            if state["calibrated"]:
                yaw_focus = max(0.0, 1.0 - abs(state["yaw"]) / (YAW_THRESH + 1e-6))
                pitch_focus = max(0.0, 1.0 - abs(state["pitch"]) / (PITCH_DOWN_THRESH + 1e-6)) if state["pitch"] >=0 else max(0.0, 1.0 + state["pitch"] / (abs(PITCH_UP_THRESH) + 1e-6))

                gh, gv = (float(np.median(gaze_hist_h)), float(np.median(gaze_hist_v))) if (len(gaze_hist_h) > 0 and len(gaze_hist_v) > 0) else (0.0, 0.0)
                gaze_focus_h = max(0.0, 1.0 - abs(gh) / (GAZE_H_ALERT + 1e-6))
                gaze_focus_v = max(0.0, 1.0 - abs(gv) / (GAZE_V_ALERT + 1e-6))

                head_focus = 0.7 * yaw_focus + 0.3 * pitch_focus
                gaze_focus = 0.6 * gaze_focus_h + 0.4 * gaze_focus_v
                focus_pct = float(np.clip(0.6 * head_focus + 0.4 * gaze_focus, 0.0, 1.0) * 100.0)

                state["focus_sum"] += focus_pct
                state["focus_count"] += 1

                socketio.emit('focus_percentage_update', {'percentage': f"{focus_pct:.1f}%"})
            else:
                socketio.emit('focus_percentage_update', {'percentage': '---'})

            # Draw HUD Text AFTER blending background
            if primary_landmarks is not None and state["calibrated"]:
                try:
                    cv2.rectangle(overlay, (5, 5), (hud_w, hud_h), (0,0,0), -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    cv2.putText(frame, f"Yaw: {int(state['yaw']):+d}°  Pitch: {int(state['pitch']):+d}°",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"GazeN H:{gx_med:+.2f} V:{gy_med:+.2f}  EAR:{state['ear']:.2f}",
                                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,0), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"HUD Drawing Error: {e}")

            # Emit alerts and score
            if alert_msg != last_alert or alert_status != last_status:
                last_alert, last_status = alert_msg, alert_status
                socketio.emit('proctor_alert', {'message': alert_msg, 'status': alert_status})

            reason = "FOCUSED"
            if "DISTRACTED" in alert_msg:
                try: reason = alert_msg.split("(", 1)[1].split(")")[0]
                except: reason = "Distracted"
            elif "AWAY" in alert_msg: reason = "User Away"
            elif "PEOPLE" in alert_msg: reason = "Multiple People"
            elif "DETECTED" in alert_msg and "SPEECH" not in alert_msg: reason = "Device Detected"
            elif "EYES CLOSED" in alert_msg: reason = "Eyes Closed"
            elif "SPEECH DETECTED" in alert_msg: reason = "Speech Detected"
            elif "TAB OUT OF FOCUS" in alert_msg: reason = "Tab Out of Focus"
            elif "Liveliness Check Failed" in alert_msg: reason = "Liveness Failed"
            elif "LIVENESS CHECK" in alert_msg: reason = "Liveness Check Active"

            socketio.emit('score_update', {'score': f"{state['score']:.1f}", 'reason': reason})

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    except GeneratorExit:
        print("Client disconnected.")
    except Exception as e:
        print("Stream error:", e)
        import traceback
        traceback.print_exc()
    finally:
        state["session"]["end_ts"] = time.time()
        print("Releasing camera...")
        cam.release()
        print("Stopping speech detector...")
        try: speech.stop()
        except: pass
        print("Stream generator finished.")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting server...")
    print("Visit: http://127.0.0.1:5000")
    print("Download report at: http://127.0.0.1:5000/report.txt")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
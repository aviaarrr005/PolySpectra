# AI Proctoring (Head/Gaze + Liveness) with Flask, MediaPipe, YOLO

Real-time, privacy-friendly proctoring system that runs locally in your browser:
- Head pose + Eye gaze distraction detection
- Liveness checks (blink + hand raise)
- Optional speech detection (microphone)
- Device detection (phone/book/remote/keyboard/mouse/laptop) via YOLO
- Focus score, session events log, downloadable report

This repo uses Flask + Socket.IO, OpenCV, MediaPipe (Face Mesh, Face Detection, Hands), Ultralytics YOLOv8, and optional SpeechRecognition.

--------------------------------------------------------------------------------

## Features

- Head pose detection (yaw/pitch) and gaze normalization (eyes only)
- Distraction detection with clear category:
  - Head distraction (large yaw/pitch)
  - Gaze distraction (large eye movement with stable head)
- Liveness checks (every few minutes)
  - Blink detection via EAR and per-window thresholds
  - Hand raise detection near face region or top portion of the frame
  - Clear pass/fail prompts and short success message
- Optional speech detection (toggle on/off)
- YOLO device detection with class thresholds
- Session score and focus percentage (live)
- Downloadable text report of events and transcript

--------------------------------------------------------------------------------

## Requirements

- OS: Windows, macOS, or Linux
- Python: 3.9–3.11 recommended
- Camera: Built-in or USB webcam
- GPU: Optional (CUDA for PyTorch speeds up YOLO)
- Microphone: Optional (for speech detection)

--------------------------------------------------------------------------------

## Install

1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install core dependencies
```bash
pip install --upgrade pip
pip install flask flask-socketio eventlet
pip install opencv-python numpy mediapipe ultralytics
```

3) Install PyTorch (choose the right command for your system)
- CPU only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 12.1 (example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
See https://pytorch.org/get-started/locally/ for the matching command.

4) Optional: SpeechRecognition
```bash
pip install SpeechRecognition
# On Windows, you may need PyAudio:
pip install pipwin && pipwin install pyaudio
# On macOS/Linux use your package manager or brew/apt to install portaudio, then:
# pip install pyaudio
```

--------------------------------------------------------------------------------

## Run

```bash
python gazetest.py
```

Then open the URL shown in the console:
- http://127.0.0.1:5000
- Video stream: http://127.0.0.1:5000/video_feed
- Download report: http://127.0.0.1:5000/report.txt

Allow Camera (and Mic if using speech) in your browser when prompted.

--------------------------------------------------------------------------------

## Quick Start Flow

1) Open the app and allow camera/microphone permissions.
2) Calibration will start:
   - Follow the on-screen prompts (Center → Eyes Left/Right → Head Left/Right → Eyes Up/Down).
   - Hold steady briefly; if it takes too long, calibration auto-relaxes and samples anyway.
3) After calibration:
   - You’ll see Head/Gaze stats, EAR (eye openness), and Focus score.
   - If you look away, you’ll see “ALERT: DISTRACTED (Head/Gaze)”.
4) Liveness (every few minutes):
   - Prompt: “LIVENESS CHECK: Blink and Raise Hand”
   - Blink once normally (don’t hold your eyes closed too long).
   - Raise your hand into the upper portion of the frame or near your face.
   - On success, “LIVENESS CHECK PASSED” shows briefly.

--------------------------------------------------------------------------------

## UI and Alerts

- STATUS: FOCUSED — Everything is fine
- ALERT: DISTRACTED (Head) — Head yaw/pitch exceeded thresholds
- ALERT: DISTRACTED (Gaze) — Eyes moved too far left/right/up/down with stable head
- ALERT: LIVENESS CHECK... — Blink and Raise Hand within the window
- ALERT: Liveliness Check Failed! — Window expired without both signals
- ALERT: MULTIPLE PEOPLE (#) — More than one face detected
- ALERT: USER AWAY — No face detected for a while
- ALERT: TAB OUT OF FOCUS — Browser tab lost focus
- ALERT: SPEECH DETECTED — Microphone activity within cooldown rules
- ALERT: <DEVICE> DETECTED! — Phone/Book/Remote (keyboard/mouse/laptop ignored in alerts by default)

--------------------------------------------------------------------------------

## Calibration Tips

- CENTER: Sit naturally and look at center; it samples automatically after a short hold.
- EYES LEFT/RIGHT/UP/DOWN: Move only your eyes; keep head relatively still.
- HEAD LEFT/RIGHT: Rotate your head slightly (~15°); keep eyes near center.
- The app relaxes stability if stuck too long (so you don’t get stuck at 0/6).
- You’ll see progress like “(3/6)”. When all phases complete, “CALIBRATION COMPLETE!” shows.

--------------------------------------------------------------------------------

## Liveness Check Details

- Triggered first after ~45 seconds, then every ~3 minutes (configurable).
- Pass criteria within 30 seconds window:
  - Blink detected via EAR pattern (open → closed → open), using per-window thresholds.
  - Raised hand detected either near the face region or in the upper portion of the frame.
- Helpful tips:
  - Blink normally once or twice; avoid super-slow eye closures.
  - Raise your hand up into the camera frame (upper ~70% of the frame) or near your face.

--------------------------------------------------------------------------------

## Scoring and Focus

- Score starts at 100 and changes based on events:
  - Major penalties for multiple people, user away, prohibited device, liveness fail
  - Minor penalties for eyes closed, distraction, speech, tab out
  - Gradual recovery when focused
- Focus percentage combines head and gaze stability:
  - Head focus weighted by yaw/pitch
  - Gaze focus weighted by normalized gaze

--------------------------------------------------------------------------------

## Device and Speech Detection

- YOLOv8 detects: cell phone, book, laptop, remote, keyboard, mouse
- Laptops/keyboards/mice are ignored in alerts by default (configurable).
- Optional custom model (e.g., earbuds/airpods): put weights at cheat_aids.pt
- Speech detection (optional):
  - Uses SpeechRecognition in a background thread
  - Toggle on/off via Socket.IO event (or code)

--------------------------------------------------------------------------------

## Configuration (edit constants in code)

- Camera and server
  - CAMERA_SOURCE: 0 (default webcam)
- Sensitivity and smoothing
  - EMA_GAZE: 0.42–0.45 (higher = responds faster)
  - GAZE_H_ALERT: medium sensitivity around 1.06–1.12
    - Lower => more sensitive (triggers sooner)
    - Higher => less sensitive
  - YAW_THRESH, PITCH_DOWN/UP_THRESH: head thresholds (degrees)
- Liveness
  - LIVENESS_FIRST_CHECK_S: first check delay (default 45s)
  - LIVENESS_INTERVAL_S: interval between checks (default 180s)
  - LIVENESS_WINDOW_S: response window (default 30s)
  - LIVENESS_BLINK_THRESH: baseline blink threshold (0.17)
- Eyes closed
  - EAR_THRESH: eyes-closed threshold (default 0.18)
  - EYES_CLOSED_ALERT_S: sustained eyes-closed duration to alert (default 4s)
- YOLO
  - YOLO_IMGSZ: 416 (smaller for CPU, larger if GPU available)
  - YOLO_INTERVAL_MS: 350ms between inferences
  - CLASS_THRESH: per-class minimum confidence for alerts
- Speech
  - ENABLE_SPEECH_DETECTION: True/False
  - SPEECH_ALERT_COOLDOWN_S: minimum time between speech alerts

Recommended medium horizontal gaze:
- EMA_GAZE ≈ 0.45
- GAZE_H_ALERT ≈ 1.06–1.08

--------------------------------------------------------------------------------

## API and Events

HTTP
- GET / — Main UI (index4.html)
- GET /video_feed — MJPEG stream (multipart)
- GET /report.txt — Session report (text/plain)

Socket.IO (Client → Server)
- ‘recalibrate’ — Reset calibration
- ‘tab_state’, payload: { "hidden": bool } — Tab visibility
- ‘toggle_speech’, payload: { "enable": bool }
- ‘toggle_stream’, payload: { "active": bool }

Socket.IO (Server → Client)
- ‘proctor_alert’, { message, status }
- ‘calibration_progress’, { phase, phase_label, collected, needed, phase_percent, overall_percent }
- ‘calibration_status_update’, { status }
- ‘focus_percentage_update’, { percentage }
- ‘speech_status’, { enabled, active }
- ‘speech_transcript’, { ts, text }
- ‘stream_status’, { active }
- ‘eyes_status’, { closed, seconds }
- ‘score_update’, { score, reason }

--------------------------------------------------------------------------------

## Performance Tips

- CPU only:
  - Use YOLOv8n or decrease YOLO_IMGSZ and increase YOLO_INTERVAL_MS
- GPU:
  - Install CUDA build of PyTorch and keep half=True in YOLO
- Frame size:
  - The pipeline rescales to ~640px width for speed; adjust if needed
- Disable features you don’t need (speech, device YOLO) for lower CPU

--------------------------------------------------------------------------------

## Troubleshooting

- Camera not opening: set CAMERA_SOURCE to 1/2, close other apps, check drivers
- Blank page/video not loading: allow camera permission in the browser
- SpeechRecognition errors:
  - Install PyAudio or disable speech (ENABLE_SPEECH_DETECTION = False)
- Mediapipe install issues:
  - Ensure Python and pip are updated; on Linux install system dependencies (OpenGL/GL libs)
- Slow performance:
  - Use YOLOv8n, increase YOLO_INTERVAL_MS, lower image size, disable speech/YOLO if testing
- Calibration stuck at 0/6:
  - Just hold steady; after ~10s the app relaxes and will still sample

--------------------------------------------------------------------------------

## Privacy

- Everything runs locally; no video/audio leaves your machine unless you change the code
- Session events and transcripts are only stored in memory and in the downloadable text report during the run
- You control mic usage (toggle), and can disable speech entirely

--------------------------------------------------------------------------------

## Notes and Acknowledgments

- Powered by MediaPipe (Face Mesh/Detection/Hands), Ultralytics YOLOv8, OpenCV, Flask, Socket.IO, (optional) SpeechRecognition
- This is not a certified proctoring product; always validate in your environment

--------------------------------------------------------------------------------

## License

AGPL-3.0

--------------------------------------------------------------------------------

## Contributing

- Open an issue with logs, OS, Python version, and what you tried
- PRs welcome (bug fixes, performance improvements, docs)

--------------------------------------------------------------------------------

## Quick Tunables Cheat Sheet

- Make gaze detection a bit more sensitive (left/right triggers sooner):
  - GAZE_H_ALERT: 1.06 (from ~1.12)
  - EMA_GAZE: 0.45 (from 0.42)
- Make it less sensitive:
  - GAZE_H_ALERT: 1.10–1.15
  - EMA_GAZE: 0.38–0.42
- Liveness easier (testing):
  - LIVENESS_FIRST_CHECK_S: 15.0
  - LIVENESS_WINDOW_S: 35.0

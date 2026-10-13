"""
Driver Drowsiness Detector
--------------------------
Real-time, software-only drowsiness detection using OpenCV + MediaPipe Face Mesh.

Signals used
  * EAR  (Eye Aspect Ratio)  - personal baseline is auto-calibrated at start
  * Microsleep               - eyes closed continuously for a long time
  * PERCLOS                  - % of time eyes were closed in the last 60 s
  * Slow blinks              - blinks that last unusually long
  * MAR  (Mouth Aspect Ratio)- yawning
  * Head pose                - head dropping / turning away for too long
  * Face lost                - driver's face not visible

Keys:  q = quit    r = recalibrate
"""
import argparse
import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------------
# Configuration (all times in seconds)
# ------------------------------------------------------------------
CALIBRATION_SECONDS = 5.0     # look at the camera with eyes open, relaxed
MIN_CALIB_SAMPLES = 30
MIN_VALID_BASELINE_EAR = 0.15  # lower than this = eyes were probably shut

CLOSED_RATIO = 0.75           # eye is "closed" when EAR < 75% of your open-eye baseline
MIN_BLINK = 0.05              # shorter closures are treated as noise
SLOW_BLINK = 0.40             # a blink longer than this is a "slow blink"
MICROSLEEP_SECONDS = 1.5      # continuous closure -> DROWSY

PERCLOS_WINDOW = 60.0
PERCLOS_MIN_DATA = 15.0       # need this much data before PERCLOS is trusted
PERCLOS_WARN = 0.12
PERCLOS_DROWSY = 0.25
SLOW_BLINKS_WARN = 3          # slow blinks within the window -> WARNING

YAWN_MAR = 0.60               # mouth open wider than this...
YAWN_SECONDS = 1.0            # ...for this long = one yawn
YAWNS_WARN = 2                # yawns within YAWN_WINDOW -> WARNING
YAWN_WINDOW = 300.0

HEAD_DEVIATION = 0.08         # change in nose/face-height ratio vs. baseline
HEAD_SECONDS = 2.0            # head off-position this long -> DROWSY

NO_FACE_WARN = 2.0
NO_FACE_DROWSY = 5.0

MAX_FAILED_FRAMES = 30        # tolerated consecutive failed reads (IP cameras)
MAX_FRAME_WIDTH = 960         # downscale big frames for speed

# ------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ------------------------------------------------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6 order for EAR
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
NOSE, FOREHEAD, CHIN = 1, 10, 152

LEVEL_LABELS = {0: "ALERT", 1: "WARNING: signs of fatigue", 2: "DROWSY! WAKE UP!"}
LEVEL_COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255)}   # BGR


# ------------------------------------------------------------------
# Geometry helpers (landmarks are normalised 0-1 -> convert to pixels)
# ------------------------------------------------------------------
def _pt(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(lm, idx, w, h):
    p = [_pt(lm, i, w, h) for i in idx]
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    return vertical / (2.0 * _dist(p[0], p[3]) + 1e-6)


def mouth_aspect_ratio(lm, w, h):
    vertical = sum(_dist(_pt(lm, a, w, h), _pt(lm, b, w, h)) for a, b in MOUTH_V)
    horizontal = _dist(_pt(lm, MOUTH_H[0], w, h), _pt(lm, MOUTH_H[1], w, h))
    return vertical / (3.0 * horizontal + 1e-6)


def head_ratio(lm, w, h):
    """Where the nose tip sits between forehead and chin (changes as the head pitches)."""
    nose, fore, chin = (_pt(lm, i, w, h)[1] for i in (NOSE, FOREHEAD, CHIN))
    return (nose - fore) / ((chin - fore) + 1e-6)


# ------------------------------------------------------------------
# Core logic (no camera / GUI code here, so it can be unit-tested)
# ------------------------------------------------------------------
class DrowsinessDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        """Start (or restart) calibration and clear all history."""
        self.calibrated = False
        self.calib_start = None
        self.calib_ear, self.calib_head = [], []
        self.base_ear = None
        self.base_head = None

        self.ear_hist = deque(maxlen=3)        # smoothing
        self.last_ear = self.last_mar = 0.0

        self.closed_since = None
        self.mouth_since = None
        self.yawn_counted = False
        self.head_since = None
        self.no_face_since = None

        self.samples = deque()                 # (t, closed, dt) for PERCLOS
        self.last_t = None
        self.blinks = deque()                  # (t, duration)
        self.yawns = deque()                   # t

    # ---- called when a face is found -----------------------------
    def update_face(self, lm, w, h, now):
        self.no_face_since = None

        ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
               eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
        self.ear_hist.append(ear)
        ear_s = sum(self.ear_hist) / len(self.ear_hist)
        mar = mouth_aspect_ratio(lm, w, h)
        hr = head_ratio(lm, w, h)
        self.last_ear, self.last_mar = ear_s, mar

        if not self.calibrated:
            if self.calib_start is None:
                self.calib_start = now
            self.calib_ear.append(ear_s)
            self.calib_head.append(hr)
            if (now - self.calib_start >= CALIBRATION_SECONDS
                    and len(self.calib_ear) >= MIN_CALIB_SAMPLES):
                # 70th percentile ignores the frames where you blinked
                base = float(np.percentile(self.calib_ear, 70))
                if base < MIN_VALID_BASELINE_EAR:      # eyes were shut: redo
                    self.calib_start, self.calib_ear, self.calib_head = None, [], []
                else:
                    self.base_ear = base
                    self.base_head = float(np.median(self.calib_head))
                    self.calibrated = True
                    self.last_t = now
            return self._evaluate(now)

        # ---- eyes -------------------------------------------------
        closed = ear_s < self.base_ear * CLOSED_RATIO
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 0.5)
        self.last_t = now
        self.samples.append((now, closed, dt))
        while self.samples and self.samples[0][0] < now - PERCLOS_WINDOW:
            self.samples.popleft()

        if closed:
            if self.closed_since is None:
                self.closed_since = now
        elif self.closed_since is not None:
            duration = now - self.closed_since
            if duration >= MIN_BLINK:
                self.blinks.append((now, duration))
            self.closed_since = None
        while self.blinks and self.blinks[0][0] < now - PERCLOS_WINDOW:
            self.blinks.popleft()

        # ---- yawning ----------------------------------------------
        if mar > YAWN_MAR:
            if self.mouth_since is None:
                self.mouth_since = now
            if not self.yawn_counted and now - self.mouth_since >= YAWN_SECONDS:
                self.yawns.append(now)
                self.yawn_counted = True
        else:
            self.mouth_since = None
            self.yawn_counted = False
        while self.yawns and self.yawns[0] < now - YAWN_WINDOW:
            self.yawns.popleft()

        # ---- head pose --------------------------------------------
        if abs(hr - self.base_head) > HEAD_DEVIATION:
            if self.head_since is None:
                self.head_since = now
        else:
            self.head_since = None

        return self._evaluate(now)

    # ---- called when no face is found ----------------------------
    def update_no_face(self, now):
        if self.no_face_since is None:
            self.no_face_since = now
        self.closed_since = self.mouth_since = self.head_since = None
        self.yawn_counted = False
        self.last_t = None
        self.ear_hist.clear()
        return self._evaluate(now)

    # ---- decide the current state --------------------------------
    def _evaluate(self, now):
        if not self.calibrated:
            elapsed = 0.0 if self.calib_start is None else now - self.calib_start
            return {"calibrating": True, "level": 0, "label": "Calibrating - look at the camera, eyes open",
                    "progress": min(elapsed / CALIBRATION_SECONDS, 1.0), "reasons": [],
                    "ear": self.last_ear, "mar": self.last_mar}

        reasons, level = [], 0

        closed_for = 0.0 if self.closed_since is None else now - self.closed_since
        if closed_for >= MICROSLEEP_SECONDS:
            level = 2
            reasons.append(f"Eyes closed {closed_for:.1f}s")

        total = sum(s[2] for s in self.samples)
        closed_time = sum(s[2] for s in self.samples if s[1])
        perclos = closed_time / total if total > 0 else 0.0
        if total >= PERCLOS_MIN_DATA:
            if perclos >= PERCLOS_DROWSY:
                level = 2
                reasons.append(f"PERCLOS {perclos:.0%}")
            elif perclos >= PERCLOS_WARN:
                level = max(level, 1)
                reasons.append(f"PERCLOS {perclos:.0%}")

        slow = sum(1 for _, d in self.blinks if d >= SLOW_BLINK)
        if slow >= SLOW_BLINKS_WARN:
            level = max(level, 1)
            reasons.append(f"{slow} slow blinks")

        if len(self.yawns) >= YAWNS_WARN:
            level = max(level, 1)
            reasons.append(f"{len(self.yawns)} yawns")

        head_for = 0.0 if self.head_since is None else now - self.head_since
        if head_for >= HEAD_SECONDS:
            level = 2
            reasons.append("Head down / away")

        if self.no_face_since is not None:
            absent = now - self.no_face_since
            if absent >= NO_FACE_DROWSY:
                level = 2
                reasons.append("Face not visible")
            elif absent >= NO_FACE_WARN:
                level = max(level, 1)
                reasons.append("Face not visible")

        window = min(total, PERCLOS_WINDOW)
        blink_rate = len(self.blinks) * 60.0 / max(window, 10.0)
        return {"calibrating": False, "level": level, "label": LEVEL_LABELS[level],
                "reasons": reasons, "ear": self.last_ear, "mar": self.last_mar,
                "baseline": self.base_ear, "perclos": perclos,
                "blink_rate": blink_rate, "yawns": len(self.yawns),
                "closed_for": closed_for}


# ------------------------------------------------------------------
# Alarm (non-blocking)
# ------------------------------------------------------------------
class Alarm:
    def __init__(self, cooldown=1.0):
        self.cooldown = cooldown
        self._last = 0.0

    def trigger(self, now):
        if now - self._last < self.cooldown:
            return
        self._last = now
        threading.Thread(target=self._beep, daemon=True).start()

    @staticmethod
    def _beep():
        try:
            import winsound                      # Windows only
            winsound.Beep(1800, 400)
        except ImportError:
            sys.stdout.write("\a")               # terminal bell on Linux / macOS
            sys.stdout.flush()


# ------------------------------------------------------------------
# Camera + drawing
# ------------------------------------------------------------------
def parse_source(arg):
    if arg is None:
        arg = input("Enter phone camera URL (e.g. http://192.168.1.15:8080/video) "
                    "or a camera index like 0 for the PC webcam: ").strip()
    return int(arg) if str(arg).isdigit() else arg


def draw_hud(frame, st):
    color = LEVEL_COLORS[st["level"]]
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, st["label"], (10, 32), font, 0.8, color, 2)

    if st["calibrating"]:
        cv2.rectangle(frame, (10, 45), (210, 60), (255, 255, 255), 1)
        cv2.rectangle(frame, (10, 45), (10 + int(200 * st["progress"]), 60), (255, 255, 255), -1)
        return

    lines = [f"EAR {st['ear']:.2f} (baseline {st['baseline']:.2f})",
             f"MAR {st['mar']:.2f}   Yawns {st['yawns']}",
             f"PERCLOS {st['perclos']:.0%}   Blinks/min {st['blink_rate']:.0f}"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 62 + 24 * i), font, 0.55, (255, 255, 255), 1)
    for i, text in enumerate(st["reasons"]):
        cv2.putText(frame, text, (10, 62 + 24 * len(lines) + 24 * i), font, 0.6, color, 2)
    if st["level"] == 2:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)


def main():
    parser = argparse.ArgumentParser(description="Driver drowsiness detector")
    parser.add_argument("--source", help="camera index (0) or IP-camera / video URL")
    parser.add_argument("--no-display", action="store_true", help="run without a window (for testing)")
    args = parser.parse_args()

    import mediapipe as mp                       # imported here so tests of the logic don't need it
    if not hasattr(mp, "solutions"):
        sys.exit("This mediapipe version has no 'solutions' API. Run: pip install mediapipe==0.10.14")

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Could not open the camera. Check the URL / index and your network.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    detector, alarm = DrowsinessDetector(), Alarm()
    mirror = isinstance(source, int)
    failed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed += 1
                if failed > MAX_FAILED_FRAMES:
                    print("Lost connection to the camera / end of video.")
                    break
                continue
            failed = 0

            if mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            now = time.monotonic()

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                state = detector.update_face(lm, w, h, now)
                for idx in (LEFT_EYE, RIGHT_EYE):
                    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idx])
                    cv2.polylines(frame, [pts], True, (255, 255, 0), 1)
            else:
                state = detector.update_no_face(now)

            if state["level"] == 2 and not state["calibrating"]:
                alarm.trigger(now)

            draw_hud(frame, state)
            if not args.no_display:
                cv2.imshow("Driver Drowsiness Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    detector.reset()
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

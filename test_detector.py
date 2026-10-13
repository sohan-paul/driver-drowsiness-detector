"""Simulated-driver tests for the detection logic (no camera needed).  Run: python test_detector.py"""
from types import SimpleNamespace as P

from drowsiness_detector import DrowsinessDetector

W = H = 1000
FPS = 30


def make_face(eye_gap=30, mouth_gap=5, nose_y=0.60):
    """Fake 468+ landmark list. eye_gap/mouth_gap in pixels; nose_y = nose position (0-1)."""
    lm = [P(x=0.5, y=0.5) for _ in range(500)]
    g = eye_gap / (2.0 * H)
    # right eye (p1..p6) = 33,160,158,133,153,144 ; eye is 100 px wide
    lm[33], lm[133] = P(x=0.20, y=0.40), P(x=0.30, y=0.40)
    lm[160], lm[144] = P(x=0.23, y=0.40 - g), P(x=0.23, y=0.40 + g)
    lm[158], lm[153] = P(x=0.27, y=0.40 - g), P(x=0.27, y=0.40 + g)
    # left eye = 362,385,387,263,373,380
    lm[362], lm[263] = P(x=0.70, y=0.40), P(x=0.80, y=0.40)
    lm[385], lm[380] = P(x=0.73, y=0.40 - g), P(x=0.73, y=0.40 + g)
    lm[387], lm[373] = P(x=0.77, y=0.40 - g), P(x=0.77, y=0.40 + g)
    # mouth: corners 78/308 (100 px wide), three vertical pairs
    m = mouth_gap / (2.0 * H)
    lm[78], lm[308] = P(x=0.45, y=0.75), P(x=0.55, y=0.75)
    for a, b, x in ((81, 178, 0.48), (13, 14, 0.50), (311, 402, 0.52)):
        lm[a], lm[b] = P(x=x, y=0.75 - m), P(x=x, y=0.75 + m)
    # head ratio landmarks
    lm[10], lm[152], lm[1] = P(x=0.5, y=0.20), P(x=0.5, y=0.90), P(x=0.5, y=nose_y)
    return lm


class Sim:
    def __init__(self):
        self.d, self.t, self.max_level, self.last = DrowsinessDetector(), 0.0, 0, None

    def run(self, seconds, **face):
        face_lm = make_face(**face)
        for _ in range(int(seconds * FPS)):
            self.t += 1.0 / FPS
            self.last = self.d.update_face(face_lm, W, H, self.t)
            self.max_level = max(self.max_level, self.last["level"])
        return self.last

    def run_no_face(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.t += 1.0 / FPS
            self.last = self.d.update_no_face(self.t)
            self.max_level = max(self.max_level, self.last["level"])
        return self.last

    def calibrate(self):
        self.run(6)
        assert self.d.calibrated, "calibration should finish"
        self.max_level = 0


def test_calibration_no_alarm():
    s = Sim()
    st = s.run(3)
    assert st["calibrating"] and st["level"] == 0
    s.run(3)
    assert s.d.calibrated and abs(s.d.base_ear - 0.30) < 0.01, s.d.base_ear


def test_normal_blinking_is_alert():
    s = Sim(); s.calibrate()
    for _ in range(20):                       # 20 normal blinks (0.15 s), one every 3 s
        s.run(2.85, eye_gap=30)
        s.run(0.15, eye_gap=3)
    assert s.max_level == 0, f"normal blinking must not alert (level {s.max_level}, {s.last['reasons']})"


def test_microsleep_triggers_drowsy():
    s = Sim(); s.calibrate()
    s.run(3, eye_gap=30)
    st = s.run(2.0, eye_gap=3)
    assert st["level"] == 2 and any("Eyes closed" in r for r in st["reasons"]), st
    st = s.run(1.0, eye_gap=30)               # opens eyes -> recovers
    assert st["level"] in (0, 1)


def test_high_perclos_triggers():
    s = Sim(); s.calibrate()
    for _ in range(12):                       # eyes closed 0.45 s every 2.5 s (~18%) -> at least WARNING
        s.run(2.05, eye_gap=30)
        s.run(0.45, eye_gap=3)
    assert s.max_level >= 1, s.last


def test_yawning_warns():
    s = Sim(); s.calibrate()
    s.run(2, mouth_gap=5)
    s.run(1.4, mouth_gap=70)                  # yawn 1
    s.run(2, mouth_gap=5)
    st = s.run(1.4, mouth_gap=70)             # yawn 2
    assert st["level"] >= 1 and st["yawns"] == 2, st


def test_short_mouth_open_is_not_yawn():
    s = Sim(); s.calibrate()
    s.run(0.5, mouth_gap=70)                  # quick talk / laugh
    st = s.run(2, mouth_gap=5)
    assert st["yawns"] == 0


def test_head_down_triggers():
    s = Sim(); s.calibrate()
    s.run(1, nose_y=0.60)
    st = s.run(2.5, nose_y=0.75)
    assert st["level"] == 2 and any("Head" in r for r in st["reasons"]), st
    st = s.run(1, nose_y=0.60)
    assert st["level"] == 0


def test_brief_head_move_ignored():
    s = Sim(); s.calibrate()
    s.run(0.8, nose_y=0.75)                   # quick glance at mirror
    s.run(2, nose_y=0.60)
    assert s.max_level == 0


def test_face_lost():
    s = Sim(); s.calibrate()
    st = s.run_no_face(3)
    assert st["level"] == 1
    st = s.run_no_face(3)
    assert st["level"] == 2


def test_recalibrate_reset():
    s = Sim(); s.calibrate()
    s.d.reset()
    assert not s.d.calibrated


def test_closed_eyes_during_calibration_restarts():
    s = Sim()
    s.run(6, eye_gap=2)                       # eyes shut the whole time
    assert not s.d.calibrated, "must not accept a closed-eye baseline"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)

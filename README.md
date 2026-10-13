# Driver Drowsiness Detector

Real-time, software-only drowsiness detection using **Python, OpenCV and MediaPipe Face Mesh**. It watches the driver's face through a webcam or a phone camera and raises a visual and audio alarm when it detects signs of fatigue. No special hardware is needed.

> **Demo:** _add a GIF or screenshot here_ (see [Adding a demo](#adding-a-demo))

---

## Features

- **Personal auto-calibration.** The first 5 seconds measure *your* open-eye baseline, so the system adapts to different eye shapes and camera angles instead of using one fixed threshold.
- **Multiple fatigue signals**, not just "eyes closed":

  | Signal | What it measures | Triggers |
  |---|---|---|
  | **Microsleep** | Eyes continuously closed | 1.5 s or more -> DROWSY |
  | **PERCLOS** | % of time eyes were closed in the last 60 s | 12% -> WARNING, 25% -> DROWSY |
  | **Slow blinks** | Blinks lasting 0.4 s or more | 3 in 60 s -> WARNING |
  | **Yawning (MAR)** | Mouth open wide for 1 s or more | 2 yawns in 5 min -> WARNING |
  | **Head pose** | Head dropping or turning away | 2 s or more -> DROWSY |
  | **Face lost** | Face not visible to the camera | 2 s -> WARNING, 5 s -> DROWSY |

- **Three clear states:** `ALERT`, `WARNING` (early fatigue signs), `DROWSY` (alarm beep and red screen border).
- **Normal blinks are ignored**, so alerts come from real fatigue, not routine blinking.
- **Works with a PC webcam or a phone camera** over Wi-Fi (IP camera apps).
- **FPS-independent**: all timing uses real seconds, not frame counts.
- **Robust to camera glitches**: dropped frames on IP streams are tolerated, and resources are released cleanly on exit.
- **Unit-tested logic**: the detection engine is separated from the camera code and tested with simulated drivers.

---

## How it works

1. **Face landmarks.** MediaPipe Face Mesh returns 468+ landmarks per frame.
2. **Eye Aspect Ratio (EAR)** from 6 landmarks per eye:

   ```
   EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
   ```

   EAR is high when the eye is open and drops toward 0 when it closes. Landmark coordinates are converted to pixels first, so the ratio isn't distorted by the frame's aspect ratio.
3. **Calibration.** The 70th percentile of EAR samples over 5 seconds (which ignores blink frames) becomes your open-eye baseline. An eye counts as *closed* when EAR falls below 75% of that baseline. If your eyes were shut during calibration, it restarts automatically.
4. **Smoothing.** EAR is averaged over 3 frames to reduce flicker.
5. **Mouth Aspect Ratio (MAR)** from inner-lip landmarks detects yawns.
6. **Head pose.** The nose tip's position between forehead and chin is compared with its calibrated baseline.
7. **Decision.** All signals are combined into one of three levels. Level 2 triggers the alarm.

---

## Installation

**Requirements:** Python 3.9 to 3.12 and a webcam or a phone with an IP camera app.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

> **Important:** `requirements.txt` pins `mediapipe==0.10.14`. Newer MediaPipe releases removed the `mp.solutions` API this project uses.

---

## Usage

```bash
python drowsiness_detector.py                # prompts for the camera source
python drowsiness_detector.py --source 0     # PC webcam
python drowsiness_detector.py --source http://192.168.1.15:8080/video   # phone camera
```

**Keys:** `q` quit, `r` recalibrate

**Steps:**
1. Sit in your normal driving position, facing the camera.
2. Keep your eyes open and relaxed for the 5-second calibration. The progress bar shows how far along it is.
3. The on-screen panel shows your state, EAR, MAR, PERCLOS, blinks per minute and yawn count.

### Using a phone as the camera
1. Install an IP camera app (for example "IP Webcam" on Android).
2. Connect the phone and computer to the **same Wi-Fi network**.
3. Start the server in the app and note the URL, e.g. `http://192.168.1.15:8080/video`.
4. Pass that URL with `--source`.

---

## Configuration

All thresholds are constants at the top of `drowsiness_detector.py`:

| Constant | Default | Meaning |
|---|---|---|
| `CALIBRATION_SECONDS` | 5.0 | Calibration duration |
| `CLOSED_RATIO` | 0.75 | Eye is closed below this fraction of baseline EAR |
| `MICROSLEEP_SECONDS` | 1.5 | Continuous closure that triggers DROWSY |
| `PERCLOS_WARN` / `PERCLOS_DROWSY` | 0.12 / 0.25 | PERCLOS levels |
| `SLOW_BLINK` | 0.40 | Blink duration counted as "slow" |
| `YAWN_MAR` / `YAWN_SECONDS` | 0.60 / 1.0 | Yawn detection |
| `HEAD_DEVIATION` / `HEAD_SECONDS` | 0.08 / 2.0 | Head-pose sensitivity and duration |

These are standard starting values. Tune them for your face and camera setup.

---

## Testing

The detection logic is separate from the camera code and is tested with simulated landmark sequences on a fake clock, so no camera is needed:

```bash
python test_detector.py
```

The 11 tests cover: calibration, normal blinking staying `ALERT`, microsleep, high PERCLOS, yawning, short mouth-opening not counted as a yawn, head down, brief head movement ignored, lost face, recalibration, and rejecting a closed-eye baseline.

---

## Project structure

```
.
├── drowsiness_detector.py   # detection engine + camera loop + alarm
├── test_detector.py         # simulated-driver unit tests
├── requirements.txt         # pinned dependencies
└── README.md
```

---

## Known limitations

- **Not validated on a real-world dataset.** The logic is unit-tested with simulated data, but accuracy on real drivers has not been formally measured.
- **Glasses, sunglasses, strong glare and low light** can reduce landmark accuracy.
- **Extreme camera angles** (far from front-facing) affect EAR and head-pose readings.
- **Head-pose check is a simple ratio**, not full 3D pose estimation, so looking down at a phone may also trigger it.
- **Alarm sound:** a real beep plays on Windows. On Linux and macOS it uses the terminal bell, which may be silent.
- **This is a learning and portfolio project, not a certified safety system.** Do not rely on it as the only safeguard while driving.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Run `pip install mediapipe==0.10.14` |
| "Could not open the camera" | Check the camera index or IP URL, and that phone and PC share the same Wi-Fi |
| Alerts too early or too late | Tune `CLOSED_RATIO` and the other constants, then press `r` to recalibrate |
| Video lags on a phone stream | Lower the resolution in the IP camera app |
| It keeps restarting calibration | Keep your eyes open and your face visible during the first 5 seconds |

---

## Future improvements

- Streamlit dashboard with an event log
- Full 3D head-pose estimation (`solvePnP`)
- Evaluation on a public drowsiness dataset with measured accuracy
- Configurable thresholds through a settings file or command-line flags

---

## Adding a demo

A short GIF makes a big difference to how your repo is judged. Record about 15 seconds showing calibration, a blink, a long eye closure triggering the alarm, and a yawn. Save it as `demo.gif`, put it in the repo, and replace the placeholder at the top with:

```markdown
![Demo](demo.gif)
```

---

## Tech stack

Python, OpenCV, MediaPipe Face Mesh, NumPy

## License

MIT (add a `LICENSE` file, or choose another license on GitHub's "Add file" > "Create new file" > `LICENSE`).

## Author

**[Your Name]**: [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

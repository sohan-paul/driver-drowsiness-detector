Driver Drowsiness Detector-
Real-time, software-only drowsiness detection using Python, OpenCV and MediaPipe Face Mesh. It watches the driver's face through a webcam or a phone camera and raises a visual and audio alarm when it detects signs of fatigue. No special hardware is needed.

Features
Personal auto-calibration. The first 5 seconds measure your open-eye baseline, so the system adapts to different eye shapes and camera angles instead of using one fixed threshold.
Multiple fatigue signals, not just "eyes closed":
Signal	What it measures	Triggers
Microsleep	Eyes continuously closed	1.5 s or more -> DROWSY
PERCLOS	% of time eyes were closed in the last 60 s	12% -> WARNING, 25% -> DROWSY
Slow blinks	Blinks lasting 0.4 s or more	3 in 60 s -> WARNING
Yawning (MAR)	Mouth open wide for 1 s or more	2 yawns in 5 min -> WARNING
Head pose	Head dropping or turning away	2 s or more -> DROWSY
Face lost	Face not visible to the camera	2 s -> WARNING, 5 s -> DROWSY
Three clear states: ALERT, WARNING (early fatigue signs), DROWSY (alarm beep and red screen border).
Normal blinks are ignored, so alerts come from real fatigue, not routine blinking.
Works with a PC webcam or a phone camera over Wi-Fi (IP camera apps).
FPS-independent: all timing uses real seconds, not frame counts.
Robust to camera glitches: dropped frames on IP streams are tolerated, and resources are released cleanly on exit.
Unit-tested logic: the detection engine is separated from the camera code and tested with simulated drivers.
How it works
Face landmarks. MediaPipe Face Mesh returns 468+ landmarks per frame.
Eye Aspect Ratio (EAR) from 6 landmarks per eye:
   EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

EAR is high when the eye is open and drops toward 0 when it closes. Landmark coordinates are converted to pixels first, so the ratio isn't distorted by the frame's aspect ratio. 3. Calibration. The 70th percentile of EAR samples over 5 seconds (which ignores blink frames) becomes your open-eye baseline. An eye counts as closed when EAR falls below 75% of that baseline. If your eyes were shut during calibration, it restarts automatically. 4. Smoothing. EAR is averaged over 3 frames to reduce flicker. 5. Mouth Aspect Ratio (MAR) from inner-lip landmarks detects yawns. 6. Head pose. The nose tip's position between forehead and chin is compared with its calibrated baseline. 7. Decision. All signals are combined into one of three levels. Level 2 triggers the alarm.

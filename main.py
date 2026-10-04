import cv2
import mediapipe as mp

# Prompt the user for the camera source
user_input = input("Enter the phone camera URL (e.g., http://192.168.1.15:8080/video) or '0' for PC webcam: ")

# OpenCV requires an integer (0) for the local webcam, but a string for an IP camera URL
if user_input.strip() == '0':
    PHONE_CAMERA_URL = 0
else:
    PHONE_CAMERA_URL = user_input.strip()

# Initialize video capture with the selected source
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to connect to the camera. Please check the URL or your webcam connection.")
        break

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # This is where your EAR math goes to calculate eye closure
            # using specific landmark points (e.g., points 33, 133, 159, 145)
            pass

    cv2.imshow('Driver Drowsiness Detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
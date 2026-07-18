import os
import face_recognition
import pickle
import cv2
import subprocess
import time
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision
import streamlit as st


with open('face_encodings.pkl', 'rb') as f:
    known_face_encoding = pickle.load(f)

start_time = time.time()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
base_options = BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
with vision.FaceLandmarker.create_from_options(options) as landmarker:

    video = cv2.VideoCapture(0)
    frame_timestamp = int((time.time() - start_time) * 1000)
    missing_frames = 0
    LOCKOUT_THRESHOLD = 90  # Number of consecutive frames to trigger lockout

    while True:
        ret, frame = video.read()
        if not ret: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp += 16
        results = landmarker.detect_for_video(mp_image, frame_timestamp)
        if results.face_landmarks:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                match = face_recognition.compare_faces([known_face_encoding], encodings[0], tolerance=0.6)
                if match[0]:
                    label, color = "Owner Detected!", (0, 255, 0)
                    missing_frames = 0  # Reset the missing frames counter
                else:
                    label, color = "Unknown Face...", (0, 0, 255)
                    missing_frames += 1
            else:
                label, color = "Scanning...", (255, 255, 255)
                missing_frames += 1
        else:
                label, color = "Owner Missing!", (0, 0, 255)
                missing_frames += 1
        cv2.putText(frame, label, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press Q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 0), 2)
        cv2.imshow('Lockout Detector', frame)
        if ord('q') == cv2.waitKey(1):
            break
        if missing_frames >= LOCKOUT_THRESHOLD:
            video.release()
            cv2.destroyAllWindows()
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            break
        
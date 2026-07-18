"""
Lockout - webcam-based screen lock security tool.

Combines the old face_saver.py (enrollment) and cv-lockout.py (monitoring)
into one script.

Usage:
    python lockout.py --setup     # capture and save your face encoding
    python lockout.py             # run the monitor (locks screen if owner missing)
"""

import argparse
import os
import pickle
import subprocess
import time

import cv2
import face_recognition
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_PATH = os.path.join(SCRIPT_DIR, 'face_encodings.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

LOCKOUT_THRESHOLD = 90  # consecutive "owner missing/unknown" frames before locking
FACE_MATCH_TOLERANCE = 0.6


def open_camera():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam.")
    return video


def capture_face(video):
    """Enrollment loop: press space to capture, q to cancel. Returns an encoding or None."""
    label, color = "", (255, 255, 255)
    encoding = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        if key == ord(' '):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                encoding = encodings[0]
                label, color = "Face Captured!", (0, 255, 0)
            else:
                label, color = "No Face Detected", (0, 0, 255)
        elif key == ord('q'):
            print("Setup cancelled.")
            break

        cv2.putText(frame, label, (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press Space to Capture", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to Cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Lockout Setup', frame)

        if encoding is not None:
            cv2.waitKey(1000)
            break

    return encoding


def run_setup():
    video = open_camera()
    try:
        encoding = capture_face(video)
    finally:
        video.release()
        cv2.destroyAllWindows()

    if encoding is not None:
        with open(ENCODING_PATH, 'wb') as f:
            pickle.dump(encoding, f)
        print(f"Saved face encoding to {ENCODING_PATH}")
    else:
        print("No face saved.")


def run_reset():
    if os.path.exists(ENCODING_PATH):
        os.remove(ENCODING_PATH)
        print(f"Removed saved face encoding at {ENCODING_PATH}")
    else:
        print("No saved face encoding to remove.")


def load_known_encoding():
    if not os.path.exists(ENCODING_PATH):
        raise FileNotFoundError(
            "No saved face found. Run setup first: python lockout.py --setup"
        )
    with open(ENCODING_PATH, 'rb') as f:
        return pickle.load(f)


def run_monitor():
    known_encoding = load_known_encoding()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    video = open_camera()
    start_time = time.time()
    missing_frames = 0

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                frame_timestamp = int((time.time() - start_time) * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = landmarker.detect_for_video(mp_image, frame_timestamp)

                if results.face_landmarks:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb)
                    if encodings:
                        match = face_recognition.compare_faces(
                            [known_encoding], encodings[0], tolerance=FACE_MATCH_TOLERANCE
                        )
                        if match[0]:
                            label, color = "Owner Detected!", (0, 255, 0)
                            missing_frames = 0
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

                if cv2.waitKey(1) == ord('q'):
                    break

                if missing_frames >= LOCKOUT_THRESHOLD:
                    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
                    break
    finally:
        video.release()
        cv2.destroyAllWindows()


def main():
    import sys
    print("DEBUG argv:", sys.argv)
    parser = argparse.ArgumentParser(description="Webcam-based screen lock.")
    parser.add_argument(
        '--setup', action='store_true',
        help="Enroll your face before running the monitor."
    )
    parser.add_argument(
        '--reset', action='store_true',
        help="Delete the saved face encoding (simulates a fresh/new-user state)."
    )
    args = parser.parse_args()

    if args.reset:
        run_reset()
    elif args.setup:
        run_setup()
    else:
        run_monitor()


if __name__ == '__main__':
    main()

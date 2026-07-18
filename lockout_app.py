"""
Lockout - webcam-based screen lock, as a single-file Streamlit app.

Run with:
    streamlit run lockout_app.py

Requires face_landmarker.task in the same directory as this script.
"""

import os
import pickle
import subprocess
import threading
import time

import av
import cv2
import face_recognition
import mediapipe as mp
import streamlit as st
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_PATH = os.path.join(SCRIPT_DIR, "face_encodings.pkl")
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

LOCKOUT_THRESHOLD = 90  # consecutive "owner missing/unknown" frames before locking
FACE_MATCH_TOLERANCE = 0.6


# ---------------------------------------------------------------------------
# Encoding storage helpers
# ---------------------------------------------------------------------------

def has_saved_face() -> bool:
    return os.path.exists(ENCODING_PATH)


def save_encoding(encoding) -> None:
    with open(ENCODING_PATH, "wb") as f:
        pickle.dump(encoding, f)


def load_encoding():
    with open(ENCODING_PATH, "rb") as f:
        return pickle.load(f)


def delete_encoding() -> bool:
    """Returns True if a saved encoding existed and was removed."""
    if has_saved_face():
        os.remove(ENCODING_PATH)
        return True
    return False


def encode_face_from_frame(frame_bgr):
    """Given a BGR numpy frame, return a face encoding, or None if no face found."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    return encodings[0] if encodings else None


# ---------------------------------------------------------------------------
# Video processors
# ---------------------------------------------------------------------------

class CaptureProcessor(VideoProcessorBase):
    """Enroll tab: passes video through unchanged, keeps the latest frame
    so the Capture button can grab it on demand."""

    def __init__(self):
        self.last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        cv2.putText(
            img, "Look at the camera, then click Capture",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 0), 2,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class LockoutProcessor(VideoProcessorBase):
    """Monitor tab: runs face detection/matching per frame and locks the
    screen once the owner has been missing for LOCKOUT_THRESHOLD frames.
    Runs on a background thread managed by streamlit-webrtc, so state read
    by the main Streamlit thread (for the status panel) is guarded by a lock."""

    def __init__(self):
        self.known_encoding = load_encoding()
        self.missing_frames = 0
        self.status_label = "Starting..."
        self.lock = threading.Lock()
        self._locked_out = False  # prevents repeated lock calls after threshold is hit

        base_options = BaseOptions(model_asset_path = MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.start_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        frame_timestamp = int((time.time() - self.start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        results = self.landmarker.detect_for_video(mp_image, frame_timestamp)

        with self.lock:
            if results.face_landmarks:
                encoding = encode_face_from_frame(img)
                if encoding is not None:
                    match = face_recognition.compare_faces(
                        [self.known_encoding], encoding, tolerance=FACE_MATCH_TOLERANCE
                    )
                    if match[0]:
                        label, color = "Owner Detected!", (0, 255, 0)
                        self.missing_frames = 0
                    else:
                        label, color = "Unknown Face...", (0, 0, 255)
                        self.missing_frames += 1
                else:
                    label, color = "Scanning...", (255, 255, 255)
                    self.missing_frames += 1
            else:
                label, color = "Owner Missing!!", (0, 0, 255)
                self.missing_frames += 1

            self.status_label = label
            missing = self.missing_frames

        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if missing >= LOCKOUT_THRESHOLD and not self._locked_out:
            self._locked_out = True
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def close(self) -> None:
        self.landmarker.close()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Lockout", page_icon="🔒")
st.title("🔒 Lockout")
st.write(
    "A webcam-based screen lock. Your computer locks itself if your face "
    "leaves the camera, or if someone else's face is detected instead."
)

enroll_tab, monitor_tab = st.tabs(["Enroll", "Monitor"])

with enroll_tab:
    if has_saved_face():
        st.info("A face is already enrolled. Capturing again will overwrite it.")

    enroll_ctx = webrtc_streamer(key="enroll", video_processor_factory=CaptureProcessor)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Capture Face", disabled=not enroll_ctx.state.playing):
            if enroll_ctx.video_processor and enroll_ctx.video_processor.last_frame is not None:
                encoding = encode_face_from_frame(enroll_ctx.video_processor.last_frame)
                if encoding is not None:
                    save_encoding(encoding)
                    st.success("Face captured and saved.")
                else:
                    st.error("No face detected in that frame — try again, closer to the camera.")
            else:
                st.warning("Camera isn't ready yet — wait a moment and try again.")
    with col2:
        if st.button("Clear Saved Face", type="secondary"):
            if delete_encoding():
                st.success("Saved face removed.")
            else:
                st.info("No saved face to remove.")

with monitor_tab:
    if not has_saved_face():
        st.error("No face enrolled yet. Switch to the **Enroll** tab first.")
    else:
        monitor_ctx = webrtc_streamer(key="monitor", video_processor_factory=LockoutProcessor)

        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        # Re-run this section periodically so the status panel reflects what
        # the background video thread (LockoutProcessor.recv) is currently seeing.
        st_autorefresh(interval=500, key="monitor_refresh")

        if monitor_ctx.video_processor:
            with monitor_ctx.video_processor.lock:
                label = monitor_ctx.video_processor.status_label
                missing = monitor_ctx.video_processor.missing_frames

            status_placeholder.markdown(f"**Status:** {label}")
            progress_placeholder.progress(min(missing / LOCKOUT_THRESHOLD, 1.0))
        else:
            status_placeholder.info("Starting camera...")

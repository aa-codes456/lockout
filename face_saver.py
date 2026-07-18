import cv2
import face_recognition
import pickle

video = cv2.VideoCapture(0)
label, color = "", (255, 255, 255)
captured = False

while True:
    ret, frame = video.read()
    if not ret: break
    key = cv2.waitKey(1)
    if key == ord(' '):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            with open('face_encodings.pkl', 'wb') as f:
                pickle.dump(encodings[0], f)
                label, color = "Face Captured!", (0, 255, 0)
                captured = True
        else:
            label, color = "No Face Detected", (0, 0, 255)
    elif ord('q') == key:
        print("Setup cancelled.")
        break
    cv2.putText(frame, label, (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Press Space to Capture", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to Cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Face Detector Setup', frame)
    
    if captured:
        cv2.waitKey(1000)
        break

video.release()
cv2.destroyAllWindows()
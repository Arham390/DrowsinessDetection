# Eye/mouth detection logic
import cv2
import mediapipe as mp
from utils import eye_aspect_ratio, mouth_aspect_ratio, log_event
from alerts import alert_sound, alert_popup
import numpy as np

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.6

class DrowsinessDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(max_num_faces=1)
        self.eye_frame_counter = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    # Collect eye landmarks (left eye 33-38)
                    left_eye = np.array([[int(face_landmarks.landmark[i].x*w),
                                          int(face_landmarks.landmark[i].y*h)] for i in range(33,39)])
                    right_eye = np.array([[int(face_landmarks.landmark[i].x*w),
                                           int(face_landmarks.landmark[i].y*h)] for i in range(263,269)])
                    # EAR calculation
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < EAR_THRESHOLD:
                        self.eye_frame_counter += 1
                        if self.eye_frame_counter >= EAR_CONSEC_FRAMES:
                            log_event("Drowsy Eyes")
                            alert_sound()
                            alert_popup()
                    else:
                        self.eye_frame_counter = 0

                    # Mouth landmarks 78-308 simplified for example
                    mouth = np.array([[int(face_landmarks.landmark[i].x*w),
                                       int(face_landmarks.landmark[i].y*h)] for i in [78,81,13,14,308,312,311,310,317,13,14,19]])
                    mar = mouth_aspect_ratio(mouth)
                    if mar > MAR_THRESHOLD:
                        log_event("Yawning")
                        alert_sound()
                        alert_popup()

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
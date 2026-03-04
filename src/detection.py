# Eye/mouth detection logic
import cv2
import mediapipe as mp
import numpy as np
import os
from utils import eye_aspect_ratio, mouth_aspect_ratio, log_event
from alerts import alert_sound, alert_popup

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.6

# MediaPipe FaceMesh landmark indices
# Left eye: 362, 385, 387, 263, 373, 380
# Right eye: 33, 160, 158, 133, 153, 144
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth: top=13, bottom=14, left=78, right=308, plus inner points
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# Path to the FaceLandmarker model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")


class DrowsinessDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.eye_frame_counter = 0

        # Set up FaceLandmarker using new Tasks API
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_timestamp_ms += 33  # ~30 FPS
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    h, w, _ = frame.shape

                    # Extract left eye landmarks
                    left_eye = np.array([[int(face_landmarks[i].x * w),
                                          int(face_landmarks[i].y * h)] for i in LEFT_EYE])
                    # Extract right eye landmarks
                    right_eye = np.array([[int(face_landmarks[i].x * w),
                                           int(face_landmarks[i].y * h)] for i in RIGHT_EYE])

                    # EAR calculation
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    # Draw eye contours
                    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

                    # Display EAR on frame
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if ear < EAR_THRESHOLD:
                        self.eye_frame_counter += 1
                        if self.eye_frame_counter >= EAR_CONSEC_FRAMES:
                            cv2.putText(frame, "DROWSY!", (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            log_event("Drowsy Eyes")
                            alert_sound()
                            alert_popup()
                    else:
                        self.eye_frame_counter = 0

                    # Extract mouth landmarks
                    mouth = np.array([[int(face_landmarks[i].x * w),
                                       int(face_landmarks[i].y * h)] for i in MOUTH])
                    mar = mouth_aspect_ratio(mouth)

                    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if mar > MAR_THRESHOLD:
                        cv2.putText(frame, "YAWNING!", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        log_event("Yawning")
                        alert_sound()
                        alert_popup()

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.landmarker.close()
        self.cap.release()
        cv2.destroyAllWindows()
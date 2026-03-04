# Helper functions (EAR, MAR, logging)
import numpy as np
import pandas as pd
from datetime import datetime
import os

logs = []

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) from 6 eye landmarks."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) from 8 mouth landmarks.
    Landmarks: [78, 81, 13, 311, 308, 402, 14, 178]
    Index:       0   1   2    3    4    5   6    7
    """
    # Vertical distances
    A = np.linalg.norm(mouth[1] - mouth[7])  # 81 - 178
    B = np.linalg.norm(mouth[3] - mouth[5])  # 311 - 402
    # Horizontal distance
    C = np.linalg.norm(mouth[0] - mouth[4])  # 78 - 308
    mar = (A + B) / (2.0 * C)
    return mar

def log_event(event_type):
    logs.append({"event": event_type, "time": datetime.now()})

def save_logs():
    if not os.path.exists("../logs"):
        os.makedirs("../logs")
    df = pd.DataFrame(logs)
    df.to_csv("../logs/drowsiness_log.csv", index=False)
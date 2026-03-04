# Helper functions (EAR, MAR, logging)
import numpy as np
import pandas as pd
from datetime import datetime
import os

logs = []

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[12] - mouth[16])
    mar = (A + B) / (2.0 * C)
    return mar

def log_event(event_type):
    logs.append({"event": event_type, "time": datetime.now()})

def save_logs():
    if not os.path.exists("../logs"):
        os.makedirs("../logs")
    df = pd.DataFrame(logs)
    df.to_csv("../logs/drowsiness_log.csv", index=False)
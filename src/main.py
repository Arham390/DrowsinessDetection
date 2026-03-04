# Entry point for Drowsiness Detector
from detection import DrowsinessDetector
from utils import save_logs

if __name__ == "__main__":
    detector = DrowsinessDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        save_logs()
        print("Logs saved to logs/drowsiness_log.csv")
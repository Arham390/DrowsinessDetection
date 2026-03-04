# Sound / GUI alerts
import winsound
import threading

# Use a cooldown to prevent alert spam
_alert_cooldown = False
_COOLDOWN_SECONDS = 3

def _reset_cooldown():
    global _alert_cooldown
    _alert_cooldown = False

def alert_sound():
    """Play a beep alert sound using Windows built-in sounds."""
    global _alert_cooldown
    if _alert_cooldown:
        return
    _alert_cooldown = True
    threading.Timer(_COOLDOWN_SECONDS, _reset_cooldown).start()
    # Play system beep in background thread (frequency=1000Hz, duration=500ms)
    threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()

def alert_popup():
    """Show a visual alert - we use on-screen text via OpenCV instead of tkinter
    to avoid threading issues. The text is drawn directly in detection.py."""
    pass  # Visual alert is handled by cv2.putText in detection.py
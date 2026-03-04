# Sound / GUI alerts
from playsound import playsound
from tkinter import messagebox, Tk
import threading

def alert_sound():
    threading.Thread(target=lambda: playsound("../data/alarm.wav")).start()

def alert_popup():
    def popup():
        root = Tk()
        root.withdraw()
        messagebox.showwarning("Drowsiness Alert", "Wake up!")
        root.destroy()
    threading.Thread(target=popup).start()
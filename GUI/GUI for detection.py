import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import threading
import time

class EpilepsyDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Epilepsy Detection System")
        master.geometry("800x600")
        master.configure(bg='white')

        #canvas
        self.canvas = tk.Canvas(master, width=750, height=400, bg='white', borderwidth=2, relief=tk.SUNKEN)
        self.canvas.pack(padx=20, pady=10)

        
        self.status_frame = tk.Frame(master, bg='white')
        self.status_frame.pack(pady=10)

    
        self.status_label = tk.Label(
            self.status_frame, 
            text="Status: Monitoring", 
            font=('Arial', 14), 
            bg='white'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

   
        self.detection_label = tk.Label(
            self.status_frame, 
            text="Seizure: Not Detected", 
            font=('Arial', 14), 
            fg='green', 
            bg='white'
        )
        self.detection_label.pack(side=tk.LEFT, padx=10)

    
        self.emergency_button = tk.Button(
            master, 
            text="Emergency Contact", 
            bg='red', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            command=self.trigger_emergency
        )
        self.emergency_button.pack(pady=10)
        self.signal_data = [0] * 500
        self.is_running = True

        self.signal_thread = threading.Thread(target=self.generate_and_update_signal)
        self.signal_thread.daemon = True
        self.signal_thread.start()

    def generate_signal(self):
        #simulate an EEG-like signal (can change it to ours)
        t = time.time()
        low_freq = np.sin(2 * np.pi * 1 * t)
        high_freq = np.sin(2 * np.pi * 10 * t)
        
        #spike generator
        seizure_prob = random.random()
        seizure_spike = 5 if seizure_prob < 0.02 else 0
        
        return low_freq + high_freq + seizure_spike

    def generate_and_update_signal(self):
        while self.is_running:
            # generate our signal
            new_signal = self.generate_signal()
            
            self.signal_data.pop(0)
            self.signal_data.append(new_signal)
            
            #check for seizure detection
            if abs(new_signal) > 6:
                self.master.after(0, self.update_seizure_status)
            
            self.master.after(0, self.draw_signal)
            
            time.sleep(0.05)  #50ms delay between updates

    def draw_signal(self):
        self.canvas.delete('all')
        width = 750
        height = 400
    
        self.canvas.create_line(0, height/2, width, height/2, fill='lightgray')
        
        points = []
        for i, value in enumerate(self.signal_data):
            x = i * (width / len(self.signal_data))
            y = height/2 - value * 30
            points.append((x, y))
        
        if len(points) > 1:
            for i in range(len(points) - 1):
                self.canvas.create_line(
                    points[i][0], points[i][1], 
                    points[i+1][0], points[i+1][1], 
                    fill='blue'
                )

    def update_seizure_status(self):
        self.detection_label.config(
            text="Seizure: DETECTED!", 
            fg='red'
        )
        messagebox.showwarning(
            "Emergency", 
            "Potential Seizure Detected!\nContacting Emergency Services..."
        )

    def trigger_emergency(self):
        messagebox.showerror(
            "Emergency Contact", 
            "Manually Triggered Emergency Services\nLocation Sharing Activated"
        )

def main():
    root = tk.Tk()
    app = EpilepsyDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
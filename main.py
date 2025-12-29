import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import random

import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageTk

import pygame

# ------------ Configuration ------------
MUSIC_DIRS = {
    "happy": "music/happy",
    "sad": "music/sad",
    "angry": "music/angry",
    "neutral": "music/neutral"
}
ANALYZE_INTERVAL = 2.0  # seconds between DeepFace analyses (reduce to speed up)
FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# ---------------------------------------

class EmotionMusicApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Emotion-Based Music Player")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Video variables
        self.cap = None
        self.running = False
        self.current_emotion = "neutral"
        self.last_analysis_time = 0.0
        self.current_song = None
        self.player_lock = threading.Lock()

        # Initialize pygame mixer
        pygame.mixer.init()

        # Build UI
        self.build_ui()

    def build_ui(self):
        mainframe = ttk.Frame(self.root, padding=8)
        mainframe.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Video panel using a label
        self.video_label = ttk.Label(mainframe)
        self.video_label.grid(row=0, column=0, columnspan=4)

        # Emotion label
        self.emotion_var = tk.StringVar(value="Emotion: -")
        self.emotion_label = ttk.Label(mainframe, textvariable=self.emotion_var, font=("Helvetica", 14))
        self.emotion_label.grid(row=1, column=0, sticky="w", pady=(8,0))

        # Buttons
        self.start_btn = ttk.Button(mainframe, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=2, column=0, pady=8)

        self.stop_btn = ttk.Button(mainframe, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=2, column=1, pady=8)

        self.pause_music_btn = ttk.Button(mainframe, text="Pause/Unpause Music", command=self.toggle_music, state="disabled")
        self.pause_music_btn.grid(row=2, column=2, pady=8)

        self.stop_music_btn = ttk.Button(mainframe, text="Stop Music", command=self.stop_music, state="disabled")
        self.stop_music_btn.grid(row=2, column=3, pady=8)

        # Dropdown to override auto emotion (optional)
        ttk.Label(mainframe, text="Override Emotion:").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.override_var = tk.StringVar(value="auto")
        options = ["auto", "happy", "sad", "angry", "neutral"]
        self.override_menu = ttk.OptionMenu(mainframe, self.override_var, *options)
        self.override_menu.grid(row=3, column=1, sticky="w", pady=(6,0))

        ttk.Label(mainframe, text="Analysis interval (s):").grid(row=3, column=2, sticky="e", pady=(6,0))
        self.interval_var = tk.DoubleVar(value=ANALYZE_INTERVAL)
        self.interval_spin = ttk.Spinbox(mainframe, from_=0.5, to=10.0, increment=0.5, textvariable=self.interval_var, width=5)
        self.interval_spin.grid(row=3, column=3, sticky="w", pady=(6,0))

    # === Camera & loop ===
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_music_btn.config(state="normal")
        self.stop_music_btn.config(state="normal")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Flip horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            # Detect face(s) quickly with Haar cascade (for drawing box)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Determine whether to run DeepFace (throttle)
            now = time.time()
            self.last_analysis_time = float(self.interval_var.get())
            analyze_every = float(self.interval_var.get())
            if now - getattr(self, "last_analysis_timestamp", 0) > analyze_every:
                # Run analysis in a separate thread to avoid blocking video
                img_for_analysis = frame.copy()
                threading.Thread(target=self._analyze_frame, args=(img_for_analysis,), daemon=True).start()
                self.last_analysis_timestamp = now

            # Show emotion label on frame
            cv2.putText(frame, f"Emotion: {self.current_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Convert frame to ImageTk
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img_pil)

            # Update UI (on main thread)
            self.video_label.imgtk = imgtk  # keep reference
            self.video_label.configure(image=imgtk)

            time.sleep(0.02)  # small delay for responsiveness

    # === Emotion analysis & music control ===
    def _analyze_frame(self, frame):
        # If user overrides emotion -> skip analysis and use override
        override = self.override_var.get()
        if override != "auto":
            detected = override
        else:
            try:
                # DeepFace expects RGB image in many backends
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Using enforce_detection=False to avoid exceptions if no clear face
                result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                # result may be a dict or list depending on version; handle both
                if isinstance(result, list):
                    result = result[0]
                detected = result.get('dominant_emotion', 'neutral')
            except Exception as e:
                # analysis failed -> keep previous or set to neutral
                print("DeepFace analyze error:", e)
                detected = self.current_emotion or "neutral"

        # Normalize to lower-case expected keys
        detected = str(detected).lower()
        if detected not in MUSIC_DIRS:
            detected = "neutral"

        # If emotion changed, update and play new music
        if detected != self.current_emotion:
            print(f"[INFO] Emotion changed: {self.current_emotion} -> {detected}")
            self.current_emotion = detected
            self.root.after(0, lambda: self.emotion_var.set(f"Emotion: {self.current_emotion}"))
            self.play_random_song_for_emotion(detected)

    def play_random_song_for_emotion(self, emotion):
        folder = MUSIC_DIRS.get(emotion, MUSIC_DIRS["neutral"])
        if not os.path.exists(folder):
            print(f"[WARN] Music folder not found: {folder}")
            return

        songs = [f for f in os.listdir(folder) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
        if not songs:
            print(f"[WARN] No songs in {folder}")
            return

        song_path = os.path.join(folder, random.choice(songs))
        # Start playback on a thread-safe block
        with self.player_lock:
            try:
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
                self.current_song = song_path
                print("[PLAYING]", song_path)
            except Exception as e:
                print("Error playing song:", e)

    def toggle_music(self):
        # Pause/unpause
        try:
            if pygame.mixer.music.get_busy():
                if not getattr(self, "_paused", False):
                    pygame.mixer.music.pause()
                    self._paused = True
                else:
                    pygame.mixer.music.unpause()
                    self._paused = False
        except Exception as e:
            print("toggle_music error:", e)

    def stop_music(self):
        try:
            pygame.mixer.music.stop()
            self.current_song = None
        except Exception as e:
            print("stop_music error:", e)

    def on_close(self):
        # Clean up
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        self.root.destroy()

if _name_ == "_main_":
    # Validate music directories exist (create empty if not)
    for k, d in MUSIC_DIRS.items():
        if not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
                print(f"Created missing folder: {d} (add some mp3/wav files there)")
            except Exception as e:
                print(f"Could not create folder {d}: {e}")

    root = tk.Tk()
    app = EmotionMusicApp(root)
    root.mainloop(

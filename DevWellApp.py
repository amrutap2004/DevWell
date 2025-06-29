import cv2
import mediapipe as mp
import time
import sqlite3
import pyttsx3
import threading
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard, mouse
from PyQt5 import QtWidgets, QtGui, QtCore
from plyer import notification
import random
import sys
import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
import traceback

class DevWellApp(QtWidgets.QMainWindow):
    # Add signal for database operations
    db_signal = QtCore.pyqtSignal(dict)
    # Add signal for stopping tracking
    stop_tracking_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.keyboard_activity = 0
        self.mouse_activity = 0
        self.tracking_active = False
        self.cap = None
        self.user_data = {}  # Dictionary to store data for each user
        self.multiple_faces_alerted = False  # Flag to track if multiple faces alert has been shown
        self.primary_user_id = None  # Track the primary user
        self.user_detection_cooldown = 5  # Seconds between user detection checks
        self.last_user_detection_time = 0  # Track last user detection check
        self.user_timeout = 3  # Seconds before considering a user no longer present
        self.current_ear = 0.45  # Default eye aspect ratio
        self.current_posture_score = 50  # Default posture score
        
        # Enhanced blink monitoring
        self.blink_count = 0  # Count blinks in the current minute
        self.blink_timer = time.time()  # Timer to track one minute intervals
        self.min_blink_threshold = 17  # Minimum blinks per minute (updated)
        self.last_blink_status = False  # Track if eyes were closed in previous frame
        self.low_blink_alert_time = 0  # Last time we alerted about low blink rate
        
        # Tiredness detection
        self.eyes_closed_start_time = None  # When eyes were closed
        self.tiredness_threshold = 30  # Alert after 30 seconds of closed eyes
        self.last_tiredness_alert_time = 0  # Last time we alerted about tiredness
        self.tiredness_alert_cooldown = 300  # 5 minutes between tiredness alerts
        
        # Enhanced posture monitoring
        self.bad_posture_start_time = None  # When bad posture was first detected
        self.bad_posture_threshold = 120  # Alert after 2 minutes (120 seconds) of bad posture
        self.last_posture_alert_time = 0  # Last time we alerted about bad posture
        self.posture_alert_cooldown = 180  # 3 minutes between posture alerts
        
        # Original variables
        self.ear_threshold = 0.45  # Threshold for eye aspect ratio (eyes closed)
        self.ear_upper_threshold = 0.50  # Upper threshold for blink detection
        self.blink_cooldown = 10  # Seconds between blink alerts
        self.last_blink_time = time.time()  # Track last blink alert
        self.posture_cooldown = 30  # 30 seconds between posture alerts
        self.last_posture_alert = 0  # Track last posture alert
        self.ear_history = []  # Store recent EAR values for smoothing
        self.ear_history_size = 5  # Number of frames to average
        self.posture_history = []  # Store recent posture scores for smoothing
        self.posture_history_size = 5  # Number of frames to average
        self.health_tips = [
            "Stay hydrated! Drink water regularly.",
            "Take deep breaths to reduce stress.",
            "Stretch your neck and shoulders.",
            "Remember to maintain good posture.",
            "Give your eyes a break by looking at distant objects.",
            "Take a short walk to improve circulation.",
            "Do some wrist exercises to prevent strain.",
            "Keep your screen at eye level.",
            "Maintain an arm's length distance from your screen.",
            "Practice the 20-20-20 rule: Look 20 feet away for 20 seconds every 20 minutes."
        ]
        self.bad_posture_count = 0
        self.posture_prev_bad = False
        self.posture_warning_time = time.time()  # Track last posture alert
        self.eyes_closed = False  # Track if eyes are currently closed
        self.last_blink_detected = time.time()  # Track last actual blink
        self.no_blink_threshold = 15  # Alert if no blink for 15 seconds
        self.last_static_image_alert = 0  # Last time we alerted about static image
        self.static_image_alert_cooldown = 300  # 5 minutes between static image alerts

        self.initUI()
        self.initDatabase()
        self.initTracking()
        self.initVoiceAlert()

        # Initialize timers
        self.break_timer = QtCore.QTimer()
        self.break_timer.timeout.connect(self.suggest_break)
        self.break_timer.start(3600000)  # 1 hour interval

        # Status update timer
        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # Update every 100ms for smooth progress bars

        # Connect the signal to the stop_tracking slot
        self.stop_tracking_signal.connect(self.stop_tracking)

    def initUI(self):
        try:
            self.setWindowTitle("DevWell - Your Health Companion")
            self.setGeometry(100, 100, 1200, 800)
            
            # Modern dark theme stylesheet
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1E1E2F;
                }

                QLabel {
                    color: #FFFFFF;
                    font-size: 16px;
                }

                QPushButton {
                    background-color: #2ECC71;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    min-width: 150px;
                    font-weight: bold;
                }

                QPushButton:hover {
                    background-color: #27AE60;
                }

                QPushButton:pressed {
                    background-color: #1E8449;
                }

                QPushButton:disabled {
                    background-color: #7F8C8D;
                    color: #BDC3C7;
                }

                QGroupBox {
                    color: white;
                    font-size: 18px;
                    border: 2px solid #3498DB;
                    border-radius: 10px;
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #1E283A;
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 10px;
                    color: #3498DB;
                    font-weight: bold;
                }

                QProgressBar {
                    border: 2px solid #3498DB;
                    border-radius: 5px;
                    text-align: center;
                    color: white;
                    background-color: #34495E;
                    font-weight: bold;
                }

                QProgressBar::chunk {
                    background-color: #2ECC71;
                    border-radius: 3px;
                }

                QDialog {
                    background-color: #1E1E2F;
                }
            """)

            # Main container widget
            main_widget = QtWidgets.QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QtWidgets.QVBoxLayout(main_widget)

            # Header with logo
            header_layout = QtWidgets.QHBoxLayout()
            
            # Create a label for the logo
            logo_label = QtWidgets.QLabel()
            # Creating a simple programmatic logo
            logo_pixmap = QtGui.QPixmap(100, 100)
            logo_pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(logo_pixmap)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#3498DB")))
            painter.drawEllipse(10, 10, 80, 80)
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#2ECC71")))
            painter.drawEllipse(30, 30, 40, 40)
            painter.end()
            logo_label.setPixmap(logo_pixmap)
            
            header_layout.addWidget(logo_label)
            
            title_label = QtWidgets.QLabel("DevWell")
            title_label.setStyleSheet("""
                font-size: 48px;
                font-weight: bold;
                color: #ECF0F1;
            """)
            
            subtitle_label = QtWidgets.QLabel("Your Developer Health Companion")
            subtitle_label.setStyleSheet("""
                font-size: 24px;
                color: #3498DB;
            """)
            
            title_layout = QtWidgets.QVBoxLayout()
            title_layout.addWidget(title_label)
            title_layout.addWidget(subtitle_label)
            
            header_layout.addLayout(title_layout)
            header_layout.addStretch()
            
            main_layout.addLayout(header_layout)

            # Create horizontal layout for camera and status
            content_layout = QtWidgets.QHBoxLayout()

            # Camera frame display
            camera_group = QtWidgets.QGroupBox("Camera Preview")
            camera_layout = QtWidgets.QVBoxLayout()
            self.camera_frame = QtWidgets.QLabel()
            self.camera_frame.setMinimumSize(640, 480)
            self.camera_frame.setStyleSheet("""
                QLabel {
                    background-color: #2C3E50;
                    border: 2px solid #3498DB;
                    border-radius: 10px;
                }
            """)
            camera_layout.addWidget(self.camera_frame)
            camera_group.setLayout(camera_layout)
            content_layout.addWidget(camera_group)

            # Status section
            status_group = QtWidgets.QGroupBox("Current Status")
            status_layout = QtWidgets.QVBoxLayout()

            # Eye Status
            self.eye_status = QtWidgets.QLabel("👁 Eye Health: Not Monitoring")
            status_layout.addWidget(self.eye_status)
            
            # Blink Counter Status
            self.blink_status = QtWidgets.QLabel("😌 Blinks: 0/17 per minute")
            status_layout.addWidget(self.blink_status)

            # Posture Status
            self.posture_status = QtWidgets.QLabel("🪑 Posture: Not Monitoring")
            status_layout.addWidget(self.posture_status)

            # Activity Status
            self.activity_status = QtWidgets.QLabel("⌨ Activity: Not Monitoring")
            status_layout.addWidget(self.activity_status)

            status_group.setLayout(status_layout)
            content_layout.addWidget(status_group)

            # Add the content layout to main layout
            main_layout.addLayout(content_layout)

            # Control buttons
            button_layout = QtWidgets.QHBoxLayout()

            self.start_button = QtWidgets.QPushButton("Start Monitoring")
            self.start_button.setStyleSheet("background-color: #2ECC71; color: white;")
            self.start_button.clicked.connect(self.start_tracking)

            self.stop_button = QtWidgets.QPushButton("Stop Monitoring")
            self.stop_button.setStyleSheet("background-color: #E74C3C; color: white;")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self.stop_tracking)

            self.report_button = QtWidgets.QPushButton("Generate Report")
            self.report_button.setStyleSheet("background-color: #3498DB; color: white;")
            self.report_button.clicked.connect(lambda: self.generate_report('weekly'))

            self.settings_button = QtWidgets.QPushButton("Settings")
            self.settings_button.setStyleSheet("background-color: #9B59B6; color: white;")
            self.settings_button.clicked.connect(self.show_settings)

            button_layout.addWidget(self.start_button)
            button_layout.addWidget(self.stop_button)
            button_layout.addWidget(self.report_button)
            button_layout.addWidget(self.settings_button)

            main_layout.addLayout(button_layout)

            # Health tips section
            tips_group = QtWidgets.QGroupBox("Health Tips")
            tips_layout = QtWidgets.QVBoxLayout()

            self.tip_label = QtWidgets.QLabel()
            self.tip_label.setStyleSheet("""
                font-size: 18px;
                color: #F1C40F;
                padding: 15px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                margin: 10px;
            """)
            self.tip_label.setAlignment(QtCore.Qt.AlignCenter)
            self.tip_label.setWordWrap(True)
            tips_layout.addWidget(self.tip_label)

            refresh_tip_button = QtWidgets.QPushButton("New Tip")
            refresh_tip_button.setStyleSheet("background-color: #F39C12; color: white;")
            refresh_tip_button.clicked.connect(self.update_health_tip)
            tips_layout.addWidget(refresh_tip_button)

            tips_group.setLayout(tips_layout)
            main_layout.addWidget(tips_group)

            # Status bar
            self.statusBar().showMessage("Ready to start monitoring")

            # Update initial health tip
            self.update_health_tip()

        except Exception as e:
            self.show_error("UI Initialization Error", str(e))

    def initDatabase(self):
        try:
            # Create a dedicated database directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            db_path = os.path.join("data", "devwell.db")
            
            # Connect to database with proper error handling
            try:
                self.conn = sqlite3.connect(db_path, check_same_thread=False)
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                self.show_error("Database Connection Error", f"Failed to connect to database: {str(e)}")
                return
            
            # Create activity table with all required columns
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    eye_alerts INTEGER DEFAULT 0,
                    posture_alerts INTEGER DEFAULT 0,
                    breaks_taken INTEGER DEFAULT 0,
                    keyboard_activity INTEGER DEFAULT 0,
                    mouse_activity INTEGER DEFAULT 0,
                    low_light_alerts INTEGER DEFAULT 0,
                    session_duration INTEGER DEFAULT 0
                )
            """)
            
            # Create index on timestamp for faster queries
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activity_timestamp 
                ON activity(timestamp)
            """)
            
            # Create settings table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_name TEXT UNIQUE,
                    setting_value TEXT
                )
            """)
            
            # Create blink data table for detailed blink tracking
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS blink_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    blink_count INTEGER,
                    minute_interval INTEGER
                )
            """)
            
            # Insert default settings if they don't exist
            default_settings = {
                'ear_threshold': '0.45',
                'min_blink_threshold': '17',
                'bad_posture_threshold': '120',
                'keyboard_limit': '2500',
                'mouse_limit': '2500'
            }
            
            for setting_name, setting_value in default_settings.items():
                self.cursor.execute("""
                    INSERT OR IGNORE INTO settings (setting_name, setting_value)
                    VALUES (?, ?)
                """, (setting_name, setting_value))
            
            self.conn.commit()
            print("Database initialized successfully")
            
        except Exception as e:
            error_msg = f"Database initialization error: {str(e)}"
            print(error_msg)
            self.show_error("Database Error", error_msg)
            traceback.print_exc()

    def initTracking(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=10,  # Allow up to 10 faces
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False
            )
            self.mp_pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.keyboard_limit = 2500
            self.mouse_limit = 2500

            self.listener_kb = keyboard.Listener(on_press=self.on_key_press)
            self.listener_mouse = mouse.Listener(on_click=self.on_mouse_click)
            self.listener_kb.start()
            self.listener_mouse.start()
        except Exception as e:
            self.show_error("Tracking Initialization Error", str(e))

    def initVoiceAlert(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            self.show_error("Voice Alert Error", str(e))

    def show_error(self, title, message):
        QtWidgets.QMessageBox.critical(self, title, message)

    def speak(self, message):
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Voice alert error: {str(e)}")

    def show_notification(self, message):
        try:
            notification.notify(title='DevWell Alert', message=message, timeout=5)
        except Exception as e:
            print(f"Notification error: {str(e)}")

    def calculate_eye_aspect_ratio(self, landmarks, indices):
        """Calculate eye aspect ratio for a single eye with improved reliability"""
        try:
            # Extract the 6 points for the eye
            points = [landmarks[i] for i in indices]
            
            # Calculate vertical distances
            v1 = np.linalg.norm(np.array([points[1].x - points[4].x, points[1].y - points[4].y]))
            v2 = np.linalg.norm(np.array([points[2].x - points[5].x, points[2].y - points[5].y]))
            
            # Calculate horizontal distance
            h = np.linalg.norm(np.array([points[0].x - points[3].x, points[0].y - points[3].y]))
            
            # Calculate eye aspect ratio with safeguard against division by zero
            if h > 0.001:  # Avoid division by very small numbers
                ear = (v1 + v2) / (2.0 * h)
            else:
                ear = 0.3  # Default value when horizontal distance is too small
            
            # Apply moving average smoothing but with less history for faster response
            self.ear_history.append(ear)
            if len(self.ear_history) > 3:  # Reduced from 5 to 3 for faster response
                self.ear_history.pop(0)
            
            # Return smoothed value
            return sum(self.ear_history) / len(self.ear_history)
            
        except Exception as e:
            print(f"Eye aspect ratio calculation error: {str(e)}")
            return 0.3  # Return default value on error

    def detect_eye_strain(self, image):
        try:
            # Check for low light conditions first
            gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray_frame)
            
            if avg_brightness < 40:  # Low light threshold
                # Reset all counters and timers during low light
                self.blink_count = 0
                self.blink_timer = time.time()
                self.last_blink_status = False
                self.eyes_closed_start_time = None
                self.last_blink_detected = time.time()
                self.eye_status.setText("👁 Eye Health: Low Light Conditions")
                return

            results = self.mp_face_mesh.process(image)
            if not results.multi_face_landmarks:
                # Reset all counters and timers when no face is detected
                self.blink_count = 0
                self.blink_timer = time.time()
                self.last_blink_status = False
                self.eyes_closed_start_time = None
                self.last_blink_detected = time.time()
                self.eye_status.setText("👁 Eye Health: No Face Detected")
                return

            # Calculate eye aspect ratio for both eyes
            left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
            right_eye_indices = [362, 385, 387, 263, 373, 380]  # Right eye landmarks
            
            for face_landmarks in results.multi_face_landmarks:
                # Calculate eye aspect ratio for both eyes
                left_eye = self.calculate_eye_aspect_ratio(face_landmarks.landmark, left_eye_indices)
                right_eye = self.calculate_eye_aspect_ratio(face_landmarks.landmark, right_eye_indices)
                
                # Average the eye aspect ratio
                current_ear = (left_eye + right_eye) / 2.0
                self.current_ear = current_ear

                current_time = time.time()
                
                # Simplified blink detection
                if current_ear <= self.ear_threshold and not self.last_blink_status:
                    # Count this as a blink
                    self.blink_count += 1
                    self.last_blink_detected = current_time
                    print(f"Blink detected! Count: {self.blink_count}, EAR: {current_ear:.3f}")
                
                # Update last blink status
                self.last_blink_status = current_ear <= self.ear_threshold
                
                # Check blink rate every minute
                if current_time - self.blink_timer >= 60:
                    if self.blink_count < self.min_blink_threshold:
                        self.show_notification(f"Low blink rate detected! Only {self.blink_count} blinks in the last minute.")
                        self.speak(f"Low blink rate detected. You only blinked {self.blink_count} times in the last minute.")
                        self.eye_status.setText("👁 Eye Health: Low Blink Rate Alert")
                        # Log eye alert to database
                        self.log_activity(eye_alerts=1)
                    print(f"Resetting blink count. Total blinks in last minute: {self.blink_count}")
                    self.blink_count = 0
                    self.blink_timer = current_time
                
                # Check for tiredness (eyes closed for 30 seconds)
                if current_ear <= self.ear_threshold:  # Eyes are closed
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = current_time
                    elif current_time - self.eyes_closed_start_time >= 30:  # 30 seconds threshold
                        if (current_time - self.last_tiredness_alert_time) >= self.tiredness_alert_cooldown:
                            self.show_notification("You seem tired! Your eyes have been closed for 30 seconds.")
                            self.speak("You seem tired. Your eyes have been closed for 30 seconds. Consider taking a short break.")
                            self.last_tiredness_alert_time = current_time
                            self.eye_status.setText("👁 Eye Health: Tiredness Alert")
                            # Log eye alert to database
                            self.log_activity(eye_alerts=1)
                else:
                    self.eyes_closed_start_time = None  # Reset timer when eyes are open
                
                # Check for static image (no blink for 50 seconds)
                time_since_last_blink = current_time - self.last_blink_detected
                if time_since_last_blink >= 50:  # If no blink for 50 seconds
                    if (current_time - self.last_static_image_alert) >= self.static_image_alert_cooldown:
                        self.show_notification("Static content alert! A real user is required. Are you there?")
                        self.speak("Static content alert.A real user is required. Are you there?")
                        self.eye_status.setText("👁 Eye Health: Static Content Alert A real user is required Are you there?")
                        # Log eye alert to database
                        self.log_activity(eye_alerts=1)
                        self.last_static_image_alert = current_time
                
                # Update eye status with more detailed information
                if time_since_last_blink >= 50:
                    self.eye_status.setText(f"👁 Eye Health: No Blinks ({int(time_since_last_blink)}s)")
                else:
                    self.eye_status.setText(f"👁 Eye Health: Good (Last blink: {int(time_since_last_blink)}s ago)")
                
                # Update blink status display
                self.blink_status.setText(f"😌 Blinks: {self.blink_count}")

        except Exception as e:
            print(f"Eye strain detection error: {str(e)}")
            traceback.print_exc()

    def log_activity(self, eye_alerts=0, posture_alerts=0, breaks_taken=0, 
                     keyboard_activity=0, mouse_activity=0, 
                     low_light_alerts=0, session_duration=0):
        """Log activity data to the database with proper transaction handling"""
        try:
            # Check if we have an active session record within the last 5 minutes
            self.cursor.execute("""
                SELECT id, session_duration FROM activity 
                WHERE timestamp >= datetime('now', '-5 minutes')
                ORDER BY timestamp DESC LIMIT 1
            """)
            recent_record = self.cursor.fetchone()
            
            if recent_record:
                # Update existing record
                self.cursor.execute("""
                    UPDATE activity SET
                        eye_alerts = eye_alerts + ?,
                        posture_alerts = posture_alerts + ?,
                        breaks_taken = breaks_taken + ?,
                        keyboard_activity = keyboard_activity + ?,
                        mouse_activity = mouse_activity + ?,
                        low_light_alerts = low_light_alerts + ?,
                        session_duration = session_duration + ?
                    WHERE id = ?
                """, (eye_alerts, posture_alerts, breaks_taken, 
                      keyboard_activity, mouse_activity, low_light_alerts,
                      session_duration, recent_record[0]))
            else:
                # Create new record
                self.cursor.execute("""
                    INSERT INTO activity (
                        eye_alerts,
                        posture_alerts,
                        breaks_taken,
                        keyboard_activity,
                        mouse_activity,
                        low_light_alerts,
                        session_duration
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (eye_alerts, posture_alerts, breaks_taken,
                      keyboard_activity, mouse_activity, low_light_alerts,
                      session_duration))
            
            self.conn.commit()
            print(f"Logged activity: eye={eye_alerts}, posture={posture_alerts}, breaks={breaks_taken}, "
                  f"keyboard={keyboard_activity}, mouse={mouse_activity}, "
                  f"low_light={low_light_alerts}, duration={session_duration}")
        except Exception as e:
            print(f"Error logging activity: {str(e)}")
            traceback.print_exc()

    def detect_users(self, image):
        """Detect and track users in the frame"""
        try:
            current_time = time.time()
            
            # Only check for multiple users periodically
            if current_time - self.last_user_detection_time < self.user_detection_cooldown:
                return

            self.last_user_detection_time = current_time
            
            # Check for faces
            face_results = self.mp_face_mesh.process(image)
            if not face_results.multi_face_landmarks:
                # No faces detected - reset all counters and timers
                if self.primary_user_id and current_time - self.user_data.get(self.primary_user_id, {}).get('last_seen', 0) > self.user_timeout:
                    self.statusBar().showMessage("Primary user no longer detected")
                    self.primary_user_id = None
                    self.activity_status.setText("⌨ Activity Level: No User Detected")
                    # Reset blink count and related variables
                    self.blink_count = 0
                    self.blink_timer = current_time
                    self.last_blink_status = False
                    self.eyes_closed_start_time = None
                    self.last_blink_detected = current_time
                    self.eye_status.setText("👁 Eye Health: No User Detected")
                return

            # Update detected users
            current_users = set()
            for face_landmarks in face_results.multi_face_landmarks:
                try:
                    user_id = self.generate_user_id(face_landmarks)
                    current_users.add(user_id)
                    
                    # Initialize user data if not exists
                    if user_id not in self.user_data:
                        self.user_data[user_id] = {
                            'last_seen': current_time,
                            'last_blink_time': current_time,
                            'posture_warning_time': current_time,
                            'eye_status': "Good",
                            'posture_status': "Good",
                            'face_landmarks': face_landmarks
                        }
                    else:
                        self.user_data[user_id]['last_seen'] = current_time
                        self.user_data[user_id]['face_landmarks'] = face_landmarks
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

            # Remove users that haven't been seen recently
            self.user_data = {k: v for k, v in self.user_data.items() 
                             if current_time - v['last_seen'] <= self.user_timeout}

            # Set primary user if not set
            if not self.primary_user_id and current_users:
                self.primary_user_id = next(iter(current_users))
                self.statusBar().showMessage("Primary user identified")
                self.multiple_faces_alerted = False
                self.activity_status.setText("⌨ Activity Level: Normal")

            # Handle multiple users
            if len(current_users) > 1:
                if not self.multiple_faces_alerted:
                    self.show_notification(f"Multiple users detected ({len(current_users)}). Only one user should be in frame.")
                    self.speak(f"Multiple users detected. Only one user should be in frame")
                    self.multiple_faces_alerted = True
                    self.activity_status.setText(f"👥 Multiple Users Detected ({len(current_users)})")
            else:
                # Only one user detected
                if self.multiple_faces_alerted:
                    # Reset multiple faces alert and update status
                    self.multiple_faces_alerted = False
                    self.activity_status.setText("⌨ Activity Level: Normal")
                    self.statusBar().showMessage("Single user detected")

            # Update status bar with current user count
            self.statusBar().showMessage(f"Detected {len(current_users)} user(s) in frame")

        except Exception as e:
            print(f"User detection error: {str(e)}")

    def detect_posture(self, image):
        try:
            results = self.mp_pose.process(image)
            if not results.pose_landmarks:
                self.posture_status.setText("🪑 Posture: No Detection")
                return

            current_time = time.time()
            
            # Calculate posture score
            self.current_posture_score = self.calculate_posture_score(results.pose_landmarks)
            print(f"Current posture score: {self.current_posture_score}")  # Debug print
            
            # Enhanced posture monitoring with clear timing stages
            if self.current_posture_score < 40:  # Bad posture threshold
                # Start tracking bad posture duration if not already tracking
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = current_time
                    print(f"Bad posture detected - starting timer at {current_time}")  # Debug print
                
                # Calculate how long bad posture has persisted
                bad_posture_duration = current_time - self.bad_posture_start_time
                print(f"Bad posture duration: {bad_posture_duration} seconds")  # Debug print
                
                # Stage 1: Warning after 30 seconds
                if 30 <= bad_posture_duration < 60:
                    self.posture_status.setText(f"🪑 Posture: Poor - Warning ({int(bad_posture_duration)}s)")
                    if not hasattr(self, 'posture_warning_shown') or not self.posture_warning_shown:
                        self.show_notification("Warning: Poor posture detected. Please adjust your position.")
                        self.speak("Warning: Poor posture detected. Please adjust your position.")
                        self.posture_warning_shown = True
                        # Log posture alert
                        self.log_activity(posture_alerts=1)
                
                # Stage 2: Alert after 2 minutes (120 seconds)
                elif bad_posture_duration >= self.bad_posture_threshold:
                    # Only alert if we haven't alerted recently
                    if (current_time - self.last_posture_alert_time) >= self.posture_alert_cooldown:
                        print("Triggering posture alert")  # Debug print
                        self.show_notification("⚠️ POSTURE ALERT: You've been in poor posture for 2 minutes! Please adjust your sitting position.")
                        self.speak("Posture alert! You've been in poor posture for too long. Please sit up straight.")
                        self.last_posture_alert_time = current_time
                        self.posture_status.setText("🪑 Posture: Poor - Needs Immediate Attention")
                        # Log posture alert
                        self.log_activity(posture_alerts=1)
                        # Reset warning flag
                        self.posture_warning_shown = False
                
                # Stage 3: Countdown to alert
                elif bad_posture_duration >= 60:
                    time_until_alert = int(self.bad_posture_threshold - bad_posture_duration)
                    self.posture_status.setText(f"🪑 Posture: Poor - Alert in {time_until_alert}s")
                    if not hasattr(self, 'posture_countdown_shown') or not self.posture_countdown_shown:
                        self.show_notification(f"Posture Alert: You have {time_until_alert} seconds to correct your posture.")
                        self.speak(f"Posture alert: You have {time_until_alert} seconds to correct your posture.")
                        self.posture_countdown_shown = True
                        # Log posture alert
                        self.log_activity(posture_alerts=1)
                else:
                    self.posture_status.setText(f"🪑 Posture: Poor ({int(bad_posture_duration)}s)")
                    # Reset warning flags when posture improves
                    self.posture_warning_shown = False
                    self.posture_countdown_shown = False
                    
            elif self.current_posture_score < 70:  # Fair posture
                # Reset bad posture timer and flags
                self.bad_posture_start_time = None
                self.posture_warning_shown = False
                self.posture_countdown_shown = False
                self.posture_status.setText("🪑 Posture: Fair")
            else:  # Good posture
                # Reset bad posture timer and flags
                self.bad_posture_start_time = None
                self.posture_warning_shown = False
                self.posture_countdown_shown = False
                self.posture_status.setText("🪑 Posture: Good")
                
        except Exception as e:
            print(f"Posture detection error: {str(e)}")
            traceback.print_exc()

    def calculate_posture_score(self, pose_landmarks):
        """Calculate posture score based on body alignment"""
        try:
            # Get relevant landmarks
            left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_ear = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
            left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            # Calculate shoulder alignment
            shoulder_angle = abs(left_shoulder.y - right_shoulder.y)
            
            # Calculate head position relative to shoulders
            head_forward = abs((left_ear.x + right_ear.x)/2 - (left_shoulder.x + right_shoulder.x)/2)
            
            # Calculate hip alignment
            hip_angle = abs(left_hip.y - right_hip.y)
            
            # Calculate neck angle (head tilt)
            neck_angle = abs((left_ear.y + right_ear.y)/2 - (left_shoulder.y + right_shoulder.y)/2)
            
            # Calculate overall score (0-100)
            score = 100 - (
                shoulder_angle * 200 +  # Shoulder alignment
                head_forward * 100 +    # Forward head posture
                hip_angle * 100 +       # Hip alignment
                neck_angle * 150        # Neck angle
            )
            
            # Add to history for smoothing
            self.posture_history.append(score)
            if len(self.posture_history) > self.posture_history_size:
                self.posture_history.pop(0)
            
            # Return smoothed value
            smoothed_score = max(0, min(100, sum(self.posture_history) / len(self.posture_history)))
            print(f"Raw posture score: {score}, Smoothed score: {smoothed_score}")  # Debug print
            return smoothed_score
            
        except Exception as e:
            print(f"Posture score calculation error: {str(e)}")
            return 50

    def generate_user_id(self, face_landmarks):
        """Generate a unique ID for a user based on face landmarks"""
        try:
            # Use more facial points for better identification
            nose = face_landmarks.landmark[1]  # Nose tip
            left_eye = face_landmarks.landmark[33]  # Left eye
            right_eye = face_landmarks.landmark[362]  # Right eye
            left_ear = face_landmarks.landmark[234]  # Left ear
            right_ear = face_landmarks.landmark[454]  # Right ear
            
            # Create a more unique identifier using multiple facial points
            return f"{nose.x:.3f}{nose.y:.3f}{left_eye.x:.3f}{right_eye.x:.3f}{left_ear.x:.3f}_{right_ear.x:.3f}"
        except Exception as e:
            print(f"Error generating user ID: {str(e)}")
            return f"user_{time.time()}"  # Fallback ID

    def generate_pose_user_id(self, pose_landmarks):
        """Generate a unique ID for a user based on pose landmarks"""
        # Use shoulder positions to generate a unique ID
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        return f"pose_{left_shoulder.x:.3f}_{right_shoulder.x:.3f}"

    def on_key_press(self, key):
        try:
            self.keyboard_activity += 1
            print(f"Keyboard activity: {self.keyboard_activity}")  # Debug print
            
            if self.keyboard_activity >= self.keyboard_limit:
                self.show_notification("Take a break from typing!")
                self.speak("You have pressed too many keys. Take a break.")
                self.keyboard_activity = 0
                self.activity_status.setText("⌨ Activity Level: High")
                # Log keyboard activity
                self.log_activity(keyboard_activity=self.keyboard_limit)
                QtCore.QTimer.singleShot(5000, lambda: self.activity_status.setText("⌨ Activity Level: Normal"))
            else:
                # Log keyboard activity periodically
                if self.keyboard_activity % 100 == 0:  # Log every 100 keystrokes
                    print(f"Logging keyboard activity: {self.keyboard_activity}")  # Debug print
                    self.log_activity(keyboard_activity=100)
        except Exception as e:
            print(f"Keyboard activity error: {str(e)}")
            traceback.print_exc()

    def on_mouse_click(self):
        try:
            self.mouse_activity += 1
            if self.mouse_activity >= self.mouse_limit:
                self.show_notification("Take a break from clicking!")
                self.speak("You have clicked too many times. Take a break.")
                self.mouse_activity = 0
                self.activity_status.setText("⌨ Activity Level: High")
                # Log mouse activity
                self.log_activity(mouse_activity=self.mouse_limit)
            else:
                # Log mouse activity periodically
                if self.mouse_activity % 100 == 0:  # Log every 100 clicks
                    self.log_activity(mouse_activity=100)
        except Exception as e:
            print(f"Mouse activity error: {str(e)}")

    def start_tracking(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Camera Error", "Unable to access webcam. Check permissions and try again.")
                return
                
            # Test camera read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.show_error("Camera Error", "Failed to read from webcam. Please check your camera connection.")
                self.cap.release()
                return
                
            self.tracking_active = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.statusBar().showMessage("Monitoring started")
            
            # Reset counters
            self.keyboard_activity = 0
            self.mouse_activity = 0
            self.blink_count = 0
            self.blink_timer = time.time()
            self.last_blink_status = False
            self.eyes_closed_start_time = None
            self.bad_posture_start_time = None
            
            # Start the tracking loop in a separate thread
            self.tracking_thread = threading.Thread(target=self.track_loop)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            
            # Reset timers
            self.break_timer.start(3600000)  # 1 hour
            self.status_timer.start(100)
            
            # Add record to database
            self.log_activity(0, 0, 0, 0, 0, 0, 0)  # Initialize with zeros
            
            # Show welcome notification
            self.show_notification("DevWell is now monitoring your health")
            self.speak("DevWell is now monitoring your health. I'll help you stay healthy during coding sessions.")
            
        except Exception as e:
            error_msg = f"Tracking Error: {str(e)}"
            print(error_msg)  # Print to console for debugging
            self.show_error("Tracking Error", error_msg)
            self.stop_tracking()  # Ensure cleanup on error

    def track_loop(self):
        consecutive_failures = 0
        max_consecutive_failures = 5
        no_face_detected_time = None
        low_light_alert_time = None
        last_session_duration_log = time.time()
        
        try:
            while self.tracking_active and self.cap is not None:
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        print(f"Failed to read frame. Attempt {consecutive_failures}/{max_consecutive_failures}")
                        if consecutive_failures >= max_consecutive_failures:
                            raise Exception("Failed to read from webcam after multiple attempts")
                        time.sleep(0.1)
                        continue
                    
                    consecutive_failures = 0  # Reset on successful frame read
                    
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Check for low light conditions
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray_frame)
                    current_time = time.time()
                    
                    if avg_brightness < 40:  # Low light threshold
                        if low_light_alert_time is None:
                            low_light_alert_time = current_time
                        elif current_time - low_light_alert_time >= 5:  # Alert after 5 seconds of low light
                            self.show_notification("Low light detected! Please improve lighting conditions.")
                            self.speak("Low light detected. Please improve your lighting conditions.")
                            self.eye_status.setText("👁 Eye Health: Low Light Alert")
                            self.log_activity(low_light_alerts=1)  # Log low light alert
                            low_light_alert_time = current_time
                    else:
                        low_light_alert_time = None
                    
                    # Log session duration every minute
                    if current_time - last_session_duration_log >= 60:
                        self.log_activity(session_duration=60)  # Log 60 seconds of session duration
                        last_session_duration_log = current_time
                    
                    # Detect users
                    self.detect_users(rgb_frame)
                    
                    # Check for face detection
                    if not self.primary_user_id:
                        if no_face_detected_time is None:
                            no_face_detected_time = current_time
                        elif current_time - no_face_detected_time >= 10:  # Alert after 10 seconds of no face
                            self.show_notification("No face detected! Please position yourself in front of the camera.")
                            self.speak("No face detected. Please position yourself in front of the camera.")
                            self.eye_status.setText("👁 Eye Health: No Face Detected")
                            no_face_detected_time = current_time
                    else:
                        no_face_detected_time = None
                    
                    # Only process if primary user is detected
                    if self.primary_user_id:
                        # Detect eye strain
                        self.detect_eye_strain(rgb_frame)
                        
                        # Detect posture
                        self.detect_posture(rgb_frame)
                    
                    # Draw eye aspect ratio on frame
                    if self.current_ear is not None:
                        cv2.putText(frame, f"EAR: {self.current_ear:.2f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw blink count on frame
                    cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw brightness level
                    cv2.putText(frame, f"Brightness: {avg_brightness:.1f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert frame to QImage and display
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_image = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                    self.camera_frame.setPixmap(QtGui.QPixmap.fromImage(q_image))
                    
                    # Small delay to prevent high CPU usage
                    time.sleep(0.01)
                    
                except Exception as frame_error:
                    print(f"Error processing frame: {str(frame_error)}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise Exception(f"Multiple frame processing errors: {str(frame_error)}")
                    time.sleep(0.1)
                    continue
                
        except Exception as e:
            print(f"Tracking loop error: {str(e)}")
            traceback.print_exc()
            # Emit signal to stop tracking
            self.stop_tracking_signal.emit()
        finally:
            self.tracking_active = False

    def stop_tracking(self):
        try:
            self.tracking_active = False
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("Monitoring stopped")
            
            # Reset status displays
            self.eye_status.setText("👁 Eye Health: Not Monitoring")
            self.blink_status.setText("😌 Blinks: 0/17 per minute")
            self.posture_status.setText("🪑 Posture: Not Monitoring")
            self.activity_status.setText("⌨ Activity: Not Monitoring")
            
            # Reset counters
            self.blink_count = 0
            self.eyes_closed_start_time = None
            self.bad_posture_start_time = None
            self.current_ear = 0.45
            self.current_posture_score = 50
            
            # Notify user
            self.show_notification("Health monitoring stopped")
            self.speak("Health monitoring stopped. Take care of yourself!")
            
        except Exception as e:
            error_msg = f"Stop Tracking Error: {str(e)}"
            print(error_msg)  # Print to console for debugging
            self.show_error("Stop Tracking Error", error_msg)

    def generate_report(self, period='weekly'):
        try:
            # Get the start date based on period
            if period == 'daily':
                start_date = datetime.datetime.now() - datetime.timedelta(days=1)
            elif period == 'weekly':
                start_date = datetime.datetime.now() - datetime.timedelta(days=7)
            else:  # monthly
                start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            
            # First check which columns exist in the activity table
            self.cursor.execute("PRAGMA table_info(activity)")
            columns = [col[1] for col in self.cursor.fetchall()]
            print(f"Available columns: {columns}")  # Debug print
            
            if not columns:
                self.show_error("Report Error", "No activity table found in database")
                return
            
            # Build the query based on available columns
            select_columns = []
            if 'eye_alerts' in columns:
                select_columns.append("SUM(eye_alerts) as eye_alerts")
            if 'posture_alerts' in columns:
                select_columns.append("SUM(posture_alerts) as posture_alerts")
            if 'breaks_taken' in columns:
                select_columns.append("SUM(breaks_taken) as breaks")
            if 'keyboard_activity' in columns:
                select_columns.append("SUM(keyboard_activity) as keyboard_activity")
            if 'mouse_activity' in columns:
                select_columns.append("SUM(mouse_activity) as mouse_activity")
            if 'low_light_alerts' in columns:
                select_columns.append("SUM(low_light_alerts) as low_light_alerts")
            if 'session_duration' in columns:
                select_columns.append("SUM(session_duration) as session_duration")
            
            if not select_columns:
                self.show_error("Report Error", "No valid columns found in activity table")
                return
            
            # Query for activity data
            query = f"""
                SELECT date(timestamp) as day,
                       {', '.join(select_columns)}
                FROM activity 
                WHERE timestamp >= ? 
                GROUP BY day
                ORDER BY day
            """
            
            print(f"Executing query: {query}")  # Debug print
            self.cursor.execute(query, (start_date.strftime("%Y-%m-%d"),))
            activity_data = self.cursor.fetchall()
            print(f"Query results: {activity_data}")  # Debug print
            
            # Query for blink data
            self.cursor.execute("""
                SELECT strftime('%H:%M', timestamp) as time,
                        blink_count
                FROM blink_data
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (start_date.strftime("%Y-%m-%d"),))
            
            blink_data = self.cursor.fetchall()
            
            if not activity_data and not blink_data:
                self.show_notification("No data available for the selected period")
                return
            
            # Create DataFrames with only available columns
            df_activity = pd.DataFrame(activity_data, columns=["Date"] + [col.split(" as ")[1] for col in select_columns])
            df_blinks = pd.DataFrame(blink_data, columns=["Time", "Blinks"])
            
            print(f"Activity DataFrame columns: {df_activity.columns}")  # Debug print
            print(f"Activity DataFrame data:\n{df_activity}")  # Debug print
            
            # Create report window
            report_window = QtWidgets.QDialog(self)
            report_window.setWindowTitle(f"DevWell {period.capitalize()} Report")
            report_window.setGeometry(200, 200, 1200, 900)
            
            layout = QtWidgets.QVBoxLayout()
            
            # Create matplotlib figure with subplots
            fig = plt.figure(figsize=(12, 12))
            plot_count = 0
            
            # Plot 1: Health alerts
            if any(col in df_activity.columns for col in ['eye_alerts', 'posture_alerts', 'low_light_alerts']):
                plot_count += 1
                ax1 = fig.add_subplot(411)
                health_cols = [col for col in ['eye_alerts', 'posture_alerts', 'low_light_alerts'] 
                             if col in df_activity.columns]
                print(f"Health columns for plot: {health_cols}")  # Debug print
                df_activity.plot(x='Date', y=health_cols, kind='bar', ax=ax1)
            ax1.set_title('Health Alerts')
            ax1.set_ylabel('Number of Alerts')
            ax1.legend(loc='upper right')
            
            # Plot 2: Blink data
            if not df_blinks.empty:
                plot_count += 1
                ax2 = fig.add_subplot(412)
                df_blinks.plot(x='Time', y='Blinks',kind ='line', ax=ax2, marker='o')
                ax2.set_title('Blink Rate Over Time')
                ax2.set_ylabel('Blinks per Minute')
                ax2.axhline(y=self.min_blink_threshold, color='r', linestyle='--', 
                           label=f'Minimum Target ({self.min_blink_threshold})')
                ax2.legend()
            
            # Plot 3: Activity levels
            if any(col in df_activity.columns for col in ['keyboard_activity', 'mouse_activity']):
                plot_count += 1
                ax3 = fig.add_subplot(413)
                activity_cols = [col for col in ['keyboard_activity', 'mouse_activity'] 
                               if col in df_activity.columns]
                print(f"Activity columns for plot: {activity_cols}")  # Debug print
                df_activity.plot(x='Date', y=activity_cols, kind='bar', ax=ax3)
            ax3.set_title('Activity Levels')
            ax3.set_ylabel('Activity Count')
            ax3.legend(loc='upper right')
            
            # Plot 4: Session duration
            if 'session_duration' in df_activity.columns:
                plot_count += 1
                ax4 = fig.add_subplot(414)
                df_activity.plot(x='Date', y='session_duration', kind='bar', ax=ax4)
                ax4.set_title('Session Duration')
                ax4.set_ylabel('Duration (seconds)')
            
            plt.tight_layout()
            
            # Add the plots to the window
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Add summary statistics
            stats_text = QtWidgets.QTextEdit()
            stats_text.setReadOnly(True)
            
            if not df_activity.empty:
                stats_html = f"""
                <h3>Summary Statistics</h3>
                    <p><b>Period:</b> {period.capitalize()} Report ({start_date.date()} to {datetime.datetime.now().date()})</p>
                """
                
                if 'eye_alerts' in df_activity.columns:
                    stats_html += f"<p><b>Total Eye Alerts:</b> {df_activity['eye_alerts'].sum()}</p>"
                if 'posture_alerts' in df_activity.columns:
                    stats_html += f"<p><b>Total Posture Alerts:</b> {df_activity['posture_alerts'].sum()}</p>"
                if 'low_light_alerts' in df_activity.columns:
                    stats_html += f"<p><b>Total Low Light Alerts:</b> {df_activity['low_light_alerts'].sum()}</p>"
                if 'breaks' in df_activity.columns:
                    stats_html += f"<p><b>Total Breaks:</b> {df_activity['breaks'].sum()}</p>"
                if not df_blinks.empty:
                    stats_html += f"<p><b>Average Blinks per Minute:</b> {df_blinks['Blinks'].mean():.1f}</p>"
                if 'keyboard_activity' in df_activity.columns:
                    total_keyboard = df_activity['keyboard_activity'].sum()
                    stats_html += f"<p><b>Total Keyboard Activity:</b> {total_keyboard:,} keystrokes</p>"
                    stats_html += f"<p><b>Average Keyboard Activity per Day:</b> {total_keyboard/len(df_activity):,.0f} keystrokes</p>"
                if 'mouse_activity' in df_activity.columns:
                    total_mouse = df_activity['mouse_activity'].sum()
                    stats_html += f"<p><b>Total Mouse Activity:</b> {total_mouse:,} clicks</p>"
                    stats_html += f"<p><b>Average Mouse Activity per Day:</b> {total_mouse/len(df_activity):,.0f} clicks</p>"
                if 'session_duration' in df_activity.columns:
                    total_duration = df_activity['session_duration'].sum()
                    stats_html += f"<p><b>Total Session Duration:</b> {total_duration/3600:.1f} hours</p>"
                    stats_html += f"<p><b>Average Session Duration per Day:</b> {total_duration/len(df_activity)/3600:.1f} hours</p>"
            else:
                stats_html = "<h3>No activity data available</h3>"
            
            stats_text.setHtml(stats_html)
            layout.addWidget(stats_text)
            
            # Save button
            save_button = QtWidgets.QPushButton("Save Report")
            save_button.clicked.connect(lambda: self.save_report(df_activity, df_blinks, period))
            layout.addWidget(save_button)
            
            report_window.setLayout(layout)
            report_window.exec_()
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            print(error_msg)
            self.show_error("Report Error", error_msg)
            traceback.print_exc()

    def save_report(self, df_activity, df_blinks, period):
        try:
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save CSV
            csv_path = os.path.join("reports", f"devwell_{period}_report_{timestamp}.csv")
            df_activity.to_csv(csv_path, index=False)
            
            # Save blink data separately
            if not df_blinks.empty:
                blink_path = os.path.join("reports", f"devwell_blinks_{period}_report_{timestamp}.csv")
                df_blinks.to_csv(blink_path, index=False)
            
            self.show_notification(f"Report saved to {csv_path}")
        except Exception as e:
            self.show_error("Save Error", f"Error saving report: {str(e)}")

    def show_settings(self):
        try:
            settings_window = QtWidgets.QDialog(self)
            settings_window.setWindowTitle("DevWell Settings")
            settings_window.setGeometry(300, 300, 400, 500)
            
            layout = QtWidgets.QVBoxLayout()
            
            # Create settings groups
            eye_group = QtWidgets.QGroupBox("Eye Health Settings")
            eye_layout = QtWidgets.QFormLayout()
            
            self.ear_threshold_input = QtWidgets.QDoubleSpinBox()
            self.ear_threshold_input.setRange(0.1, 0.5)
            self.ear_threshold_input.setValue(self.ear_threshold)
            self.ear_threshold_input.setSingleStep(0.01)
            eye_layout.addRow("Eye Aspect Ratio Threshold:", self.ear_threshold_input)
            
            self.blink_threshold_input = QtWidgets.QSpinBox()
            self.blink_threshold_input.setRange(5, 20)
            self.blink_threshold_input.setValue(self.min_blink_threshold)
            eye_layout.addRow("Minimum Blinks per Minute:", self.blink_threshold_input)
            
            eye_group.setLayout(eye_layout)
            layout.addWidget(eye_group)
            
            # Posture settings
            posture_group = QtWidgets.QGroupBox("Posture Settings")
            posture_layout = QtWidgets.QFormLayout()
            
            self.posture_threshold_input = QtWidgets.QSpinBox()
            self.posture_threshold_input.setRange(30, 90)
            self.posture_threshold_input.setValue(self.bad_posture_threshold)
            posture_layout.addRow("Bad Posture Alert Time (seconds):", self.posture_threshold_input)
            
            posture_group.setLayout(posture_layout)
            layout.addWidget(posture_group)
            
            # Activity settings
            activity_group = QtWidgets.QGroupBox("Activity Settings")
            activity_layout = QtWidgets.QFormLayout()
            
            self.keyboard_limit_input = QtWidgets.QSpinBox()
            self.keyboard_limit_input.setRange(1000, 5000)
            self.keyboard_limit_input.setValue(self.keyboard_limit)
            activity_layout.addRow("Keyboard Activity Limit:", self.keyboard_limit_input)
            
            self.mouse_limit_input = QtWidgets.QSpinBox()
            self.mouse_limit_input.setRange(1000, 5000)
            self.mouse_limit_input.setValue(self.mouse_limit)
            activity_layout.addRow("Mouse Activity Limit:", self.mouse_limit_input)
            
            activity_group.setLayout(activity_layout)
            layout.addWidget(activity_group)
            
            # Save button
            save_button = QtWidgets.QPushButton("Save Settings")
            save_button.clicked.connect(lambda: self.save_settings(settings_window))
            layout.addWidget(save_button)
            
            settings_window.setLayout(layout)
            settings_window.exec_()
            
        except Exception as e:
            self.show_error("Settings Error", str(e))

    def save_settings(self, settings_window):
        try:
            # Update settings
            self.ear_threshold = self.ear_threshold_input.value()
            self.min_blink_threshold = self.blink_threshold_input.value()
            self.bad_posture_threshold = self.posture_threshold_input.value()
            self.keyboard_limit = self.keyboard_limit_input.value()
            self.mouse_limit = self.mouse_limit_input.value()
            
            # Save to database
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY,
                    setting_name TEXT,
                    setting_value TEXT
                )
            """)
            
            settings = {
                'ear_threshold': self.ear_threshold,
                'min_blink_threshold': self.min_blink_threshold,
                'bad_posture_threshold': self.bad_posture_threshold,
                'keyboard_limit': self.keyboard_limit,
                'mouse_limit': self.mouse_limit
            }
            
            for name, value in settings.items():
                self.cursor.execute("""
                    INSERT OR REPLACE INTO settings (setting_name, setting_value)
                    VALUES (?, ?)
                """, (name, str(value)))
            
            self.conn.commit()
            settings_window.accept()
            self.show_notification("Settings saved successfully")
            
        except Exception as e:
            self.show_error("Save Settings Error", str(e))

    def update_health_tip(self):
        try:
            tip = random.choice(self.health_tips)
            self.tip_label.setText(tip)
        except Exception as e:
            print(f"Health tip update error: {str(e)}")

    def suggest_break(self):
        try:
            self.show_notification("Time for a break! Take a few minutes to stretch and rest your eyes.")
            self.speak("It's time for a break. Please take a few minutes to stretch and rest your eyes.")
            
            # Log break taken
            self.log_activity(breaks_taken=1)
            
        except Exception as e:
            print(f"Break suggestion error: {str(e)}")

    def update_status(self):
        try:
            # Update status bar with current monitoring state
            if self.tracking_active:
                status = "Monitoring active"
                if self.primary_user_id:
                    status += f" - User detected"
                self.statusBar().showMessage(status)
            else:
                self.statusBar().showMessage("Monitoring inactive")
                
        except Exception as e:
            print(f"Status update error: {str(e)}")

    def closeEvent(self, event):
        try:
            # Clean up resources
            self.stop_tracking()
            if hasattr(self, 'conn'):
                self.conn.close()
            event.accept()
        except Exception as e:
            print(f"Close event error: {str(e)}")
            event.accept()

    def handle_db_operation(self, data):
        try:
            # Handle different types of database operations
            operation = data.get('operation')
            
            if operation == 'insert_activity':
                self.cursor.execute("""
                    INSERT INTO activity (
                        eye_alerts,
                        posture_alerts,
                        breaks_taken,
                        keyboard_activity,
                        mouse_activity
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    data.get('eye_alerts', 0),
                    data.get('posture_alerts', 0),
                    data.get('breaks_taken', 0),
                    data.get('keyboard_activity', 0),
                    data.get('mouse_activity', 0)
                ))
                self.conn.commit()
                
            elif operation == 'update_settings':
                for setting_name, setting_value in data.get('settings', {}).items():
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO settings (setting_name, setting_value)
                        VALUES (?, ?)
                    """, (setting_name, str(setting_value)))
                self.conn.commit()
                
            elif operation == 'get_settings':
                self.cursor.execute("SELECT setting_name, setting_value FROM settings")
                settings = dict(self.cursor.fetchall())
                return settings
                
        except Exception as e:
            print(f"Database operation error: {str(e)}")
            self.show_error("Database Error", str(e))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DevWellApp()
    window.show()
    sys.exit(app.exec_())
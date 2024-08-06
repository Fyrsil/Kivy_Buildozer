import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def draw_angle(image, a, b, c, angle, color):
    cv2.line(image, a, b, color, 2)
    cv2.line(image, b, c, color, 2)
    cv2.putText(image, str(int(angle)), (b[0] - 20, b[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

class PostureApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        self.img1 = Image()
        layout.add_widget(self.img1)
        
        self.stop_button = Button(text='Stop', size_hint=(1, 0.1))
        self.stop_button.bind(on_press=self.stop_app)
        layout.add_widget(self.stop_button)
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        Window.bind(on_request_close=self.on_request_close)
        return layout

    def on_start(self):
        self.capture = cv2.VideoCapture(0)
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.calibration_frames = 0
        self.is_calibrated = False
        
        self.alert_cooldown = 10
        self.last_alert_time = 0
        self.sound_file = "alert_sound.mp3"

    def on_stop(self):
        self.capture.release()

    def on_request_close(self, *args):
        self.capture.release()
        self.stop()
        return True

    def stop_app(self, instance):
        self.capture.release()
        self.stop()

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                              int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
            left_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
            right_ear = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))

            shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
            neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

            if not self.is_calibrated and self.calibration_frames < 30:
                self.calibration_shoulder_angles.append(shoulder_angle)
                self.calibration_neck_angles.append(neck_angle)
                self.calibration_frames += 1
                cv2.putText(frame, f"Calibrating... {self.calibration_frames}/30", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            elif not self.is_calibrated:
                self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 10
                self.neck_threshold = np.mean(self.calibration_neck_angles) - 10
                self.is_calibrated = True
                print(f"Calibration complete. Shoulder threshold: {self.shoulder_threshold:.1f}, Neck threshold: {self.neck_threshold:.1f}")

            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
            draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
            draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

            if self.is_calibrated:
                current_time = time.time()
                if shoulder_angle < self.shoulder_threshold or neck_angle < self.neck_threshold:
                    status = "Poor Posture"
                    color = (0, 0, 255)
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        print("Poor posture detected! Please sit up straight.")
                        if os.path.exists(self.sound_file):
                            playsound(self.sound_file)
                        self.last_alert_time = current_time
                else:
                    status = "Good Posture"
                    color = (0, 255, 0)

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{self.shoulder_threshold:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{self.neck_threshold:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        buf = cv2.flip(frame, 0).tostring()  # Flip the frame vertically before displaying
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = image_texture

if __name__ == '__main__':
    PostureApp().run()

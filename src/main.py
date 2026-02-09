"""
Crossy Road Body Pose Controller
Control Crossy Road game using webcam body pose detection with MediaPipe.
"""

import cv2
import numpy as np
import time
import subprocess
import os
from pynput.keyboard import Controller, Key
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Processing resolution (MediaPipe runs on this size for speed, lower = faster)
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Calibration settings
CALIBRATION_SECONDS = 2.0

# Jump detection thresholds
JUMP_VELOCITY_THRESHOLD = 0.012  # Upward velocity threshold to trigger jump (lower = more sensitive)
JUMP_COOLDOWN_SECONDS = 0.3      # Minimum time between jumps (reduced for faster response)
MIN_JUMP_DISPLACEMENT = -0.04    # Minimum upward displacement (negative = up)

# Smoothing parameters (0-1, higher = more smoothing, lower = more responsive)
POSITION_SMOOTHING = 0.4
VELOCITY_SMOOTHING = 0.5

# Zone boundaries (divide screen into 3 equal vertical zones)
LEFT_ZONE_THRESHOLD = 1/3
RIGHT_ZONE_THRESHOLD = 2/3

# MediaPipe model path
MODEL_PATH = "models/pose_landmarker.task"

# Crossy Road application path
CROSSY_ROAD_APP = "/Applications/Crossy Road.app"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def launch_crossy_road():
    """Launch Crossy Road application if it exists."""
    if os.path.exists(CROSSY_ROAD_APP):
        try:
            subprocess.Popen(['open', CROSSY_ROAD_APP])
            print(f"Launching Crossy Road...")
            time.sleep(2)  # Give the game time to start
            return True
        except Exception as e:
            print(f"Failed to launch Crossy Road: {e}")
            return False
    else:
        print(f"Crossy Road not found at: {CROSSY_ROAD_APP}")
        return False


# ============================================================================
# EXPONENTIAL MOVING AVERAGE CLASS
# ============================================================================

class EMA:
    """Exponential Moving Average for smooth signal filtering."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

    def reset(self):
        self.value = None


# ============================================================================
# POSE CONTROLLER CLASS
# ============================================================================

class PoseController:
    """Main controller for pose-based game input."""

    def __init__(self):
        # Initialize keyboard controller
        self.keyboard = Controller()

        # Initialize MediaPipe PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Camera setup
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Calibration state
        self.calibrated = False
        self.calibration_start_time = None
        self.baseline_hip_y = []
        self.baseline_shoulder_y = []
        self.baseline_hip_avg = None
        self.baseline_shoulder_avg = None

        # Smoothed tracking
        self.hip_y_ema = EMA(alpha=POSITION_SMOOTHING)
        self.velocity_ema = EMA(alpha=VELOCITY_SMOOTHING)
        self.prev_hip_y = None
        self.prev_time = None

        # Jump detection state
        self.last_jump_time = 0
        self.jump_triggered = False

        # Current state for HUD
        self.current_zone = "MIDDLE"
        self.current_displacement = 0.0
        self.current_velocity = 0.0

    def get_torso_center_x(self, landmarks, frame_width):
        """Calculate torso center x coordinate from hip landmarks."""
        # Get left and right hip landmarks (indices 23 and 24)
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Calculate midpoint
        center_x = (left_hip.x + right_hip.x) / 2
        return center_x * frame_width

    def get_hip_y(self, landmarks):
        """Calculate average hip y coordinate (normalized)."""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        return (left_hip.y + right_hip.y) / 2

    def get_shoulder_y(self, landmarks):
        """Calculate average shoulder y coordinate (normalized)."""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        return (left_shoulder.y + right_shoulder.y) / 2

    def determine_zone(self, center_x, frame_width):
        """Determine which zone (LEFT, MIDDLE, RIGHT) the torso is in."""
        normalized_x = center_x / frame_width

        if normalized_x < LEFT_ZONE_THRESHOLD:
            return "LEFT"
        elif normalized_x > RIGHT_ZONE_THRESHOLD:
            return "RIGHT"
        else:
            return "MIDDLE"

    def calibrate(self, landmarks):
        """Collect baseline measurements during calibration phase."""
        if self.calibration_start_time is None:
            self.calibration_start_time = time.time()

        elapsed = time.time() - self.calibration_start_time

        if elapsed < CALIBRATION_SECONDS:
            # Collect samples
            hip_y = self.get_hip_y(landmarks)
            shoulder_y = self.get_shoulder_y(landmarks)
            self.baseline_hip_y.append(hip_y)
            self.baseline_shoulder_y.append(shoulder_y)
            return False
        else:
            # Finish calibration
            if len(self.baseline_hip_y) > 0:
                self.baseline_hip_avg = np.mean(self.baseline_hip_y)
                self.baseline_shoulder_avg = np.mean(self.baseline_shoulder_y)
                self.calibrated = True
            return True

    def detect_jump(self, landmarks):
        """Detect jump based on hip displacement and velocity."""
        if not self.calibrated:
            return False

        current_time = time.time()

        # Check cooldown
        if current_time - self.last_jump_time < JUMP_COOLDOWN_SECONDS:
            return False

        # Get current hip position
        hip_y = self.get_hip_y(landmarks)
        smoothed_hip_y = self.hip_y_ema.update(hip_y)

        # Calculate displacement from baseline (negative = jumping up)
        displacement = smoothed_hip_y - self.baseline_hip_avg
        self.current_displacement = displacement

        # Calculate velocity
        if self.prev_hip_y is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                raw_velocity = (smoothed_hip_y - self.prev_hip_y) / dt
                velocity = self.velocity_ema.update(raw_velocity)
                self.current_velocity = velocity

                # Jump detection logic: upward velocity crosses threshold
                # and significant upward displacement
                if (velocity < -JUMP_VELOCITY_THRESHOLD and
                    displacement < MIN_JUMP_DISPLACEMENT and
                    not self.jump_triggered):
                    self.jump_triggered = True

                # Reset jump trigger when velocity returns to normal
                elif velocity > -JUMP_VELOCITY_THRESHOLD / 2:
                    if self.jump_triggered:
                        # Jump detected!
                        self.last_jump_time = current_time
                        self.jump_triggered = False
                        return True
                    self.jump_triggered = False

        self.prev_hip_y = smoothed_hip_y
        self.prev_time = current_time

        return False

    def execute_jump_command(self, zone):
        """Execute keyboard commands based on jump and zone."""
        # Always press Up
        self.keyboard.press(Key.up)
        self.keyboard.release(Key.up)

        # Press Left or Right based on zone
        if zone == "LEFT":
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)
            print(f"JUMP + LEFT")
        elif zone == "RIGHT":
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)
            print(f"JUMP + RIGHT")
        else:
            print(f"JUMP")

    def draw_zones(self, frame):
        """Draw vertical zone boundaries on frame."""
        height, width = frame.shape[:2]

        # Draw zone lines
        left_x = int(width * LEFT_ZONE_THRESHOLD)
        right_x = int(width * RIGHT_ZONE_THRESHOLD)

        cv2.line(frame, (left_x, 0), (left_x, height), (0, 255, 0), 2)
        cv2.line(frame, (right_x, 0), (right_x, height), (0, 255, 0), 2)

        # Label zones
        cv2.putText(frame, "LEFT", (left_x // 2 - 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "MIDDLE", (width // 2 - 40, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (right_x + (width - right_x) // 2 - 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_hud(self, frame):
        """Draw debug HUD with current state."""
        height, width = frame.shape[:2]

        # Background for HUD
        hud_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - hud_height), (width, height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # HUD text
        y_offset = height - hud_height + 25

        if self.calibrated:
            status_text = "STATUS: CALIBRATED"
            status_color = (0, 255, 0)
        else:
            elapsed = 0
            if self.calibration_start_time:
                elapsed = time.time() - self.calibration_start_time
            status_text = f"CALIBRATING: {elapsed:.1f}/{CALIBRATION_SECONDS:.1f}s"
            status_color = (0, 255, 255)

        cv2.putText(frame, status_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        zone_color = (255, 255, 255)
        if self.current_zone == "LEFT":
            zone_color = (255, 0, 0)
        elif self.current_zone == "RIGHT":
            zone_color = (0, 0, 255)

        cv2.putText(frame, f"ZONE: {self.current_zone}", (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        cv2.putText(frame, f"DISPLACEMENT: {self.current_displacement:.4f}",
                   (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"VELOCITY: {self.current_velocity:.4f}",
                   (10, y_offset + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_pose(self, frame, landmarks):
        """Draw pose landmarks on frame."""
        height, width = frame.shape[:2]

        # Draw key landmarks
        key_points = [11, 12, 23, 24]  # shoulders and hips
        for idx in key_points:
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        # Draw torso center
        center_x = self.get_torso_center_x(landmarks, width)
        hip_y = self.get_hip_y(landmarks)
        center_y = int(hip_y * height)
        cv2.circle(frame, (int(center_x), center_y), 8, (255, 0, 255), -1)

    def process_frame(self, frame):
        """Process a single frame for pose detection and control."""
        # Resize frame for faster MediaPipe processing
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        detection_result = self.detector.detect(mp_image)

        # Draw zones
        self.draw_zones(frame)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]

            # Draw pose
            self.draw_pose(frame, landmarks)

            # Calibration phase
            if not self.calibrated:
                self.calibrate(landmarks)
            else:
                # Get torso position and zone
                center_x = self.get_torso_center_x(landmarks, frame.shape[1])
                zone = self.determine_zone(center_x, frame.shape[1])
                self.current_zone = zone

                # Detect jump
                if self.detect_jump(landmarks):
                    self.execute_jump_command(zone)

        # Draw HUD
        self.draw_hud(frame)

        return frame

    def run(self):
        """Main loop."""
        print("=" * 60)
        print("Crossy Road Body Pose Controller")
        print("=" * 60)
        print("Stand in view of camera and stay still for calibration.")
        print("After calibration, jump and lean left/right to control.")
        print("Press 'q' to quit.")
        print("=" * 60)

        # Launch Crossy Road
        launch_crossy_road()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Mirror the frame horizontally
                frame = cv2.flip(frame, 1)

                # Process frame
                frame = self.process_frame(frame)

                # Display
                cv2.imshow("Crossy Road Controller", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("\nController stopped.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    controller = PoseController()
    controller.run()


if __name__ == "__main__":
    main()

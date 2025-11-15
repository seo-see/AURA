import numpy as np
import cv2
import time

class MovementAnalyzer:
    def __init__(self, keypoint_indices, window_size=10, speed_threshold=0.05, analyze_movement=True):
        """
        keypoint_indices: List of keypoint indices to analyze (e.g., [15, 16, 13, 14, 11, 12])
        window_size: Number of frames for cumulative movement calculation
        speed_threshold: Agitation detection threshold
        analyze_movement: True to enable analysis, False to disable
        """
        self.keypoint_indices = keypoint_indices
        self.window_size = window_size
        self.speed_threshold = speed_threshold
        self.analyze_movement = analyze_movement
        self.prev_landmarks = None
        self.velocity_history = []  # [Average velocity per frame]
        self.agitation_flag = False
        self.last_agitation_time = 0
        self.agitation_hold_duration = 6.0  # Alarm hold duration (seconds)
        self.MAX_HUMAN_VELOCITY = 0.5 

    def update(self, curr_landmarks):
        """
        curr_landmarks: MediaPipe pose_landmarks.landmark list
        """
        if not self.analyze_movement:
            self.agitation_flag = False
            return 0.0, 0.0, 0.0, False
        if self.prev_landmarks is None:
            self.prev_landmarks = curr_landmarks
            self.velocity_history.append(0.0)
            return 0.0, 0.0, 0.0, False

        velocities = []
        valid_count = 0
        for idx in self.keypoint_indices:
            prev = self.prev_landmarks[idx]
            curr = curr_landmarks[idx]
            if prev.visibility < 0.7 or curr.visibility < 0.7:
                continue
            dist = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2 + (curr.z - prev.z)**2)
            if dist > self.MAX_HUMAN_VELOCITY:
                continue  # Ignore unrealistic velocities
            velocities.append(dist)
            valid_count += 1

        # If valid keypoints are less than 50% of total, maintain previous state
        if valid_count < len(self.keypoint_indices) * 0.5:
            # Maintain last value from velocity_history
            if self.velocity_history:
                last_velocity = self.velocity_history[-1]
                self.velocity_history.append(last_velocity)
            else:
                self.velocity_history.append(0.0)
            self.prev_landmarks = curr_landmarks
            # Return previous state values
            mean_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
            peak_velocity = max(self.velocity_history) if self.velocity_history else 0.0
            cumulative_velocity = sum(self.velocity_history)
            return mean_velocity, peak_velocity, cumulative_velocity, self.agitation_flag

        mean_velocity = np.mean(velocities) if velocities else 0.0
        self.velocity_history.append(mean_velocity)
        if len(self.velocity_history) > self.window_size:
            self.velocity_history.pop(0)
        peak_velocity = max(self.velocity_history) if self.velocity_history else 0.0
        cumulative_velocity = sum(self.velocity_history)
        # Agitation flag: if any of mean, peak, cumulative exceeds threshold
        self.agitation_flag = (
            mean_velocity > self.speed_threshold or
            peak_velocity > self.speed_threshold or
            cumulative_velocity > self.speed_threshold * self.window_size
        )
        if self.agitation_flag:
            self.last_agitation_time = time.time()
        self.prev_landmarks = curr_landmarks
        return mean_velocity, peak_velocity, cumulative_velocity, self.agitation_flag

    def get_velocity_text(self, mean_velocity, peak_velocity, cumulative_velocity):
        """
        Return Mean/Peak/Cum velocity as single line text
        """
        return f"VEL: M:{mean_velocity:.2f}  P:{peak_velocity:.2f}  C:{cumulative_velocity:.2f}"

    def draw_status(self, image, mean_velocity, peak_velocity, cumulative_velocity, agitation):
        """
        image: Frame to visualize (np.ndarray)
        mean_velocity, peak_velocity, cumulative_velocity: Velocity values
        agitation: Unrest/agitation detection status (bool)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)  # Yellow
        x, y = 50, 80  # Top left
        # Display text during alarm hold duration (don't display initially when 0)
        if self.last_agitation_time > 0 and time.time() - self.last_agitation_time < self.agitation_hold_duration:
            cv2.putText(image, "AGITATION DETECTED!", (x, y), font, 1.0, color, 3)
        return image 
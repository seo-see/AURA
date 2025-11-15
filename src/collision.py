import cv2
import mediapipe as mp
import numpy as np
import time

class CollisionConfig:
    """
    Configuration class for managing all collision detection parameters.
    """
    def __init__(self):
        # === [Mode Selection] ===
        # If True: use relative radii (proportional to head/hand size), False: use absolute pixel size
        self.use_relative_radius = False

        # === [Collision Detection Parameters] ===
        self.base_threshold = 0.3            # 3D normalized distance threshold for collision
        self.collision_duration = 0.3        # Minimum duration (seconds) to sustain danger state
        self.visibility_threshold = 0.7      # Minimum required visibility for MediaPipe landmarks
        self.camera_distance_factor = 1.0    # Placeholder for correction due to camera distance

        # === [Absolute Pixel Mode] ===
        self.mouth_radius = 150              # Mouth aura circle (pixels)
        self.hand_radius = 100               # Hand aura circle (pixels)

        # === [Relative Pixel Mode] ===
        self.mouth_radius_ratio = 2.0        # Aura radius (multiplied by head width)
        self.hand_radius_ratio = 2.0         # Aura radius (multiplied by hand width)

        # === [Weights and Score Thresholds] ===
        self.alpha = 0.7                     # Weight for 2D overlap score (0-1)
        self.beta = 0.3                      # Weight for 3D proximity score (0-1)
        self.score_threshold = 0.3           # Score threshold to decide danger

        # Fallback values when dimensions can't be computed (for missing landmarks)
        self.fallback_mouth_radius = 150
        self.fallback_hand_radius = 100

class CollisionManager:
    """
    Logic to manage collision state, duration, and alarm holding.
    """
    def __init__(self, config=None):
        self.config = config if config else CollisionConfig()
        self.is_colliding = False
        self.collision_frames = []
        self.frame_count = 0
        self.collision_start_time = 0
        self.consecutive_collisions = 0
        self.last_collision_time = 0      # Timestamp for last detected collision
        self.alarm_hold_duration = 6.0    # Keep alarm (seconds) after last detected danger

    def update(self, is_collision, current_time, frame_count):
        if is_collision:
            if self.collision_start_time == 0:
                self.collision_start_time = current_time
                self.consecutive_collisions = 1
            else:
                self.consecutive_collisions += 1
            # If collision persists, treat as active collision
            if current_time - self.collision_start_time >= self.config.collision_duration:
                self.is_colliding = True
                self.collision_frames.append(frame_count)
                self.last_collision_time = current_time
        else:
            if self.consecutive_collisions > 0:
                self.consecutive_collisions -= 1
            if self.consecutive_collisions == 0:
                self.collision_start_time = 0
                self.is_colliding = False
        # Maintain is_colliding True for alarm_hold_duration
        if current_time - self.last_collision_time < self.alarm_hold_duration:
            self.is_colliding = True
        else:
            self.is_colliding = False
        return False  # No alarm sound; just managing state

def calculate_head_width(landmarks, image_width, image_height):
    """
    Estimate head width: maximum of ear-to-ear distance and double eyebrow-to-mouth distance.
    """
    try:
        # 1. Ear-to-ear distance
        left_ear = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
        ear_width = None
        if left_ear.visibility >= 0.7 and right_ear.visibility >= 0.7:
            left_ear_x = int(left_ear.x * image_width)
            left_ear_y = int(left_ear.y * image_height)
            right_ear_x = int(right_ear.x * image_width)
            right_ear_y = int(right_ear.y * image_height)
            ear_width = np.sqrt((right_ear_x - left_ear_x)**2 + (right_ear_y - left_ear_y)**2)

        # 2. Double distance from eyebrow to mouth, for both sides
        left_eye_inner = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER]
        right_eye_inner = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER]
        mouth_left = landmarks.landmark[mp.solutions.pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = landmarks.landmark[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT]

        brow_mouth_width_1 = None
        brow_mouth_width_2 = None

        if left_eye_inner.visibility >= 0.7 and mouth_left.visibility >= 0.7:
            left_eye_x = int(left_eye_inner.x * image_width)
            left_eye_y = int(left_eye_inner.y * image_height)
            mouth_left_x = int(mouth_left.x * image_width)
            mouth_left_y = int(mouth_left.y * image_height)
            brow_mouth_width_1 = np.sqrt((mouth_left_x - left_eye_x)**2 + (mouth_left_y - left_eye_y)**2) * 2

        if right_eye_inner.visibility >= 0.7 and mouth_right.visibility >= 0.7:
            right_eye_x = int(right_eye_inner.x * image_width)
            right_eye_y = int(right_eye_inner.y * image_height)
            mouth_right_x = int(mouth_right.x * image_width)
            mouth_right_y = int(mouth_right.y * image_height)
            brow_mouth_width_2 = np.sqrt((mouth_right_x - right_eye_x)**2 + (mouth_right_y - right_eye_y)**2) * 2

        # Return the largest available measure
        widths = [w for w in [ear_width, brow_mouth_width_1, brow_mouth_width_2] if w is not None]
        if widths:
            return max(widths)
        else:
            return None
    except Exception as e:
        print(f"Error calculating head width: {e}")
        return None

def calculate_hand_width(landmarks, image_width, image_height, is_left_hand=True):
    """
    Estimate hand width: maximum of wrist-to-index_tip and thumb-to-pinky distances.
    """
    try:
        if is_left_hand:
            wrist = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            index_tip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
            thumb = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_THUMB]
            pinky = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_PINKY]
        else:
            wrist = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            index_tip = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_INDEX]
            thumb = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_THUMB]
            pinky = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_PINKY]
        
        # 1. Wrist-to-index-tip
        wrist_index_width = None
        if wrist.visibility >= 0.7 and index_tip.visibility >= 0.7:
            wrist_x = int(wrist.x * image_width)
            wrist_y = int(wrist.y * image_height)
            index_x = int(index_tip.x * image_width)
            index_y = int(index_tip.y * image_height)
            wrist_index_width = np.sqrt((index_x - wrist_x)**2 + (index_y - wrist_y)**2)

        # 2. Thumb-to-pinky
        thumb_pinky_width = None
        if thumb.visibility >= 0.7 and pinky.visibility >= 0.7:
            thumb_x = int(thumb.x * image_width)
            thumb_y = int(thumb.y * image_height)
            pinky_x = int(pinky.x * image_width)
            pinky_y = int(pinky.y * image_height)
            thumb_pinky_width = np.sqrt((pinky_x - thumb_x)**2 + (pinky_y - thumb_y)**2)

        widths = [w for w in [wrist_index_width, thumb_pinky_width] if w is not None]
        if widths:
            return max(widths)
        else:
            return None
    except Exception as e:
        print(f"Error calculating hand width: {e}")
        return None

def calculate_dynamic_radii(landmarks, image_width, image_height, config):
    """
    Dynamically calculate aura radii for mouth (head) and hands, using proportional scaling when available.
    """
    # Head width
    head_width = calculate_head_width(landmarks, image_width, image_height)

    # Hand width, use the largest of the two hands present
    left_hand_width = calculate_hand_width(landmarks, image_width, image_height, is_left_hand=True)
    right_hand_width = calculate_hand_width(landmarks, image_width, image_height, is_left_hand=False)
    hand_width = None
    if left_hand_width is not None and right_hand_width is not None:
        hand_width = max(left_hand_width, right_hand_width)
    elif left_hand_width is not None:
        hand_width = left_hand_width
    elif right_hand_width is not None:
        hand_width = right_hand_width

    if head_width is not None:
        mouth_radius = int(head_width * config.mouth_radius_ratio)
    else:
        mouth_radius = config.fallback_mouth_radius

    if hand_width is not None:
        hand_radius = int(hand_width * config.hand_radius_ratio)
    else:
        hand_radius = config.fallback_hand_radius

    return mouth_radius, hand_radius

def calculate_3d_distance(point1, point2):
    """
    Calculate Euclidean distance between two points in MediaPipe's normalized 3D coordinate space.
    """
    return np.sqrt((point1.x - point2.x)**2 + 
                  (point1.y - point2.y)**2 + 
                  (point1.z - point2.z)**2)

def calculate_normalized_distance(point1, point2):
    """
    2D (x,y) normalized Euclidean distance.
    """
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    return np.sqrt(dx**2 + dy**2)

def calculate_dynamic_threshold(image_width, image_height, config):
    """
    Optionally adapt collision threshold based on camera properties (not currently used).
    """
    return config.base_threshold * config.camera_distance_factor

def check_collision(landmarks, hand_idx, mouth_idx, image_width, image_height, threshold, config):
    """
    Checks for collision between a mouth landmark and a hand landmark, with both 2D and 3D checks.
    """
    if landmarks is None:
        return False

    hand = landmarks.landmark[hand_idx]
    mouth = landmarks.landmark[mouth_idx]

    # Visibility check
    if hand.visibility < config.visibility_threshold or mouth.visibility < config.visibility_threshold:
        return False

    # 2D pixel-space distance
    hand_x = int(hand.x * image_width)
    hand_y = int(hand.y * image_height)
    mouth_x = int(mouth.x * image_width)
    mouth_y = int(mouth.y * image_height)
    dist_2d = np.sqrt((hand_x - mouth_x)**2 + (hand_y - mouth_y)**2)

    # Aura overlap in 2D
    is_overlap = dist_2d < (config.hand_radius + config.mouth_radius)

    # 3D normalized-space distance
    dist_3d = np.sqrt(
        (hand.x - mouth.x)**2 +
        (hand.y - mouth.y)**2 +
        (hand.z - mouth.z)**2
    )
    is_close = dist_3d < config.base_threshold

    if is_overlap and is_close:
        return True

    return False

def draw_aura_layer(image, center, radius, color, alpha=0.1):
    """
    Draw an aura-like, multi-layered, semi-transparent circle at the given center/radius/color.
    """
    overlay = image.copy()
    # Outermost
    cv2.circle(overlay, center, radius, color, -1)
    # Middle
    cv2.circle(overlay, center, int(radius * 0.7), color, -1)
    # Innermost
    cv2.circle(overlay, center, int(radius * 0.4), color, -1)
    # Blend layers
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_landmarks_on_image(rgb_image, results, collision_manager, config=None):
    """
    Draw aura overlays and put collision-related scores on image. Returns annotated image and score strings.
    """
    if config is None:
        config = CollisionConfig()
    annotated_image = rgb_image.copy()
    current_time = time.time()
    if results.pose_landmarks:
        image_width = rgb_image.shape[1]
        image_height = rgb_image.shape[0]
        
        # Set aura radii based on mode
        if config.use_relative_radius:
            mouth_radius, hand_radius = calculate_dynamic_radii(
                results.pose_landmarks, image_width, image_height, config
            )
        else:
            mouth_radius = config.mouth_radius
            hand_radius = config.hand_radius
        
        # Save computed radii for use in collision checks
        config.mouth_radius = mouth_radius
        config.hand_radius = hand_radius
        
        mouth_landmarks = [mp.solutions.pose.PoseLandmark.NOSE]
        is_collision = False
        left_hand_landmark = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
        right_hand_landmark = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_INDEX]
        for mouth_landmark in mouth_landmarks:
            mouth = results.pose_landmarks.landmark[mouth_landmark]
            scores = []
            for hand_landmark in [left_hand_landmark, right_hand_landmark]:
                # Lowered visibility: set score to zero
                if hand_landmark.visibility < config.visibility_threshold:
                    scores.append((0.0, 0.0, 0.0))
                    continue
                # 2D-pixel distance and sum of aura radii
                hand_x = int(hand_landmark.x * image_width)
                hand_y = int(hand_landmark.y * image_height)
                mouth_x = int(mouth.x * image_width)
                mouth_y = int(mouth.y * image_height)
                dist_2d = np.sqrt((hand_x - mouth_x)**2 + (hand_y - mouth_y)**2)
                radius_sum = config.hand_radius + config.mouth_radius
                overlap_score = max(0, 1 - dist_2d / radius_sum)
                # 3D normalized-space distance and score
                dist_3d = np.sqrt(
                    (hand_landmark.x - mouth.x)**2 +
                    (hand_landmark.y - mouth.y)**2 +
                    (hand_landmark.z - mouth.z)**2
                )
                proximity_score = max(0, 1 - dist_3d / config.base_threshold)
                # Weighted total danger score
                score = config.alpha * overlap_score + config.beta * proximity_score
                scores.append((score, overlap_score, proximity_score))
            # If any hand's score exceeds danger threshold: treat as potential collision
            if any(s[0] > config.score_threshold for s in scores):
                is_collision = True
                break

        # Update collision state for alarm/duration
        collision_manager.update(is_collision, current_time, collision_manager.frame_count)
        color = (255, 0, 0) if collision_manager.is_colliding else (0, 255, 0)

        # Aura overlays: mouth
        for mouth_landmark in mouth_landmarks:
            landmark = results.pose_landmarks.landmark[mouth_landmark]
            mouth_point = (int(landmark.x * image_width), int(landmark.y * image_height))
            draw_aura_layer(annotated_image, mouth_point, config.mouth_radius, color, alpha=0.2)
        
        # Aura overlays: hands
        left_hand_point = (int(left_hand_landmark.x * image_width), int(left_hand_landmark.y * image_height))
        right_hand_point = (int(right_hand_landmark.x * image_width), int(right_hand_landmark.y * image_height))
        # Red if colliding, green otherwise, per hand
        left_hand_color = (255, 0, 0) if scores[0][0] > config.score_threshold else (0, 255, 0)
        right_hand_color = (255, 0, 0) if scores[1][0] > config.score_threshold else (0, 255, 0)
        if left_hand_landmark.visibility >= config.visibility_threshold:
            draw_aura_layer(annotated_image, left_hand_point, config.hand_radius, left_hand_color, alpha=0.2)
        if right_hand_landmark.visibility >= config.visibility_threshold:
            draw_aura_layer(annotated_image, right_hand_point, config.hand_radius, right_hand_color, alpha=0.2)
        
        # Draw score/detection info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 255)  # Yellow
        w = annotated_image.shape[1]

        # Score display for left/right hands (overlap and proximity, plus weighted score)
        lh_score, lh_overlap, lh_prox = scores[0]
        rh_score, rh_overlap, rh_prox = scores[1]
        lh_text = f"LH: 2D:{lh_overlap:.2f} 3D:{lh_prox:.2f} S:{lh_score:.2f}"
        rh_text = f"RH: 2D:{rh_overlap:.2f} 3D:{rh_prox:.2f} S:{rh_score:.2f}"

        return annotated_image, lh_text, rh_text
    return annotated_image, None, None

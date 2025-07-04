import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# --- Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Cannot read initial frame from webcam.")
    cap.release()
    exit()
h, w, c = frame.shape

canvas = np.zeros((h, w, c), dtype=np.uint8)
prev_x, prev_y = 0, 0
draw_color = (255, 255, 255)
erase_color = (0, 0, 0)
thickness = 5
drawing_dot_color = (0, 255, 0)
erasing_indicator_color = (0, 0, 255)
erasing_indicator_radius = 20
# Increased touch_threshold further for more drawing flexibility
touch_threshold = 45 # Adjusted from 35. Experiment with this!
min_move_threshold = 1

# --- Hysteresis / State Confirmation ---
drawing_confirm_frames_count = 0
erasing_confirm_frames_count = 0
# Slightly increased confirm_threshold for more robustness against flickers
confirm_threshold = 7 # Adjusted from 5. Experiment with this!

# --- Kalman Filter Initialization ---
class KalmanFilter:
    def __init__(self, state_dim=2, meas_dim=2, control_dim=0):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.control_dim = control_dim

        self.A = np.eye(state_dim)
        self.H = np.eye(meas_dim)
        self.Q = np.eye(state_dim) * 20
        self.R_initial = np.eye(meas_dim) * 0.5
        self.R = np.copy(self.R_initial)

        self.P = np.eye(state_dim) * 1000
        self.x = np.zeros((state_dim, 1))

        self.innovation_history = deque(maxlen=30)
        self.alpha = 0.05

    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.control_dim, 1))
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)

        self.innovation_history.append(np.array([[y[0][0]**2], [y[1][0]**2]]))

        if len(self.innovation_history) > 0:
            avg_innovation_sq = np.mean(list(self.innovation_history), axis=0)
            self.R = (1 - self.alpha) * self.R + self.alpha * np.diag(avg_innovation_sq.flatten())
            self.R = np.clip(self.R, 0.1, 50.0)

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.state_dim) - np.dot(K, self.H), self.P)

# Initialize Kalman filter
kf = KalmanFilter()

drawing_state = False  # Current confirmed drawing state

print("Starting Virtual Smart Board...")
print("Touch index finger and thumb to start drawing (green dot).")
print("Show open palm (red circle) to erase.")
print("Press 'q' to quit.")

# Helper function to check if a finger is extended (straight)
def is_finger_extended(landmarks, finger_tip, finger_pip, finger_mcp):
    # Calculate Euclidean distance between tip and PIP, and PIP and MCP
    # If finger is straight, tip-PIP distance should be substantial,
    # and PIP should be roughly above MCP (considering Y-coords in image space)
    # This also helps with various hand rotations.
    # More simply: check if the Y-coordinate of the tip is significantly above the PIP.
    # Using a threshold based on screen height can normalize this.
    tip_y = landmarks[finger_tip].y * h
    pip_y = landmarks[finger_pip].y * h
    mcp_y = landmarks[finger_mcp].y * h

    # A simple check: tip is above PIP AND PIP is above MCP
    # This implies the finger is relatively straight and extended upwards
    return (tip_y < pip_y) and (pip_y < mcp_y)

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("Error: Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    output_frame = np.zeros_like(frame)

    # Flags for current frame's detected gestures (before hysteresis)
    current_frame_is_drawing_gesture = False
    current_frame_is_erasing_gesture = False

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark

        try:
            idx_tip_x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            idx_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            thumb_tip_x = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w)
            thumb_tip_y = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h)

            # Calculate distance for drawing gesture
            distance = math.sqrt((idx_tip_x - thumb_tip_x) ** 2 + (idx_tip_y - thumb_tip_y) ** 2)
            current_frame_is_drawing_gesture = distance < touch_threshold

            # --- Improved Erasing Gesture Detection ---
            # Check if all four fingers (index, middle, ring, pinky) are extended
            fingers_extended = (
                is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP) and
                is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP) and
                is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP) and
                is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)
            )

            # Check if the thumb is also extended/away from the palm
            # A common way is to check the distance between thumb tip and index MCP (or wrist)
            thumb_idx_mcp_dist = math.sqrt(
                (landmarks[mp_hands.HandLandmark.THUMB_TIP].x - landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)**2 +
                (landmarks[mp_hands.HandLandmark.THUMB_TIP].y - landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)**2
            ) * w # Scale to pixel coordinates

            # Define a threshold for thumb being "open". This needs tuning.
            # A value like 0.15-0.25 relative to screen width might work.
            thumb_open_threshold = 0.18 * w # Example, tune this. 18% of screen width.

            # Eraser gesture: all four fingers extended AND thumb is open/away AND NOT current_frame_is_drawing_gesture
            # The last part is crucial: a drawing gesture should never be an erasing gesture
            current_frame_is_erasing_gesture = fingers_extended and (thumb_idx_mcp_dist > thumb_open_threshold) and not current_frame_is_drawing_gesture


        except Exception as e:
            # print(f"Error processing landmarks: {e}") # Suppress frequent error messages
            pass # Continue if landmarks are temporarily missing

        # --- State Confirmation Logic (Hysteresis and Prioritization) ---
        # Prioritize drawing: If drawing gesture is detected, focus on confirming that.
        if current_frame_is_drawing_gesture:
            drawing_confirm_frames_count += 1
            erasing_confirm_frames_count = 0 # Immediately reset erasing count if drawing is detected
            if drawing_confirm_frames_count >= confirm_threshold:
                drawing_state = True # Confirm drawing
        elif current_frame_is_erasing_gesture:
            erasing_confirm_frames_count += 1
            drawing_confirm_frames_count = 0 # Immediately reset drawing count if erasing is detected
            if erasing_confirm_frames_count >= confirm_threshold:
                drawing_state = False # Confirm erasing (stop drawing)
        else: # Neither primary gesture is detected, or they are ambiguous
            # Decay counts, don't just reset to 0 to allow for slight flickers within the threshold
            drawing_confirm_frames_count = max(0, drawing_confirm_frames_count - 1)
            erasing_confirm_frames_count = max(0, erasing_confirm_frames_count - 1)
            
            # If no gesture is strong enough, or both have decayed, revert to non-drawing state
            if drawing_confirm_frames_count == 0 and erasing_confirm_frames_count == 0:
                 drawing_state = False


        # --- Drawing/Erasing Execution based on confirmed state ---
        if drawing_state: # This means drawing is the confirmed state
            # Display a green dot at the index fingertip for drawing
            cv2.circle(output_frame, (idx_tip_x, idx_tip_y), 10, drawing_dot_color, cv2.FILLED)

            # Kalman Filter for drawing
            measurement = np.array([[idx_tip_x], [idx_tip_y]], dtype=np.float32)
            kf.predict()
            kf.update(measurement)
            smoothed_x, smoothed_y = int(kf.x[0]), int(kf.x[1])

            # Movement Threshold and Drawing
            if prev_x != 0 and prev_y != 0:
                distance_moved = math.sqrt((smoothed_x - prev_x) ** 2 + (smoothed_y - prev_y) ** 2)
                if distance_moved > min_move_threshold:
                    cv2.line(canvas, (int(prev_x), int(prev_y)), (smoothed_x, smoothed_y), draw_color, thickness)
                    prev_x, prev_y = smoothed_x, smoothed_y
            else:
                prev_x, prev_y = smoothed_x, smoothed_y
            
        # Only erase if confirmed erasing state AND not currently in drawing state
        elif not drawing_state and (erasing_confirm_frames_count >= confirm_threshold):
            # Display a red circle to indicate erasing
            palm_center_x = int(landmarks[mp_hands.HandLandmark.WRIST].x * w)
            palm_center_y = int(landmarks[mp_hands.HandLandmark.WRIST].y * h)
            cv2.circle(output_frame, (palm_center_x, palm_center_y), erasing_indicator_radius, erasing_indicator_color,
                       cv2.FILLED)
            # Interactive Erasing (Draw with black)
            if prev_x != 0 and prev_y != 0:
                # Use Kalman filter for erasing motion as well for smoother erasing
                measurement = np.array([[palm_center_x], [palm_center_y]], dtype=np.float32)
                kf.predict()
                kf.update(measurement)
                smoothed_erase_x, smoothed_erase_y = int(kf.x[0]), int(kf.x[1])

                cv2.line(canvas, (int(prev_x), int(prev_y)), (smoothed_erase_x, smoothed_erase_y), erase_color, thickness * 3)
                prev_x, prev_y = smoothed_erase_x, smoothed_erase_y
            else:
                # Initialize prev_x, prev_y for erasing based on current palm position
                measurement = np.array([[palm_center_x], [palm_center_y]], dtype=np.float32)
                kf.predict()
                kf.update(measurement)
                prev_x, prev_y = int(kf.x[0]), int(kf.x[1])
                

        else: # Neither state is confirmed (or intermediate), reset prev_x/y
            prev_x, prev_y = 0, 0

    else: # No hand detected
        prev_x, prev_y = 0, 0
        drawing_state = False
        drawing_confirm_frames_count = 0
        erasing_confirm_frames_count = 0


    final_frame = cv2.add(output_frame, canvas)
    cv2.imshow("Virtual Smart Board - Press 'q' to quit", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

print("Releasing resources.")
cap.release()
cv2.destroyAllWindows()
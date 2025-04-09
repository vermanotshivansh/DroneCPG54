import cv2
import mediapipe as mp
import numpy as np
import time

model_path = "/Users/shivanshverma/Desktop/Sem 6/DroneCapstone/face_landmarker.task"


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

face_locked = False
locked_box = None
last_seen = 0
lock_timeout = 2
search_box_size = 150

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global face_locked, locked_box, last_seen
    if result.face_landmarks:
        face = result.face_landmarks[0]
        xs = [lm.x for lm in face]
        ys = [lm.y for lm in face]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        h, w, _ = output_image.numpy_view().shape
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        sx, sy = w // 2 - search_box_size // 2, h // 2 - search_box_size // 2
        if not face_locked and sx < cx < sx + search_box_size and sy < cy < sy + search_box_size:
            face_locked = True
            print("✅ Face locked!")

        if face_locked:
            locked_box = (x1, y1, x2, y2)
            last_seen = time.time()



def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pip = [2, 6, 10, 14, 18]
    fingers = []
    for tip, base in zip(tips, pip):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            fingers.append(1)  # Extended
        else:
            fingers.append(0)
    return fingers  

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist (Stop)"
    if fingers == [0, 1, 0, 0, 0]:
        return "Move Forward"
    if fingers == [0, 1, 1, 1, 1]:
        return "Swipe"
    if fingers == [1, 1, 1, 1, 1]:
        return "Open Palm (Idle)"
    return "Unknown"


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_faces=1
)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        h, w, _ = frame.shape

        
        landmarker.detect_async(mp_image, timestamp_ms)

        if face_locked:
            if time.time() - last_seen > lock_timeout:
                face_locked = False
                locked_box = None
                print("❌ Face lost!")
            elif locked_box:
                x1, y1, x2, y2 = locked_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            sx, sy = w // 2 - search_box_size // 2, h // 2 - search_box_size // 2
            cv2.rectangle(frame, (sx, sy), (sx + search_box_size, sy + search_box_size), (0, 0, 255), 2)

        
        if face_locked:
            hand_result = hands.process(frame_rgb)
            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Right":
                        draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        fingers = get_finger_states(hand_landmarks)
                        gesture = detect_gesture(fingers)
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        print("Gesture detected:", gesture)

                        # === ACTION MAPPING ===
                        if gesture == "Move Forward":
                            print("🚀 FORWARD")
                        elif gesture == "Fist (Stop)":
                            print("🛑 STOP")
                        elif gesture == "Open Palm (Idle)":
                            print("🌀 HOVER")
                        elif gesture == "Swipe":
                            print("➡️ SWIPE MOVE")  

        cv2.imshow("Face + Right Hand Tracker", frame)
        if cv2.waitKey(5) & 0xFF == ord("a"):
            break

cap.release()
cv2.destroyAllWindows()

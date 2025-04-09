import cv2
import mediapipe as mp
import numpy as np
import time

# Setup
model_path = "/Users/shivanshverma/Desktop/DroneCapstone/face_landmarker.task"  # Update if needed

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global face lock status
face_locked = False
last_seen = 0
locked_box = None
lock_timeout = 2  # seconds
search_box_size = 150


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global face_locked, last_seen, locked_box
    if result.face_landmarks:
        # Get bounding box from landmarks
        face_landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        frame_h, frame_w, _ = output_image.numpy_view().shape
        x1, y1 = int(xmin * frame_w), int(ymin * frame_h)
        x2, y2 = int(xmax * frame_w), int(ymax * frame_h)

        # Only lock if inside search box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sx, sy = frame_w // 2 - search_box_size // 2, frame_h // 2 - search_box_size // 2

        if not face_locked:
            if sx < cx < sx + search_box_size and sy < cy < sy + search_box_size:
                face_locked = True
                print("Face locked!")

        if face_locked:
            locked_box = (x1, y1, x2, y2)
            last_seen = time.time()


# Set options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Start webcam and landmarker
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        # Face tracking
        if face_locked:
            if time.time() - last_seen > lock_timeout:
                face_locked = False
                locked_box = None
                print("Face lost!")
            elif locked_box:
                x1, y1, x2, y2 = locked_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Show red center box when waiting
            sx, sy = w // 2 - search_box_size // 2, h // 2 - search_box_size // 2
            cv2.rectangle(frame, (sx, sy), (sx + search_box_size, sy + search_box_size), (0, 0, 255), 2)

        cv2.imshow("MediaPipe Face Lock", frame)
        if cv2.waitKey(5) == ord("a"):
            break

cap.release()
cv2.destroyAllWindows()

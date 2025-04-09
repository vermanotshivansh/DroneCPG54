import cv2

face_cap = cv2.CascadeClassifier("/Users/shivanshverma/Desktop/DroneCapstone/haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

face_locked = False
locked_face = None
lock_tolerance = 50  # pixels for movement
search_box_color = (0, 0, 255)  # red
lock_box_color = (0, 255, 0)    # green

while True:
    ret, video_data = video_capture.read()
    if not ret:
        break

    height, width, _ = video_data.shape

    # Define the center "search" box
    center_x, center_y = width // 2, height // 2
    search_box_size = 150
    search_box = (
        center_x - search_box_size // 2,
        center_y - search_box_size // 2,
        search_box_size,
        search_box_size
    )

    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if face_locked:
        still_present = False
        for (x, y, w, h) in faces:
            lx, ly, lw, lh = locked_face
            if abs(x - lx) < lock_tolerance and abs(y - ly) < lock_tolerance:
                locked_face = (x, y, w, h)
                still_present = True
                break
        if not still_present:
            face_locked = False
            locked_face = None

    if not face_locked:
        # Show search box
        (sx, sy, sw, sh) = search_box
        cv2.rectangle(video_data, (sx, sy), (sx + sw, sy + sh), search_box_color, 2)

        # Try to lock face within search box
        for (x, y, w, h) in faces:
            face_center = (x + w // 2, y + h // 2)
            if (sx < face_center[0] < sx + sw) and (sy < face_center[1] < sy + sh):
                locked_face = (x, y, w, h)
                face_locked = True
                break

    if locked_face:
        x, y, w, h = locked_face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), lock_box_color, 2)

    cv2.imshow("Face Tracker", video_data)
    if cv2.waitKey(5) == ord("a"):
        break

video_capture.release()
cv2.destroyAllWindows()

import cv2
import time
import numpy as np
from ultralytics import YOLO

from camera_utils import select_camera
from face_recognition_module import FaceRecognitionSystem
from cat_color_classifier import classify_cat_color

print("OpenCV version:", cv2.__version__)

# ====================== PERFORMANCE SETTINGS ======================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

YOLO_IMG_SIZE = 320
YOLO_CONF = 0.45                    # Slightly increased for better boxes

FACE_RECOGNITION_ENABLED = False

# COCO class id for cat
CAT_CLASS_ID = 15

# ====================== BOX SETTINGS ======================
BOX_SHRINK_FACTOR = 0.085           # Higher = tighter box (0.05 ~ 0.12 recommended)

# ====================== DEBUG SETTINGS ======================
CAT_COLOR_DEBUG = False             # Set to True to start in debug mode

# ====================== CAMERA SELECTION ======================
FORCE_CAMERA_INDEX = None

if FORCE_CAMERA_INDEX is not None:
    cam_index = FORCE_CAMERA_INDEX
    print(f"Forcing camera index: {cam_index}")
else:
    cam_index = select_camera()
    if cam_index is None:
        print("Exiting Program")
        exit()

# ====================== LOAD YOLO MODEL ======================
print("Loading YOLO model...")

model = YOLO("yolo26s.pt")
model.to("cuda")

print(f"✅ YOLO Model loaded on: {next(model.model.parameters()).device}")

# ====================== YOLO WARMUP ======================
print("Warming up YOLO model...")
dummy = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=YOLO_IMG_SIZE, verbose=False, device="cuda")
print("YOLO warm-up completed.\n")

# ====================== FACE RECOGNITION ======================
if FACE_RECOGNITION_ENABLED:
    face_system = FaceRecognitionSystem()
    face_recognition_enabled = True
    face_add_mode = False
    print("Face Recognition System initialized.\n")
else:
    face_recognition_enabled = False
    face_add_mode = False

# ====================== CAMERA ======================
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera Opened Successfully.")
print(
    "Controls:\n"
    "'q' or ESC - Quit\n"
    "'a' - Add Face\n"
    "'r' - Toggle Face Recognition\n"
    "'d' - Toggle Cat Color Debug\n"
)

# ====================== FPS TRACKING ======================
fps_display = 0
fps_counter = 0
fps_time = time.time()

# ====================== MAIN LOOP ======================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ====================== YOLO DETECTION ======================
    results = model(
        frame,
        imgsz=YOLO_IMG_SIZE,
        conf=YOLO_CONF,
        verbose=False,
        device="cuda",
        half=True
    )

    annotated_frame = frame.copy()

    # ====================== PROCESS DETECTIONS ======================
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls != CAT_CLASS_ID:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Safety clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            cat_crop = frame[y1:y2, x1:x2]

            if cat_crop.size == 0:
                continue

            # ====================== COLOR CLASSIFICATION ======================
            cat_label, box_color, color_confidence = classify_cat_color(
                cat_crop, 
                debug=CAT_COLOR_DEBUG
            )

            # ====================== TIGHTEN BOUNDING BOX ======================
            width = x2 - x1
            height = y2 - y1
            
            shrink_x = int(width * BOX_SHRINK_FACTOR)
            shrink_y = int(height * BOX_SHRINK_FACTOR)

            x1_tight = x1 + shrink_x
            y1_tight = y1 + shrink_y
            x2_tight = x2 - shrink_x
            y2_tight = y2 - shrink_y

            # ====================== DRAW TIGHTENED BOX ======================
            display_text = (
                f"{cat_label} "
                f"YOLO:{confidence:.2f} "
                f"CLR:{color_confidence:.2f}"
            )

            cv2.rectangle(
                annotated_frame,
                (x1_tight, y1_tight),
                (x2_tight, y2_tight),
                box_color,
                2
            )

            cv2.putText(
                annotated_frame,
                display_text,
                (x1_tight, y1_tight - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    # ====================== FACE RECOGNITION ======================
    if face_recognition_enabled:
        face_locations, face_names = face_system.recognize_faces(frame)

        if face_locations:
            annotated_frame = face_system.draw_face_results(
                annotated_frame, face_locations, face_names
            )

        if face_add_mode:
            cv2.putText(
                annotated_frame,
                "FACE ADD MODE - Press 'a' again to add detected face",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

    # ====================== FPS CALCULATION ======================
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_time = time.time()

    # ====================== STATUS TEXT ======================
    status_text = f"FPS: {fps_display} | YOLO:{YOLO_IMG_SIZE}"

    if face_recognition_enabled:
        face_count = len(face_locations) if 'face_locations' in locals() else 0
        status_text += f" | Faces: {face_count}"

    if CAT_COLOR_DEBUG:
        status_text += " | DEBUG ON"

    cv2.putText(
        annotated_frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ====================== HELP TEXT ======================
    cv2.putText(
        annotated_frame,
        "'q'-Quit | 'a'-Add Face | 'r'-Toggle Rec | 'd'-Toggle Debug",
        (10, annotated_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )

    # ====================== SHOW WINDOW ======================
    window_name = "opencv-yolo"
    cv2.imshow(window_name, annotated_frame)

    # ====================== KEYBOARD CONTROLS ======================
    key = cv2.waitKey(1) & 0xFF

    if key in [ord("q"), ord("c"), 27]:
        print("Exiting...")
        break

    elif key == ord("a") and face_recognition_enabled:
        if not face_add_mode:
            face_add_mode = True
            print("\n📸 FACE ADD MODE ACTIVATED")
            print("Press 'a' again when ready to add the face")
        else:
            print("\n📸 Attempting to add face...")
            success = face_system.add_face_interactive(frame)
            if success:
                print("✅ Face added successfully!\n")
                face_add_mode = False
            else:
                print("❌ Failed to add face. Try again.\n")

    elif key == ord("r"):
        face_recognition_enabled = not face_recognition_enabled
        print("✅ Face recognition ENABLED" if face_recognition_enabled else "❌ Face recognition DISABLED")
        if not face_recognition_enabled:
            face_add_mode = False

    elif key == ord("d"):
        CAT_COLOR_DEBUG = not CAT_COLOR_DEBUG
        if CAT_COLOR_DEBUG:
            print("🐱 Cat Color Debug Mode ENABLED")
        else:
            print("🐱 Cat Color Debug Mode DISABLED")

    # Window closed manually
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed by user. Exiting...")
        break

# ====================== CLEANUP ======================
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
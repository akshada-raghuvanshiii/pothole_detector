from ultralytics import YOLO
import cv2
from collections import deque

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "pothole.pt"
VIDEO_PATH = "ruralRoad_potHoles.mp4"
 
BASE_CONF = 0.25
CENTER_CONF = 0.40
SIDE_CONF = 0.25

SEVERITY_WINDOW = 5  # temporal smoothing

# =====================================================
# LOAD MODEL
# =====================================================
model = YOLO(MODEL_PATH)

# =====================================================
# OPEN VIDEO
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

paused = False
severity_buffer = deque(maxlen=SEVERITY_WINDOW)

print("Controls:")
print("SPACE : Pause / Resume")
print("Q     : Quit")

# =====================================================
# VIDEO LOOP
# =====================================================
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break

        h, w, _ = frame.shape

        # -----------------------------------------------
        # ROAD REGION OF INTEREST (LOWER 60%)
        # -----------------------------------------------
        roi_y_start = int(0.4 * h)
        roi = frame[roi_y_start:h, :]

        # -----------------------------------------------
        # YOLO INFERENCE
        # -----------------------------------------------
        results = model(roi, conf=BASE_CONF, verbose=False)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if label != "pothole":
                    continue

                # Bounding box (ROI coords)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y1 += roi_y_start
                y2 += roi_y_start

                cx = (x1 + x2) / 2

                # -----------------------------------------------
                # ADAPTIVE CONFIDENCE (CENTER vs SIDES)
                # -----------------------------------------------
                if w * 0.3 < cx < w * 0.7:
                    if conf < CENTER_CONF:
                        continue
                    lane_risk = 1.0
                else:
                    if conf < SIDE_CONF:
                        continue
                    lane_risk = 0.6

                # -----------------------------------------------
                # RULE-BASED SEVERITY (NO SCORE COLLAPSE)
                # -----------------------------------------------
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                area_ratio = box_area / frame_area

                # Strong rescaling for dashcam video
                scaled_area = min(area_ratio * 30, 1.0)

                # Base severity from size
                if scaled_area > 0.45:
                    base_severity = 3   # HIGH
                elif scaled_area > 0.20:
                    base_severity = 2   # MEDIUM
                else:
                    base_severity = 1   # LOW

                # Upgrade if in center lane
                if lane_risk == 1.0 and base_severity < 3:
                    base_severity += 1

                # -----------------------------------------------
                # TEMPORAL STABILIZATION
                # -----------------------------------------------
                severity_buffer.append(base_severity)
                final_level = round(sum(severity_buffer) / len(severity_buffer))

                if final_level == 3:
                    severity = "HIGH"
                    color = (0, 0, 255)      # Red
                elif final_level == 2:
                    severity = "MEDIUM"
                    color = (0, 165, 255)    # Orange
                else:
                    severity = "LOW"
                    color = (0, 255, 0)      # Green

                # -----------------------------------------------
                # DRAW OUTPUT
                # -----------------------------------------------
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f} | {severity}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("Pothole Detection with Severity", frame)

    # -----------------------------------------------
    # KEYBOARD CONTROLS
    # -----------------------------------------------
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE
        paused = not paused

cap.release()
cv2.destroyAllWindows()
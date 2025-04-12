import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Open video
cap = cv2.VideoCapture("666.mp4")

# === ROI Selection ===
roi_points = [(161, 290), (826, 290), (863, 697), (161, 699)]

# === Detection Loop ===
def point_inside_polygon(x, y, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), (x, y), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1)
    detections = results[0].boxes.data.cpu().numpy()

    people_inside_roi = 0
    for det in detections:
        class_id = int(det[5])
        if class_id == 0:
            x1, y1, x2, y2 = map(int, det[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if point_inside_polygon(cx, cy, roi_points):
                people_inside_roi += 1
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Estimate fill
    roi_area = cv2.contourArea(np.array(roi_points, dtype=np.int32))
    print(roi_points)
    print(roi_area)
    person_area = 5000  # Adjust this
    estimated_fill = (people_inside_roi * person_area) / roi_area * 100
    print(estimated_fill)

    # Determine alert
    if estimated_fill > 60:
        alert = "HIGH"
        color = (0, 0, 255)
    else:
        alert = "NORMAL"
        color = (0, 255, 0)

    # Draw ROI and info
    cv2.polylines(frame, [np.array(roi_points, np.int32)], isClosed=True, color=color, thickness=3)
    cv2.putText(frame, f"Alert: {alert} ({estimated_fill:.1f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show frame
    cv2.imshow("Crowd Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

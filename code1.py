import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Confidence threshold
threshold = 0.6
centroid_threshold = 100  # Distance threshold for centroid proximity (adjust as needed)

def is_wearing_white(cropped_img):
    # Convert to LAB color space (better for brightness detection)
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    _, bright_mask = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    # Convert to HSV to isolate white color
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    color_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine both masks
    final_mask = cv2.bitwise_and(color_mask, bright_mask)

    white_pixels = cv2.countNonZero(final_mask)
    total_pixels = cropped_img.shape[0] * cropped_img.shape[1]
    
    white_ratio = white_pixels / total_pixels
    return white_ratio > 0.15

def calculate_centroid(xmin, ymin, xmax, ymax):
    return ((xmin + xmax) // 2, (ymin + ymax) // 2)

# Extract detection results in structured format
def detect_boxes(results):
    boxes = results[0].boxes
    rois = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        score = int(box.conf[0] * 100)
        class_id = int(box.cls[0])
        tracker_id = int(box.id[0]) if box.id is not None else None
        rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
    return rois

# Open video
cap = cv2.VideoCapture("video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("safety_output.mp4", fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

# Persistent safety status dictionary
persistent_status = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use tracking with YOLOv8
    results = model.track(frame, persist=True, conf=threshold, iou=0.5, agnostic_nms=True)
    rois = detect_boxes(results)

    person_data = {}
    white_objects = []

    # First pass: Collect person and potential white object data
    for roi in rois:
        x1, y1, x2, y2, class_id, score, tracker_id = roi
        if tracker_id is None:
            continue

        centroid = calculate_centroid(x1, y1, x2, y2)
        
        if model.names[class_id] == 'person':
            cropped = frame[y1:y2, x1:x2]
            has_white = is_wearing_white(cropped)
            person_data[tracker_id] = {
                "bbox": [x1, y1, x2, y2],
                "centroid": centroid,
                "white_detected": has_white
            }
        else:
            # Check if non-person object might be white (potential safety item)
            cropped = frame[y1:y2, x1:x2]
            if is_wearing_white(cropped):
                white_objects.append({"centroid": centroid, "tracker_id": tracker_id})

    # Second pass: Check proximity of white objects to persons
    for tracker_id in person_data:
        person_centroid = person_data[tracker_id]["centroid"]
        person_data[tracker_id]["near_white"] = False
        
        for white_obj in white_objects:
            distance = math.sqrt(
                (white_obj["centroid"][0] - person_centroid[0])**2 +
                (white_obj["centroid"][1] - person_centroid[1])**2
            )
            if distance < centroid_threshold:
                person_data[tracker_id]["near_white"] = True
                # Update persistent status if white object is close
                persistent_status[tracker_id] = True
                break

    # Visualize results
    for tracker_id in person_data:
        x1, y1, x2, y2 = person_data[tracker_id]["bbox"]
        
        # Determine safety status
        if tracker_id in persistent_status and persistent_status[tracker_id]:
            safe = True
        else:
            safe = person_data[tracker_id]["white_detected"] or person_data[tracker_id]["near_white"]

        color = (0, 255, 0) if safe else (0, 0, 255)
        label = "Safe" if safe else "Not Safe"
        # Uncomment to include tracker ID
        # label += f" ID: {tracker_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
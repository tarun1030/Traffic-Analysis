from ultralytics import YOLO
import cv2
import math

# Load custom-trained YOLOv8 model
model = YOLO("yolo8n.pt")  # Update this path

# Open video
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties and output setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "output_video2.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

conf_threshold = 0.1  # Lowered to capture more detections
centroid_threshold = 110  # Distance threshold for centroid proximity (adjust as needed)

def calculate_centroid(xmin, ymin, xmax, ymax):
    return ((xmin + xmax) // 2, (ymin + ymax) // 2)

def detect_boxes(results):
    boxes = results[0].boxes
    rois = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        score = box.conf[0]
        class_id = int(box.cls[0])
        tracker_id = int(box.id[0]) if box.id is not None else -1
        if class_id in [0, 10]:  # Person, Helmet
            rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
    return rois

# Persistent safety status dictionary
persistent_status = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform tracking
    results = model.track(frame, persist=True, conf=conf_threshold, iou=0.5, agnostic_nms=True)
    detections = detect_boxes(results)

    person_ppe = {}
    # Store all bounding boxes and centroids first
    for det in detections:
        xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
        if tracker_id == -1:  # Skip if no tracker ID
            continue
        centroid = calculate_centroid(xmin, ymin, xmax, ymax)
        if class_id == 0:  # Person
            person_ppe[tracker_id] = {
                "helmet": False,
                "bbox": [xmin, ymin, xmax, ymax],
                "centroid": centroid
            }
        elif class_id == 10:  # Helmet
            person_ppe[tracker_id] = {"bbox": [xmin, ymin, xmax, ymax], "centroid": centroid} if tracker_id not in person_ppe else person_ppe[tracker_id]

    # Assign helmet status based on centroid proximity
    for det in detections:
        xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
        if tracker_id == -1 or class_id == 0:  # Skip if no tracker ID or if it's a person
            continue
        if class_id == 10:  # Helmet
            helmet_centroid = calculate_centroid(xmin, ymin, xmax, ymax)
            for tid in person_ppe:
                if tid != -1 and tid != tracker_id and "centroid" in person_ppe[tid]:
                    person_centroid = person_ppe[tid]["centroid"]
                    distance = math.sqrt((helmet_centroid[0] - person_centroid[0])**2 + 
                                       (helmet_centroid[1] - person_centroid[1])**2)
                    if distance < centroid_threshold:
                        person_ppe[tid]["helmet"] = True
                        # Update persistent status
                        persistent_status[tid] = True

    # Visualize results with persistent status
    for det in detections:
        xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
        if class_id == 0 and tracker_id != -1:
            # Check persistent status first
            if tracker_id in persistent_status and persistent_status[tracker_id]:
                has_helmet = True
            else:
                has_helmet = person_ppe[tracker_id]["helmet"]
            
            label = "Safe (Helmet)" if has_helmet else "Unsafe (No Helmet)"
            color = (0, 255, 0) if has_helmet else (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(frame)
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_path}")
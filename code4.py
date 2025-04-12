import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

def calculate_overlap_percentage(bbox, roi_coordinates):
    xmin, ymin, xmax, ymax = bbox
    mask = np.zeros((max(1000, ymax + 10), max(1000, xmax + 10)), dtype=np.uint8)
    roi_np = np.array(roi_coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [roi_np], 255)

    bbox_mask = np.zeros_like(mask)
    cv2.rectangle(bbox_mask, (xmin, ymin), (xmax, ymax), 255, -1)

    intersection = cv2.bitwise_and(mask, bbox_mask)
    intersection_area = cv2.countNonZero(intersection)
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area == 0:
        return 0
    return intersection_area / bbox_area

def process_video(video_path, roi_data, output_name="output_intrusion.mp4",
                  detection_threshold=0.7, overlap_threshold=0.5, frame_interval=1):

    os.makedirs("output_videos", exist_ok=True)
    output_path = os.path.join("output_videos", output_name)

    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_intrusion_time = 0
    max_persons_in_roi = 0
    time_per_frame = 1 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        display_frame = frame.copy()
        cv2.polylines(display_frame, [np.array(roi_data, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.putText(display_frame, "Restricted Area", roi_data[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        results = model.predict(frame, conf=detection_threshold, classes=[0])
        persons_in_roi = 0

        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            overlap = calculate_overlap_percentage((xmin, ymin, xmax, ymax), roi_data)
            if overlap >= overlap_threshold:
                persons_in_roi += 1
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(display_frame, "INTRUDER", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if persons_in_roi > max_persons_in_roi:
            max_persons_in_roi = persons_in_roi

        if persons_in_roi > 0:
            total_intrusion_time += time_per_frame
            cv2.putText(display_frame, f"ALERT: {persons_in_roi} person(s)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(display_frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    print("\n✅ Intrusion Detection Complete")
    print(f"Total intrusion time: {total_intrusion_time:.2f} seconds")
    print(f"Max persons detected in ROI: {max_persons_in_roi}")

if __name__ == "__main__":
    # Define the ROI as a list of points (example for demo)
    roi = [(200, 100), (500, 100), (500, 400), (200, 400)]
    
    # Run the intrusion detection
    process_video(
        video_path="sample_video.mp4",        # Replace with your video path
        roi_data=roi,
        output_name="demo_intrusion_output.mp4",
        detection_threshold=0.5,              # YOLO detection confidence threshold
        overlap_threshold=0.4,                # Minimum % bbox inside ROI
        frame_interval=2                      # Skip every N frames
    )

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_blur(frame, threshold=180):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def detect_scene_change(frame1, frame2, threshold=0.75):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold, score

# Video
cap = cv2.VideoCapture("3.mp4")
ret, prev_frame = cap.read()

# Persistent tampering state
tamper_state = {
    "blur": False,
    "scene_change": False,
    "occlusion": False
}

alert_duration = 100  # number of frames to persist the alert
alert_counter = {
    "blur": 0,
    "scene_change": 0,
    "occlusion": 0
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect blur
    blur_flag, blur_score = detect_blur(frame)
    if blur_flag:
        tamper_state["blur"] = True
        alert_counter["blur"] = alert_duration
    elif alert_counter["blur"] > 0:
        alert_counter["blur"] -= 1
    else:
        tamper_state["blur"] = False

    # Detect scene change
    scene_change_flag, scene_score = detect_scene_change(prev_frame, frame)
    if scene_change_flag:
        tamper_state["scene_change"] = True
        alert_counter["scene_change"] = alert_duration
    elif alert_counter["scene_change"] > 0:
        alert_counter["scene_change"] -= 1
    else:
        tamper_state["scene_change"] = False

    # Occlusion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_var = np.var(hist)
    occlusion_flag = hist_var < 100

    if occlusion_flag:
        tamper_state["occlusion"] = True
        alert_counter["occlusion"] = alert_duration
    elif alert_counter["occlusion"] > 0:
        alert_counter["occlusion"] -= 1
    else:
        tamper_state["occlusion"] = False

    # Display tamper alerts
    if any(tamper_state.values()):
        y = 50
        for t_type, active in tamper_state.items():
            if active:
                msg = f"TAMPERING DETECTED: {t_type.upper()}"
                cv2.putText(frame, msg, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1.5)
                y += 30

    cv2.imshow("Camera Tampering Detection", frame)
    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

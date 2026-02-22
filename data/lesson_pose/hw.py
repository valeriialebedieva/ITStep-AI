import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')


def get_angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])  # Knee (center point)
    c = np.array([x3, y3])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


cap = cv2.VideoCapture('squat.mp4')
counter = 0
stage = "up"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    if results.keypoints:
        points = results.keypoints.xy[0].cpu().numpy()

        if len(points) > 16:
            x1, y1 = points[12]  # Right Hip
            x2, y2 = points[14]  # Right Knee
            x3, y3 = points[16]  # Right Ankle

            angle = get_angle(x1, y1, x2, y2, x3, y3)

            if angle < 90:
                stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1

            cv2.putText(frame, f"Angle: {int(angle)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {counter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Squat Counter (YOLO)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
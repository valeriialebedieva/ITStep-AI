import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'meetings.mp4'
cap = cv2.VideoCapture(video_path)

# task 1
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("Task 1 - Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# task 2
cap = cv2.VideoCapture(video_path)

show_video = False

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame, classes=[0], verbose=False)

    people_count = len(results[0].boxes)

    if people_count == 5:
        show_video = True
        print("Target reached: 5 people detected. Starting video display...")

    if show_video:
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f'People count: {people_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Task 2 - Trigger on 5 People", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
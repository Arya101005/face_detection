import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

save_folder = "captured_faces"
os.makedirs(save_folder, exist_ok=True)

model = YOLO("models/yolov8n-face.pt")
tracker = DeepSort(max_age=30)
saved_ids = set()

video_path = "test.mp4"

if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
    print("Using webcam instead...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(video_path)

face_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    results = model(frame)

    tracks_input = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            width = x2 - x1
            height = y2 - y1

            if confidence > 0.5 and width > 40 and height > 40:
                tracks_input.append(([x1, y1, width, height], confidence, "face"))

    tracks = tracker.update_tracks(tracks_input, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        if track_id not in saved_ids:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_filename = os.path.join(save_folder, f"face_{track_id}.jpg")
                cv2.imwrite(face_filename, face_crop)
                saved_ids.add(track_id)

    face_count = len(saved_ids)

    cv2.putText(
        frame,
        f"Faces Count: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total unique faces detected: {face_count}")
print(f"Faces saved to: {save_folder}")

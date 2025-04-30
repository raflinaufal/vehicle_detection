import os
import cv2
import numpy as np
import gdown
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ====================
# === SETUP VIDEO ====
# ====================
# Ganti dengan ID file Google Drive Anda
GOOGLE_DRIVE_FILE_ID = "1IiW1T8NUgchlLXXwmgNZiZVc-wAHFJfN"  # <--- GANTI DI SINI
VIDEO_FILENAME = "video_drive.mp4"

# Unduh video jika belum ada
if not os.path.exists(VIDEO_FILENAME):
    print("Mengunduh video dari Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", VIDEO_FILENAME, quiet=False)

# Buka video
cap = cv2.VideoCapture(VIDEO_FILENAME)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS video: {fps}")

# ======================
# === LOAD MODEL & TRACKER ===
# ======================
model = YOLO("yolov8n.pt")  # Gunakan model kecil, bisa diganti 'yolov8s.pt' dll
model.fuse()
tracker = DeepSort(max_age=30)

# =====================
# === KONFIGURASI ====
# =====================
PIXEL_TO_METER = 0.05   # Perkiraan konversi pixel ke meter
line_y = 300            # Y-position untuk garis counting

# Counter
enter_count = 0
exit_count = 0
positions = {}

# ===================
# === MAIN LOOP ====
# ===================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in ["car", "truck"]:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Hitung kecepatan
        if track_id in positions:
            prev_cx, prev_cy = positions[track_id]
            pixel_distance = np.linalg.norm([cx - prev_cx, cy - prev_cy])
            meter_per_frame = pixel_distance * PIXEL_TO_METER
            speed = (meter_per_frame * fps) * 3.6  # m/s ke km/h
        else:
            speed = 0.0

        # Simpan posisi saat ini
        positions[track_id] = (cx, cy)

        # Hitung kendaraan yang masuk atau keluar
        if (cy < line_y and positions[track_id][1] >= line_y):
            enter_count += 1
        elif (cy > line_y and positions[track_id][1] <= line_y):
            exit_count += 1

        # Gambar bounding box dan info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{track.det_class} {int(speed)}km/h", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Gambar garis counting
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    # Tampilkan info counting
    cv2.putText(frame, f"Count Entering: {enter_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Count Exiting: {exit_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import sys
import os
import requests
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.utils.video_processing import read_video, display_video_with_ui, display_video_with_opencv
from src.utils.yolo_inference import run_yolo_inference
from src.tracking.tracker import VehicleTracker
from src.tracking.speed_estimation import SpeedEstimator
from src.counting.vehicle_counter import VehicleCounter
from src.config.settings import VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH
from src.config.settings import SPEED_ESTIMATION_FACTOR
from src.config.settings import VEHICLE_CLASSES, DETECT_ONLY_CLASSES, DETECTION_COLORS, COUNT_TEXT_COLOR

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def download_yolov8_model():
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8", "yolov8.pt")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print("Downloading YOLOv8 model...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("YOLOv8 model downloaded successfully.")
        else:
            print("Failed to download YOLOv8 model. Please check the URL or your internet connection.")
    else:
        print("YOLOv8 model already exists.")

def process_frame(frame, vehicle_tracker, speed_estimator, vehicle_counter):
    detections = vehicle_tracker.update(frame)
    if not detections or not isinstance(detections, list):
        return frame

    frame_height, frame_width = frame.shape[:2]
    divider_y = frame_height // 2
    cv2.line(frame, (0, divider_y), (frame_width, divider_y), (0, 255, 0), 2)  # Green line

    for vehicle in detections:
        bbox = vehicle['bbox']
        vehicle_id = vehicle['id']
        class_id = vehicle.get('class_id', 2)
        vehicle_type = VEHICLE_CLASSES.get(class_id, 'car')
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        speed = speed_estimator.estimate_speed(vehicle_id, (center_x, center_y))
        color = DETECTION_COLORS.get(vehicle_type, (255, 255, 255))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        label = f"{vehicle_id}:{vehicle_type} {int(speed)}km/h"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_size[1] - 10), (bbox[0] + text_size[0], bbox[1]), color, -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Update counter: crossing line
        vehicle_counter.update_counts(vehicle_id, center_y, vehicle_type)

    counts = vehicle_counter.get_counts()
    enter_count = counts['total']['enter']
    leave_count = counts['total']['leave']

    # --- Display counts ---
    x_left = 10
    y_top = 30
    cv2.putText(frame, "Count of Vehicles Leaving", (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
    cv2.putText(frame, f"Total Count: {leave_count}", (x_left, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
    y_offset = y_top + 60
    for vehicle_type, count in counts['by_type'].items():
        if count['leave'] > 0:
            cv2.putText(frame, f"{vehicle_type}:{count['leave']}", (x_left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
            y_offset += 30

    # Top-right: Entering
    enter_title = "Count of Vehicles Entering"
    total_enter_text = f"Total Count: {enter_count}"
    (title_w, _), _ = cv2.getTextSize(enter_title, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    (total_w, _), _ = cv2.getTextSize(total_enter_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    x_right = frame_width - title_w - 10
    cv2.putText(frame, enter_title, (x_right, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
    cv2.putText(frame, total_enter_text, (frame_width - total_w - 10, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
    y_offset = y_top + 60
    for vehicle_type, count in counts['by_type'].items():
        if count['enter'] > 0:
            pertype_text = f"{vehicle_type}:{count['enter']}"
            (w, _), _ = cv2.getTextSize(pertype_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
            cv2.putText(frame, pertype_text, (frame_width - w - 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_TEXT_COLOR, 3)
            y_offset += 30

    return frame

def main():
    """Main function to run the object tracking pipeline."""
    setup_logging()
    logging.info("Starting object tracking pipeline.")

    # Ensure YOLOv8 model is downloaded
    download_yolov8_model()

    # Initialize video input
    video_capture = read_video(VIDEO_INPUT_PATH)
    
    # Get original video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Video properties: {frame_width}x{frame_height}, {original_fps} FPS")

    # Initialize YOLOv8 model
    yolo_model = run_yolo_inference(VIDEO_INPUT_PATH)

    # Initialize trackers and counters
    vehicle_tracker = VehicleTracker(model=yolo_model)
    speed_estimator = SpeedEstimator(fps=original_fps, real_world_distance_per_pixel=SPEED_ESTIMATION_FACTOR)

    # Inisialisasi VehicleCounter berbasis divider_y
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    divider_y = frame_height // 2
    vehicle_counter = VehicleCounter(divider_y=divider_y)

    def process_frame_with_tracking(frame):
        try:
            return process_frame(frame, vehicle_tracker, speed_estimator, vehicle_counter)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    display_video_with_opencv(video_capture, process_frame_with_tracking)

    logging.info("Object tracking pipeline finished.")

if __name__ == "__main__":
    main()
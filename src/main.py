import sys
import os
import requests
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.utils.video_processing import read_video, write_video
from src.utils.yolo_inference import run_yolo_inference
from src.tracking.tracker import VehicleTracker
from src.tracking.speed_estimation import SpeedEstimator
from src.config.settings import VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH
from src.config.settings import SPEED_ESTIMATION_FACTOR

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,  # Change to DEBUG to enable detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def download_yolov8_model():
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"  # YOLOv8m model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8", "yolov8m.pt")

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

def process_frame(frame, vehicle_tracker, speed_estimator):
    detections = vehicle_tracker.update(frame)
    if not detections or not isinstance(detections, list):
        return frame

    for vehicle in detections:
        bbox = vehicle['bbox']
        vehicle_id = vehicle['id']
        class_id = vehicle.get('class_id', 2)  # Default to 'car' if class_id is missing
        
        # Map class_id to vehicle type
        vehicle_type = 'car'  # Default type
        if class_id == 2:  # Example: COCO class ID for car
            vehicle_type = 'car'
        elif class_id == 7:  # Example: COCO class ID for truck
            vehicle_type = 'truck'
        elif class_id == 3:  # Example: COCO class ID for motorcycle
            vehicle_type = 'motorcycle'
        
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        speed = speed_estimator.estimate_speed(vehicle_id, (center_x, center_y), vehicle_type)
        
        # Draw bounding box
        color = (255, 0, 0)  # Blue for all vehicles
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Display speed only
        speed_text = f"{int(speed)} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Calculate text position
        text_y = bbox[1] - 10
        if text_y < 20:
            text_y = bbox[1] + 20
        
        # Draw speed text
        cv2.putText(frame, speed_text, (bbox[0], text_y), font, font_scale, color, thickness)

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

    # Dynamically calculate SPEED_ESTIMATION_FACTOR based on drone altitude and camera FOV
    fov_horizontal = 90  # Example: 90 degrees (adjust based on your camera specs)
    fov_vertical = 60    # Example: 60 degrees (adjust based on your camera specs)
    speed_estimation_factor = SpeedEstimator.calculate_real_world_distance_per_pixel(
        frame_width, frame_height, fov_horizontal, fov_vertical
    )
    logging.info(f"Calculated SPEED_ESTIMATION_FACTOR: {speed_estimation_factor}")

    # Initialize YOLOv8 model
    yolo_model = run_yolo_inference(VIDEO_INPUT_PATH)

    # Initialize trackers and speed estimator
    vehicle_tracker = VehicleTracker(model=yolo_model)
    speed_estimator = SpeedEstimator(fps=original_fps, real_world_distance_per_pixel=speed_estimation_factor)

    # Initialize video writer for output
    output_dir = os.path.dirname(VIDEO_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    video_writer = write_video(VIDEO_OUTPUT_PATH, frame_width, frame_height, original_fps)

    frame_count = 0
    while True:  # Process all frames until the end of the video
        ret, frame = video_capture.read()
        if not ret:
            logging.info("End of video reached.")
            break

        # Process the frame
        try:
            processed_frame = process_frame(frame, vehicle_tracker, speed_estimator)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            processed_frame = frame  # Use the original frame if processing fails

        # Write the processed frame to the output video
        video_writer.write(processed_frame)
        frame_count += 1

    # Release resources
    video_capture.release()
    video_writer.release()
    logging.info("Output video saved successfully.")

if __name__ == "__main__":
    main()
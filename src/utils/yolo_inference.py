import sys
import os
import logging
import subprocess

# Check if ultralytics is installed, if not, try to install it automatically
try:
    from ultralytics import YOLO
    logging.info("Successfully imported YOLO")
except ImportError:
    print("Ultralytics package not found. Attempting to install it now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        print("Successfully installed and imported ultralytics.")
    except Exception as e:
        print(f"Error installing ultralytics: {e}")
        print("Please manually install the required package by running: pip install ultralytics")
        sys.exit(1)

from src.config.settings import MODEL_PATH, VEHICLE_CLASSES, DETECT_ONLY_CLASSES, TRACKING_CONFIDENCE_THRESHOLD

import cv2

def run_yolo_inference(video_path):
    """
    Load the YOLOv8 model
    
    Args:
        video_path: Path to the video file (not used for model loading)
        
    Returns:
        YOLO model object that can be called for inference
    """
    try:
        # Load the YOLOv8 model using path from settings
        model = YOLO(MODEL_PATH)
        logging.info(f"Successfully loaded YOLOv8 model from {MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        raise

def process_detections(detections):
    """
    Process YOLOv8 detections into a standardized format
    
    Args:
        detections: Raw detection results from YOLOv8
        
    Returns:
        List of formatted detection dictionaries
    """
    # Define the classes we're interested in from settings
    target_classes = DETECT_ONLY_CLASSES if DETECT_ONLY_CLASSES else list(VEHICLE_CLASSES.keys())
    confidence_threshold = TRACKING_CONFIDENCE_THRESHOLD
    
    processed_results = []
    
    try:
        for det in detections:
            boxes = det.boxes
            
            # Apply Non-Maximum Suppression with higher threshold to avoid missing detections
            # This is handled automatically by YOLOv8, but we can control the processing
            
            for box in boxes:
                # Get box coordinates in (top, left, bottom, right) format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence
                confidence = box.conf[0].item()
                
                # Get class ID
                class_id = int(box.cls[0].item())
                
                # Special case for motorcycle detection - use lower threshold
                motorcycle_threshold = confidence_threshold * 0.8 if class_id == 3 else confidence_threshold
                
                # Only include vehicles we're interested in
                if class_id in target_classes and confidence >= motorcycle_threshold:
                    processed_results.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
    except Exception as e:
        logging.error(f"Error processing YOLOv8 results: {e}")
    
    return processed_results
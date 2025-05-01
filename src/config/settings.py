# Configuration settings for the object tracking project

# Path settings
MODEL_PATH = "models/yolov8/yolov8.pt"
INPUT_VIDEO_DIR = "data/input_videos/"
OUTPUT_VIDEO_DIR = "data/output_results/"
VIDEO_INPUT_PATH = "data/video_drive.mp4"
VIDEO_OUTPUT_PATH = OUTPUT_VIDEO_DIR + "output_video.mp4"

# Detection settings
TRACKING_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to detect more vehicles including motorcycles
SPEED_ESTIMATION_FACTOR = 0.3  # Increased for more realistic speed values

# Vehicle classes we want to detect (COCO dataset indices)
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
VEHICLE_CLASSES = {
    2: 'car',      # car
    7: 'truck'     # truck
}

# Only detect these classes - set to empty list to detect all
# Sekarang hanya mendeteksi mobil dan truk
DETECT_ONLY_CLASSES = [2, 7]  # Cars and trucks only

# UI settings
DETECTION_COLORS = {
    'car': (255, 0, 255),       # Magenta for cars
    'truck': (255, 165, 0),     # Orange for trucks
}
COUNT_TEXT_COLOR = (255, 0, 0)   # Blue text for counts
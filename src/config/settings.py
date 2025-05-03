# Configuration settings for the object tracking project

# Path settings
MODEL_PATH = "models/yolov8/yolov8l.pt"  # Use the medium YOLOv8 model for better accuracy
INPUT_VIDEO_DIR = "data/input_videos/"
OUTPUT_VIDEO_DIR = "data/output_results/"
VIDEO_INPUT_PATH = "data/video_drive.mp4"
VIDEO_OUTPUT_PATH = OUTPUT_VIDEO_DIR + "output_simpang.mp4"

# Detection settings
TRACKING_CONFIDENCE_THRESHOLD = 0.5  # Increase threshold for better accuracy

# Placeholder for drone altitude (in meters)
DRONE_ALTITUDE = 50  # Example: 50 meters (adjust based on actual altitude)

# Ensure this factor is calculated dynamically based on drone altitude and camera calibration
SPEED_ESTIMATION_FACTOR = 0.05  # Example: 1 pixel = 0.05 meters (adjust dynamically)

# Detect all objects (empty list means no filtering by class)
DETECT_ONLY_CLASSES = []  # Detect all objects

# UI settings
COUNT_TEXT_COLOR = (255, 0, 0)  # Blue text for counts
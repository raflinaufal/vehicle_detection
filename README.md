# Object Tracking with YOLOv8

This project implements vehicle tracking, counting, and speed estimation using
the YOLOv8 model. It processes video input to detect and analyze vehicle
movements in real-time.

## Project Structure

```
object-tracking-yolov8
├── src
│   ├── main.py                # Entry point of the application
│   ├── tracking
│   │   ├── tracker.py         # Vehicle tracking logic
│   │   └── speed_estimation.py # Speed estimation logic
│   ├── counting
│   │   └── vehicle_counter.py  # Vehicle counting logic
│   ├── utils
│   │   ├── video_processing.py  # Video input/output utilities
│   │   └── yolo_inference.py    # YOLOv8 inference functions
│   └── config
│       └── settings.py         # Configuration settings
├── data
│   ├── input_videos           # Directory for input video files
│   └── output_results          # Directory for output results
├── models
│   └── yolov8                 # Directory for YOLOv8 model files
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files to ignore in version control
```

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd object-tracking-yolov8
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Input Video

1. Place your input video files in the `data/input_videos` directory.
2. Update the `VIDEO_INPUT_PATH` in `src/config/settings.py` to point to your
   input video file. For example:
   ```python
   VIDEO_INPUT_PATH = "data/input_videos/your_video.mp4"
   ```

### Output Video

1. The processed video will be saved in the `data/output_results` directory.
2. Update the `VIDEO_OUTPUT_PATH` in `src/config/settings.py` to specify the
   output file name. For example:
   ```python
   VIDEO_OUTPUT_PATH = OUTPUT_VIDEO_DIR + "processed_video.mp4"
   ```

### Running the Application

1. Run the main script:

   ```
   python src/main.py
   ```

2. The application will:
   - Load the YOLOv8 model.
   - Process the input video frame by frame.
   - Detect and track vehicles.
   - Estimate the speed of each vehicle.
   - Save the processed video with bounding boxes and speed annotations to the
     output directory.

3. **Real-Time Display**:
   - The application will display the video in real-time with bounding boxes and speed annotations.
   - Press `q` to stop the video playback.

### Example Command

To run the application, execute the following command in your terminal:
```
python src/main.py
```

Ensure that the input video path and output video path are correctly configured in `src/config/settings.py`.

## Components

- **VehicleTracker**: Handles the tracking of vehicles in video frames.
- **SpeedEstimator**: Calculates the speed of tracked vehicles.
- **VehicleCounter**: Counts vehicles entering and leaving a specified area.
- **Video Processing Utilities**: Functions for reading and writing video files.
- **YOLOv8 Inference**: Loads the YOLOv8 model and performs vehicle detection.

## Example Configuration

Here is an example configuration in `src/config/settings.py`:

```python
# Path settings
VIDEO_INPUT_PATH = "data/input_videos/sample_video.mp4"
VIDEO_OUTPUT_PATH = "data/output_results/processed_video.mp4"

# Detection settings
TRACKING_CONFIDENCE_THRESHOLD = 0.5

# Drone altitude (in meters)
DRONE_ALTITUDE = 50

# Speed estimation factor (calculated dynamically)
SPEED_ESTIMATION_FACTOR = 0.05
```

## Troubleshooting

1. **YOLOv8 Model Not Found**:

   - Ensure the YOLOv8 model file exists in the `models/yolov8` directory.
   - If not, the application will attempt to download it automatically.

2. **Input Video Not Found**:

   - Ensure the input video file exists in the `data/input_videos` directory.
   - Update the `VIDEO_INPUT_PATH` in `src/config/settings.py` to point to the
     correct file.

3. **Dependencies Not Installed**:
   - Run the following command to install all required dependencies:
     ```
     pip install -r requirements.txt
     ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any
enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for
details.

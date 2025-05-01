# Object Tracking with YOLOv8

This project implements vehicle tracking, counting, and speed estimation using the YOLOv8 model. It processes video input to detect and analyze vehicle movements in real-time.

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

1. Place your input video files in the `data/input_videos` directory.
2. Configure the settings in `src/config/settings.py` as needed.
3. Run the application:
   ```
   python src/main.py
   ```

## Components

- **VehicleTracker**: Handles the tracking of vehicles in video frames.
- **SpeedEstimator**: Calculates the speed of tracked vehicles.
- **VehicleCounter**: Counts vehicles entering and leaving a specified area.
- **Video Processing Utilities**: Functions for reading and writing video files.
- **YOLOv8 Inference**: Loads the YOLOv8 model and performs vehicle detection.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
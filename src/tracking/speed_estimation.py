import time
import numpy as np
import logging

class SpeedEstimator:
    def __init__(self, fps, real_world_distance_per_pixel):
        self.fps = fps
        self.real_world_distance_per_pixel = real_world_distance_per_pixel
        self.previous_positions = {}
        self.position_history = {}
        self.speed_history = {}
        self.last_update_time = {}
        self.current_time = time.time()
        self.history_length = 5  # Track positions over multiple frames for smoother speed calculation
        self.speed_smoothing_factor = 0.7  # Higher value = more smoothing (0-1)
    
    def estimate_speed(self, vehicle_id, current_position):
        """
        Estimate the speed of a vehicle based on its movement between frames.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            current_position: Current position (x, y) of the vehicle center
            
        Returns:
            Estimated speed in km/h
        """
        # Update current time
        self.current_time = time.time()
        
        # Get center point of bounding box if it's not already a center point
        if isinstance(current_position, tuple) and len(current_position) == 2:
            center_x, center_y = current_position
        else:
            logging.warning(f"Expected center point, got: {current_position}")
            return 0.0
        
        # Initialize speed history for new vehicles
        if vehicle_id not in self.speed_history:
            self.speed_history[vehicle_id] = 0.0
        
        # Initialize position history for new vehicles
        if vehicle_id not in self.position_history:
            self.position_history[vehicle_id] = []
            self.previous_positions[vehicle_id] = (center_x, center_y)
            self.last_update_time[vehicle_id] = self.current_time
            return self.speed_history[vehicle_id]  # Return 0 for first detection
        
        # Calculate time elapsed since last update (real time, not just frames)
        time_elapsed = self.current_time - self.last_update_time[vehicle_id]
        if time_elapsed < 0.001:  # Avoid division by zero
            time_elapsed = 0.001
            
        # Store position history
        self.position_history[vehicle_id].append((center_x, center_y))
        
        # Keep only the last N positions
        if len(self.position_history[vehicle_id]) > self.history_length:
            self.position_history[vehicle_id].pop(0)
        
        # Use previous position from history if available
        if len(self.position_history[vehicle_id]) > 1:
            prev_x, prev_y = self.position_history[vehicle_id][0]
            frames_elapsed = min(len(self.position_history[vehicle_id]), self.history_length)
        else:
            prev_x, prev_y = self.previous_positions[vehicle_id]
            frames_elapsed = 1
        
        # Calculate distance moved in pixels
        pixel_distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
        
        # Convert to real-world distance (in meters)
        distance = pixel_distance * self.real_world_distance_per_pixel
        
        # Calculate speed (distance / time)
        # For frame-based: speed = distance * (self.fps / frames_elapsed) * 3.6
        # For real-time: speed = distance / time_elapsed * 3.6
        if frames_elapsed > 1:
            # Use frame-based calculation for smoother results
            speed = distance * (self.fps / frames_elapsed) * 3.6  # Convert to km/h
        else:
            # Use real-time calculation
            speed = distance / time_elapsed * 3.6  # Convert to km/h
        
        # Apply low-pass filter for smoother speed estimates
        if self.speed_history[vehicle_id] > 0:
            smoothed_speed = (self.speed_smoothing_factor * self.speed_history[vehicle_id] + 
                              (1 - self.speed_smoothing_factor) * speed)
        else:
            smoothed_speed = speed
        
        # Store for next calculation
        self.previous_positions[vehicle_id] = (center_x, center_y)
        self.last_update_time[vehicle_id] = self.current_time
        
        # Apply some constraints to avoid unrealistic values
        if smoothed_speed > 120:  # Cap maximum speed
            smoothed_speed = 120
        
        # Update speed history
        self.speed_history[vehicle_id] = max(0, smoothed_speed)
        
        return round(self.speed_history[vehicle_id], 1)
    
    def reset(self):
        self.previous_positions.clear()
        self.position_history.clear()
        self.speed_history.clear()
        self.last_update_time.clear()
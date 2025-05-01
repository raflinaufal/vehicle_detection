import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.settings import VEHICLE_CLASSES

class VehicleCounter:
    def __init__(self, divider_y):
        self.divider_y = divider_y
        self.last_positions = {}  # vehicle_id: last_y
        self.enter_count = 0
        self.leave_count = 0
        self.type_counts = {}
        for _, vehicle_type in VEHICLE_CLASSES.items():
            self.type_counts[vehicle_type] = {'enter': 0, 'leave': 0}

    def update_counts(self, vehicle_id, center_y, vehicle_type):
        if vehicle_type not in self.type_counts:
            self.type_counts[vehicle_type] = {'enter': 0, 'leave': 0}
        last_y = self.last_positions.get(vehicle_id, None)
        if last_y is not None:
            # Crossing from below to above: entering
            if last_y > self.divider_y and center_y <= self.divider_y:
                self.enter_count += 1
                self.type_counts[vehicle_type]['enter'] += 1
            # Crossing from above to below: leaving
            elif last_y <= self.divider_y and center_y > self.divider_y:
                self.leave_count += 1
                self.type_counts[vehicle_type]['leave'] += 1
        self.last_positions[vehicle_id] = center_y

    def get_counts(self):
        return {
            'total': {'enter': self.enter_count, 'leave': self.leave_count},
            'by_type': self.type_counts
        }

    def reset_counts(self):
        self.enter_count = 0
        self.leave_count = 0
        self.last_positions.clear()
        for vehicle_type in self.type_counts:
            self.type_counts[vehicle_type] = {'enter': 0, 'leave': 0}
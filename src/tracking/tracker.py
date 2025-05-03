import logging
import numpy as np
from src.config.settings import DETECT_ONLY_CLASSES, TRACKING_CONFIDENCE_THRESHOLD
from src.utils.yolo_inference import process_detections

def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class VehicleTracker:
    def __init__(self, model, iou_threshold=0.3, max_lost=5):
        self.model = model
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # id: {'bbox':..., 'class_id':..., 'lost':..., 'last_frame':...}
        self.frame_count = 0

    def update(self, frame):
        self.frame_count += 1
        results = self.model(frame)
        detections = process_detections(results)
        updated_tracks = {}
        assigned = set()
        # Match detections to existing tracks
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det = None
            best_idx = -1
            for idx, det in enumerate(detections):
                if idx in assigned: continue
                iou_score = iou(track['bbox'], det['bbox'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = det
                    best_idx = idx
            if best_iou > self.iou_threshold:
                updated_tracks[track_id] = {
                    'bbox': best_det['bbox'],
                    'class_id': best_det['class_id'],
                    'lost': 0,
                    'last_frame': self.frame_count
                }
                assigned.add(best_idx)
            else:
                # Not matched, increase lost
                if track['lost'] + 1 < self.max_lost:
                    updated_tracks[track_id] = {
                        'bbox': track['bbox'],
                        'class_id': track['class_id'],
                        'lost': track['lost'] + 1,
                        'last_frame': track['last_frame']
                    }
        # Add new detections as new tracks
        for idx, det in enumerate(detections):
            if idx not in assigned:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],  # Class ID is still included for general use
                    'lost': 0,
                    'last_frame': self.frame_count
                }
                self.next_id += 1
        # Remove old tracks
        self.tracks = {tid: t for tid, t in updated_tracks.items() if t['lost'] < self.max_lost}
        # Prepare output
        tracked_vehicles = []
        for tid, t in self.tracks.items():
            tracked_vehicles.append({
                'id': tid,
                'bbox': t['bbox'],
                'class_id': t['class_id']
            })
        return tracked_vehicles

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
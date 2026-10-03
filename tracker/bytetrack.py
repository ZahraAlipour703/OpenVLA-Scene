
from typing import List
import numpy as np
from openvla_scene.detector.detection import Detection

try:
    from supervision import ByteTrack, Detections
except ImportError:
    raise ImportError("Install supervision: pip install supervision")


class ByteTracker:
    """Simple ByteTrack wrapper for OpenVLA-Scene."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.tracker = ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
        )

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return detections

        # Convert to supervision format
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.score for d in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)  # not used for now

        sv_dets = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self.tracker.update_with_detections(sv_dets)

        # Map track_ids back
        for i, det in enumerate(detections):
            if i < len(tracked) and tracked.tracker_id is not None:
                det.track_id = int(tracked.tracker_id[i])
            else:
                det.track_id = None

        return detections
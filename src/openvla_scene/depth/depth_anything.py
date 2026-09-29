
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

from openvla_scene.detector.detection import Detection


class DepthAnything:
    """Monocular depth estimation using Depth Anything."""

    def __init__(
        self,
        model_name: str = "LiheYoung/depth-anything-small-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def estimate(self, image: Image.Image) -> np.ndarray:
        """Returns depth map as numpy array (H, W)."""
        result = self.pipe(image)
        depth = np.array(result["depth"])
        return depth

    def assign_to_detections(
        self,
        depth_map: np.ndarray,
        detections: List[Detection],
    ) -> List[Detection]:
        """Assign median depth value inside each bounding box."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(depth_map.shape[1], x2)
            y2 = min(depth_map.shape[0], y2)

            if x2 > x1 and y2 > y1:
                region = depth_map[y1:y2, x1:x2]
                det.depth = float(np.median(region))
            else:
                det.depth = None
        return detections
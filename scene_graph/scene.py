from dataclasses import dataclass, field
from typing import List, Optional, Any
from PIL import Image
from openvla_scene.detector.detection import Detection


@dataclass
class Scene:
    image: Image.Image
    detections: List[Detection] = field(default_factory=list)
    masks: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    depth: Optional[Any] = None
    point_cloud: Optional[Any] = None
    graph: Optional[dict] = None
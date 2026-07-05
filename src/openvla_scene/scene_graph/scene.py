from dataclasses import dataclass, field
from typing import List
from PIL import Image

from openvla_scene.detector.detection import Detection


@dataclass
class Scene:

    image: Image.Image

    detections: List[Detection] = field(default_factory=list)

    masks: list = field(default_factory=list)

    tracks: list = field(default_factory=list)

    depth = None

    point_cloud = None

    graph = None
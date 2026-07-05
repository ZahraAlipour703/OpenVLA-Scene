from dataclasses import dataclass, field
from typing import List
from PIL import Image


@dataclass
class Detection:
    label: str
    score: float
    box: tuple


@dataclass
class Scene:
    image: Image.Image
    detections: List[Detection] = field(default_factory=list)

    def add_detection(self, detection: Detection):
        self.detections.append(detection)

    def get_objects(self):
        return self.detections
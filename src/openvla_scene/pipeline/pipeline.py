# src/openvla_scene/pipeline/pipeline.py
from PIL import Image

from openvla_scene.scene_graph.scene import Scene
from openvla_scene.detector.grounding_dino import GroundingDINODetector
from openvla_scene.segmenter.sam import SAMSegmenter


class PerceptionPipeline:

    def __init__(self):
        self.detector = GroundingDINODetector()
        self.segmenter = SAMSegmenter()

    def run(
        self,
        image: Image.Image,
        labels: list[str],
    ) -> Scene:

        scene = Scene(image=image)

        # 1. Detect
        scene.detections = self.detector.detect(
            image=image,
            labels=labels,
        )

        # 2. Segment
        scene.detections = self.segmenter.segment(
            image=image,
            detections=scene.detections,
        )

        # Store masks list for convenience
        scene.masks = [d.mask for d in scene.detections if d.mask is not None]

        return scene
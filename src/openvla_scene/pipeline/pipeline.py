from PIL import Image

from openvla_scene.scene_graph.scene import Scene
from openvla_scene.detector.grounding_dino import GroundingDINODetector


class PerceptionPipeline:

    def __init__(self):

        self.detector = GroundingDINODetector()

    def run(
        self,
        image: Image.Image,
        labels: list[str],
    ) -> Scene:

        scene = Scene(image=image)

        scene.detections = self.detector.detect(
            image=image,
            labels=labels,
        )

        return scene
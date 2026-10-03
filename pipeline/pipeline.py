
from PIL import Image
from openvla_scene.scene_graph.scene import Scene
from openvla_scene.detector.grounding_dino import GroundingDINODetector
from openvla_scene.segmenter.sam import SAMSegmenter
from openvla_scene.tracker.bytetrack import ByteTracker
from openvla_scene.depth.depth_anything import DepthAnything
from openvla_scene.scene_graph.graph import SceneGraph


class PerceptionPipeline:

    def __init__(self):
        self.detector = GroundingDINODetector()
        self.segmenter = SAMSegmenter()
        self.tracker = ByteTracker()
        self.depth_estimator = DepthAnything()
        self.scene_graph = SceneGraph()

    def run(self, image: Image.Image, labels: list[str]) -> Scene:
        scene = Scene(image=image)

        # 1. Detect
        scene.detections = self.detector.detect(image=image, labels=labels)

        # 2. Segment
        scene.detections = self.segmenter.segment(image=image, detections=scene.detections)

        # 3. Track
        scene.detections = self.tracker.update(scene.detections)

        # 4. Depth
        depth_map = self.depth_estimator.estimate(image)
        scene.depth = depth_map
        scene.detections = self.depth_estimator.assign_to_detections(depth_map, scene.detections)

        # 5. Scene Graph
        scene.graph = self.scene_graph.build(scene.detections)

        scene.masks = [d.mask for d in scene.detections if d.mask is not None]
        scene.tracks = [d.track_id for d in scene.detections if d.track_id is not None]

        return scene
from transformers import pipeline
from torchvision.ops import nms
import torch

from openvla_scene.detector.detection import Detection


class GroundingDINODetector:

    def __init__(
        self,
        model_name="IDEA-Research/grounding-dino-base",
        confidence_threshold=0.30,
        iou_threshold=0.50,
    ):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.pipe = pipeline(
            task="zero-shot-object-detection",
            model=model_name,
        )

    def detect(
        self,
        image,
        labels,
    ):

        outputs = self.pipe(
            image,
            candidate_labels=labels,
        )

        detections = []

        for obj in outputs:

            score = obj["score"]

            if score < self.confidence_threshold:
                continue

            box = obj["box"]

            detections.append(
                Detection(
                    label=obj["label"],
                    score=score,
                    bbox=(
                        box["xmin"],
                        box["ymin"],
                        box["xmax"],
                        box["ymax"],
                    ),
                )
            )

        return self._apply_nms(detections)

    def _apply_nms(self, detections):

        if len(detections) == 0:
            return detections

        boxes = torch.tensor(
            [d.bbox for d in detections],
            dtype=torch.float32,
        )

        scores = torch.tensor(
            [d.score for d in detections],
            dtype=torch.float32,
        )

        keep = nms(
            boxes,
            scores,
            self.iou_threshold,
        )

        return [detections[i] for i in keep]
from typing import List

from transformers import pipeline

from .detector import Detector
from .detection import Detection


class GroundingDINODetector(Detector):
    """
    GroundingDINO detector wrapper.
    Returns a list of Detection objects.
    """

    def __init__(self):
        self.pipe = pipeline(
            task="zero-shot-object-detection",
            model="IDEA-Research/grounding-dino-base",
        )

    def detect(
        self,
        image,
        labels: List[str],
    ) -> List[Detection]:

        outputs = self.pipe(
            image,
            candidate_labels=labels,
        )

        detections = []

        for obj in outputs:

            box = obj["box"]

            detections.append(
                Detection(
                    label=obj["label"],
                    score=float(obj["score"]),
                    bbox=(
                        int(box["xmin"]),
                        int(box["ymin"]),
                        int(box["xmax"]),
                        int(box["ymax"]),
                    ),
                )
            )

        return detections
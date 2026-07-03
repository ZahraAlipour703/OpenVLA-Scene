from transformers import pipeline


class GroundingDINODetector:

    def __init__(self):

        self.pipe = pipeline(

            task="zero-shot-object-detection",

            model="IDEA-Research/grounding-dino-base",

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

        return outputs
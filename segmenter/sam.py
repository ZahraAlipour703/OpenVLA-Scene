
from typing import List
import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor

from openvla_scene.detector.detection import Detection


class SAMSegmenter:
    """Segment objects from bounding boxes using SAM."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def segment(
        self,
        image: Image.Image,
        detections: List[Detection],
    ) -> List[Detection]:
        if not detections:
            return detections

        # Prepare input boxes (SAM expects [[x1,y1,x2,y2], ...])
        input_boxes = [[list(det.bbox) for det in detections]]

        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (N, 1, H, W) or (N, 3, H, W)

        # Take the best mask for each detection
        for i, det in enumerate(detections):
            if i < len(masks):
                # masks[i] shape: (num_masks, H, W) → pick highest quality
                mask = masks[i]
                if mask.ndim == 3:
                    # choose the mask with highest score if available
                    mask = mask[0]
                det.mask = mask.numpy().astype(bool)

        return detections
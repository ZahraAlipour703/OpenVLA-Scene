
from typing import List, Dict, Any
from openvla_scene.detector.detection import Detection


class SceneGraph:
    """Builds a simple spatial scene graph from detections."""

    def __init__(self):
        self.nodes = []
        self.edges = []

    def build(self, detections: List[Detection]) -> Dict[str, Any]:
        self.nodes = []
        self.edges = []

        for i, det in enumerate(detections):
            node = {
                "id": i,
                "label": det.label,
                "score": det.score,
                "bbox": det.bbox,
                "track_id": det.track_id,
                "depth": det.depth,
            }
            self.nodes.append(node)

        # Spatial relations
        for i, a in enumerate(detections):
            for j, b in enumerate(detections):
                if i >= j:
                    continue
                relation = self._get_relation(a, b)
                if relation:
                    self.edges.append({
                        "source": i,
                        "target": j,
                        "relation": relation,
                    })

        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def _get_relation(self, a: Detection, b: Detection) -> str | None:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2

        # Left / Right
        if ax2 < bx1:
            return "left_of"
        if bx2 < ax1:
            return "right_of"

        # Above / Below
        if ay2 < by1:
            return "above"
        if by2 < ay1:
            return "below"

        # Depth-based (if available)
        if a.depth is not None and b.depth is not None:
            if a.depth < b.depth * 0.85:
                return "in_front_of"
            if b.depth < a.depth * 0.85:
                return "behind"

        return "near"
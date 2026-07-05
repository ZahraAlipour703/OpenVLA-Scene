import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
from PIL import Image, ImageDraw

from openvla_scene.pipeline.pipeline import PerceptionPipeline

# --------------------------------------------------
# Load image
# --------------------------------------------------

image_path = r"D:\zra\PROJECTS\Github\OpenVLA-Scene\demo\CARDS_COURTYARD_B_T_frame_0113.jpg"

image = Image.open(image_path).convert("RGB")

# --------------------------------------------------
# Object categories
# --------------------------------------------------

labels = [
    "person",
    "cup",
    "bottle",
    "keyboard",
    "monitor",
]

# --------------------------------------------------
# Run perception pipeline
# --------------------------------------------------

pipeline = PerceptionPipeline()

scene = pipeline.run(
    image=image,
    labels=labels,
)

# --------------------------------------------------
# Draw detections
# --------------------------------------------------

draw = ImageDraw.Draw(scene.image)

CONFIDENCE = 0.30

for det in scene.detections:

    if det.score < CONFIDENCE:
        continue

    xmin, ymin, xmax, ymax = det.bbox

    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        outline="red",
        width=3,
    )

    draw.text(
        (xmin, max(0, ymin - 20)),
        f"{det.label}: {det.score:.2f}",
        fill="red",
    )

# --------------------------------------------------
# Save result
# --------------------------------------------------

output_path = "output.jpg"

scene.image.save(output_path)

print(f"\nSaved visualization to {output_path}\n")

# --------------------------------------------------
# Print detections
# --------------------------------------------------

print("=" * 70)
print("Detections")
print("=" * 70)

for det in scene.detections:

    print(
        f"{det.label:<15}"
        f"{det.score:.2f}    "
        f"{det.bbox}"
    )

print("=" * 70)
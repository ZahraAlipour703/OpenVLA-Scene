from transformers import pipeline
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# -----------------------------
# Load image
# -----------------------------
image_path = r"D:\zra\PROJECTS\Github\OpenVLA-Scene\demo\CARDS_COURTYARD_B_T_frame_0113.jpg"
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Load OWLv2 detector
# -----------------------------
detector = pipeline(
    task="zero-shot-object-detection",
    model="google/owlv2-base-patch16-ensemble"
)

# -----------------------------
# Language queries
# -----------------------------
candidate_labels = [
    "person",
    "cup",
    "bottle",
    "keyboard",
    "monitor"
]

# -----------------------------
# Detect
# -----------------------------
outputs = detector(
    image,
    candidate_labels=candidate_labels
)

# -----------------------------
# Draw detections
# -----------------------------
draw = ImageDraw.Draw(image)

CONFIDENCE = 0.30

for obj in outputs:

    score = obj["score"]

    if score < CONFIDENCE:
        continue

    box = obj["box"]

    xmin = box["xmin"]
    ymin = box["ymin"]
    xmax = box["xmax"]
    ymax = box["ymax"]

    label = f'{obj["label"]}: {score:.2f}'

    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        outline="red",
        width=3
    )

    draw.text(
        (xmin, max(0, ymin-20)),
        label,
        fill="red"
    )

# -----------------------------
# Save
# -----------------------------
output_path = "output.jpg"
image.save(output_path)

print(f"Saved visualization to {output_path}")

# -----------------------------
# Show
# -----------------------------
print("Done! Check putput.jpg")
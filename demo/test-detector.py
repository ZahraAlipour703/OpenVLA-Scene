import sys
from pathlib import Path
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from openvla_scene.detector.grounding_dino import GroundingDINODetector


image = Image.open("D:\zra\PROJECTS\Github\OpenVLA-Scene\demo\CARDS_COURTYARD_B_T_frame_0113.jpg")

model = GroundingDINODetector()

results = model.detect(

    image,

    [

        "person",

        "cup",

        "keyboard",

        "monitor",

        "bottle",

    ],

)

print(results)
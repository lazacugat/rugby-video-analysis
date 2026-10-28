"""Paths, the matches in each split and the post-processing parameters."""

from pathlib import Path

import pandas as pd

# Folders
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
ANNOTATIONS = DATA / "annotations"
FRAMES_1FPS = DATA / "frames_1fps"
FRAMES_SMALL = DATA / "frames_1fps_384"            # 384 × 384
MODELS = ROOT / "models"
IMAGENET_WEIGHTS = "yolo26m-cls.pt"                # Ultralytics YOLO26m-cls, ImageNet-pretrained (downloaded automatically)
DEFAULT_MODEL = MODELS / "model_69" / "yolo_cls_69.pt"     # line + scrum + maul (trained on TRAIN only)
MANIFEST = DATA / "manifest.csv"

# Annotated matches in each split (split by venue / camera)
manifest = pd.read_csv(MANIFEST, dtype=str)
manifest["match"] = manifest.file.str.rsplit(".", n=1).str[0]      # without the extension
annotated = manifest[manifest.annotated == "yes"]
TRAIN = annotated.match[annotated.split == "train"].tolist()
VAL = annotated.match[annotated.split == "val"].tolist()
TEST = annotated.match[annotated.split == "test"].tolist()

# Post-processing: moving average of WINDOW s, threshold, segments separated by <= MAX_GAP s are merged,
# and segments shorter than MIN_DURATION are dropped
WINDOW = {"line": 15, "scrum": 9, "maul": 5}
MAX_GAP = 2
MIN_DURATION = {"line": 6, "scrum": 4, "maul": 6}
THRESHOLD = {"line": 0.05, "scrum": 0.30, "maul": 0.10}
# A maul only counts if it starts between 2 s before and 16 s after the p_line peak of a detected line
# (in train, 99 % of mauls start before throw + 15.5 s)
MAUL_AFTER_LINE = (-2, 16)
# Line recovered from a maul: if there is a very clear maul (p_maul >= 0.9 for at least 3 s), with no detected
# line before it and no scrum (p_scrum >= 0.5) in the previous 30 s, and p_line reaches 0.01 in some second
# of the previous 20 s, a line is added 6 s before the maul starts
LINE_FROM_MAUL = {"p_maul": 0.9, "seconds": 3, "no_scrum": 30, "before": 20, "p_line": 0.01, "offset": 6}

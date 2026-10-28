"""Shrink the frames, choose which seconds are used and build the folders YOLO expects."""
import cv2
import numpy as np
import pandas as pd

import config

SIDE = 384
NOTHING_PER_EVENT = 4
SEED = 0


def shrink(args):
    """Shrinks seconds ts of a match to SIDE × SIDE. Takes (match, ts) together so it can be used with Pool."""
    match, ts = args
    folder = config.FRAMES_SMALL / match
    folder.mkdir(parents=True, exist_ok=True)
    for t in ts:
        target = folder / f"{t:05d}.jpg"
        if not target.exists():
            image = cv2.imread(str(config.FRAMES_1FPS / match / f"{t:05d}.jpg"))
            # Stretched, not cropped: lines are often at the edges of the image
            cv2.imwrite(str(target), cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    return match


def sample(seconds):
    """All line, scrum and maul seconds, and NOTHING_PER_EVENT random "nothing" seconds per line and scrum second
    (maul seconds add no "nothing", so the amount of "nothing" is the same as without mauls)."""
    rng = np.random.default_rng(SEED)
    events = seconds[seconds.label.isin(["line", "scrum", "maul"])]
    count = NOTHING_PER_EVENT * seconds.label.isin(["line", "scrum"]).sum()
    nothing = seconds[seconds.label == "nothing"]
    nothing = nothing.iloc[rng.choice(len(nothing), min(len(nothing), count), replace=False)]
    return pd.concat([events, nothing])


def build_folders(seconds, folder, part):
    """<folder>/<part>/<class>/<match>_<t>.jpg, with part = train or val. Symlinks, not copies."""
    for row in seconds.itertuples():
        name = f"{row.t:05d}.jpg"
        target = folder / part / row.label / f"{row.match}_{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(config.FRAMES_SMALL / row.match / name)

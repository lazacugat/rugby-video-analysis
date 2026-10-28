"""
train(folder, data)          trains YOLO26m-cls on <data>/train/ and checks <data>/val/ after each epoch
predict(weights, seconds)    adds p_line and p_scrum (and p_maul, if the model has it) to each second
"""
import numpy as np
import ultralytics.models.yolo.classify.train as yolo_cls_train
import ultralytics.models.yolo.classify.val as yolo_cls_val
from ultralytics import YOLO

import config

SIDE = 384
EPOCHS = 15                  # passes over all the train images
BATCH = 32                   # images per step
VALIDATE_EVERY_EPOCH = True
WORKERS = 2                  # image-loading processes; with 4 the machine ran out of RAM
SEED = 0                     # fixes the training randomness (image order, flips...)


# On WSL, pinned memory (pin_memory) crashes training with "CUDA error: unknown error".
# YOLO doesn't allow turning it off, so its dataloader-building function is replaced.
def _no_pinned_memory(function):
    def wrapped(*args, **kwargs):
        kwargs["pin_memory"] = False
        return function(*args, **kwargs)
    return wrapped


yolo_cls_train.build_dataloader = _no_pinned_memory(yolo_cls_train.build_dataloader)
yolo_cls_val.build_dataloader = _no_pinned_memory(yolo_cls_val.build_dataloader)


def train(folder, data):
    model = YOLO(config.IMAGENET_WEIGHTS)
    model.train(data=str(data), imgsz=SIDE, epochs=EPOCHS, batch=BATCH, workers=WORKERS,
                project=str(folder), name="yolo", exist_ok=True, seed=SEED, deterministic=False,
                erasing=0.0,        # no random erasing: it could erase the line or the scrum itself
                scale=0.0,          # no zoom: the event may be at the edge
                fliplr=0.5, plots=False, verbose=False, val=VALIDATE_EVERY_EPOCH)
    return folder / "yolo" / "weights" / "last.pt"


def predict(weights, seconds):
    model = YOLO(str(weights))
    names = model.names                                     # {0: 'line', 1: 'maul', 2: 'nothing', 3: 'scrum'}
    paths = [str(config.FRAMES_SMALL / m / f"{t:05d}.jpg") for m, t in zip(seconds.match, seconds.t)]
    probs = []
    for i in range(0, len(paths), 256):
        for r in model.predict(paths[i:i + 256], imgsz=SIDE, verbose=False, batch=64):
            probs.append(r.probs.data.cpu().numpy())
        print(f"  {i + 256} of {len(paths)}", flush=True)
    probs = np.array(probs)
    column = {v: k for k, v in names.items()}
    seconds = seconds.assign(p_line=probs[:, column["line"]], p_scrum=probs[:, column["scrum"]])
    if "maul" in column:
        seconds = seconds.assign(p_maul=probs[:, column["maul"]])
    return seconds

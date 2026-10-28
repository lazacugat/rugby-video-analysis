"""
Trains YOLO-cls on TRAIN_MATCHES and checks VAL_MATCHES at the end of each epoch.
For the final model (once everything was chosen with VAL): TRAIN_MATCHES = TRAIN + VAL.
Saves models/model_<n>/yolo_cls_<n>.pt (n = number of train matches).
Usage (from the repo root):  python events/train.py        (~2 h on an RTX 3070)
"""
import shutil
from multiprocessing import Pool

import pandas as pd

import build_dataset
import config
import labels
import model

TRAIN_MATCHES = config.TRAIN
VAL_MATCHES = config.VAL

NAME = f"{len(TRAIN_MATCHES)}"
FOLDER = config.MODELS / f"model_{NAME}"
WEIGHTS = FOLDER / f"yolo_cls_{NAME}.pt"


def main():
    if WEIGHTS.exists():
        print(f"{WEIGHTS} already exists; delete it first to retrain.")
        return
    train = build_dataset.sample(labels.label_all(TRAIN_MATCHES))
    val = build_dataset.sample(labels.label_all(VAL_MATCHES))
    print(f"train: {len(TRAIN_MATCHES)} matches  {train.label.value_counts().to_dict()}")
    print(f"val:   {len(VAL_MATCHES)} matches  {val.label.value_counts().to_dict()}", flush=True)
    # Shrinks the chosen frames to 384 × 384 (the ones already shrunk are skipped)
    chosen = pd.concat([train, val])
    with Pool(4) as pool:
        pool.map(build_dataset.shrink, [(m, s.t.tolist()) for m, s in chosen.groupby("match")])
    build_dataset.build_folders(train, FOLDER / "data", "train")
    build_dataset.build_folders(val, FOLDER / "data", "val")
    weights = model.train(FOLDER, FOLDER / "data")
    shutil.copy(weights, WEIGHTS)
    print(f"Done: {WEIGHTS}")


if __name__ == "__main__":
    main()

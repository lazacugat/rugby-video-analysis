"""
Measures a model on the VAL matches by events (lines, scrums and mauls detected, false alarms)
and tries different thresholds, smoothing windows and minimum durations.

1. Runs the model on every second of each val match (GPU, ~3 min per match) and saves
   the probabilities in <model folder>/val_probs/<match>.csv. Matches already saved are skipped.
2. Tries every combination (CPU only, no retraining) and shows:
   - the result of the post-processing in config (the current one);
   - the best combinations: for each number of false alarms, the one that detects the most events.

Usage (from the repo root):  python events/evaluate.py
                             python events/evaluate.py models/other/weights.pt
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

import build_dataset
import config
import model
import postprocess

WEIGHTS = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DEFAULT_MODEL
PROBS_FOLDER = WEIGHTS.parent / "val_probs"

# What is tried
THRESHOLDS = {"line": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3],
              "scrum": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
              "maul": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}
WINDOWS = {"line": [1, 3, 5, 9, 15, 21], "scrum": [1, 3, 5, 9], "maul": [1, 3, 5, 9]}
MIN_DURATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]      # (maul: after the detected lines, MAUL_AFTER_LINE)


def probabilities():
    """{match: table with t, p_line, p_scrum, p_maul}, running the model only on the missing ones."""
    PROBS_FOLDER.mkdir(exist_ok=True)
    probs = {}
    for match in config.VAL:
        file = PROBS_FOLDER / f"{match}.csv"
        if not file.exists():
            print(f"Model {WEIGHTS.name}: {match}", flush=True)
            n = len(list((config.FRAMES_1FPS / match).glob("*.jpg")))
            build_dataset.shrink((match, range(n)))
            model.predict(WEIGHTS, pd.DataFrame({"match": match, "t": range(n)})).to_csv(file, index=False)
        probs[match] = pd.read_csv(file)
    return probs


def measure(probs, kind, threshold, window, min_duration):
    """Detected events, events that count and false alarms per match, adding up the val matches."""
    detected = total = false_alarms = 0
    for match, table in probs.items():
        hits, f = postprocess.evaluate_match(match, table.t, table[f"p_{kind}"], kind,
                                             threshold, window, min_duration, table.p_line,
                                             table.get("p_scrum"), table.get("p_maul"))
        detected += sum(hits)
        total += len(hits)
        false_alarms += f
    return detected, total, false_alarms / len(probs)


def main():
    probs = probabilities()
    print(f"\nModel {WEIGHTS.name} on {len(probs)} val matches")
    kinds = [k for k in ("line", "scrum", "maul") if f"p_{k}" in next(iter(probs.values()))]
    for kind in kinds:
        current = (config.THRESHOLD[kind], config.WINDOW[kind], config.MIN_DURATION[kind])
        d, total, f = measure(probs, kind, *current)
        print(f"\n{kind.upper()}  ({total} events)")
        print(f"  current (threshold {current[0]}, smoothing {current[1]} s, min. duration {current[2]} s): "
              f"{d}/{total} detected, {f:.1f} false alarms per match")

        rows = [(threshold, window, min_duration, *measure(probs, kind, threshold, window, min_duration))
                for threshold, window, min_duration
                in itertools.product(THRESHOLDS[kind], WINDOWS[kind], MIN_DURATIONS)]
        # The best ones: sorted by false alarms, keeping only those that detect more than all the previous ones
        print("  best combinations (from fewest to most false alarms):")
        print("    threshold  smoothing  min.dur.   detected   false/match")
        best = -1
        for threshold, window, min_duration, d, total, f in sorted(rows, key=lambda x: (x[5], -x[3])):
            if d > best:
                best = d
                print(f"    {threshold:<9}  {window:>5} s   {min_duration:>4} s    {d:>4}/{total}      {f:6.1f}")


if __name__ == "__main__":
    main()

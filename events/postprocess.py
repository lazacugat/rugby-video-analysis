"""
From the probability of each second to detections, and how many of them are hits.

A detection hits an event if it overlaps its window:
    line   [throw − 5, throw + 3]
    scrum  [formation − 5, set + 4]  (with the note "no set", set + 2)
    maul   [start − 5, end]
Events with the note "ignore" don't count, and a detection that overlaps them is not a false alarm.
Seconds inside an "ignore" row (pre-match, half-time...) are not used.
"""
import numpy as np
import pandas as pd

import config
import labels

HIT_LINE = (-5, 3)
HIT_SCRUM_BEFORE = -5
HIT_SCRUM_AFTER = 4
HIT_NO_SET = 2
HIT_MAUL_BEFORE = -5


def smooth(p, window):
    """Centred moving average of `window` seconds."""
    return pd.Series(p).rolling(window, center=True, min_periods=1).mean().to_numpy()


def detections(t, p, kind, threshold=None, window=None, min_duration=None):
    """[(start, end), ...]. Whatever is not given (threshold, window, min_duration) comes from config."""
    if threshold is None:
        threshold = config.THRESHOLD[kind]
    if window is None:
        window = config.WINDOW[kind]
    if min_duration is None:
        min_duration = config.MIN_DURATION[kind]
    above = t[smooth(p, window) >= threshold]
    if len(above) == 0:
        return []
    cuts = np.flatnonzero(np.diff(above) > config.MAX_GAP) + 1
    segments = [(seg[0], seg[-1]) for seg in np.split(above, cuts)]
    return [(start, end) for start, end in segments if end - start + 1 >= min_duration]


def peak(t, p, segment):
    """The second with the highest p inside the segment (its start, if the segment has no seconds)."""
    inside = (t >= segment[0]) & (t <= segment[1])
    return t[inside][p[inside].argmax()] if inside.any() else segment[0]


def line_detections(t, p_line, p_scrum=None, p_maul=None, threshold=None, window=None, min_duration=None):
    """Like detections, plus the lines recovered from a very clear maul (LINE_FROM_MAUL), which are
    one-second segments. Without p_scrum and p_maul, nothing is recovered."""
    dets = detections(t, p_line, "line", threshold, window, min_duration)
    if p_scrum is None or p_maul is None:
        return dets
    r = config.LINE_FROM_MAUL
    before, after = config.MAUL_AFTER_LINE
    peaks = [peak(t, p_line, d) for d in dets]
    new = []
    for a, b in detections(t, p_maul, "maul"):
        inside = (t >= a) & (t <= b)
        if (p_maul[inside] >= r["p_maul"]).sum() < r["seconds"]:
            continue
        if any(x + before <= a <= x + after for x in peaks):
            continue
        if p_scrum[(t >= a - r["no_scrum"]) & (t <= b)].max(initial=0) >= 0.5:
            continue
        previous = (t >= a - r["before"]) & (t < a)
        if p_line[previous].max(initial=0) >= r["p_line"]:
            new.append((a - r["offset"], a - r["offset"]))
    return sorted(dets + new)


def maul_detections(t, p_maul, p_line, threshold=None, window=None, min_duration=None, p_scrum=None):
    """Like detections, but only the mauls that start shortly after a detected line (MAUL_AFTER_LINE).
    With p_scrum, lines recovered from a maul (line_detections) also count."""
    before, after = config.MAUL_AFTER_LINE
    peaks = [peak(t, p_line, d) for d in line_detections(t, p_line, p_scrum, p_maul)]
    return [d for d in detections(t, p_maul, "maul", threshold, window, min_duration)
            if any(x + before <= d[0] <= x + after for x in peaks)]


def windows(events, kind):
    """[(a, b, ignore), ...]: the hit window of each event of that kind."""
    ev = events[events.kind == kind]
    if kind == "line":
        a = ev.mark + HIT_LINE[0]
        b = ev.mark + HIT_LINE[1]
    elif kind == "maul":
        a = ev.mark + HIT_MAUL_BEFORE
        b = ev.end_mark
    else:
        a = ev.mark + HIT_SCRUM_BEFORE
        b = ev.end_mark + np.where(ev.notes.str.contains("no set"), HIT_NO_SET, HIT_SCRUM_AFTER)
    return list(zip(a, b, ev.ignore))


def evaluate(dets, windows):
    """detected: True / False for each event that counts. false_alarms: detections that overlap no window."""
    def overlaps(det, window):
        return det[0] <= window[1] and det[1] >= window[0]

    detected = [any(overlaps(d, w) for d in dets) for w in windows if not w[2]]
    false_alarms = sum(not any(overlaps(d, w) for w in windows) for d in dets)
    return detected, false_alarms


def evaluate_match(match, t, p, kind, threshold=None, window=None, min_duration=None,
                   p_line=None, p_scrum=None, p_maul=None):
    """For one match: detected (True / False per event) and how many false alarms there were.
    Mauls need p_line (mauls are searched after the detected lines).
    With p_scrum and p_maul, lines recovered from a maul are added (LINE_FROM_MAUL)."""
    t, p = np.asarray(t), np.asarray(p)
    events = labels.read_events(match)
    use = np.ones(len(t), dtype=bool)
    for segment in events[events.kind == "ignore"].itertuples():
        use &= (t < segment.start) | (t > segment.end)
    keep = lambda x: None if x is None else np.asarray(x)[use]
    if kind == "maul":
        dets = maul_detections(t[use], p[use], keep(p_line), threshold, window, min_duration, keep(p_scrum))
    elif kind == "line":
        dets = line_detections(t[use], p[use], keep(p_scrum), keep(p_maul), threshold, window, min_duration)
    else:
        dets = detections(t[use], p[use], kind, threshold, window, min_duration)
    return evaluate(dets, windows(events, kind))

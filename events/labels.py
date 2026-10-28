"""
The label of every second of a match, from its annotation CSV:

    line     from throw - BEFORE_LINE to throw + AFTER_LINE
    scrum    from the formation (a pair of second-rows kneels down) - BEFORE_SCRUM to the set (the packs engage) + AFTER_SCRUM
    maul     from the jumper landing to the ball coming out (or the maul collapsing); only if MAUL = True
    ignore   a margin around each event, the events with the note "ignore"
             and the rows of class "ignore" (pre-match, half-time...: from t_start to t_end)
    nothing  everything else

Annotation columns: t_start (line: throw; scrum: formation; maul: start), t_end (scrum: set; maul: end),
class, team, notes.
"""
import numpy as np
import pandas as pd

import config

BEFORE_LINE = 0
AFTER_LINE = 3
BEFORE_SCRUM = 0
AFTER_SCRUM = 2
MARGIN_BEFORE = 5
MARGIN_AFTER = {"line": 10, "scrum": 5, "maul": 5}
MAUL = True              # True: mauls are a class; False: they are ignored


def read_events(match):
    """Table with kind (the class), mark, end_mark, notes, ignore and the labelled segment (start, end)."""
    a = pd.read_csv(config.ANNOTATIONS / f"{match}.csv", dtype={"notes": str}).fillna({"notes": ""})
    a["class"] = a["class"].str.strip()
    lines = a[a["class"] == "line"]
    scrums = a[a["class"] == "scrum"].copy()
    segments = a[a["class"] == "ignore"]
    # A scrum with the note "ignore" may have no set (for example, the referee reset it):
    # it is ignored from the formation
    no_set = scrums.t_end.isna() & scrums.notes.str.contains("ignore")
    scrums.loc[no_set, "t_end"] = scrums.t_start[no_set]
    # Same for an "ignore" maul with no end
    mauls = a[a["class"] == "maul"].copy()
    no_end = mauls.t_end.isna() & mauls.notes.str.contains("ignore")
    mauls.loc[no_end, "t_end"] = mauls.t_start[no_end]
    if scrums.t_end.isna().any() or segments.t_end.isna().any() or mauls.t_end.isna().any():
        raise ValueError(f"{match}: scrums, mauls or ignore rows without a second mark")
    if (scrums.t_end < scrums.t_start).any():
        raise ValueError(f"{match}: scrums with the set before the formation")
    events = pd.concat([
        pd.DataFrame({"kind": "line", "mark": lines.t_start, "end_mark": np.nan, "notes": lines.notes,
                      "ignore": lines.notes.str.contains("ignore"),
                      "start": lines.t_start - BEFORE_LINE, "end": lines.t_start + AFTER_LINE}),
        pd.DataFrame({"kind": "scrum", "mark": scrums.t_start, "end_mark": scrums.t_end,
                      "notes": scrums.notes, "ignore": scrums.notes.str.contains("ignore"),
                      "start": scrums.t_start - BEFORE_SCRUM, "end": scrums.t_end + AFTER_SCRUM}),
        pd.DataFrame({"kind": "maul", "mark": mauls.t_start, "end_mark": mauls.t_end,
                      "notes": mauls.notes, "ignore": mauls.notes.str.contains("ignore"),
                      "start": mauls.t_start, "end": mauls.t_end}),
        pd.DataFrame({"kind": "ignore", "mark": segments.t_start, "end_mark": np.nan, "notes": segments.notes,
                      "ignore": True, "start": segments.t_start, "end": segments.t_end}),
    ], ignore_index=True)
    return events.sort_values("start")


def label(match):
    """One row per second of the match: match, t, label."""
    seconds = len(list((config.FRAMES_1FPS / match).glob("*.jpg")))
    t = np.arange(seconds)
    labels = np.full(seconds, "nothing", dtype=object)
    events = read_events(match)
    if not MAUL:
        events = events[events.kind != "maul"]
    real_events = events[events.kind != "ignore"]
    # Mauls are painted first: if a maul starts before throw + AFTER_LINE, those seconds stay line
    real_events = pd.concat([real_events[real_events.kind == "maul"],
                             real_events[real_events.kind != "maul"]])

    def inside(ev):
        return (t >= ev.start) & (t <= ev.end)

    def margin(ev):
        before = (t >= ev.start - MARGIN_BEFORE) & (t < ev.start)
        after = (t > ev.end) & (t <= ev.end + MARGIN_AFTER[ev.kind])
        return before | after

    # What is painted later overwrites what came before
    for ev in real_events.itertuples():
        labels[margin(ev)] = "ignore"
    for ev in real_events.itertuples():
        labels[inside(ev)] = ev.kind
    for ev in events[events.ignore].itertuples():
        labels[inside(ev)] = "ignore"

    return pd.DataFrame({"match": match, "t": t, "label": labels})


def label_all(matches):
    return pd.concat([label(m) for m in matches], ignore_index=True)

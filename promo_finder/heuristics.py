"""Model-free scoring: a rules baseline, and the fallback when no model is configured.

Scores come from the script's own annotations: delivery tags such as (Furious) or
(Gloating), impact words in ACTION/CAMERA/SOUND lines (slap, explosion, chains), and
dialogue shape (questions, threats). Crude, but free, instant and explainable, and it
is the bar the model-based picks have to clear.
"""
from __future__ import annotations

import re

from .screenplay import Block, Episode
from .timeline import Segment, timeline

INTENSE_TAGS = {
    "furious": 1.0, "screaming": 1.0, "hysterical": 1.0, "enraged": 1.0, "outraged": 0.9, "shocked": 0.9,
    "horrified": 0.9, "terrified": 0.9, "panicked": 0.8, "venomous": 0.8, "menacing": 0.8, "threatening": 0.9,
    "sinister": 0.8, "vicious": 0.8, "devastated": 0.8, "betrayed": 0.8, "sobbing": 0.8, "stunned": 0.7,
    "gloating": 0.7, "cruel": 0.7, "accusing": 0.7, "desperate": 0.7, "mocking": 0.6, "pleading": 0.6,
    "weeping": 0.6, "defiant": 0.6, "scornful": 0.6, "contemptuous": 0.6, "booming": 0.6, "gasping": 0.6,
    "smug": 0.5, "trembling": 0.5, "incredulous": 0.5, "commanding": 0.5, "seductive": 0.5, "cold": 0.4,
    "breathless": 0.4, "dismissive": 0.4,
}
IMPACT_WORDS = {
    "slap": 1.0, "slaps": 1.0, "explosion": 1.0, "explodes": 1.0, "red-hot": 1.0, "noose": 1.0, "gallows": 1.0,
    "cliffhanger": 1.0, "strike": 0.9, "strikes": 0.9, "struck": 0.9, "punch": 0.9, "knife": 0.9, "dagger": 0.9,
    "ablaze": 0.9, "screams": 0.9, "poison": 0.9, "arsenic": 0.9, "cut to black": 0.9, "burst": 0.8, "bursts": 0.8,
    "shatter": 0.8, "shatters": 0.8, "blood": 0.8, "bleeding": 0.8, "flames": 0.8, "blade": 0.8, "sword": 0.8,
    "collapses": 0.8, "faints": 0.8, "unmask": 0.8, "witch": 0.8, "arrest": 0.8, "crash": 0.7, "fire": 0.7,
    "chains": 0.7, "chained": 0.7, "kiss": 0.7, "kisses": 0.7, "seizes": 0.7, "drags": 0.7, "hurls": 0.7,
    "slams": 0.7, "reveal": 0.7, "reveals": 0.7, "kneel": 0.6, "kneels": 0.6, "grabs": 0.6, "throws": 0.6,
    "gasps": 0.6, "crest": 0.5, "dungeon": 0.5, "iron": 0.5, "tears": 0.4, "ring": 0.4,
}
_IMPACT_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, IMPACT_WORDS), key=len, reverse=True)) + r")\b", re.I)
DIALOGUE_CUES = [
    (re.compile(r"\b(kill|die|dead|hang|burn|poison|witch|arrest|guards|execute|execution)\b", re.I), 0.6),
    (re.compile(r"\b(liar|lying|betray\w*|traitor|murder\w*|thief)\b", re.I), 0.6),
    (re.compile(r"\b(duke|duchess|heir|crown|rightful|imposter)\b", re.I), 0.4),
    (re.compile(r"\b(marry|marrying|wedding|husband|wife)\b", re.I), 0.3),
]

MIN_LEN, MAX_LEN, IDEAL = 20.0, 60.0, (25.0, 45.0)


def block_intensity(b: Block) -> float:
    if b.kind == "dialogue":
        s = max((INTENSE_TAGS.get(t.lower(), 0.0) for t in b.tags), default=0.0)
        for rx, w in DIALOGUE_CUES:
            if rx.search(b.text):
                s = max(s, w) + 0.1
        if b.text.rstrip().endswith(("!", "?")):
            s += 0.1
        return min(s, 1.0)
    hits = [IMPACT_WORDS[m.group(1).lower()] for m in _IMPACT_RE.finditer(b.text)]
    return max(hits, default=0.0)


def hook_score(ep: Episode, start: float) -> float:
    tl = timeline(ep)
    first = [b for b in ep.blocks if start - 0.12 <= b.start <= start + 3.0]
    s = max((block_intensity(b) for b in first), default=0.0)
    line = tl.first_line_start(start, start + 3.0)
    if line is not None and line - start <= 1.5:
        s += 0.2
    elif line is None:
        s -= 0.15
    return max(0.0, min(s, 1.0))


def cliff_score(ep: Episode, end: float) -> float:
    last = [b for b in ep.blocks if end - 4.0 <= b.end <= end + 0.12]
    s = max((block_intensity(b) for b in last), default=0.0)
    lines = [b for b in last if b.kind == "dialogue"]
    if lines and lines[-1].text.rstrip().endswith(("?", "...", "…", "—", "-")):
        s += 0.3
    if any(re.search(r"cliffhanger|cut to black|ominous", b.text, re.I) for b in last if b.kind in ("sfx", "camera")):
        s += 0.3
    if end >= ep.duration - 0.5:
        s += 0.2
    return min(s, 1.0)


def standalone_score(ep: Episode, start: float, end: float) -> float:
    speakers = {b.speaker for b in timeline(ep).blocks_between(start, end) if b.kind == "dialogue"}
    return max(0.0, 1.0 - 0.12 * max(0, len(speakers) - 3))


def score_window(ep: Episode, start: float, end: float) -> dict:
    h, c, a = hook_score(ep, start), cliff_score(ep, end), standalone_score(ep, start, end)
    d = end - start
    fit = 1.0 if IDEAL[0] <= d <= IDEAL[1] else 0.85
    return {"hook": round(h, 3), "cliffhanger": round(c, 3), "standalone": round(a, 3),
            "total": round((0.4 * h + 0.4 * c + 0.2 * a) * fit, 4)}


def candidate_windows(ep: Episode):
    tl = timeline(ep)
    cuts = [c for c in tl.cuts if tl.is_clean(c)]
    for i, s in enumerate(cuts):
        for e in cuts[i + 1:]:
            if e - s > MAX_LEN:
                break
            if e - s >= MIN_LEN:
                yield s, e


def best_windows(ep: Episode, k: int = 3, max_overlap: float = 0.3) -> list:
    scored = sorted(((score_window(ep, s, e), s, e) for s, e in candidate_windows(ep)),
                    key=lambda x: -x[0]["total"])
    picked = []
    for sc, s, e in scored:
        if all(min(e, pe) - max(s, ps) <= max_overlap * min(e - s, pe - ps) for _, ps, pe in picked):
            picked.append((sc, s, e))
        if len(picked) == k:
            break
    return [(Segment(ep.number, s, e), sc) for sc, s, e in picked]


def baseline_opening(ep: Episode, target: float = 40.0) -> Segment:
    """What a rushed editor does: post the first ~40 s of the episode."""
    tl = timeline(ep)
    return Segment(ep.number, 0.0, tl.snap_end(min(target, ep.duration)))


def baseline_ending(ep: Episode, target: float = 40.0) -> Segment:
    """The other shortcut: the last ~40 s, which ends on the episode's own cliffhanger."""
    tl = timeline(ep)
    return Segment(ep.number, tl.snap_start(max(0.0, ep.duration - target)), ep.duration)

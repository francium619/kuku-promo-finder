"""Cut points, snapping and clip rendering.

The model never writes timestamps. It names blocks, and this module turns them into
cut points that (a) never split a line of dialogue and (b) land on an existing camera
cut when one is close, so the promo cuts where the episode already cuts.
"""
from __future__ import annotations

from dataclasses import dataclass

from .screenplay import Block, Episode, Show, fmt_ts

TOL = 0.12          # seconds of slack when deciding whether a time is "inside" a line
MAX_PREROLL = 1.0   # how far back a start may move to reach a camera cut
MAX_TAIL = 1.5      # how far forward an end may move to reach a camera cut


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_id: str = ""
    end_id: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def label(self) -> str:
        return f"Ep {self.episode} {fmt_ts(self.start)}-{fmt_ts(self.end)}"


class Timeline:
    def __init__(self, ep: Episode):
        self.ep = ep
        self.lines = sorted((b.start, b.end) for b in ep.blocks if b.kind == "dialogue")
        cams = [b for b in ep.blocks if b.kind == "camera"] or ep.blocks
        pts = {0.0, ep.duration}
        for b in cams:
            pts.update((b.start, b.end))
        self.cuts = sorted(p for p in pts if 0.0 <= p <= ep.duration)

    def line_at(self, t: float):
        for s, e in self.lines:
            if s + TOL < t < e - TOL:
                return s, e
        return None

    def is_clean(self, t: float) -> bool:
        return self.line_at(t) is None

    def snap_start(self, t: float) -> float:
        t = min(max(t, 0.0), self.ep.duration)
        ln = self.line_at(t)
        if ln:
            t = ln[0]
        cands = [c for c in self.cuts if t - MAX_PREROLL - 1e-6 <= c <= t + 1e-6 and self.is_clean(c)]
        return round(max(cands), 2) if cands else round(t, 2)

    def snap_end(self, t: float) -> float:
        t = min(max(t, 0.0), self.ep.duration)
        ln = self.line_at(t)
        if ln:
            t = ln[1]
        cands = [c for c in self.cuts
                 if t - 1e-6 <= c <= t + MAX_TAIL + 1e-6 and self.is_clean(c)
                 and not any(t + TOL < s < c - TOL for s, _ in self.lines)]
        return round(min(cands), 2) if cands else round(t, 2)

    def blocks_between(self, start: float, end: float) -> list:
        return [b for b in self.ep.blocks if b.end > start + TOL and b.start < end - TOL]

    def first_line_start(self, start: float, end: float):
        for s, _ in self.lines:
            if start - TOL <= s < end:
                return s
        return None


_TL_CACHE: dict = {}


def timeline(ep: Episode) -> Timeline:
    key = id(ep)
    if key not in _TL_CACHE:
        _TL_CACHE[key] = Timeline(ep)
    return _TL_CACHE[key]


def render_block(b: Block) -> str:
    if b.kind == "dialogue":
        who = b.speaker + (f" [{b.alias}]" if b.alias else "")
        how = f" ({', '.join(b.tags)})" if b.tags else ""
        return f'{who}{how}: "{b.text}"'
    if b.kind == "camera":
        return f"SEE: {b.text}"
    if b.kind == "action":
        return f"ACTION: {b.text}"
    if b.kind == "sfx":
        return f"SOUND: {b.text}"
    return f"{b.label}: {b.text}"


def render_episode(ep: Episode) -> str:
    """Episode transcript with block IDs, for the mining prompt."""
    out, scene = [], None
    headings = {s.number: s.heading for s in ep.scenes}
    for b in ep.blocks:
        if b.scene != scene:
            scene = b.scene
            out.append(f"== SCENE {scene}: {headings.get(scene, '')} ==")
        out.append(f"{b.bid} [{fmt_ts(b.start)}-{fmt_ts(b.end)}] {render_block(b)}")
    return "\n".join(out)


def render_clip(show: Show, segments: list) -> str:
    """What a viewer of the cut clip would see and hear, with times from 0 (for cold reads/judges)."""
    out, offset = [], 0.0
    for k, seg in enumerate(segments, 1):
        tl = timeline(show.episode(seg.episode))
        if len(segments) > 1:
            out.append(f"--- cut {k} ---")
        for b in tl.blocks_between(seg.start, seg.end):
            rs = offset + max(b.start, seg.start) - seg.start
            re_ = offset + min(b.end, seg.end) - seg.start
            out.append(f"[{rs:4.1f}-{re_:4.1f}s] {render_block(b)}")
        offset += seg.end - seg.start
    return "\n".join(out)


def clip_lines(show: Show, segments: list) -> list:
    """Structured excerpt for reports: (segment index, absolute time, block)."""
    rows = []
    for k, seg in enumerate(segments):
        tl = timeline(show.episode(seg.episode))
        for b in tl.blocks_between(seg.start, seg.end):
            rows.append((k, b))
    return rows


def overlap_seconds(a: list, b: list) -> float:
    tot = 0.0
    for x in a:
        for y in b:
            if x.episode == y.episode:
                tot += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return tot


def iou(a: list, b: list) -> float:
    inter = overlap_seconds(a, b)
    union = sum(s.duration for s in a) + sum(s.duration for s in b) - inter
    return inter / union if union > 0 else 0.0

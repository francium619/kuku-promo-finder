"""Parse timestamped microdrama screenplays into Episode objects.

Input can be a .zip, a folder of .txt files, or a single .txt file. The parser is
lenient on purpose: header and scene fields match case-insensitively, unknown
block labels are kept as "other", and anything it cannot read becomes a warning
instead of an exception, because the next show may be formatted a little
differently from this one.
"""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TS = r"(\d+(?::\d+){1,2}(?:\.\d+)?)"
RANGE_RE = re.compile(
    r"^start:\s*\(?\s*" + _TS + r"\s*\)?\s*\|\s*end:\s*\(?\s*" + _TS + r"\s*\)?\s*$", re.I
)
BLOCK_RE = re.compile(r"^\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
SCENE_NUM_RE = re.compile(r"^scene\s+(\d+)\s*$", re.I)
SCENE_FIELD_RE = re.compile(r"^scene\s+(heading|start|end)\s*:\s*(.*)$", re.I)
HEADER_RE = re.compile(r"^(episode number|episode logline|continuity notes)\s*:\s*(.*)$", re.I)
PAREN_RE = re.compile(r"\(([^()]*)\)")

CAMERA_LABELS = {"CAMERA", "CAM", "SHOT"}
ACTION_LABELS = {"ACTION"}
SFX_PREFIXES = ("SOUND", "SFX", "MUSIC", "AMBIENCE")
OTHER_LABELS = {"TEXT", "TITLE", "SUPER", "CAPTION", "ON-SCREEN TEXT", "TRANSITION", "INSERT"}
EXTENSIONS = {"OS", "VO", "OC", "CONTD", "CONT'D"}
TITLES = ("LADY ", "LORD ", "SIR ", "THE ", "MISS ", "MR. ", "MRS. ", "DUKE ", "DUCHESS ")


def parse_ts(s: str) -> float:
    """'01:15.50' -> 75.5 ; also accepts 'h:mm:ss.xx'."""
    parts = s.strip().split(":")
    secs = float(parts[-1])
    mins = int(parts[-2]) if len(parts) >= 2 else 0
    hrs = int(parts[-3]) if len(parts) >= 3 else 0
    return hrs * 3600 + mins * 60 + secs


def fmt_ts(t: float) -> str:
    """75.5 -> '01:15.50' (same style as the input pack)."""
    m, s = divmod(max(t, 0.0) + 1e-9, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Block:
    bid: str        # b001, b002, ... in time order
    kind: str       # camera | action | sfx | dialogue | other
    label: str      # raw label, e.g. "LADY MARA (Confused, Breathless)"
    speaker: str    # dialogue only: label without parentheticals, e.g. "LADY MARA"
    alias: str      # name given in a parenthetical, e.g. "Elias" in "WET-COAT STRANGER (Elias) (Grim)"
    tags: list      # delivery/emotion tags, e.g. ["Confused", "Breathless"]
    start: float
    end: float
    text: str
    scene: int


@dataclass
class Scene:
    number: int
    heading: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class Episode:
    number: int
    source: str
    logline: str = ""
    continuity: str = ""
    scenes: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    _index: dict = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.blocks] + [s.end for s in self.scenes if s.end is not None]
        return round(max(ends), 2) if ends else 0.0

    def block(self, bid: str) -> Block | None:
        if not self._index:
            self._index = {b.bid: b for b in self.blocks}
        return self._index.get(str(bid).strip().lower())


@dataclass
class Show:
    title: str
    episodes: list
    warnings: list = field(default_factory=list)
    total_episodes: int = 0   # episodes in the input, even when --episodes selects a subset

    def episode(self, number: int) -> Episode:
        for e in self.episodes:
            if e.number == number:
                return e
        raise KeyError(f"episode {number} not found")

    @property
    def runtime(self) -> float:
        return round(sum(e.duration for e in self.episodes), 2)


def split_label(label: str):
    """'WET-COAT STRANGER (Elias) (Cold, Hard)' -> ('WET-COAT STRANGER', 'Elias', '', ['Cold', 'Hard'])."""
    base = label.split("(", 1)[0].strip().upper()
    parens = [p.strip() for p in PAREN_RE.findall(label)]
    alias, ext, tags = "", "", []
    for i, p in enumerate(parens):
        key = p.upper().replace(".", "").replace(" ", "")
        if key in EXTENSIONS:
            ext = p
        elif i == 0 and len(parens) >= 2 and "," not in p and re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", p):
            alias = p
        else:
            tags += [t.strip() for t in p.split(",") if t.strip()]
    return base, alias, ext, tags


def classify(base: str) -> str:
    if base in CAMERA_LABELS:
        return "camera"
    if base in ACTION_LABELS:
        return "action"
    if base.startswith(SFX_PREFIXES):
        return "sfx"
    if base in OTHER_LABELS:
        return "other"
    return "dialogue"


def parse_episode(text: str, source: str) -> Episode:
    ep = Episode(number=0, source=source)
    lines = text.splitlines()
    raw = []  # (start, order, label, text, scene, end)
    scene_no = 0
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        m = HEADER_RE.match(s)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "episode number":
                num = re.search(r"\d+", val)
                ep.number = int(num.group()) if num else 0
            elif key == "episode logline":
                ep.logline = val
            else:
                ep.continuity = val
            i += 1
            continue
        m = SCENE_NUM_RE.match(s)
        if m:
            scene_no = int(m.group(1))
            ep.scenes.append(Scene(scene_no))
            i += 1
            continue
        m = SCENE_FIELD_RE.match(s)
        if m:
            if not ep.scenes:
                scene_no += 1
                ep.scenes.append(Scene(scene_no))
            fld, val = m.group(1).lower(), m.group(2).strip()
            sc = ep.scenes[-1]
            if fld == "heading":
                sc.heading = val
            else:
                try:
                    setattr(sc, fld, parse_ts(val))
                except ValueError:
                    ep.warnings.append(f"line {i + 1}: unreadable scene {fld} '{val}'")
            i += 1
            continue
        m = BLOCK_RE.match(s)
        if m:
            label = m.group(1).strip()
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            r = RANGE_RE.match(lines[j].strip()) if j < n else None
            if not r:
                ep.warnings.append(f"line {i + 1}: block '{label}' has no Start/End range; skipped")
                i += 1
                continue
            k = j + 1
            body = []
            while k < n:
                t = lines[k].strip()
                if not t or BLOCK_RE.match(t) or SCENE_NUM_RE.match(t) or SCENE_FIELD_RE.match(t) or t.startswith("---"):
                    break
                body.append(t)
                k += 1
            start, end = parse_ts(r.group(1)), parse_ts(r.group(2))
            if end < start:
                ep.warnings.append(f"line {j + 1}: end before start in '{label}'; swapped")
                start, end = end, start
            raw.append((start, len(raw), label, " ".join(body), scene_no, end))
            i = k
            continue
        i += 1

    raw.sort(key=lambda x: (x[0], x[1]))
    for idx, (start, _, label, body, scene, end) in enumerate(raw, 1):
        base, alias, _ext, tags = split_label(label)
        kind = classify(base)
        if kind == "dialogue" and len(body) >= 2 and body[0] in "\"“" and body[-1] in "\"”":
            body = body[1:-1].strip()
        ep.blocks.append(Block(f"b{idx:03d}", kind, label, base if kind == "dialogue" else "",
                               alias, tags, round(start, 2), round(end, 2), body, scene))
    if ep.number == 0:
        num = re.search(r"(\d+)", Path(source).name)
        ep.number = int(num.group(1)) if num else 0
    if ep.blocks and not any(b.kind == "camera" for b in ep.blocks):
        ep.warnings.append("no CAMERA blocks: cut points fall back to all block boundaries")
    return ep


def _read_inputs(path: Path):
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                name = info.filename
                if info.is_dir() or not name.lower().endswith(".txt") or "__MACOSX" in name:
                    continue
                yield name, z.read(info).decode("utf-8-sig", errors="replace")
    elif path.is_dir():
        for f in sorted(path.rglob("*.txt")):
            yield str(f.relative_to(path)), f.read_text(encoding="utf-8-sig", errors="replace")
    elif path.is_file():
        yield path.name, path.read_text(encoding="utf-8-sig", errors="replace")
    else:
        raise FileNotFoundError(path)


def show_title(path: Path) -> str:
    name = path.stem if path.is_file() else path.name
    name = re.sub(r"[-_ ]*scripts?\b.*$", "", name, flags=re.I)
    return name.strip(" -_") or "Untitled show"


def load_show(path, episodes: set | None = None) -> Show:
    path = Path(path)
    show = Show(title=show_title(path), episodes=[])
    for name, text in _read_inputs(path):
        ep = parse_episode(text, name)
        if not ep.blocks:
            show.warnings.append(f"{name}: no timed blocks found; ignored")
            continue
        show.total_episodes += 1
        if episodes and ep.number not in episodes:
            continue
        show.episodes.append(ep)
    show.episodes.sort(key=lambda e: (e.number, e.source))
    seen = Counter(e.number for e in show.episodes)
    for num, c in seen.items():
        if c > 1:
            show.warnings.append(f"episode number {num} appears {c} times")
    return show


def cast_summary(show: Show) -> dict:
    """Speaker labels, merged where an honorific is the only difference ('LADY MARA' ->
    'MARA'), plus aliases the script itself gives ('WET-COAT STRANGER (Elias)')."""
    counts = Counter(b.speaker for e in show.episodes for b in e.blocks if b.kind == "dialogue")
    canon = {}
    for sp in counts:
        canon[sp] = sp
        for t in TITLES:
            if sp.startswith(t) and sp[len(t):] in counts:
                canon[sp] = sp[len(t):]
                break
    hints = Counter((b.speaker, b.alias.upper()) for e in show.episodes for b in e.blocks if b.alias)
    return {"lines_per_label": dict(counts.most_common()), "merged": {k: v for k, v in canon.items() if k != v},
            "aliases_in_labels": {f"{a} = {b}": c for (a, b), c in hints.items()}}

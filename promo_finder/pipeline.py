"""Mine -> check/repair -> montage -> cold read -> rank.

Division of labour:
  model  : judgement (which moment hooks, where it is unresolved, what a stranger would follow)
  code   : everything checkable (IDs exist, cut points never split a line, 20-60 s, hook <= 3 s,
           the promo ends on its cliffhanger, de-duplication, ranking arithmetic)
  human  : approves/rejects in review.html before anything is posted
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import prompts as P
from .heuristics import best_windows, block_intensity, score_window
from .llm import LLM, LLMError, LLMFatal
from .screenplay import Episode, Show, fmt_ts
from .timeline import Segment, overlap_seconds, render_block, render_clip, render_episode, timeline

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0      # brief: the hook must land in the first 3 seconds
CLIFF_TAIL = 6.0       # a promo may run at most this long past its cliffhanger beat
DUP_OVERLAP = 0.5      # promos sharing more than half their footage are duplicates


@dataclass
class Promo:
    pid: str
    kind: str                       # single | montage | rules
    segments: list
    title: str = ""
    hook: str = ""
    hook_id: str = ""
    hook_at: float | None = None    # seconds from promo start to the hook block
    cliffhanger: str = ""
    cliffhanger_id: str = ""
    why: str = ""
    hook_type: str = "other"
    context_gap: str = ""
    model_scores: dict = field(default_factory=dict)   # hook / cliffhanger / standalone, 1-10
    cold_read: dict = field(default_factory=dict)      # fresh-viewer read of the cut clip
    rule_scores: dict = field(default_factory=dict)    # model-free scores, for reference
    flags: list = field(default_factory=list)          # problems that lower the score
    notes: list = field(default_factory=list)          # things an editor should check (no penalty)
    repairs: list = field(default_factory=list)        # fixes code applied to the model's proposal
    score: float = 0.0
    rank: int = 0
    duplicate_of: str = ""
    post_reason: str = ""

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list:
        return sorted({s.episode for s in self.segments})


@dataclass
class Beat:
    bid: str
    segment: Segment
    what: str
    role: str
    intensity: int


@dataclass
class Result:
    show: Show
    promos: list
    duplicates: list
    posting: list
    beats: list
    events: list
    meta: dict


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return lo


def _merge(segs: list) -> list:
    out = []
    for s in segs:
        if out and out[-1].episode == s.episode and s.start <= out[-1].end + 0.3 and s.end >= out[-1].start:
            prev = out[-1]
            out[-1] = Segment(s.episode, min(prev.start, s.start), max(prev.end, s.end), prev.start_id, s.end_id)
        else:
            out.append(s)
    return out


def _rule_scores(show: Show, segs: list) -> dict:
    first, last = show.episode(segs[0].episode), show.episode(segs[-1].episode)
    a = score_window(first, segs[0].start, segs[0].end)
    b = score_window(last, segs[-1].start, segs[-1].end)
    return {"hook": a["hook"], "cliffhanger": b["cliffhanger"], "standalone": min(a["standalone"], b["standalone"])}


def _editor_notes(show: Show, p: Promo) -> None:
    seg = p.segments[0]
    tl = timeline(show.episode(seg.episode))
    if tl.first_line_start(seg.start, seg.start + HOOK_WINDOW) is None:
        p.notes.append("hook is visual only (no line in the first 3 s): check the rendered shot really reads")
    if p.cold_read and p.cold_read.get("clarity", 5) <= 2:
        p.notes.append("a fresh viewer could not follow it: " + (p.cold_read.get("confusing") or "").strip())


# ---- single-episode promos ---------------------------------------------------------------
def build_single(show: Show, ep: Episode, prop: dict, pid: str):
    """Turn one model proposal into a checked Promo. Returns (promo or None, problems, repairs)."""
    tl = timeline(ep)
    errs, repairs, flags = [], [], []
    segs = []
    for sg in (prop.get("segments") or [])[:3]:
        b1, b2 = ep.block(sg.get("start_id", "")), ep.block(sg.get("end_id", ""))
        if b1 is None or b2 is None:
            errs.append(f"segment {sg.get('start_id')}-{sg.get('end_id')} uses a block ID that does not exist")
            continue
        s, e = tl.snap_start(b1.start), tl.snap_end(max(b1.end, b2.end))
        if e - s < 1.0:
            errs.append(f"segment {b1.bid}-{b2.bid} is empty or runs backwards")
            continue
        segs.append(Segment(ep.number, s, e, b1.bid, b2.bid))
    if not segs:
        return None, errs or ["no usable segments"], repairs
    segs = _merge(segs)

    hook_at = None
    hb = ep.block(prop.get("hook_id", ""))
    if hb is None:
        errs.append(f"hook_id {prop.get('hook_id')!r} does not exist")
    elif not (segs[0].start - 0.15 <= hb.start <= segs[0].end):
        errs.append(f"hook block {hb.bid} is not inside the first segment")
    else:
        hook_at = hb.start - segs[0].start
        if hook_at > HOOK_WINDOW:
            new_start = tl.snap_start(hb.start)
            if sum(x.duration for x in segs) - (new_start - segs[0].start) >= MIN_LEN:
                repairs.append(f"trimmed {new_start - segs[0].start:.1f}s of slow opening "
                               f"(hook was at {hook_at:.1f}s, now {hb.start - new_start:.1f}s)")
                segs[0] = Segment(ep.number, new_start, segs[0].end, hb.bid, segs[0].end_id)
                hook_at = hb.start - new_start
            else:
                errs.append(f"hook block {hb.bid} starts {hook_at:.1f}s in (limit 3 s) and trimming "
                            f"the opening would leave the promo under 20 s")

    cb = ep.block(prop.get("cliffhanger_id", ""))
    if cb is None:
        flags.append("cliffhanger block ID does not exist")
    elif not (segs[-1].start - 0.15 <= cb.start <= segs[-1].end + 0.15):
        flags.append(f"cliffhanger block {cb.bid} is not in the final segment")
    elif segs[-1].end - cb.end > CLIFF_TAIL:
        new_end = tl.snap_end(cb.end)
        if sum(x.duration for x in segs) - (segs[-1].end - new_end) >= MIN_LEN:
            repairs.append(f"cut {segs[-1].end - new_end:.1f}s that played after the cliffhanger")
            segs[-1] = Segment(ep.number, segs[-1].start, new_end, segs[-1].start_id, cb.bid)
        else:
            flags.append(f"runs {segs[-1].end - cb.end:.1f}s past its cliffhanger")

    total = sum(x.duration for x in segs)
    if total > MAX_LEN:
        errs.append(f"total length is {total:.1f}s (limit 60 s)")
    elif total < MIN_LEN:
        errs.append(f"total length is {total:.1f}s (minimum 20 s)")
    if errs:
        return None, errs, repairs

    p = Promo(pid=pid, kind="single", segments=segs, title=prop.get("title", ""), hook=prop.get("hook", ""),
              hook_id=hb.bid, hook_at=round(hook_at, 2), cliffhanger=prop.get("cliffhanger", ""),
              cliffhanger_id=cb.bid if cb else "", why=prop.get("why", ""),
              hook_type=prop.get("hook_type", "other"), context_gap=prop.get("context_gap", ""),
              model_scores={"hook": _clamp(prop.get("hook_score"), 1, 10),
                            "cliffhanger": _clamp(prop.get("cliffhanger_score"), 1, 10),
                            "standalone": _clamp(prop.get("standalone_score"), 1, 10)},
              flags=flags, repairs=repairs)
    return p, [], repairs


def build_beat(ep: Episode, raw: dict, k: int):
    tl = timeline(ep)
    b1, b2 = ep.block(raw.get("start_id", "")), ep.block(raw.get("end_id", ""))
    if b1 is None or b2 is None:
        return None
    s, e = tl.snap_start(b1.start), tl.snap_end(max(b1.end, b2.end))
    if not 2.0 <= e - s <= 25.0:
        return None
    return Beat(f"E{ep.number:02d}.k{k}", Segment(ep.number, s, e, b1.bid, b2.bid), raw.get("what", ""),
                raw.get("role", "setup"), _clamp(raw.get("intensity"), 1, 10))


def mine_episode(show: Show, ep: Episode, llm: LLM):
    user = P.MINE_USER.format(title=show.title, number=ep.number, total=show.total_episodes or len(show.episodes),
                              runtime=fmt_ts(ep.duration), logline=ep.logline, continuity=ep.continuity,
                              transcript=render_episode(ep))
    data = llm.complete_json(P.MINE_SYSTEM, user, P.MINE_SCHEMA, tag=f"mine_ep{ep.number:03d}")
    promos, events, problems = [], [], []

    def take(props, rnd):
        for prop in props[:3]:
            pid = f"E{ep.number:02d}-P{len(promos) + 1}"
            p, errs, reps = build_single(show, ep, prop, pid)
            status = "rejected" if p is None else ("repaired" if reps else "accepted")
            events.append({"episode": ep.number, "round": rnd, "title": prop.get("title", ""),
                           "status": status, "problems": errs, "repairs": reps,
                           "pid": p.pid if p else None})
            if p is not None:
                if any(overlap_seconds(p.segments, q.segments) > DUP_OVERLAP * min(p.duration, q.duration)
                       for q in promos):
                    events[-1]["status"] = "duplicate"
                    continue
                promos.append(p)
            elif rnd == 1:
                problems.append(f"- promo '{prop.get('title', '')}': " + "; ".join(errs))

    take(data.get("promos") or [], 1)
    if problems:
        fix = user + "\n\n" + P.FIX_USER.format(problems="\n".join(problems))
        try:
            data2 = llm.complete_json(P.MINE_SYSTEM, fix, P.MINE_SCHEMA, tag=f"fix_ep{ep.number:03d}")
            take(data2.get("promos") or [], 2)
        except LLMError as e:
            events.append({"episode": ep.number, "round": 2, "status": "fix-call-failed", "problems": [str(e)]})
    beats = [b for k, raw in enumerate((data.get("beats") or [])[:5], 1) if (b := build_beat(ep, raw, k))]
    return promos[:3], beats, events


# ---- rules-only promos (no model, or the model failed) ------------------------------------
def rules_promos(show: Show, ep: Episode) -> list:
    out = []
    for i, (seg, sc) in enumerate(best_windows(ep, k=3), 1):
        tl = timeline(ep)
        blocks = tl.blocks_between(seg.start, seg.end)
        head = [b for b in blocks if b.start <= seg.start + HOOK_WINDOW] or blocks[:1]
        tail = [b for b in blocks if b.end >= seg.end - 4.0] or blocks[-1:]
        hb = max(head, key=block_intensity)
        cb = max(tail, key=block_intensity)
        p = Promo(pid=f"E{ep.number:02d}-R{i}", kind="rules", segments=[seg],
                  title=f"Episode {ep.number}: rules pick {i}", hook=render_block(hb), hook_id=hb.bid,
                  hook_at=round(max(0.0, hb.start - seg.start), 2), cliffhanger=render_block(cb),
                  cliffhanger_id=cb.bid,
                  why=f"rule scores: hook {sc['hook']:.2f}, cliffhanger {sc['cliffhanger']:.2f}, "
                      f"standalone {sc['standalone']:.2f}",
                  model_scores={k: 1 + round(9 * sc[k]) for k in ("hook", "cliffhanger", "standalone")})
        out.append(p)
    return out


# ---- cross-episode montages ---------------------------------------------------------------
def mine_montages(show: Show, beats: list, llm: LLM, n: int = 6):
    if len(beats) < 4 or len({b.segment.episode for b in beats}) < 2:
        return [], []
    by_id = {b.bid: b for b in beats}
    lines = [f"{b.bid} | ep {b.segment.episode} | {b.segment.duration:.1f} | {b.role} | {b.intensity} | {b.what}"
             for b in beats]
    background = " ".join(f"Ep {e.number}: {e.logline}" for e in show.episodes)
    user = P.MONTAGE_USER.format(title=show.title, background=background, beats="\n".join(lines))
    data = llm.complete_json(P.MONTAGE_SYSTEM.replace("{n}", str(n)), user, P.MONTAGE_SCHEMA, tag="montage")
    promos, events = [], []
    for i, prop in enumerate((data.get("promos") or [])[:n], 1):
        errs, segs = [], []
        for bid in (prop.get("beat_ids") or [])[:4]:
            bt = by_id.get(str(bid).strip())
            if bt is None:
                errs.append(f"unknown beat {bid}")
            else:
                segs.append(bt.segment)
        if len(segs) < 2:
            errs.append("needs at least 2 beats")
        total = sum(s.duration for s in segs)
        if segs and not MIN_LEN <= total <= MAX_LEN:
            errs.append(f"total length {total:.1f}s is outside 20-60 s")
        if any(overlap_seconds([a], [b]) > 0 for j, a in enumerate(segs) for b in segs[j + 1:]):
            errs.append("beats overlap")
        status = "rejected" if errs else "accepted"
        events.append({"episode": "montage", "round": 1, "title": prop.get("title", ""), "status": status,
                       "problems": errs, "repairs": []})
        if errs:
            continue
        segs = _merge(segs)
        first_block = show.episode(segs[0].episode).block(segs[0].start_id)
        p = Promo(pid=f"M{len(promos) + 1}", kind="montage", segments=segs, title=prop.get("title", ""),
                  hook=prop.get("hook", ""), hook_id=segs[0].start_id,
                  hook_at=round(max(0.0, first_block.start - segs[0].start), 2),
                  cliffhanger=prop.get("cliffhanger", ""), cliffhanger_id=segs[-1].end_id,
                  why=prop.get("why", ""), hook_type=prop.get("hook_type", "other"),
                  context_gap=prop.get("context_gap", ""),
                  model_scores={"hook": _clamp(prop.get("hook_score"), 1, 10),
                                "cliffhanger": _clamp(prop.get("cliffhanger_score"), 1, 10),
                                "standalone": _clamp(prop.get("standalone_score"), 1, 10)})
        events[-1]["pid"] = p.pid
        if len(p.episodes) < 2:
            p.flags.append("all beats come from one episode")
        promos.append(p)
    return promos, events


# ---- cold read: a fresh 'viewer' who only sees the cut clip -------------------------------
def cold_read(show: Show, p: Promo, llm: LLM) -> None:
    user = P.COLD_USER.format(duration=p.duration, clip=render_clip(show, p.segments)) + "\n\n" + P.COLD_SCALE
    data = llm.complete_json(P.COLD_SYSTEM, user, P.COLD_SCHEMA, tag=f"cold_{p.pid}")
    for k in ("hook", "cliffhanger", "clarity"):
        data[k] = _clamp(data.get(k), 1, 5)
    p.cold_read = data
    said = (data.get("contradictions") or "").strip().strip('"').strip()
    if said and said.lower().rstrip(".") not in ("none", "n/a", "no", "nothing"):
        # Splicing beats from different episodes can genuinely mislead, so that costs score. Inside one
        # continuous scene a "contradiction" is the show's own content (often just two wordings of a
        # camera direction), so it goes to the editor as a note instead.
        if p.kind == "montage":
            p.flags.append("fresh viewer saw a contradiction: " + said)
        else:
            p.notes.append("fresh viewer noticed: " + said)


# ---- scoring and ranking ------------------------------------------------------------------
# The brief asks for all three, so the score is a geometric mean: a promo that is weak on any
# one of them cannot hide behind the other two. Each criterion averages the mining model's
# 1-10 score and the fresh viewer's 1-5 score, mapped to 0.05-1.
CRITERIA = (("hook", "hook"), ("cliffhanger", "cliffhanger"), ("standalone", "clarity"))
FLAG_PENALTY = 0.9


def composite(p: Promo, ep_range: tuple, early_bonus: float) -> float:
    ms, cr = p.model_scores, p.cold_read
    parts = []
    for mk, ck in CRITERIA:
        vals = []
        if mk in ms:
            vals.append(0.05 + 0.95 * (ms[mk] - 1) / 9)
        if ck in cr:
            vals.append(0.05 + 0.95 * (cr[ck] - 1) / 4)
        parts.append(sum(vals) / len(vals) if vals else 0.05)
    s = (parts[0] * parts[1] * parts[2]) ** (1 / 3)
    s *= FLAG_PENALTY ** len(p.flags)
    lo, hi = ep_range
    if hi > lo and early_bonus:
        s += early_bonus * (1 - (min(p.episodes) - lo) / (hi - lo))
    return round(s, 4)


def post_reason(p: Promo) -> str:
    bits = []
    if p.hook_at is not None:
        bits.append(f"{p.hook_type} hook at {p.hook_at:.1f}s")
    cr = p.cold_read
    if cr:
        bits.append(f"fresh viewer rated hook {cr['hook']}/5, cliffhanger {cr['cliffhanger']}/5, "
                    f"clarity {cr['clarity']}/5")
        if cr.get("question_left"):
            bits.append(f"left asking: \"{cr['question_left'].strip()}\"")
    if p.kind == "montage":
        bits.append("one mini-story across episodes " + ", ".join(map(str, p.episodes)))
    elif min(p.episodes) <= 5:
        bits.append("early episode, so a click lands straight on this story")
    return "; ".join(bits)


def rank(promos: list):
    """Sort by score and drop near-duplicates: >50% shared footage within a kind, >80% across
    kinds (a montage that borrows one beat from a single-episode promo is a different product)."""
    ordered = sorted(promos, key=lambda p: (-p.score, p.pid))
    kept, dups = [], []
    for p in ordered:
        twin = next((q for q in kept if overlap_seconds(p.segments, q.segments)
                     > (DUP_OVERLAP if (p.kind == "montage") == (q.kind == "montage") else 0.8)
                     * min(p.duration, q.duration)), None)
        if twin:
            p.duplicate_of = twin.pid
            dups.append(p)
        else:
            kept.append(p)
    for i, p in enumerate(kept, 1):
        p.rank = i
    return kept, dups


def posting_order(ranked: list, n: int = 10, per_episode: int = 2) -> list:
    """Post-first list: best scores, but no more than `per_episode` promos from one episode."""
    used, out = {}, []
    for p in ranked:
        if all(used.get(e, 0) < per_episode for e in p.episodes):
            out.append(p)
            for e in p.episodes:
                used[e] = used.get(e, 0) + 1
        if len(out) == n:
            break
    for p in out:
        p.post_reason = post_reason(p)
    return [p.pid for p in out]


def run(show: Show, llm: LLM, cold_llm: LLM | None = None, workers: int = 4, montage: bool = True,
        early_bonus: float = 0.05, n_post: int = 10, log=print) -> Result:
    t0 = time.time()
    cold_llm = cold_llm or llm
    singles, beats, events = [], [], []

    def one(ep):
        if not llm.enabled:
            return rules_promos(show, ep), [], []
        try:
            return mine_episode(show, ep, llm)
        except LLMFatal:
            raise  # wrong model name, key or CLI version: stop instead of quietly using rules
        except Exception as e:  # a bad or malformed reply must not sink the whole show
            return rules_promos(show, ep), [], [{"episode": ep.number, "round": 1, "status": "model-failed",
                                                 "problems": [f"{type(e).__name__}: {e}"], "repairs": [],
                                                 "fallback": "rules"}]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ep, (ps, bs, evs) in zip(show.episodes, ex.map(one, show.episodes)):
            singles += ps
            beats += bs
            events += evs
            log(f"  episode {ep.number:>3}: {len(ps)} promos, {len(bs)} beats")

    montages = []
    if montage and llm.enabled:
        try:
            montages, mev = mine_montages(show, beats, llm)
            events += mev
            log(f"  montages: {len(montages)} valid")
        except LLMFatal:
            raise
        except Exception as e:
            events.append({"episode": "montage", "round": 1, "status": "model-failed", "problems": [str(e)]})

    promos = singles + montages
    for p in promos:
        p.rule_scores = _rule_scores(show, p.segments)

    if cold_llm.enabled:
        def cr(p):
            try:
                cold_read(show, p, cold_llm)
            except LLMFatal:
                raise
            except Exception as e:
                p.flags.append(f"cold read failed: {e}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(cr, promos))
        log(f"  cold reads: {sum(1 for p in promos if p.cold_read)} of {len(promos)}")

    nums = [e.number for e in show.episodes]
    for p in promos:
        _editor_notes(show, p)
        p.score = composite(p, (min(nums), max(nums)), early_bonus)
    ranked, dups = rank(promos)
    posting = posting_order(ranked, n_post)
    meta = {"seconds": round(time.time() - t0, 1), "model": llm.spec, "cold_model": cold_llm.spec,
            "prompt_version": P.PROMPT_VERSION, "usage": {"model": dict(llm.usage)},
            "early_bonus": early_bonus, "score": "geometric mean of hook, cliffhanger, standalone; x0.9 per flag"}
    if cold_llm is not llm:
        meta["usage"]["cold_model"] = dict(cold_llm.usage)
    return Result(show, ranked, dups, posting, beats, events, meta)

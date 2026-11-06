"""Write the run's outputs: promos.json, promos_ranked.csv, top_promos.md, review.html, logs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .pipeline import Promo, Result
from .screenplay import fmt_ts
from .timeline import timeline


def seg_text(p: Promo) -> str:
    return " | ".join(f"Ep {s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)


def excerpt(result: Result, p: Promo) -> list:
    rows, offset = [], 0.0
    for k, s in enumerate(p.segments):
        tl = timeline(result.show.episode(s.episode))
        for b in tl.blocks_between(s.start, s.end):
            rel = offset + max(b.start, s.start) - s.start
            rows.append({"cut": k + 1, "t": round(rel, 1), "abs": f"Ep {s.episode} {fmt_ts(b.start)}",
                         "kind": b.kind, "who": b.speaker, "how": ", ".join(b.tags), "text": b.text,
                         "id": b.bid, "hook": b.bid == p.hook_id and k == 0,
                         "cliff": b.bid == p.cliffhanger_id and k == len(p.segments) - 1})
        offset += s.duration
    return rows


def promo_dict(result: Result, p: Promo, with_excerpt: bool = False) -> dict:
    d = {"rank": p.rank, "post_order": (result.posting.index(p.pid) + 1) if p.pid in result.posting else None,
         "id": p.pid, "kind": p.kind, "title": p.title, "episodes": p.episodes,
         "segments": [{"episode": s.episode, "start": fmt_ts(s.start), "end": fmt_ts(s.end),
                       "start_s": s.start, "end_s": s.end, "seconds": s.duration,
                       "first_block": s.start_id, "last_block": s.end_id} for s in p.segments],
         "duration_s": p.duration, "hook_at_s": p.hook_at, "hook": p.hook, "hook_type": p.hook_type,
         "cliffhanger": p.cliffhanger, "why_it_works": p.why, "why_post_first": p.post_reason,
         "context_gap": p.context_gap, "score": p.score, "model_scores": p.model_scores,
         "fresh_viewer": p.cold_read, "rule_scores": p.rule_scores, "flags": p.flags, "notes": p.notes,
         "repairs": p.repairs,
         "duplicate_of": p.duplicate_of or None}
    if with_excerpt:
        d["excerpt"] = excerpt(result, p)
        d["episode_lengths"] = {s.episode: result.show.episode(s.episode).duration for s in p.segments}
    return d


def validation_stats(events: list) -> dict:
    proposals = [e for e in events if e.get("status") in ("accepted", "repaired", "rejected", "duplicate")]
    c = Counter(e["status"] for e in proposals)
    first = [e for e in proposals if e.get("round") == 1]
    reasons = Counter()
    for e in proposals:
        for prob in e.get("problems", []):
            key = ("length out of range" if "length" in prob else "hook after 3 s" if "hook block" in prob
                   and "starts" in prob else "hook not in first segment" if "first segment" in prob
                   else "unknown block ID" if "does not exist" in prob else "other")
            reasons[key] += 1
    repairs = Counter()
    for e in proposals:
        for r in e.get("repairs", []):
            repairs["trimmed slow opening" if r.startswith("trimmed") else "cut tail after cliffhanger"] += 1
    return {"proposals": len(proposals), "by_status": dict(c),
            "first_round": dict(Counter(e["status"] for e in first)),
            "rejection_reasons": dict(reasons), "repairs": dict(repairs),
            "model_failures": sum(1 for e in events if e.get("status") == "model-failed")}


def write_all(result: Result, out: Path, command: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    show = result.show
    stats = validation_stats(result.events)
    meta = dict(result.meta)
    meta.update({"show": show.title, "episodes": len(show.episodes), "runtime_s": show.runtime,
                 "generated": datetime.now().isoformat(timespec="seconds"), "command": command,
                 "promos": len(result.promos), "duplicates_removed": len(result.duplicates),
                 "by_kind": dict(Counter(p.kind for p in result.promos)), "validation": stats,
                 "parse_warnings": show.warnings + [f"Ep {e.number}: {w}" for e in show.episodes for w in e.warnings]})

    promos = [promo_dict(result, p) for p in result.promos]
    (out / "promos.json").write_text(json.dumps({"meta": meta, "post_first": result.posting, "promos": promos,
                                                 "duplicates": [promo_dict(result, p) for p in result.duplicates]},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "validation_log.json").write_text(json.dumps({"stats": stats, "events": result.events},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    _csv(result, out / "promos_ranked.csv")
    _markdown(result, meta, out / "top_promos.md")
    _html(result, meta, out / "review.html")
    return meta


def _csv(result: Result, path: Path) -> None:
    cols = ["rank", "post_order", "id", "kind", "title", "segments", "duration_s", "hook_at_s", "hook",
            "cliffhanger", "why_it_works", "why_post_first", "hook_type", "score", "model_hook",
            "model_cliffhanger", "model_standalone", "viewer_hook", "viewer_cliffhanger", "viewer_clarity",
            "viewer_question_left", "viewer_confusing", "flags", "notes", "repairs"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in result.promos:
            cr, ms = p.cold_read, p.model_scores
            w.writerow([p.rank, (result.posting.index(p.pid) + 1) if p.pid in result.posting else "", p.pid, p.kind,
                        p.title, seg_text(p), p.duration, p.hook_at, p.hook, p.cliffhanger, p.why, p.post_reason,
                        p.hook_type, p.score, ms.get("hook"), ms.get("cliffhanger"), ms.get("standalone"),
                        cr.get("hook"), cr.get("cliffhanger"), cr.get("clarity"), cr.get("question_left"),
                        cr.get("confusing"), "; ".join(p.flags), "; ".join(p.notes), "; ".join(p.repairs)])


def _markdown(result: Result, meta: dict, path: Path) -> None:
    show = result.show
    by_id = {p.pid: p for p in result.promos}
    L = [f"# Promos: {show.title}", "",
         f"{len(show.episodes)} episodes, {show.runtime / 60:.0f} min of screenplay. Model `{meta['model']}`, "
         f"fresh-viewer check `{meta['cold_model']}`. {len(result.promos)} promos after de-duplication "
         f"({', '.join(f'{v} {k}' for k, v in meta['by_kind'].items())}). Generated {meta['generated']}.", "",
         "Timestamps are episode time (mm:ss.xx), cut on existing camera cuts and never inside a line.", "",
         "## Post these first", ""]
    for i, pid in enumerate(result.posting, 1):
        p = by_id[pid]
        L += [f"### {i}. {p.title}  (rank {p.rank}, score {p.score:.2f})", "",
              f"- **Segments:** {seg_text(p)}  ({p.duration:.1f} s)",
              f"- **Hook** (at {p.hook_at:.1f} s): {p.hook}",
              f"- **Cliffhanger:** {p.cliffhanger}",
              f"- **Why it works:** {p.why}",
              f"- **Why post it first:** {p.post_reason}"]
        if p.flags or p.notes:
            L.append(f"- **Check:** {'; '.join(p.flags + p.notes)}")
        L.append("")
    L += ["## All promos, ranked", ""]
    for p in result.promos:
        star = " (post first)" if p.pid in result.posting else ""
        L += [f"**{p.rank}. {p.title}**{star}: {seg_text(p)} ({p.duration:.1f} s), score {p.score:.2f}",
              f"Hook: {p.hook} | Cliffhanger: {p.cliffhanger} | Why: {p.why}"
              + (f" | Check: {'; '.join(p.flags + p.notes)}" if p.flags or p.notes else ""), ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _html(result: Result, meta: dict, path: Path) -> None:
    payload = {"meta": {k: meta[k] for k in ("show", "episodes", "runtime_s", "model", "cold_model", "generated",
                                             "promos", "by_kind")},
               "posting": result.posting,
               "promos": [promo_dict(result, p, with_excerpt=True) for p in result.promos]}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    tpl = (Path(__file__).parent / "review_template.html").read_text(encoding="utf-8")
    path.write_text(tpl.replace("/*__DATA__*/null", data), encoding="utf-8")

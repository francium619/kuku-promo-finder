"""Evaluate a run without ground-truth labels.

1. Blind pairwise judging. For every episode, the tool's best single-episode promo is put against
   the shortcuts an editor would take: the first ~40 s, the last ~40 s (which end on the episode's
   own cliffhanger) and the best rules-only window. A judge model that did NOT generate the picks
   sees only the two clip transcripts, in random A/B order.
2. Position-bias check: a sample of pairs is judged again with A and B swapped.
3. Model swap: the best pick per episode from a second run (another model) is compared with the
   main run, by overlap and head-to-head.
4. Rule checks done by code alone: lengths, cuts inside lines, how fast the first line lands.

    python eval/evaluate.py INPUT --run RUN/promos.json [--alt ALT/promos.json]
           --judge claude-cli:claude-opus-5-5 --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promo_finder import prompts as P  # noqa: E402
from promo_finder.heuristics import baseline_ending, baseline_opening, best_windows  # noqa: E402
from promo_finder.llm import LLM, LLMError, LLMFatal  # noqa: E402
from promo_finder.screenplay import load_show  # noqa: E402
from promo_finder.timeline import Segment, iou, render_clip, timeline  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_run(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    best = {}
    for p in data["promos"]:  # already in rank order
        if p["kind"] in ("single", "rules") and len(p["episodes"]) == 1:
            ep = p["episodes"][0]
            if ep not in best:
                best[ep] = {"id": p["id"], "title": p["title"],
                            "segments": [Segment(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]}
    return data, best


def judge_pair(show, llm, a_segs, b_segs, tag):
    da = sum(s.duration for s in a_segs)
    db = sum(s.duration for s in b_segs)
    user = P.JUDGE_USER.format(a=render_clip(show, a_segs), b=render_clip(show, b_segs), da=da, db=db)
    return llm.complete_json(P.JUDGE_SYSTEM, user, P.JUDGE_SCHEMA, tag=tag)


def run_pairs(show, llm, pairs, workers, rng):
    """pairs: list of (key, tool_segs, other_segs). Returns verdicts from the tool's point of view."""
    jobs = []
    for key, tool, other in pairs:
        tool_is_a = rng.random() < 0.5
        jobs.append((key, tool_is_a, tool, other))

    def one(job):
        key, tool_is_a, tool, other = job
        a, b = (tool, other) if tool_is_a else (other, tool)
        try:
            v = judge_pair(show, llm, a, b, f"judge_{key}")
        except LLMFatal:
            raise
        except LLMError as e:
            return {"key": key, "error": str(e)}
        conv = {"A": "tool" if tool_is_a else "other", "B": "other" if tool_is_a else "tool", "tie": "tie"}
        return {"key": key, "tool_was": "A" if tool_is_a else "B", "reason": v["reason"],
                **{c: conv.get(v[c], "tie") for c in ("winner", "hook", "cliffhanger", "standalone")}}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, jobs))


def tally(rows):
    ok = [r for r in rows if "error" not in r]
    out = {"n": len(ok), "pairs": len(rows), "errors": len(rows) - len(ok)}
    for c in ("winner", "hook", "cliffhanger", "standalone"):
        t = sum(r[c] == "tool" for r in ok)
        o = sum(r[c] == "other" for r in ok)
        out[c] = {"tool": t, "other": o, "tie": len(ok) - t - o}
    t, o = out["winner"]["tool"], out["winner"]["other"]
    out["win_rate_excl_ties"] = round(t / (t + o), 3) if t + o else None
    out["ci95"] = [round(x, 3) for x in wilson(t, t + o)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--run", required=True, help="promos.json of the run to evaluate")
    ap.add_argument("--alt", help="promos.json of a run with another model (for the model-swap check)")
    ap.add_argument("--judge", default="claude-cli:claude-opus-5-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--swap-check", type=int, default=25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parent.parent / ".cache" / "llm"))
    ap.add_argument("--cached-only", action="store_true",
                    help="make no model calls: use cached verdicts and report the rest as pending")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    show = load_show(args.input)
    run, best = load_run(Path(args.run))
    llm = LLM(args.judge, cache_dir=args.cache or None, log_dir=out / "judge_responses",
              retries=0 if args.cached_only else 2)
    if args.cached_only:
        def _not_cached(*a):
            raise LLMError("pending: not judged yet (--cached-only run)")
        llm._call = _not_cached
    else:
        llm.preflight()
    rng = random.Random(args.seed)
    eps = [e for e in show.episodes if e.number in best]

    baselines = {
        "first_40s": {e.number: [baseline_opening(e)] for e in eps},
        "last_40s": {e.number: [baseline_ending(e)] for e in eps},
        "rules_only": {e.number: [best_windows(e, k=1)[0][0]] for e in eps},
    }
    results, verdicts = {}, []
    for name, segs in baselines.items():
        pairs = [(f"{name}_ep{n:03d}", best[n]["segments"], segs[n]) for n in sorted(segs)]
        rows = run_pairs(show, llm, pairs, args.workers, rng)
        for r in rows:
            r["comparison"] = name
        verdicts += rows
        results[name] = tally(rows)
        print(f"{name}: {results[name]['winner']}  win rate {results[name]['win_rate_excl_ties']} CI {results[name]['ci95']}")

    # position-bias check: same pairs, opposite order. The sample is drawn from all pairs (judged or not)
    # with its own RNG, so it is the same on every run and finished re-judgements come from the cache.
    srng = random.Random(args.seed + 1)
    sample = srng.sample(verdicts, min(args.swap_check, len(verdicts)))
    same = done = pending = 0
    for v in sample:
        if "error" in v:
            pending += 1
            continue
        name, ep = v["comparison"], int(v["key"].rsplit("ep", 1)[1])
        tool, other = best[ep]["segments"], baselines[name][ep]
        a, b = (other, tool) if v["tool_was"] == "A" else (tool, other)
        try:
            again = judge_pair(show, llm, a, b, f"swap_{v['key']}")
        except LLMFatal:
            raise
        except LLMError:
            pending += 1
            continue
        done += 1
        tool_now = "B" if v["tool_was"] == "A" else "A"
        w = again["winner"]
        now = "tool" if w == tool_now else ("tie" if w == "tie" else "other")
        same += now == v["winner"]
    results["position_bias_check"] = {"pairs": done, "same_verdict": same, "pending": pending}
    print(f"position check: {same}/{done} unchanged when A/B swapped ({pending} pending)")

    # model swap
    if args.alt:
        alt_run, alt_best = load_run(Path(args.alt))
        common = sorted(set(best) & set(alt_best))
        ious = {n: iou(best[n]["segments"], alt_best[n]["segments"]) for n in common}
        differ = [n for n in common if ious[n] < 0.8]
        rows = run_pairs(show, llm, [(f"alt_ep{n:03d}", best[n]["segments"], alt_best[n]["segments"]) for n in differ],
                         args.workers, rng)
        for r in rows:
            r["comparison"] = "main_vs_alt"
        verdicts += rows
        t = tally(rows)
        results["model_swap"] = {
            "main_model": run["meta"]["model"], "alt_model": alt_run["meta"]["model"], "episodes": len(common),
            "same_moment_iou_ge_0.3": sum(v >= 0.3 for v in ious.values()),
            "near_identical_iou_ge_0.8": len(common) - len(differ), "judged_where_different": t,
            "main_cost_usd": run["meta"]["usage"], "alt_cost_usd": alt_run["meta"]["usage"]}
        print(f"model swap: same moment in {results['model_swap']['same_moment_iou_ge_0.3']}/{len(common)}; "
              f"head-to-head where different: {t['winner']}")

    # rule checks by code alone
    def first_line_delay(segs):
        s = segs[0]
        t = timeline(show.episode(s.episode)).first_line_start(s.start, s.end)
        return None if t is None else t - s.start

    def check(segsets):
        n = len(segsets)
        delays = [first_line_delay(s) for s in segsets]
        clean = sum(all(timeline(show.episode(x.episode)).is_clean(x.start)
                        and timeline(show.episode(x.episode)).is_clean(x.end) for x in s) for s in segsets)
        in_len = sum(20 <= sum(x.duration for x in s) <= 60 for s in segsets)
        known = sorted(d for d in delays if d is not None)
        return {"n": n, "line_within_3s": sum(d is not None and d <= 3.0 for d in delays),
                "median_first_line_s": round(known[len(known) // 2], 2) if known else None,
                "no_cut_inside_a_line": clean, "length_20_60": in_len}

    all_tool = [[Segment(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]] for p in run["promos"]]
    results["rule_checks"] = {
        "tool_all_promos": check(all_tool),
        "tool_best_per_episode": check([best[n]["segments"] for n in sorted(best)]),
        "first_40s": check([v for v in baselines["first_40s"].values()]),
        "last_40s": check([v for v in baselines["last_40s"].values()]),
    }
    results["judge"] = args.judge
    results["judge_usage"] = llm.usage
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    with (out / "judgments.jsonl").open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    write_summary(results, out / "summary.md")
    print(f"judge usage: {llm.usage}")
    return 0


def write_summary(r, path):
    names = {"first_40s": "first 40 s of the episode", "last_40s": "last 40 s (episode's own cliffhanger)",
             "rules_only": "best rules-only window (no model)"}
    L = ["# Evaluation", "", f"Judge: `{r['judge']}`, which did not generate the picks. It sees only the two clip "
         "transcripts, in random A/B order, and applies the brief's three criteria.", "",
         "## Tool's best promo per episode vs an editor's shortcuts", "",
         "| Compared with | Judged | Tool wins | Shortcut wins | Ties | Tool win rate excl. ties (95% CI) | Hook | Cliffhanger | Stands alone |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, label in names.items():
        t = r[k]
        crit = " | ".join(f"{t[c]['tool']}-{t[c]['other']}" for c in ("hook", "cliffhanger", "standalone"))
        rate = (f"{t['win_rate_excl_ties']:.0%} ({t['ci95'][0]:.0%}-{t['ci95'][1]:.0%})"
                if t["win_rate_excl_ties"] is not None else "-")
        L.append(f"| {label} | {t['n']} of {t.get('pairs', t['n'])} | {t['winner']['tool']} | {t['winner']['other']} | "
                 f"{t['winner']['tie']} | {rate} | {crit} |")
    pb = r["position_bias_check"]
    pending = [f"{names[k].split(' (')[0]}: {r[k]['errors']}" for k in names if r[k].get("errors")]
    L += ["", "Per-criterion cells are tool-shortcut wins.", ""]
    if pending:
        L += [f"Not judged yet ({'; '.join(pending)}): the judge hit its plan usage limit. Re-running the same "
              "command finishes them; judged pairs come from the cache.", ""]
    L += [f"Position-bias check: {pb['same_verdict']} of {pb['pairs']} re-judged verdicts were unchanged when A and B "
          f"were swapped" + (f"; {pb['pending']} not re-judged yet." if pb.get("pending") else "."), ""]
    if "model_swap" in r:
        m = r["model_swap"]
        t = m["judged_where_different"]
        h2h = (f"- Head-to-head where the picks differ ({t['n']} of {t.get('pairs', t['n'])} judged): main model wins "
               f"{t['winner']['tool']}, alt model wins {t['winner']['other']}, ties {t['winner']['tie']}."
               if t["n"] else f"- Head-to-head where the picks differ: not judged yet ({t.get('pairs', 0)} pairs pending).")
        L += ["## Model swap", "",
              f"- `{m['main_model']}` vs `{m['alt_model']}` over {m['episodes']} episodes.",
              f"- Same moment picked (footage overlap IoU >= 0.3): {m['same_moment_iou_ge_0.3']} episodes; "
              f"near-identical (IoU >= 0.8): {m['near_identical_iou_ge_0.8']}.", h2h, ""]
    L += ["## Rule checks (code only, no model)", "",
          "| Set | n | Length 20-60 s | No cut inside a line | Spoken line within 3 s | Median time to first line |",
          "|---|---|---|---|---|---|"]
    for k, v in r["rule_checks"].items():
        L.append(f"| {k} | {v['n']} | {v['length_20_60']} | {v['no_cut_inside_a_line']} | {v['line_within_3s']} | "
                 f"{v['median_first_line_s']} s |")
    L += ["", "Caveat: an LLM judge is a proxy for an audience, not a measurement of one. The real test is posting "
          "and reading 3-second hold rate, completion rate and click-through."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

"""Command line: `python -m promo_finder parse|run ...`  (see README.md)."""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from . import __version__
from .llm import LLM, LLMError
from .pipeline import run
from .report import seg_text, write_all
from .screenplay import cast_summary, fmt_ts, load_show


def parse_episodes(spec: str | None):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def make_llm(spec, args, log_dir=None):
    price = tuple(float(x) for x in args.price.split(",")) if getattr(args, "price", None) else None
    return LLM(spec, base_url=args.base_url, api_key_env=args.api_key_env, cache_dir=args.cache or None,
               log_dir=log_dir, effort=args.effort, price=price)


def cmd_parse(args) -> int:
    show = load_show(args.input, parse_episodes(args.episodes))
    print(f"Show: {show.title}")
    print(f"Episodes: {len(show.episodes)}   total runtime {show.runtime / 60:.1f} min")
    for e in show.episodes:
        kinds = {}
        for b in e.blocks:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        print(f"  Ep {e.number:>3}  {fmt_ts(e.duration)}  scenes {len(e.scenes):>2}  " +
              "  ".join(f"{k} {v}" for k, v in sorted(kinds.items())) +
              (f"  WARN: {'; '.join(e.warnings)}" if e.warnings else ""))
    for w in show.warnings:
        print("WARN:", w)
    cast = cast_summary(show)
    if cast["merged"] or cast["aliases_in_labels"]:
        print("Speaker labels merged:", cast["merged"])
        print("Aliases written in labels:", cast["aliases_in_labels"])
    return 0


def cmd_run(args) -> int:
    show = load_show(args.input, parse_episodes(args.episodes))
    if not show.episodes:
        print("No episodes found in", args.input, file=sys.stderr)
        return 2
    out = Path(args.out)
    llm = make_llm(args.model, args, log_dir=out / "llm_responses")
    cold = make_llm(args.cold_model, args, log_dir=out / "llm_responses") if args.cold_model else llm
    for m in {id(llm): llm, id(cold): cold}.values():
        m.preflight()  # missing key / URL / CLI: stop now, not after a silent rules-only run
    print(f"{show.title}: {len(show.episodes)} episodes, {show.runtime / 60:.1f} min. "
          f"Model {llm.spec}" + (f", fresh-viewer check {cold.spec}" if cold is not llm else ""))
    if not llm.enabled:
        print("No model: using rules only (pass --model to use one; see README).")
    try:
        result = run(show, llm, cold, workers=args.workers, montage=not args.no_montage,
                     early_bonus=args.early_bonus, n_post=args.top)
    except LLMError as e:
        print("Model error:", e, file=sys.stderr)
        return 1
    command = "python -m promo_finder " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    meta = write_all(result, out, command)
    by_id = {p.pid: p for p in result.promos}
    print(f"\n{len(result.promos)} promos written to {out.resolve()}")
    print("Post first:")
    for i, pid in enumerate(result.posting[:5], 1):
        p = by_id[pid]
        print(f"  {i}. [{p.score:.2f}] {p.title} :: {seg_text(p)} ({p.duration:.0f} s)")
    u = meta["usage"]
    for name, us in u.items():
        print(f"{name}: {us['calls']} calls ({us['cache_hits']} cached), {us['input_tokens']:,} in / "
              f"{us['output_tokens']:,} out tokens, est. ${us['cost_usd']:.2f}")
        if us["cache_hits"]:
            print(f"  (cached answers originally cost {us.get('cached_input_tokens', 0):,} in / "
                  f"{us.get('cached_output_tokens', 0):,} out tokens, est. ${us.get('cached_cost_usd', 0):.2f})")
    print(f"Done in {meta['seconds']:.0f} s. Open review.html to approve or reject.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find ranked promo clips in microdrama screenplays.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help="check that a show's screenplays parse; prints a summary")
    p.add_argument("input", help=".zip, folder of .txt files, or one .txt file")
    p.add_argument("--episodes", help="subset, e.g. 1-5,9")
    p.set_defaults(func=cmd_parse)

    r = sub.add_parser("run", help="find and rank promos")
    r.add_argument("input", help=".zip, folder of .txt files, or one .txt file")
    r.add_argument("--out", default="outputs/run", help="output folder")
    r.add_argument("--model", default=os.environ.get("PROMO_MODEL", "none"),
                   help="provider:model, e.g. claude-cli:claude-sonnet-5-5, openai:gpt-4o-mini, "
                        "gemini:gemini-2.5-flash, compat:<name>, or none (rules only)")
    r.add_argument("--cold-model", default=os.environ.get("PROMO_COLD_MODEL"),
                   help="cheaper model for the fresh-viewer check (default: same as --model)")
    r.add_argument("--base-url", help="endpoint for compat/openai-style providers")
    r.add_argument("--api-key-env", help="name of the env var holding the API key")
    r.add_argument("--effort", help="claude-cli only: low|medium|high")
    r.add_argument("--price", help="USD per million tokens 'in,out' for the cost estimate")
    r.add_argument("--episodes", help="subset, e.g. 1-5,9")
    r.add_argument("--workers", type=int, default=4, help="parallel model calls")
    r.add_argument("--top", type=int, default=10, help="size of the post-first list")
    r.add_argument("--no-montage", action="store_true", help="skip cross-episode promos")
    r.add_argument("--early-bonus", type=float, default=0.05,
                   help="ranking bonus for early-episode promos (0 to switch off)")
    r.add_argument("--cache", default=str(Path(__file__).resolve().parent.parent / ".cache" / "llm"),
                   help="response cache folder ('' to disable)")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, FileNotFoundError) as e:
        print("Error:", e, file=sys.stderr)
        return 2

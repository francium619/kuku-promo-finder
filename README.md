# kuku-promo-finder

Reads the timestamped screenplays of a microdrama and returns ranked 20-60 second promos: for each one,
the episode and exact start/end of every segment, the hook, the cliffhanger, one line on why it works,
and why it should (or should not) be posted first. An editor reviews the list in a single HTML page
instead of watching every episode.

Built for the Kuku AI Generalist assignment, Problem 1. The input pack (Kuku's screenplays) is
confidential and **not** in this repo; the tests use a two-episode sample show written for the repo.

## Run it

Python 3.10+. The core needs no packages (standard library only).

```bash
# 1. Check a new show parses (episodes, runtimes, format warnings)
python -m promo_finder parse "path/to/Show-scripts.zip"

# 2. No model, no key, free: rules-only picks
python -m promo_finder run "path/to/Show-scripts.zip" --model none --out outputs/rules

# 3. With a model (pick one; the model is one flag)
python -m promo_finder run SHOW.zip --out outputs/run --model gemini:gemini-2.5-flash      # GEMINI_API_KEY (free tier)
python -m promo_finder run SHOW.zip --out outputs/run --model openai:gpt-4o-mini           # OPENAI_API_KEY
python -m promo_finder run SHOW.zip --out outputs/run --model anthropic:claude-sonnet-5-5  # ANTHROPIC_API_KEY, pip install anthropic
python -m promo_finder run SHOW.zip --out outputs/run --model claude-cli:claude-sonnet-5-5 # Claude Code login, no key
python -m promo_finder run SHOW.zip --out outputs/run --model compat:<model-name> \
       --base-url https://your-gateway/v1 --api-key-env YOUR_KEY_VAR                       # any OpenAI-compatible API
```

Errors that a retry cannot fix (an unknown model, a bad key, a `claude` CLI too old for the model)
stop the run instead of quietly falling back to rules. With `claude-cli`, run `claude update` or set
`CLAUDE_BIN` to a newer Claude Code binary if you see "version ... required".

Input can be a `.zip`, a folder of `.txt` files, or one `.txt` file. Useful options: `--cold-model`
(a cheaper model for the fresh-viewer check), `--episodes 1-10`, `--workers 6`, `--no-montage`,
`--early-bonus 0`. Responses are cached in `.cache/`, so a re-run on the same input is free and
gives the same output.

The raw outputs in the submission were made with `--model claude-cli:claude-sonnet-5-5
--cold-model claude-cli:claude-haiku-5-5`. The OpenAI-compatible adapter, which the `gemini`,
`openai`, `groq`, `openrouter`, `ollama` and `compat` providers share, is unit-tested against a local
mock server. The `anthropic` adapter is not tested. Neither was run live for the submission.

## What comes out

| File | What it is |
|---|---|
| `top_promos.md` | The post-first list with reasons, then every promo in rank order |
| `promos_ranked.csv` | One row per promo: segments, hook, cliffhanger, why, scores, flags |
| `review.html` | Editor review page: script of each cut with the first 3 s, hook and cliffhanger highlighted; approve / maybe / reject; export decisions as CSV |
| `promos.json` | Everything, including the fresh-viewer read and the scores |
| `validation_log.json` | Every model proposal and what code did with it (accepted, repaired, sent back, rejected) |
| `llm_responses/` | Unedited model replies |

## How it works

```
screenplays ─► parse (code) ─► mine each episode (model) ─► check + repair (code) ──┐
                                   │  ▲ problems code can't fix go back once        │
                                   ▼  │                                             ▼
                               beats ─► cross-episode montages (model) ─► fresh-viewer cold read (cheap model)
                                                                              │
                         review.html (human approves) ◄─ rank + de-duplicate (code)
```

* **Code** does everything checkable. The model names blocks by ID (`b014`) and never writes a
  timestamp; code turns IDs into cut points that never split a line of dialogue and snap to the
  episode's own camera cuts. Code enforces 20-60 s, the hook within 3 s (it trims a slow opening
  itself when it can), and the end on the cliffhanger (it trims a tail that plays past it). It also
  removes duplicates and does the ranking arithmetic.
* **The model** does judgement: which moment stops a scroll, where it is still unresolved, and whether
  a stranger would follow it.
* **A second, cheaper model** plays a viewer who has never seen the show and reads only the cut clip.
  Its scores feed the ranking and its confusion notes go to the editor.
* **A human** approves or rejects in `review.html`. Nothing is posted automatically.

Ranking is the geometric mean of hook, cliffhanger and stands-alone, so a promo cannot hide a weak
criterion behind two strong ones. Each criterion averages the mining model's score and the fresh
viewer's. Problems that code finds or the viewer reports (for example a contradiction) cost 10% each;
early episodes get a small bonus, switchable with `--early-bonus`. The post-first list allows at most
two promos per episode.

## Evaluate

```bash
python eval/evaluate.py SHOW.zip --run outputs/run/promos.json --alt outputs/alt/promos.json \
       --judge claude-cli:claude-opus-5-5 --out outputs/eval
```

The script blind-judges each episode's best promo against the first 40 s, the last 40 s and the
rules-only pick, re-judges a sample with A and B swapped to check position bias, compares two models'
picks, and runs rule checks in code. Results go to `summary.md`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Known limits

* It reads the script, not the video. A hook that exists only in a camera description may not read on
  screen; those promos are marked for the editor to check.
* Cross-episode montages can mislead (two weddings that look like one). The fresh-viewer check catches
  some of these; an editor must approve every montage.
* The scores are model judgements, not audience data. Before the ranking is trusted, its weights
  should be tuned on real hold, completion and click-through rates.

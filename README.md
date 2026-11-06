# kuku-promo-finder

This tool reads the timestamped screenplays of a microdrama and suggests ranked promos of 20 to 60
seconds. For each one you get the episode and exact start and end of every segment, the hook, the
cliffhanger, a line on why it works, and why it should (or shouldn't) be posted first. An editor then
goes through the list on one HTML page instead of watching every episode.

I built it for the Kuku AI Generalist assignment (Problem 1). Kuku's screenplays are confidential, so
they are not in this repo. The tests use a short two-episode sample show I wrote for it.

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

1. Code parses the screenplays into numbered blocks (dialogue, action, camera, sound).
2. A model reads each episode and proposes up to three promos, plus a few short "beats" that could
   work in a montage. It refers to script blocks by ID (like `b014`) and never writes a timestamp.
3. Code turns the IDs into cut points. A cut never splits a line of dialogue, and it snaps to the
   episode's own camera cuts. Code also enforces the rules: 20 to 60 seconds, a hook within 3 seconds
   (it trims a slow opening itself when it can) and an ending on the cliffhanger (it trims a tail
   that plays past it). Anything it can't fix goes back to the model once, with the exact problem.
4. The model joins beats from different episodes into a few montages, as long as the result stays
   true to the story.
5. A second, cheaper model plays a viewer who has never seen the show. It reads only the cut clip and
   says what it understood, what confused it and what it wants to know next.
6. Code removes near-duplicates and ranks what's left.
7. An editor approves or rejects each promo in `review.html`. Nothing is posted automatically.

The split is simple: code does everything that can be checked, the model does the judgement (which
moment stops a scroll, where it is still unresolved, whether a stranger would follow it), and a person
makes the final call.

The rank is the geometric mean of the hook, cliffhanger and stand-alone scores, so a promo can't hide
one weak score behind two strong ones. Each score averages the first model's rating and the fresh
viewer's. Each problem that code finds or the viewer reports (a contradiction, for example) costs 10%,
and early episodes get a small bonus that `--early-bonus` controls. The post-first list takes at most
two promos from any one episode.

## Evaluate

```bash
python eval/evaluate.py SHOW.zip --run outputs/run/promos.json --alt outputs/alt/promos.json \
       --judge claude-cli:claude-opus-5-5 --out outputs/eval
```

This has a separate model judge, blind, each episode's best promo against three shortcuts: the
episode's first 40 seconds, its last 40 seconds and the best window found by rules alone. It re-judges
a sample with the two clips swapped to check for order bias, compares the picks of two models, and runs
the rule checks in code. Results go to `summary.md`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Known limits

- It reads the script, not the video. A hook that exists only in a camera description may not come
  across on screen, so those promos are flagged for the editor.
- Cross-episode montages can mislead (two different weddings can look like one). The fresh-viewer
  check catches some of these, but an editor should approve every montage.
- The scores are model judgements, not audience data. The ranking weights should be tuned on real
  hold, completion and click-through rates before anyone relies on them.

"""Prompts and JSON schemas. Kept in one place so they can be read, tuned and versioned."""

PROMPT_VERSION = "2026-11-06.2"

HOOK_TYPES = ["betrayal", "humiliation", "violence", "reveal", "threat", "injustice", "rescue",
              "romance", "mystery", "other"]

CRAFT = """A promo works when it does three things:
1. HOOK - something in the FIRST 3 SECONDS makes a scrolling stranger stop: a slap or other violence, a
   shocking or cruel line, a reveal, a threat, a public humiliation, a striking image. The clip starts ON
   that moment or a beat before it. Never open on establishing shots, walking, or quiet set-up.
2. CLIFFHANGER - it cuts at an unresolved moment: right after a question, threat, accusation, reveal or
   action whose outcome we do not see. It never includes the answer or the resolution.
3. STANDS ALONE - someone who has never seen the show understands who wants what within seconds, from
   what is said and shown in the clip. Prefer moments where the dialogue states the situation. Clips
   that depend on names, backstory or earlier episodes are weaker.
Length: 20-60 seconds in total; 25-45 is ideal."""

MINE_SYSTEM = f"""You are a senior promo editor for vertical microdrama apps. You cut 20-60 second promos from a
show's own footage for TikTok, Instagram Reels and YouTube Shorts, to pull in viewers who have never
heard of the show.

{CRAFT}

You receive ONE episode as timed blocks. Each block has an ID (b001, b002, ...). SEE = what is on screen,
ACTION = what characters do, SOUND = sound effects; dialogue appears as SPEAKER (delivery): "line".

Return up to 3 promo candidates for this episode, best first, plus 2-5 "beats": short, intense,
self-explanatory moments (3-15 s) that could be joined with beats from other episodes into a montage.

Rules:
- Refer to blocks ONLY by ID. Never write timestamps: code turns IDs into exact cut points and checks them.
- A segment runs from the START of its start_id block to the END of its end_id block.
- A promo may join up to 3 segments from this episode to skip slow parts (story order).
- hook_id: the block that stops the scroll. It must START within the first 3 seconds of the first segment,
  so start the first segment on it or one short block before it.
- cliffhanger_id: the last strong unresolved beat; the final segment must end on it or right after it.
- Scores are 1-10 and must be honest: 9-10 only for a moment a professional would post today.
- context_gap: what a new viewer would NOT understand from the clip alone ("" if nothing).
- If this episode has nothing promo-worthy, return the best you can with honest low scores."""

MINE_USER = """SHOW: {title}
EPISODE {number} of {total} - runtime {runtime}
Background for you only (a new viewer will NOT know this): {logline} {continuity}

TRANSCRIPT
{transcript}"""

MINE_SCHEMA = {
    "type": "object",
    "properties": {
        "promos": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "segments": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"start_id": {"type": "string"}, "end_id": {"type": "string"}},
                    "required": ["start_id", "end_id"], "additionalProperties": False}},
                "hook_id": {"type": "string"},
                "hook": {"type": "string"},
                "cliffhanger_id": {"type": "string"},
                "cliffhanger": {"type": "string"},
                "why": {"type": "string"},
                "hook_type": {"type": "string", "enum": HOOK_TYPES},
                "hook_score": {"type": "integer"},
                "cliffhanger_score": {"type": "integer"},
                "standalone_score": {"type": "integer"},
                "context_gap": {"type": "string"},
            },
            "required": ["title", "segments", "hook_id", "hook", "cliffhanger_id", "cliffhanger", "why",
                         "hook_type", "hook_score", "cliffhanger_score", "standalone_score", "context_gap"],
            "additionalProperties": False}},
        "beats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "start_id": {"type": "string"}, "end_id": {"type": "string"},
                "what": {"type": "string"},
                "role": {"type": "string", "enum": ["hook", "setup", "escalation", "cliffhanger"]},
                "intensity": {"type": "integer"}},
            "required": ["start_id", "end_id", "what", "role", "intensity"], "additionalProperties": False}},
    },
    "required": ["promos", "beats"],
    "additionalProperties": False,
}

FIX_USER = """Your previous answer for this episode had problems that code could not fix:
{problems}

Return the full JSON again with corrected promos (same rules). Use the [start-end] times in the
transcript to keep every promo between 20 and 60 s in total and the hook inside the first 3 s."""

COLD_SYSTEM = """You are an ordinary person scrolling TikTok / Instagram Reels. You have NEVER seen or heard of this
show. Below is everything in a short clip that just appeared in your feed: SEE lines are what is on
screen, the other lines are what you hear. Times are seconds from the start of the clip; "--- cut ---"
marks a jump to another moment. Judge ONLY from what is in the clip. Be blunt: most clips do not make
people stop."""

COLD_USER = """CLIP ({duration:.0f} s)
{clip}"""

COLD_SCHEMA = {
    "type": "object",
    "properties": {
        "what_happens": {"type": "string"},
        "who": {"type": "string"},
        "first_3_seconds": {"type": "string"},
        "stop_scrolling": {"type": "boolean"},
        "question_left": {"type": "string"},
        "confusing": {"type": "string"},
        "contradictions": {"type": "string"},
        "hook": {"type": "integer"},
        "cliffhanger": {"type": "integer"},
        "clarity": {"type": "integer"},
        "want_next_episode": {"type": "boolean"},
    },
    "required": ["what_happens", "who", "first_3_seconds", "stop_scrolling", "question_left", "confusing",
                 "contradictions", "hook", "cliffhanger", "clarity", "want_next_episode"],
    "additionalProperties": False,
}
COLD_SCALE = ("Scores are 1-5: hook = how strongly the first 3 seconds stop you; cliffhanger = how badly you "
              "need to know what happens next; clarity = how well you understand who wants what without "
              "having seen the show. confusing = what you did not understand (\"\" if nothing). "
              "contradictions = anything in the clip that seems to contradict itself, such as the same person "
              "appearing to be two people or one event looking like two (\"\" if nothing; do not list things "
              "that are merely unexplained).")

MONTAGE_SYSTEM = f"""You are a senior promo editor for vertical microdrama apps. Build CROSS-EPISODE promos: join
2-4 short beats from different episodes of the same show into one 20-60 second clip.

{CRAFT}

Extra rules for montages:
- The joined beats must read as ONE clear mini-story to a stranger (for example humiliation -> rescue ->
  threat), not a trailer of unrelated moments. Jumps in place or costume must not confuse.
- Be TRUE to the show. Never splice beats so that two different people look like one person, or two
  different events look like one (for example two different weddings, or two different villains in
  similar gowns). A viewer who then watches the show must not feel tricked. Use the series background
  to check who is who in each beat.
- The first beat must hook in its first 3 seconds; the last beat must end unresolved.
- Use each beat's given duration; the sum must be 20-60 s.
- Refer to beats only by ID. Propose up to {{n}} montages, best first."""

MONTAGE_USER = """SHOW: {title}
Series background for you only: {background}

BEATS (id | episode | seconds | role | intensity | what happens)
{beats}"""

MONTAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "promos": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "beat_ids": {"type": "array", "items": {"type": "string"}},
                "hook": {"type": "string"},
                "cliffhanger": {"type": "string"},
                "why": {"type": "string"},
                "hook_type": {"type": "string", "enum": HOOK_TYPES},
                "hook_score": {"type": "integer"},
                "cliffhanger_score": {"type": "integer"},
                "standalone_score": {"type": "integer"},
                "context_gap": {"type": "string"},
            },
            "required": ["title", "beat_ids", "hook", "cliffhanger", "why", "hook_type", "hook_score",
                         "cliffhanger_score", "standalone_score", "context_gap"],
            "additionalProperties": False}},
    },
    "required": ["promos"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = f"""You are a performance-marketing lead at a microdrama app. You decide which promo clip to post
to pull in NEW viewers who have never seen the show. Two candidate clips are shown as transcripts
(SEE = on screen; other lines = what you hear; times in seconds from the clip start).

{CRAFT}

Compare the two clips on each criterion, then pick the one you would post. Ties are allowed but rare.
Judge the content, not the length, unless a clip breaks the 20-60 s rule."""

JUDGE_USER = """CLIP A ({da:.0f} s)
{a}

CLIP B ({db:.0f} s)
{b}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string", "enum": ["A", "B", "tie"]},
        "cliffhanger": {"type": "string", "enum": ["A", "B", "tie"]},
        "standalone": {"type": "string", "enum": ["A", "B", "tie"]},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["hook", "cliffhanger", "standalone", "winner", "reason"],
    "additionalProperties": False,
}

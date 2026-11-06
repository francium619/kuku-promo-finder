"""Tests on a two-episode sample show written for this repo (no Kuku data). Run: python -m unittest -v"""
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promo_finder import pipeline  # noqa: E402
from promo_finder.heuristics import baseline_ending, baseline_opening, candidate_windows  # noqa: E402
from promo_finder.llm import LLM, extract_json  # noqa: E402
from promo_finder.report import write_all  # noqa: E402
from promo_finder.screenplay import fmt_ts, load_show, parse_ts  # noqa: E402
from promo_finder.timeline import timeline  # noqa: E402

SAMPLE = ROOT / "tests" / "sample_show"


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.show = load_show(SAMPLE)

    def test_episodes_and_durations(self):
        self.assertEqual([e.number for e in self.show.episodes], [1, 2])
        self.assertEqual(self.show.episode(1).duration, 48.0)
        self.assertEqual(self.show.episode(2).duration, 30.0)

    def test_uppercase_scene_fields(self):
        sc = self.show.episode(2).scenes[0]
        self.assertTrue(sc.heading.startswith("EXT. HARBOUR"))
        self.assertEqual((sc.start, sc.end), (0.0, 30.0))

    def test_labels_alias_extension_and_tags(self):
        ep = self.show.episode(2)
        stranger = next(b for b in ep.blocks if b.speaker == "WET-COAT STRANGER")
        self.assertEqual(stranger.alias, "Elias")
        self.assertEqual(stranger.tags, ["Grim"])
        theo = next(b for b in ep.blocks if b.speaker == "THEO")
        self.assertEqual(theo.tags, ["Shouting"])  # (O.S.) is an extension, not a tag
        self.assertEqual(theo.alias, "")

    def test_quotes_stripped_and_time_order(self):
        ep = self.show.episode(1)
        first_line = next(b for b in ep.blocks if b.kind == "dialogue")
        self.assertEqual(first_line.text, "To the lighthouse, and to us!")
        starts = [b.start for b in ep.blocks]
        self.assertEqual(starts, sorted(starts))

    def test_timestamps(self):
        self.assertEqual(parse_ts("01:15.50"), 75.5)
        self.assertEqual(parse_ts("1:02:03.5"), 3723.5)
        self.assertEqual(fmt_ts(75.5), "01:15.50")

    def test_zip_input(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "Sample Show-scripts (2).zip"
            with zipfile.ZipFile(z, "w") as zf:
                for f in SAMPLE.glob("*.txt"):
                    zf.write(f, f"Sample Show/{f.name}")
            show = load_show(z)
            self.assertEqual(show.title, "Sample Show")
            self.assertEqual(len(show.episodes), 2)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.ep = load_show(SAMPLE).episode(1)
        self.tl = timeline(self.ep)

    def test_snaps_out_of_a_line(self):
        self.assertLessEqual(self.tl.snap_start(9.0), 8.0)       # inside THEO 8.0-10.6
        self.assertGreaterEqual(self.tl.snap_end(9.0), 10.6)

    def test_never_cuts_inside_a_line(self):
        for i in range(0, 481):
            t = i / 10
            self.assertTrue(self.tl.is_clean(self.tl.snap_start(t)), t)
            self.assertTrue(self.tl.is_clean(self.tl.snap_end(t)), t)

    def test_prefers_camera_cuts(self):
        self.assertEqual(self.tl.snap_start(8.0), 7.5)   # camera cut 0.5 s before the line
        self.assertEqual(self.tl.snap_end(39.5), 40.0)   # camera cut 0.5 s after the line


class HeuristicTests(unittest.TestCase):
    def test_windows_and_baselines_in_range(self):
        ep = load_show(SAMPLE).episode(1)
        wins = list(candidate_windows(ep))
        self.assertTrue(wins)
        self.assertTrue(all(20 <= e - s <= 60 for s, e in wins))
        for seg in (baseline_opening(ep), baseline_ending(ep)):
            self.assertTrue(20 <= seg.duration <= 60, seg)


def fake_model(system, user, schema, tag):
    if tag == "mine_ep001":
        return {"promos": [
            {"title": "ok", "segments": [{"start_id": "b012", "end_id": "b020"}], "hook_id": "b012", "hook": "h",
             "cliffhanger_id": "b020", "cliffhanger": "c", "why": "w", "hook_type": "betrayal",
             "hook_score": 8, "cliffhanger_score": 8, "standalone_score": 8, "context_gap": ""},
            {"title": "slow open", "segments": [{"start_id": "b001", "end_id": "b013"}], "hook_id": "b005",
             "hook": "h", "cliffhanger_id": "b014", "cliffhanger": "c", "why": "w", "hook_type": "reveal",
             "hook_score": 6, "cliffhanger_score": 6, "standalone_score": 6, "context_gap": ""},
            {"title": "bad id", "segments": [{"start_id": "b001", "end_id": "b099"}], "hook_id": "b001",
             "hook": "h", "cliffhanger_id": "b099", "cliffhanger": "c", "why": "w", "hook_type": "other",
             "hook_score": 5, "cliffhanger_score": 5, "standalone_score": 5, "context_gap": ""}],
            "beats": [{"start_id": "b010", "end_id": "b010", "what": "deed", "role": "hook", "intensity": 8},
                      {"start_id": "b019", "end_id": "b019", "what": "stranger", "role": "cliffhanger", "intensity": 9}]}
    if tag == "fix_ep001":
        assert "b099" in user  # the problem is reported back to the model
        return {"promos": [
            {"title": "ok again", "segments": [{"start_id": "b012", "end_id": "b020"}], "hook_id": "b012",
             "hook": "h", "cliffhanger_id": "b020", "cliffhanger": "c", "why": "w", "hook_type": "betrayal",
             "hook_score": 8, "cliffhanger_score": 8, "standalone_score": 8, "context_gap": ""}], "beats": []}
    if tag == "mine_ep002":
        return {"promos": [
            {"title": "storm", "segments": [{"start_id": "b001", "end_id": "b011"}], "hook_id": "b002", "hook": "h",
             "cliffhanger_id": "b011", "cliffhanger": "c", "why": "w", "hook_type": "threat",
             "hook_score": 7, "cliffhanger_score": 9, "standalone_score": 7, "context_gap": ""}],
            "beats": [{"start_id": "b004", "end_id": "b004", "what": "debt", "role": "setup", "intensity": 6},
                      {"start_id": "b009", "end_id": "b011", "what": "burn it", "role": "cliffhanger", "intensity": 9}]}
    if tag == "montage":
        return {"promos": [
            {"title": "m", "beat_ids": ["E01.k1", "E01.k2", "E02.k1", "E02.k2"], "hook": "h", "cliffhanger": "c",
             "why": "w", "hook_type": "betrayal", "hook_score": 7, "cliffhanger_score": 8, "standalone_score": 6,
             "context_gap": ""},
            {"title": "too short", "beat_ids": ["E01.k1", "E02.k1"], "hook": "h", "cliffhanger": "c", "why": "w",
             "hook_type": "other", "hook_score": 5, "cliffhanger_score": 5, "standalone_score": 5, "context_gap": ""}]}
    if tag.startswith("cold_"):
        return {"what_happens": "x", "who": "x", "first_3_seconds": "x", "stop_scrolling": True,
                "question_left": "what next?", "confusing": "", "contradictions": "two doors" if tag == "cold_M1" else "",
                "hook": 4, "cliffhanger": 4, "clarity": 4, "want_next_episode": True}
    raise AssertionError(f"unexpected call {tag}")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.show = load_show(SAMPLE)
        self.llm = LLM("fake", cache_dir=None)
        self.llm.fake = fake_model
        self.result = pipeline.run(self.show, self.llm, workers=2, log=lambda *a: None)

    def test_every_promo_obeys_the_rules(self):
        self.assertTrue(self.result.promos)
        for p in self.result.promos:
            self.assertTrue(20 <= p.duration <= 60, (p.pid, p.duration))
            for s in p.segments:
                tl = timeline(self.show.episode(s.episode))
                self.assertTrue(tl.is_clean(s.start) and tl.is_clean(s.end), (p.pid, s))
            if p.kind == "single":
                self.assertLessEqual(p.hook_at, 3.0, p.pid)

    def test_validation_outcomes(self):
        st = {(e["title"], e["status"]) for e in self.result.events}
        self.assertIn(("ok", "accepted"), st)
        self.assertIn(("slow open", "repaired"), st)       # code trimmed 7.5 s of slow opening
        self.assertIn(("bad id", "rejected"), st)          # sent back to the model ...
        self.assertIn(("ok again", "duplicate"), st)       # ... whose fix duplicated an accepted promo
        self.assertIn(("too short", "rejected"), st)

    def test_montage_and_contradiction_flag(self):
        m = [p for p in self.result.promos if p.kind == "montage"]
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].episodes, [1, 2])
        self.assertTrue(any("contradiction" in f for f in m[0].flags))

    def test_malformed_reply_falls_back_to_rules(self):
        llm = LLM("fake", cache_dir=None)
        llm.fake = lambda s, u, sc, tag: ({"promos": "garbage", "beats": None} if tag == "mine_ep001"
                                          else fake_model(s, u, sc, tag))
        res = pipeline.run(self.show, llm, workers=1, montage=False, log=lambda *a: None)
        self.assertTrue(any(e["status"] == "model-failed" and e["episode"] == 1 for e in res.events))
        self.assertTrue(any(p.kind == "rules" and p.episodes == [1] for p in res.promos + res.duplicates))

    def test_fatal_model_error_stops_the_run(self):
        from promo_finder.llm import LLMFatal

        def broken(*a):
            raise LLMFatal("unknown model")
        llm = LLM("fake", cache_dir=None, retries=3)
        llm.fake = broken
        with self.assertRaises(LLMFatal):
            pipeline.run(self.show, llm, workers=1, log=lambda *a: None)
        self.assertLessEqual(llm.usage["failures"], 2)  # at most one per episode: no retries (would be 4 each)

    def test_geometric_mean_punishes_a_weak_criterion(self):
        a = pipeline.Promo("a", "single", [], model_scores={"hook": 6, "cliffhanger": 6, "standalone": 6})
        b = pipeline.Promo("b", "single", [], model_scores={"hook": 10, "cliffhanger": 10, "standalone": 1})
        a.segments = b.segments = [pipeline.Segment(1, 0, 30)]
        self.assertGreater(pipeline.composite(a, (1, 1), 0), pipeline.composite(b, (1, 1), 0))

    def test_reports_written(self):
        with tempfile.TemporaryDirectory() as d:
            write_all(self.result, Path(d), "test")
            for name in ("promos.json", "promos_ranked.csv", "top_promos.md", "review.html", "validation_log.json"):
                self.assertTrue((Path(d) / name).exists(), name)
            html = (Path(d) / "review.html").read_text(encoding="utf-8")
            self.assertNotIn("/*__DATA__*/null", html)
            data = json.loads((Path(d) / "promos.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["post_first"]), len(self.result.posting))


class _Handler(http.server.BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append({"auth": self.headers.get("Authorization"), "body": body})
        if "response_format" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format is not supported"}')
            return
        reply = {"choices": [{"message": {"content": 'Sure!\n```json\n{"answer": "ok", "score": 7}\n```'}}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *a):
        pass


class LLMTests(unittest.TestCase):
    def test_preflight_catches_bad_configuration(self):
        from promo_finder.llm import LLMFatal
        os.environ.pop("PF_MISSING_KEY", None)
        with self.assertRaises(LLMFatal):
            LLM("compat:x", cache_dir=None).preflight()                       # no --base-url
        with self.assertRaises(LLMFatal):
            LLM("compat:x", base_url="http://127.0.0.1:9/v1", api_key_env="PF_MISSING_KEY",
                cache_dir=None).preflight()                                   # key variable not set
        LLM("none").preflight()                                               # rules-only needs nothing

    def test_extract_json_tolerates_chatter(self):
        self.assertEqual(extract_json('Here:\n```json\n{"a": [1, 2,], "b": "x}"}\n```'), {"a": [1, 2], "b": "x}"})

    def test_openai_compatible_endpoint_swap_and_cache(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        os.environ["PF_TEST_KEY"] = "secret"
        schema = {"type": "object", "properties": {"answer": {"type": "string"}, "score": {"type": "integer"}},
                  "required": ["answer", "score"]}
        with tempfile.TemporaryDirectory() as d:
            llm = LLM("compat:test-model", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                      api_key_env="PF_TEST_KEY", cache_dir=d)
            self.assertEqual(llm.complete_json("sys", "user", schema, "t"), {"answer": "ok", "score": 7})
            self.assertEqual(_Handler.requests[0]["auth"], "Bearer secret")
            self.assertNotIn("response_format", _Handler.requests[-1]["body"])   # dropped after the 400
            self.assertEqual(llm.usage["input_tokens"], 11)
            n = len(_Handler.requests)
            llm.complete_json("sys", "user", schema, "t")                       # second call: cache hit
            self.assertEqual(len(_Handler.requests), n)
            self.assertEqual(llm.usage["cache_hits"], 1)
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    unittest.main()

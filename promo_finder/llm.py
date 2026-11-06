"""One small interface over many model providers, so swapping the model is one flag.

    --model none                         rules only, no model
    --model claude-cli:claude-sonnet-5-5 Claude Code CLI (uses your Claude login; no API key)
    --model anthropic:claude-sonnet-5-5  Anthropic API (ANTHROPIC_API_KEY)
    --model openai:gpt-4o-mini           OpenAI (OPENAI_API_KEY)
    --model gemini:gemini-2.5-flash      Gemini free tier, OpenAI-compatible endpoint (GEMINI_API_KEY)
    --model groq:llama-3.3-70b-versatile Groq (GROQ_API_KEY)
    --model openrouter:<model>           OpenRouter (OPENROUTER_API_KEY)
    --model ollama:llama3.1              local Ollama, no key
    --model compat:<model> --base-url URL --api-key-env VAR
                                         any OpenAI-compatible gateway (e.g. one Kuku provides)

Every call asks for JSON that matches a schema. Responses are cached on disk keyed by
(model, prompts, schema), so re-running on the same input is free and repeatable.
Standard library only, except the anthropic provider, which uses the official SDK.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "compat": (None, None),
}

# USD per million tokens (input, output), Anthropic list prices as of 2026-10, for the cost
# estimate only (the Batch API halves them). Edit, or pass --price in,out for other models.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(RuntimeError):
    pass


class LLMFatal(LLMError):
    """An error a retry cannot fix (unknown model, bad key, CLI too old): fail fast."""


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    raise ValueError("unterminated JSON object in reply")


def check_required(data, schema) -> None:
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise ValueError(f"reply is missing keys: {missing}")


class LLM:
    def __init__(self, spec: str = "none", base_url: str | None = None, api_key_env: str | None = None,
                 cache_dir: str | None = ".cache/llm", log_dir: str | None = None, timeout: int = 420,
                 retries: int = 2, effort: str | None = None, price: tuple | None = None):
        provider, _, model = (spec or "none").partition(":")
        self.provider, self.model = provider.strip().lower(), model.strip()
        self.spec = f"{self.provider}:{self.model}" if self.model else self.provider
        if self.provider not in OPENAI_COMPAT and self.provider not in ("none", "claude-cli", "anthropic", "fake"):
            raise LLMError(f"unknown provider '{self.provider}' in --model {spec}")
        if self.provider not in ("none", "fake") and not self.model:
            raise LLMError(f"--model {spec}: give a model name after the colon")
        self.base_url, self.api_key_env = base_url, api_key_env
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.timeout, self.retries, self.effort = timeout, retries, effort
        self.price = price or PRICES.get(self.model)
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0,
                      "cost_usd": 0.0, "model_seconds": 0.0, "failures": 0}
        self._lock = threading.Lock()
        self._optional = {"temperature": 0.3, "response_format": {"type": "json_object"}}
        self.fake = None  # tests: callable(system, user, schema, tag) -> dict

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    # ---- public -------------------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema: dict, tag: str = "call") -> dict:
        key = hashlib.sha256(json.dumps([self.spec, self.base_url, system, user, schema],
                                        sort_keys=True).encode()).hexdigest()[:32]
        if self.cache_dir:
            hit = self.cache_dir / f"{key}.json"
            if hit.exists():
                rec = json.loads(hit.read_text(encoding="utf-8"))
                u = rec.get("usage", {})
                with self._lock:
                    self.usage["cache_hits"] += 1
                    # what producing this cached answer originally cost, so re-runs still report it
                    self.usage["cached_input_tokens"] = self.usage.get("cached_input_tokens", 0) + u.get("input_tokens", 0)
                    self.usage["cached_output_tokens"] = self.usage.get("cached_output_tokens", 0) + u.get("output_tokens", 0)
                    if self.price:
                        self.usage["cached_cost_usd"] = round(self.usage.get("cached_cost_usd", 0.0) + (
                            u.get("input_tokens", 0) * self.price[0] + u.get("output_tokens", 0) * self.price[1]) / 1e6, 6)
                self._log(tag, rec["data"], rec.get("usage", {}), cached=True)
                return rec["data"]
        last = None
        for attempt in range(self.retries + 1):
            try:
                t0 = time.time()
                data, usage = self._call(system, user, schema, tag)
                check_required(data, schema)
                usage["seconds"] = round(time.time() - t0, 2)
                self._account(usage)
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / f"{key}.json").write_text(
                        json.dumps({"spec": self.spec, "tag": tag, "data": data, "usage": usage}, ensure_ascii=False),
                        encoding="utf-8")
                self._log(tag, data, usage, cached=False)
                return data
            except LLMFatal:
                with self._lock:
                    self.usage["failures"] += 1
                raise
            except (LLMError, ValueError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired,
                    urllib.error.URLError, OSError) as e:
                last = e
                with self._lock:
                    self.usage["failures"] += 1
                if attempt < self.retries:
                    time.sleep(min(30, 3 * (2 ** attempt)))
        raise LLMError(f"{self.spec} failed on {tag} after {self.retries + 1} tries: {last}")

    def preflight(self) -> None:
        """Fail before any work if the model cannot possibly be called (missing key, URL, CLI or SDK)."""
        if self.provider in ("none", "fake"):
            return
        if self.provider == "claude-cli":
            self._claude_bin()
        elif self.provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                raise LLMFatal("--model anthropic:... needs the official SDK: pip install anthropic") from None
            if self.api_key_env:
                self._key(self.api_key_env)
        else:
            base, env = OPENAI_COMPAT[self.provider]
            if not (self.base_url or base):
                raise LLMFatal("--model compat:<name> needs --base-url")
            self._key(env)

    # ---- providers ----------------------------------------------------------------------
    def _call(self, system, user, schema, tag):
        if self.provider == "fake":
            return self.fake(system, user, schema, tag), {"input_tokens": 0, "output_tokens": 0}
        if self.provider == "claude-cli":
            return self._claude_cli(system, user, schema)
        if self.provider == "anthropic":
            return self._anthropic(system, user, schema)
        if self.provider in OPENAI_COMPAT:
            return self._openai_compat(system, user, schema)
        raise LLMError("no model configured (--model none)")

    @staticmethod
    def _schema_note(schema) -> str:
        return ("\n\nReply with ONE JSON object only, no prose, matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False))

    def _claude_bin(self) -> str:
        if os.environ.get("CLAUDE_BIN"):
            return os.environ["CLAUDE_BIN"]
        found = shutil.which("claude")
        if not found:
            raise LLMFatal("claude CLI not found: install Claude Code or pick another --model")
        p = Path(found)
        exe = p.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if p.suffix.lower() in (".cmd", ".ps1", "") and exe.exists():
            return str(exe)  # call the binary directly; .cmd shims mangle quotes in long prompts
        return found

    def _claude_cli(self, system, user, schema):
        cmd = [self._claude_bin(), "-p", "--model", self.model, "--output-format", "json",
               "--json-schema", json.dumps(schema), "--system-prompt", system, "--tools", "",
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
        if self.effort:
            cmd += ["--effort", self.effort]
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout)
        line = next((ln for ln in proc.stdout.splitlines() if ln.lstrip().startswith("{")), None)
        if not line:
            raise LLMError(f"claude CLI gave no JSON (exit {proc.returncode}): {proc.stderr[-400:]}")
        out = json.loads(line)
        if out.get("is_error"):
            msg = f"claude CLI error: {str(out.get('result'))[:400]}"
            if re.search(r"hit your .{0,30}limit|usage limit", msg, re.I):
                raise LLMFatal(msg + " (plan usage limit: wait for the reset or use another --model)")
            if out.get("api_error_status") in (400, 401, 403, 404):
                raise LLMFatal(msg + " (set CLAUDE_BIN to a newer Claude Code if the model is unsupported)")
            raise LLMError(msg)
        data = out.get("structured_output")
        if data is None:
            data = extract_json(out.get("result") or "")
        u = out.get("usage") or {}
        usage = {"input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                 + (u.get("cache_read_input_tokens") or 0),
                 "output_tokens": u.get("output_tokens") or 0,
                 "cli_cost_usd": out.get("total_cost_usd")}
        return data, usage

    def _post(self, url, body, headers):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:600]
            finally:
                e.close()
            kind = LLMFatal if e.code in (400, 401, 403, 404) else LLMError
            raise kind(f"HTTP {e.code} from {url}: {detail}") from None

    def _key(self, default_env):
        env = self.api_key_env or default_env
        if not env:
            return None
        val = os.environ.get(env)
        if not val:
            raise LLMFatal(f"set {env} to use --model {self.spec}")
        return val

    def _openai_compat(self, system, user, schema):
        base, env = OPENAI_COMPAT[self.provider]
        base = self.base_url or base
        if not base:
            raise LLMFatal("--model compat:<name> needs --base-url")
        key = self._key(env)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user + self._schema_note(schema)}]
        while True:
            body = {"model": self.model, "messages": msgs, **self._optional}
            try:
                resp = self._post(base.rstrip("/") + "/chat/completions", body, headers)
                break
            except LLMError as e:
                # Some models reject temperature or JSON mode; drop the optional knob and retry.
                dropped = next((k for k in list(self._optional) if k in str(e) and "HTTP 400" in str(e)), None)
                if not dropped:
                    raise
                with self._lock:
                    self._optional.pop(dropped, None)
        text = resp["choices"][0]["message"]["content"] or ""
        u = resp.get("usage") or {}
        return extract_json(text), {"input_tokens": u.get("prompt_tokens", 0),
                                    "output_tokens": u.get("completion_tokens", 0)}

    def _anthropic(self, system, user, schema):
        try:
            import anthropic  # official SDK: pip install anthropic
        except ImportError:
            raise LLMFatal("--model anthropic:... needs the official SDK: pip install anthropic") from None
        kwargs = {"base_url": self.base_url} if self.base_url else {}
        if self.api_key_env:
            kwargs["api_key"] = self._key(self.api_key_env)
        client = anthropic.Anthropic(max_retries=2, timeout=self.timeout, **kwargs)
        try:
            msg = client.messages.create(model=self.model, max_tokens=16000, system=system,
                                         messages=[{"role": "user", "content": user + self._schema_note(schema)}])
        except anthropic.APIStatusError as e:
            kind = LLMFatal if e.status_code in (400, 401, 403, 404) else LLMError
            raise kind(f"Anthropic API {e.status_code}: {str(e)[:400]}") from None
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Anthropic API connection error: {e}") from None
        if msg.stop_reason == "refusal":
            raise LLMError("model declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return extract_json(text), {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}

    # ---- bookkeeping --------------------------------------------------------------------
    def _account(self, usage):
        cost = None
        if self.price:
            cost = (usage.get("input_tokens", 0) * self.price[0] + usage.get("output_tokens", 0) * self.price[1]) / 1e6
        elif usage.get("cli_cost_usd") is not None:
            cost = usage["cli_cost_usd"]
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += usage.get("input_tokens", 0)
            self.usage["output_tokens"] += usage.get("output_tokens", 0)
            self.usage["model_seconds"] = round(self.usage["model_seconds"] + usage.get("seconds", 0.0), 2)
            if cost is not None:
                self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 6)

    def _log(self, tag, data, usage, cached):
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
        (self.log_dir / f"{safe}.json").write_text(
            json.dumps({"model": self.spec, "tag": tag, "cached": cached, "usage": usage, "response": data},
                       ensure_ascii=False, indent=1), encoding="utf-8")

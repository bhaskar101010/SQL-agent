"""Two-tier LLM router with cohort fallback and per-cohort key rotation.

Tiers:
  STRONG  -> SQL generation / optimize / debug / explain
  CHEAP   -> intent classify / injection guard

Cohort order (per tier): Gemini (round-robin keys) -> Groq -> OpenRouter.
On any provider error, the next cohort is tried. Within a Gemini cohort, a
failing key advances the rotation before the cohort is marked failed.

Usage:
    from app.llm_router import LLMRouter, Tier
    router = LLMRouter()
    text = router.chat(Tier.STRONG, system="...", user="...", temperature=0.0)
"""
from __future__ import annotations

import os
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


class Tier(str, Enum):
    STRONG = "strong"
    CHEAP = "cheap"


# ---------------------------------------------------------------------------
# Provider adapters -- each returns a plain string response or raises.
# ---------------------------------------------------------------------------

_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))


def _call_gemini(api_key: str, model: str, system: str, user: str,
                 temperature: float) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model, system_instruction=system or None)
    resp = m.generate_content(
        user,
        generation_config={"temperature": temperature},
        request_options={"timeout": _REQUEST_TIMEOUT},
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty gemini response")
    return text


def _call_groq(api_key: str, model: str, system: str, user: str,
               temperature: float) -> str:
    from groq import Groq
    client = Groq(api_key=api_key, timeout=_REQUEST_TIMEOUT)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty groq response")
    return text


def _call_openrouter(api_key: str, model: str, system: str, user: str,
                     temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                    timeout=_REQUEST_TIMEOUT)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty openrouter response")
    return text


# ---------------------------------------------------------------------------
# Cohort definition
# ---------------------------------------------------------------------------

@dataclass
class Cohort:
    name: str
    keys: list[str]                                     # one or more api keys
    model: str
    call: Callable[[str, str, str, str, float], str]   # (key, model, sys, user, temp) -> text
    min_interval: float = 0.0                           # min seconds between any two calls in this cohort
    cooldown_on_429: float = 60.0                       # skip this cohort for N seconds after a 429
    _cycle: Optional[Iterable[str]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_call_ts: float = 0.0
    _cooldown_until: float = 0.0

    def __post_init__(self):
        self._cycle = itertools.cycle(self.keys) if self.keys else None

    @property
    def available(self) -> bool:
        return bool(self.keys) and bool(self.model)

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def _throttle(self) -> None:
        """Block until min_interval has passed since the last call in this cohort."""
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.time() - self._last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call_ts = time.time()

    def try_call(self, system: str, user: str, temperature: float) -> str:
        """Try each key once (round-robin); raise if every key fails.

        On a rate-limit (429) failure, mark the cohort as cooling down so the
        router skips us for `cooldown_on_429` seconds instead of hammering.
        """
        if not self.available:
            raise RuntimeError(f"cohort {self.name} not configured")
        if self.in_cooldown:
            raise RuntimeError(
                f"cohort {self.name} in cooldown "
                f"({int(self._cooldown_until - time.time())}s remaining)"
            )
        last_err: Optional[Exception] = None
        for _ in range(len(self.keys)):
            self._throttle()
            key = next(self._cycle)
            try:
                return self.call(key, self.model, system, user, temperature)
            except Exception as e:  # noqa: BLE001 -- any provider error -> next key
                last_err = e
                log.debug("cohort=%s key=***%s error=%s",
                          self.name, key[-4:], e)
                # If this looks like a rate limit, trip the cohort cooldown
                # right away — no point trying the other keys in this project.
                es = str(e).lower()
                if ("429" in es or "quota" in es or "rate limit" in es
                        or "resourceexhausted" in es):
                    self._cooldown_until = time.time() + self.cooldown_on_429
                    log.info("cohort=%s cooldown %.0fs (rate limited)",
                             self.name, self.cooldown_on_429)
                    break
        raise RuntimeError(f"cohort {self.name} exhausted: {last_err}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Strip whitespace and surrounding single/double quotes defensively."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()


def _split_csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [c for c in (_clean(x) for x in v.split(",")) if c]


class LLMRouter:
    def __init__(self) -> None:
        gemini_keys = _split_csv(os.getenv("GEMINI_API_KEYS"))
        groq_key = _clean(os.getenv("GROQ_API_KEY", ""))
        openrouter_key = _clean(os.getenv("OPENROUTER_API_KEY", ""))

        # Per-cohort min interval (seconds). Gemini free tier is 10 RPM per
        # project (keys share quota), so a 7s floor keeps us safely under.
        gemini_delay = float(os.getenv("GEMINI_MIN_INTERVAL", "7"))
        groq_delay = float(os.getenv("GROQ_MIN_INTERVAL", "0"))
        openrouter_delay = float(os.getenv("OPENROUTER_MIN_INTERVAL", "0"))

        # Cohort order: Groq first (fast, no per-call throttle), then Gemini
        # (slower + daily quota + 7s floor), then OpenRouter free as last resort.
        self._chains: dict[Tier, list[Cohort]] = {
            Tier.STRONG: [
                Cohort("groq",
                       [groq_key] if groq_key else [],
                       os.getenv("GROQ_STRONG_MODEL", "openai/gpt-oss-120b"),
                       _call_groq,
                       min_interval=groq_delay),
                Cohort("gemini",
                       gemini_keys,
                       os.getenv("GEMINI_STRONG_MODEL", "gemini-2.5-flash"),
                       _call_gemini,
                       min_interval=gemini_delay),
                Cohort("openrouter",
                       [openrouter_key] if openrouter_key else [],
                       os.getenv("OPENROUTER_STRONG_MODEL",
                                 "deepseek/deepseek-chat-v3-0324:free"),
                       _call_openrouter,
                       min_interval=openrouter_delay),
            ],
            Tier.CHEAP: [
                Cohort("groq",
                       [groq_key] if groq_key else [],
                       os.getenv("GROQ_CHEAP_MODEL", "openai/gpt-oss-20b"),
                       _call_groq,
                       min_interval=groq_delay),
                Cohort("gemini",
                       gemini_keys,
                       os.getenv("GEMINI_CHEAP_MODEL", "gemini-2.5-flash-lite"),
                       _call_gemini,
                       min_interval=gemini_delay),
                Cohort("openrouter",
                       [openrouter_key] if openrouter_key else [],
                       os.getenv("OPENROUTER_CHEAP_MODEL",
                                 "google/gemini-2.0-flash-exp:free"),
                       _call_openrouter,
                       min_interval=openrouter_delay),
            ],
        }

    def available_cohorts(self, tier: Tier) -> list[str]:
        return [c.name for c in self._chains[tier] if c.available]

    def chat(self, tier: Tier, *, system: str = "", user: str,
             temperature: float = 0.0) -> str:
        chain = [c for c in self._chains[tier] if c.available]
        if not chain:
            raise RuntimeError(
                f"no cohort configured for tier={tier}. "
                "fill in .env (GEMINI_API_KEYS / GROQ_API_KEY / OPENROUTER_API_KEY)."
            )
        last_err: Optional[Exception] = None
        for cohort in chain:
            try:
                return cohort.try_call(system, user, temperature)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.info("tier=%s falling through from %s", tier.value, cohort.name)
        raise RuntimeError(f"all cohorts failed for tier={tier.value}: {last_err}")

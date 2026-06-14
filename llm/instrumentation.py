"""Lightweight call instrumentation for LLM / VLM endpoints.

Writes one JSONL record per call to ``$SKILLNAV_LOG_DIR`` (default
``/tmp/skillnav_logs``), split by run via ``$SKILLNAV_RUN_ID``.

The aggregator in ``scripts/aggregate_calls.py`` post-processes these
files into the per-episode efficiency table reported in the paper.

Token counts are character-based approximations (``~4 chars / token``
for English, ``~2`` for CJK). Exact counts from model APIs (DeepSeek
``response.usage``, Ollama ``prompt_eval_count``) are honoured when the
caller passes them through. We document this caveat in the paper.

Thread safety: a single ``threading.Lock`` guards file writes within a
process. Concurrent processes must use distinct ``$SKILLNAV_RUN_ID``
to avoid interleaving.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, Tuple

_LOG_DIR = os.environ.get("SKILLNAV_LOG_DIR", "/tmp/skillnav_logs")
_RUN_ID = os.environ.get("SKILLNAV_RUN_ID", "default")
_INSTRUMENTATION_ENABLED = os.environ.get("SKILLNAV_INSTRUMENT", "1") != "0"

_episode_id: Optional[int] = None
_episode_scene: Optional[str] = None
_episode_target: Optional[str] = None
_option_state: Optional[str] = None
_lock = threading.Lock()


# ---------- mutators called from habitat_evaluation.py ----------

def set_episode(eid: int, scene: Optional[str] = None, target: Optional[str] = None) -> None:
    """Record the current episode context. Call once at episode start."""
    global _episode_id, _episode_scene, _episode_target
    _episode_id = int(eid)
    _episode_scene = scene
    _episode_target = target


def set_option_state(state: Optional[str]) -> None:
    """Record the current SMDP option. Call when the Exploration agent
    transitions between {broad-explore, directed-search,
    target-approach, verification}. ``None`` is allowed and means the
    SMDP scheduler is not yet wired up."""
    global _option_state
    _option_state = state


# ---------- internals ----------

def _approx_tokens(text: str) -> int:
    """Rough English-leaning token estimate.

    We use 4 chars/token for ASCII and 2 chars/token for non-ASCII
    (covers Chinese targets without underestimating). Good enough for
    EMNLP efficiency tables; document caveat in paper.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + cjk_chars // 2)


def _log_path(kind: str) -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    return os.path.join(_LOG_DIR, f"{kind}_{_RUN_ID}.jsonl")


def _emit(kind: str, payload: dict) -> None:
    if not _INSTRUMENTATION_ENABLED:
        return
    rec = {
        "ts": time.time(),
        "episode_id": _episode_id,
        "scene": _episode_scene,
        "target": _episode_target,
        "option_state": _option_state,
    }
    rec.update(payload)
    try:
        with _lock:
            with open(_log_path(kind), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        # Instrumentation must never crash the planner.
        print(f"[instrumentation] failed to write {kind}: {e}")


# ---------- public loggers ----------

def log_llm_call(
    client: str,
    prompt: str,
    response: str,
    latency_ms: float,
    prompt_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None,
    model: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one LLM call event."""
    payload = {
        "kind": "llm",
        "client": client,
        "model": model,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else _approx_tokens(prompt),
        "response_tokens": response_tokens if response_tokens is not None else _approx_tokens(response or ""),
        "prompt_chars": len(prompt or ""),
        "response_chars": len(response or ""),
        "latency_ms": float(latency_ms),
    }
    if extra:
        payload["extra"] = extra
    _emit("llm_calls", payload)


def log_vlm_call(
    server: str,
    endpoint: str,
    txt: str,
    latency_ms: float,
    img_shape: Optional[Tuple[int, int, int]] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one VLM call event."""
    payload = {
        "kind": "vlm",
        "server": server,
        "endpoint": endpoint,
        "txt_chars": len(txt or ""),
        "img_h": img_shape[0] if img_shape else None,
        "img_w": img_shape[1] if img_shape else None,
        "latency_ms": float(latency_ms),
    }
    if extra:
        payload["extra"] = extra
    _emit("vlm_calls", payload)


# ---------- convenience: a context-manager-style timer ----------

class TimedCall:
    """Stopwatch helper: ``with TimedCall() as t: ...; t.elapsed_ms``."""

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (time.time() - self._t0) * 1000.0
        return False

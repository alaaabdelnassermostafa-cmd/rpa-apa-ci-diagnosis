import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

# Approximate list prices in USD per 1M tokens.
# These are used for rough spend monitoring, not billing-grade accounting.
_PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    # DeepSeek list prices (USD / 1M tokens). Used for estimation since
    # DeepSeek does not return a per-call cost field.
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # OpenRouter-slugged variants, in case the provider is switched back.
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek/deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_LOCK = threading.Lock()
_SESSION_TOTALS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
}


def log_transcript(label: str, model: str, messages, response, extra: dict | None = None) -> None:
    """Write the full prompt + raw answer for one LLM call to a transcript file.

    Enabled by setting LLM_TRANSCRIPT_PATH to a file path. Captures everything
    a debugger wants: system+user messages sent, and the raw content returned
    (including any <think> reasoning the model emitted). No-op when unset, so
    it costs nothing on normal runs.
    """
    path = os.environ.get("LLM_TRANSCRIPT_PATH")
    if not path:
        return
    try:
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = "(no content)"
        rec = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "label": label,
            "model": model,
            "messages": [
                {"role": m.get("role"), "content": m.get("content")}
                for m in (messages or [])
            ],
            "answer": content,
        }
        if extra:
            rec.update(extra)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def usage_kwargs() -> dict:
    """Extra create() kwargs so OpenRouter reports real per-call cost.

    OpenRouter returns usage.cost only when the request opts in via
    extra_body={"usage": {"include": true}}. Harmless on other providers
    (ignored), so callers can always spread it into create().
    """
    if (os.environ.get("LLM_PROVIDER", "").lower() == "openrouter"):
        return {"extra_body": {"usage": {"include": True}}}
    return {}


def _normalize_model(model: str) -> str:
    if not model:
        return ""
    m = model.strip()
    if "/" in m:
        parts = m.split("/")
        if len(parts) == 2:
            provider, name = parts
            if provider in {"openai", "deepseek", "anthropic"}:
                return name if provider != "openai" else m
    return m


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    candidates = [model, _normalize_model(model)]
    pricing = None
    for c in candidates:
        pricing = _PRICING_PER_1M.get(c)
        if pricing:
            break
    if not pricing:
        return None
    return (
        (prompt_tokens * pricing["input"]) / 1_000_000
        + (completion_tokens * pricing["output"]) / 1_000_000
    )


def _extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "real_cost": None}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    # OpenRouter returns the actual USD cost of the call in usage.cost when the
    # request is made with usage accounting enabled. Use it when present so the
    # budget guard is exact rather than estimated. It may be an attribute or,
    # for some SDK versions, a dict key.
    real_cost = getattr(usage, "cost", None)
    if real_cost is None and isinstance(getattr(usage, "model_extra", None), dict):
        real_cost = usage.model_extra.get("cost")
    try:
        real_cost = float(real_cost) if real_cost is not None else None
    except (TypeError, ValueError):
        real_cost = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "real_cost": real_cost,
    }


def _append_jsonl(record: Dict[str, Any]) -> None:
    path = os.environ.get("LLM_USAGE_LOG_PATH")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass


def record_usage(response: Any, model: str, call_type: str = "chat", label: str = "") -> Dict[str, Any]:
    usage = _extract_usage(response)
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]

    # Prefer the provider's real cost (OpenRouter); fall back to the local
    # price-table estimate only when the provider did not report a cost.
    real_cost = usage.get("real_cost")
    cost_source = "real" if real_cost is not None else "estimate"
    est_cost = real_cost if real_cost is not None else _estimate_cost_usd(
        model, prompt_tokens, completion_tokens
    )

    with _LOCK:
        _SESSION_TOTALS["calls"] += 1
        _SESSION_TOTALS["prompt_tokens"] += prompt_tokens
        _SESSION_TOTALS["completion_tokens"] += completion_tokens
        _SESSION_TOTALS["total_tokens"] += total_tokens
        if est_cost is not None:
            _SESSION_TOTALS["estimated_cost_usd"] += est_cost
        session_snapshot = dict(_SESSION_TOTALS)

    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "call_type": call_type,
        "label": label,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": est_cost,
        "cost_source": cost_source,
        "session": session_snapshot,
    }

    # Default is enabled; set LLM_USAGE_LOG=0 to silence stdout logs.
    if os.environ.get("LLM_USAGE_LOG", "1") != "0":
        cost_txt = "n/a" if est_cost is None else f"${est_cost:.6f}"
        print(
            "[llm-usage] "
            f"type={call_type} label={label or '-'} model={model} "
            f"prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} "
            f"cost={cost_txt}({cost_source}) "
            f"session_total=${session_snapshot['estimated_cost_usd']:.6f}"
        )

    _append_jsonl(record)
    return record


def get_session_totals() -> Dict[str, Any]:
    with _LOCK:
        return dict(_SESSION_TOTALS)


def get_session_cost() -> float:
    with _LOCK:
        return float(_SESSION_TOTALS["estimated_cost_usd"])


class BudgetExceeded(RuntimeError):
    """Raised when accumulated session spend exceeds the configured cap."""


def check_budget(cap_usd: float) -> None:
    """Raise BudgetExceeded if session spend has passed cap_usd.

    Callers should invoke this between cases (not mid-case) so an abort
    happens at a clean checkpoint and in-flight work is already saved.
    """
    spent = get_session_cost()
    if spent >= cap_usd:
        raise BudgetExceeded(
            f"session spend ${spent:.4f} reached cap ${cap_usd:.2f}"
        )

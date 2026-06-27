"""ci_monitor_llm.py

LLM-assisted early failure predictor for CI runs.

Goal
----
Provide a cheap, *one-LLM-call-per-run* failure prediction that can fire mid-run
once enough high-signal clues are collected.

Design
------
- Wraps the deterministic :class:`ci_monitor.MonitorState`.
- Collects "highest clues" primarily from the monitor's one-shot `risk_log`
  signals and a small curated set of error-flavoured log lines.
- Makes at most one LLM call per run, gated by a configurable heuristic.

This module intentionally does NOT trigger APA triage; it only produces an early
prediction (p_fail) + rationale that you can compare against ground truth or use
as a trigger in a higher-level orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.apa.ci_monitor import MonitorState
from src.apa.llm_config import make_client, provider_env_name


DEFAULT_PROVIDER = os.environ.get("CI_MONITOR_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "deepseek"
DEFAULT_MODEL = os.environ.get("CI_MONITOR_LLM_MODEL") or os.environ.get("CI_AGENT_MODEL") or "deepseek-chat"


GIVEAWAY_SIGNALS = {
    # Highly diagnostic / usually post-failure
    "error_marker",
    "exit_code_nonzero",
    "process_exit_nonzero",
    "command_failed",
    "error_event_received",
    "step_failed",
    "job_failed",
    "run_confirmed_failed",
}


GIVEAWAY_LINE_SUBSTRINGS = (
    "##[error]",
    "process completed with exit code",
    "exit code ",
    "command failed",
)


@dataclass
class LLMPrediction:
    p_fail: float
    predicted_label: str
    reason: str
    model: str
    provider: str
    raw_json: Dict[str, Any]


class LLMEarlyFailurePredictor:
    """Wrap a MonitorState and call an LLM once when evidence is strong enough."""

    def __init__(
        self,
        run_id: str,
        *,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        call_gate_risk: float = 0.35,
        min_signals: int = 2,
        max_signal_clues: int = 8,
        max_error_lines: int = 20,
        max_recent_chunks: int = 2,
        max_recent_chars: int = 2000,
        temperature: float = 0.0,
        max_tokens: int = 200,
        pred_threshold: float = 0.65,
        cache_path: Optional[str] = None,
        exclude_giveaway_signals: bool = True,
        exclude_giveaway_lines: bool = True,
        require_pre_giveaway: bool = True,
    ) -> None:
        self.monitor = MonitorState(run_id=run_id)

        self.provider = provider
        self.model = model

        self.call_gate_risk = call_gate_risk
        self.min_signals = min_signals
        self.max_signal_clues = max_signal_clues
        self.max_error_lines = max_error_lines
        self.max_recent_chunks = max_recent_chunks
        self.max_recent_chars = max_recent_chars

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pred_threshold = pred_threshold

        self.cache_path = cache_path or os.environ.get("CI_MONITOR_LLM_CACHE")

        self.exclude_giveaway_signals = exclude_giveaway_signals
        self.exclude_giveaway_lines = exclude_giveaway_lines
        self.require_pre_giveaway = require_pre_giveaway

        self._giveaway_seen_in_text: bool = False

        self.llm_called: bool = False
        self.llm_call_event_count: Optional[int] = None
        self.llm_call_line: Optional[int] = None
        self.prediction: Optional[LLMPrediction] = None

        self._client = None
        self._llm_unavailable: bool = False

    @property
    def p_fail(self) -> Optional[float]:
        return self.prediction.p_fail if self.prediction else None

    @property
    def predicted_failure(self) -> bool:
        return bool(self.prediction and self.prediction.p_fail >= self.pred_threshold)

    def process_event(self, event: Dict[str, Any]) -> List[str]:
        # Track giveaway text regardless of whether we exclude it from LLM evidence.
        if self.require_pre_giveaway and event.get("type") == "log_chunk":
            text = (event.get("text") or "").lower()
            if text and any(s in text for s in GIVEAWAY_LINE_SUBSTRINGS):
                self._giveaway_seen_in_text = True

        fired = self.monitor.process_event(event)

        # Record the latest observed line number if provided.
        line_no = event.get("line_no")
        if isinstance(line_no, int) and line_no > 0:
            self.llm_call_line = self.llm_call_line or None  # no-op; set on call
            self._latest_line_no = line_no
        else:
            self._latest_line_no = getattr(self, "_latest_line_no", None)

        if (not self.llm_called) and self._should_call_llm():
            self._call_llm_once()

        return fired

    def _non_giveaway_evidence(self) -> List[Dict[str, Any]]:
        rl = list(self.monitor.risk_log)
        return [e for e in rl if (e.get("signal") not in GIVEAWAY_SIGNALS)]

    def _non_giveaway_evidence_score(self) -> float:
        # Risk deltas are additive and positive in ci_monitor; treat as a simple score.
        score = 0.0
        for e in self._non_giveaway_evidence():
            try:
                score += float(e.get("delta") or 0.0)
            except Exception:
                continue
        return score

    def _should_call_llm(self) -> bool:
        if self._llm_unavailable:
            return False

        # Never call once we have giveaway evidence; after that it's not interesting.
        if self.require_pre_giveaway:
            if (self.monitor.signals_fired & GIVEAWAY_SIGNALS) or self._giveaway_seen_in_text:
                return False

        # Must have some non-giveaway evidence.
        n_signals = len([s for s in self.monitor.signals_fired if s not in GIVEAWAY_SIGNALS])
        if n_signals < self.min_signals:
            return False

        # Gate on pre-giveaway evidence score rather than full monitor risk.
        if self._non_giveaway_evidence_score() < self.call_gate_risk:
            return False

        # Avoid calling if we have literally no textual evidence.
        if not self.monitor.risk_log and not self.monitor.current_error_lines and not self.monitor.seen_log_chunks:
            return False

        return True

    def _build_clues(self) -> Dict[str, Any]:
        # Risk log is already one-shot per signal; choose the largest deltas.
        risk_log = list(self.monitor.risk_log)
        if self.exclude_giveaway_signals:
            risk_log = [e for e in risk_log if (e.get("signal") not in GIVEAWAY_SIGNALS)]
        risk_log_sorted = sorted(risk_log, key=lambda e: abs(float(e.get("delta") or 0.0)), reverse=True)
        top_signals = risk_log_sorted[: self.max_signal_clues]

        def _clip(s: str, n: int = 220) -> str:
            s = (s or "").strip()
            if len(s) <= n:
                return s
            return s[: n - 3] + "..."

        # Error-flavoured lines: keep recent unique lines.
        error_lines: List[str] = []
        seen = set()
        for line in reversed(self.monitor.current_error_lines):
            ln = _clip(line, 240)
            if not ln:
                continue
            if self.exclude_giveaway_lines:
                lnl = ln.lower()
                if any(s in lnl for s in GIVEAWAY_LINE_SUBSTRINGS):
                    continue
            if ln in seen:
                continue
            seen.add(ln)
            error_lines.append(ln)
            if len(error_lines) >= self.max_error_lines:
                break
        error_lines = list(reversed(error_lines))

        # Recent chunk excerpts.
        recent_chunks = [c for c in self.monitor.seen_log_chunks[-self.max_recent_chunks :] if (c or "").strip()]
        recent_excerpt = "\n\n".join(recent_chunks)
        recent_excerpt = _clip(recent_excerpt, self.max_recent_chars)

        clues: Dict[str, Any] = {
            "run_id": self.monitor.run_id,
            "repo": self.monitor.repo,
            "workflow": self.monitor.workflow,
            "branch": self.monitor.branch,
            "event_trigger": self.monitor.event_trigger,
            "event_count": self.monitor.event_count,
            "failure_risk_heuristic": round(float(self.monitor.failure_risk), 3),
            "evidence_score_non_giveaway": round(self._non_giveaway_evidence_score(), 3),
            "signals_fired": sorted(self.monitor.signals_fired),
            "signals_fired_non_giveaway": sorted(s for s in self.monitor.signals_fired if s not in GIVEAWAY_SIGNALS)
            if self.exclude_giveaway_signals
            else None,
            "giveaway_signals_seen": sorted(self.monitor.signals_fired & GIVEAWAY_SIGNALS),
            "top_risk_signals": top_signals,
            "n_jobs_seen": len(self.monitor.jobs_seen),
            "n_steps_seen": len(self.monitor.steps_seen),
            "n_failed_steps": len(self.monitor.failed_steps),
            "error_lines": error_lines,
            "recent_excerpt": recent_excerpt,
        }
        return clues

    def _prompt_from_clues(self, clues: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        system = (
            "You predict whether a CI run will ultimately FAIL given partial evidence. "
            "Be conservative: only assign high p_fail when evidence strongly suggests eventual failure. "
            "Return a JSON object with keys: p_fail (0..1), reason (short), key_clues (array of strings)."
        )

        user_obj = {
            "task": "Predict probability this CI run ultimately fails.",
            "constraints": {
                "one_call": True,
                "cheap": True,
                "keep_reason_short": True,
            },
            "evidence": clues,
        }

        user = json.dumps(user_obj, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Cache key should be stable across whitespace differences.
        cache_key = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()
        return messages, cache_key

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_path:
            return None
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("cache_key") == cache_key:
                        return obj.get("response")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return None

    def _append_cache(self, cache_key: str, response_obj: Dict[str, Any]) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"cache_key": cache_key, "response": response_obj}, ensure_ascii=False) + "\n")
        except Exception:
            # Cache must be best-effort and never break monitoring.
            return

    def _call_llm_once(self) -> None:
        clues = self._build_clues()
        messages, cache_key = self._prompt_from_clues(clues)

        cached = self._read_cache(cache_key)
        if cached is not None:
            self.llm_called = True
            self.llm_call_event_count = self.monitor.event_count
            self.llm_call_line = getattr(self, "_latest_line_no", None)
            self.prediction = self._parse_prediction(cached)
            return

        if self._client is None:
            # Lazy init so importing this module stays cheap.
            try:
                self._client = make_client(provider=self.provider)
            except RuntimeError:
                # Most common: missing API key. Abstain without crashing.
                self._llm_unavailable = True
                return

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
        except Exception as e:
            # Hard failure → mark as called but return a neutral prediction.
            data = {"p_fail": 0.5, "reason": f"LLM error: {e}", "key_clues": []}

        self._append_cache(cache_key, data)

        self.llm_called = True
        self.llm_call_event_count = self.monitor.event_count
        self.llm_call_line = getattr(self, "_latest_line_no", None)
        self.prediction = self._parse_prediction(data)

    def _parse_prediction(self, data: Dict[str, Any]) -> LLMPrediction:
        try:
            p_fail = float(data.get("p_fail"))
        except Exception:
            p_fail = 0.5
        p_fail = max(0.0, min(1.0, p_fail))
        reason = str(data.get("reason") or "").strip()
        predicted_label = "failure" if p_fail >= self.pred_threshold else "success"
        return LLMPrediction(
            p_fail=p_fail,
            predicted_label=predicted_label,
            reason=reason,
            model=self.model,
            provider=self.provider,
            raw_json=data,
        )


def explain_missing_api_key(provider: str = DEFAULT_PROVIDER) -> str:
    env = provider_env_name(provider)
    return f"Missing LLM API key: set {env} (or LLM_API_KEY)"

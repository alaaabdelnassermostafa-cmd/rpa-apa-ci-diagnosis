"""
APA framework: a small public API over the agentic CI-failure diagnosis pipeline.

The framework turns one failed CI run into a structured diagnosis: a failure category,
a confidence, the reasoning, and a concrete fix recommendation, together with the
investigation trace (tools called, belief evolution). It wraps the same components used
throughout the thesis (intake -> deterministic preprocessing -> Bayesian belief tracker ->
EIG-planned agent loop -> classification + remediation).

Two entry points:

    from apa.framework import diagnose, load_case
    case = load_case("brooooooklyn/image")     # a stored, curated case (for demos / reproduction)
    dx   = diagnose(case)                       # run the APA agent on it
    print(dx.pretty())

For a brand-new failure (a raw GitHub Actions / GHALogs record), use the raw path:

    from apa.framework import diagnose_raw
    dx = diagnose_raw(raw_github_actions_record)

Requirements: an LLM key in the environment (DEEPSEEK_API_KEY by default; set CI_AGENT_MODEL
to switch models) and, for live evidence gathering, optionally a GITHUB_TOKEN.
"""
from __future__ import annotations
import os, json, gzip, glob
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv; load_dotenv()

import re
from dataclasses import asdict as _asdict
from .bayesian_tracker import BeliefState
from .intake_parser import RunEvent
from .agent import (build_agent_graph, run_agent, _initial_tools_for_event,
                    _fetch_commit_diff, _summarize_patch_files, _fetch_run_history,
                    _failed_step_summary, DEFAULT_MODEL)


def _event_from_case(c: dict) -> RunEvent:
    """Build a RunEvent from a curated (intake/extraction) case record."""
    ik = c.get("intake", {}); md = c.get("metadata", {})
    run_number = ik.get("run_number") or md.get("run_number")
    attempt = ik.get("attempt") or md.get("attempt")
    if run_number is None:
        m = re.search(r"_(\d+)_(\d+)$", ik.get("run_id", ""))
        if m:
            run_number = int(m.group(1)); attempt = attempt or int(m.group(2))
    attempt = attempt or 1
    branch = ik.get("branch", "")
    _PROTECTED = ("main", "master", "release", "releases/", "prod", "production")
    bl = (branch or "").lower()
    is_protected = any(bl == p or bl.startswith(p) for p in _PROTECTED)
    detection = ik.get("failure_detection", "")
    if not detection or detection == "not_a_failure":
        detection = "job_level_fallback"
    available = ["error_text", "many_jobs_failed", "branch_type", "commit_message",
                 "previous_runs", "parent_commit_run", "detection_mode"]
    return RunEvent(
        source="github", run_id=ik.get("run_id", ""), repo=ik.get("repo", ""),
        event=ik.get("event", ""), conclusion=ik.get("conclusion", "failure"),
        started_at=ik.get("started_at", ""), n_jobs=ik.get("n_jobs", 0) or 1,
        failed_jobs_count=ik.get("failed_jobs_count", 0) or 1, duration_sec=None,
        branch=branch, commit_sha=ik.get("commit_sha", ""),
        commit_title=ik.get("commit_title", ""), commit_author=ik.get("commit_author", ""),
        workflow=ik.get("workflow", ""), run_number=run_number, attempt=attempt,
        is_protected_branch=is_protected, failure_detection=detection,
        available_signals=available,
    )

_DATASETS = ("data/dataset_remote_250.jsonl.gz", "data/dataset_remote_120.jsonl.gz",
             "data/dataset_remote_next.jsonl.gz", "data/dataset_topup.jsonl.gz")


@dataclass
class Diagnosis:
    """The structured output of a single APA diagnosis."""
    repo: str
    category: str
    confidence: float
    severity: str
    reasoning: str
    recommended_action: str
    error_lines: list = field(default_factory=list)
    implicated_files: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    steps: int = 0
    beliefs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        top = sorted(self.beliefs.items(), key=lambda x: -x[1])[:3]
        belief_str = ", ".join(f"{k} {v:.0%}" for k, v in top)
        files = ", ".join(self.implicated_files[:5]) or "-"
        return (
            f"\n  repo:        {self.repo}\n"
            f"  category:    {self.category}   (confidence {self.confidence:.0%}, {self.severity})\n"
            f"  reasoning:   {self.reasoning}\n"
            f"  fix:         {self.recommended_action or '(none)'}\n"
            f"  files:       {files}\n"
            f"  investigation: {self.steps} step(s); tools = {', '.join(self.tools_used) or 'none'}\n"
            f"  top beliefs: {belief_str}\n"
        )


# ── loading stored cases (for demos / reproduction) ─────────────────────────

def load_case(query: str) -> dict:
    """Return the first curated case whose run_id or repo contains `query`."""
    q = query.lower()
    for path in _DATASETS:
        if not os.path.exists(path):
            continue
        for line in gzip.open(path, "rt", encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ik = r.get("intake", {})
            if q in ik.get("run_id", "").lower() or q in ik.get("repo", "").lower():
                return r
    raise KeyError(f"no stored case matching {query!r}")


def list_cases(limit: int = 20) -> list[str]:
    out = []
    for path in _DATASETS:
        if not os.path.exists(path):
            continue
        for line in gzip.open(path, "rt", encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line)["intake"]["repo"])
                if len(out) >= limit:
                    return out
    return out


# ── core diagnosis ──────────────────────────────────────────────────────────

def _state_from_case(case: dict, model: str) -> dict:
    """Build the agent state for a curated (intake/extraction) case, matching the
    evaluation pipeline so framework output reproduces the thesis numbers."""
    event = _event_from_case(case)
    ext = case.get("extraction", {})
    excs = ext.get("log_excerpts", []) or []
    log_texts = [e.get("text", "") for e in excs if isinstance(e, dict) and "text" in e]
    error_lines = ext.get("sample_error_lines", []) or []
    cf, cds = [], {}
    if event.repo and event.commit_sha:
        try:
            cf = _fetch_commit_diff(event.repo, event.commit_sha).get("files", [])
            cds = _summarize_patch_files(cf)
        except Exception:
            pass
    rh = {}
    if event.repo and event.branch and event.run_number:
        try:
            rh = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception:
            pass
    p = BeliefState()
    return {
        "run_event": _asdict(event), "raw_run": case,
        "beliefs": dict(p.probabilities), "belief_history": list(p.history),
        "confidence": p.confidence(), "entropy": p.entropy(),
        "tools_available": _initial_tools_for_event(case, event), "tools_called": [],
        "investigation_log": [], "current_step": 0, "done": False,
        "error_lines": error_lines, "mentioned_files": ext.get("mentioned_files", []),
        "log_excerpt_texts": log_texts, "changed_files": cf, "commit_diff": cds,
        "failed_step_context": _failed_step_summary(event), "dependency_changes": {},
        "run_history": rh, "similar_failures": [], "workflow_contents": [],
        "runner_environment": {}, "pr_context": {}, "semantic_diff_links": [],
        "preprocessing_summary": {"top_category": "CODE_REGRESSION"}, "classification": {},
        "api_key": "", "model": model, "_next_tool": "",
    }


def _to_diagnosis(repo: str, final_state: dict) -> Diagnosis:
    cl = final_state.get("classification", {}) or {}
    files = [m.get("path", "") for m in (final_state.get("mentioned_files") or []) if isinstance(m, dict)]
    return Diagnosis(
        repo=repo,
        category=cl.get("category", "UNKNOWN"),
        confidence=float(cl.get("confidence", 0.0)),
        severity=cl.get("severity", "MODERATE"),
        reasoning=cl.get("reasoning", ""),
        recommended_action=cl.get("recommended_action", ""),
        error_lines=final_state.get("error_lines", [])[:10],
        implicated_files=[f for f in files if f][:10],
        tools_used=final_state.get("tools_called", []),
        steps=final_state.get("current_step", 0),
        beliefs=final_state.get("beliefs", {}),
    )


def diagnose(case: dict, model: str | None = None, recommend: bool = True) -> Diagnosis:
    """Diagnose a curated (stored) CI failure case with the APA agent."""
    model = model or os.environ.get("CI_AGENT_MODEL", DEFAULT_MODEL)
    prev = os.environ.get("CI_AGENT_SKIP_ACTION")
    os.environ["CI_AGENT_SKIP_ACTION"] = "0" if recommend else "1"
    try:
        final = build_agent_graph().invoke(_state_from_case(case, model))
    finally:
        if prev is None:
            os.environ.pop("CI_AGENT_SKIP_ACTION", None)
        else:
            os.environ["CI_AGENT_SKIP_ACTION"] = prev
    return _to_diagnosis(case.get("intake", {}).get("repo", "?"), final)


def diagnose_raw(raw_run: dict, model: str | None = None) -> Diagnosis:
    """Diagnose a brand-new raw GitHub Actions / GHALogs record (deployment path)."""
    res = run_agent(raw_run, model=model or os.environ.get("CI_AGENT_MODEL", DEFAULT_MODEL))
    repo = (raw_run.get("repo") or raw_run.get("intake", {}).get("repo") or "?")
    # run_agent already returns a result dict shaped like final_state's projection
    state_like = {
        "classification": res.get("classification", {}),
        "mentioned_files": res.get("mentioned_files", []),
        "error_lines": res.get("error_lines", []),
        "tools_called": res.get("tools_used", []),
        "current_step": res.get("steps_taken", 0),
        "beliefs": res.get("beliefs", {}),
    }
    return _to_diagnosis(repo, state_like)

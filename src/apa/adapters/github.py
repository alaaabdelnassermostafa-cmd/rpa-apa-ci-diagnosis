import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any

from .base import FailureAdapter, RunEvent, FailedStepInfo


_PROTECTED = ("main", "master", "release", "releases/", "prod", "production")

_TOOLING_ARTIFACT_PATTERNS = (
    "bash-command-extractor",
    "Converting circular structure to JSON",
    "BashWord",
    "Parser exception",
)

def _is_protected(branch: str) -> bool:
    if not branch:
        return False
    b = branch.lower()
    return any(b == p or b.startswith(p) for p in _PROTECTED)

def _duration(started: Optional[str], ended: Optional[str]) -> Optional[float]:
    if not started or not ended:
        return None
    try:
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        return (e - s).total_seconds()
    except Exception:
        return None

def _describe_step(step: dict) -> Tuple[str, str]:
    step_type = step.get("type", "unknown")
    if step_type == "action":
        repo = step.get("repository", "")
        action = step.get("action", "")
        version = step.get("version", "")
        label = f"{repo}/{action}@{version}".strip("/@") or "action"
    else:
        label = (
            step.get("name")
            or step.get("category")
            or step.get("command")
            or step.get("code", "")[:80]
            or step_type
        )
    return str(label)[:200], step_type

def _extract_error(step: dict) -> Optional[str]:
    for key in ("error", "errors", "error_message", "failure_reason"):
        val = step.get(key)
        if not val:
            continue
        if isinstance(val, str):
            return val[:500]
        if isinstance(val, dict):
            return json.dumps(val)[:500]
        if isinstance(val, list) and val:
            return str(val[0])[:500]
    return None

def _looks_like_tooling_artifact(error_text: Optional[str]) -> bool:
    if not error_text:
        return False
    return any(p in error_text for p in _TOOLING_ARTIFACT_PATTERNS)

def _parse_step_start(step: dict) -> Optional[datetime]:
    raw = step.get("start_date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None

def _find_failed_step_in_job(job: dict) -> Tuple[Optional[int], str]:
    steps = job.get("steps") or []
    if not steps:
        return None, "unknown_failure"

    for i, step in enumerate(steps):
        if _extract_error(step):
            return i, "per_step_error"

    if len(steps) == 1:
        return 0, "single_step_inferred"

    indexed_with_time = [
        (i, _parse_step_start(s)) for i, s in enumerate(steps)
    ]
    indexed_with_time = [(i, t) for i, t in indexed_with_time if t is not None]
    if indexed_with_time:
        last_idx = max(indexed_with_time, key=lambda pair: pair[1])[0]
        return last_idx, "job_level_fallback"

    return len(steps) - 1, "job_level_fallback"


class GitHubAdapter(FailureAdapter):
    @property
    def source_name(self) -> str:
        return "github"

    @property
    def available_signals(self) -> List[str]:
        return [
            "error_text",
            "many_jobs_failed",
            "branch_type",
            "commit_message",
            "previous_runs",
            "parent_commit_run",
            "detection_mode"
        ]

    def parse(self, raw_run: dict) -> RunEvent:
        meta = raw_run.get("metadata") or {}
        head_commit = meta.get("head_commit") or {}
        author = head_commit.get("author") or {}
        actor = meta.get("actor") or {}

        commit_msg = head_commit.get("message") or ""
        commit_title = commit_msg.split("\\n", 1)[0] if commit_msg else ""

        branch = meta.get("head_branch") or ""
        conclusion = meta.get("conclusion") or "unknown"
        log_insights = raw_run.get("log_insights") or []

        failed_steps: List[FailedStepInfo] = []
        failed_jobs_count = 0
        run_detection_mode = "not_a_failure"

        if conclusion == "failure":
            if not log_insights:
                run_detection_mode = "unknown_failure"
            else:
                per_job_modes: List[str] = []
                for job in log_insights:
                    idx, mode = _find_failed_step_in_job(job)
                    per_job_modes.append(mode)

                    if idx is None:
                        continue

                    failed_jobs_count += 1
                    steps = job.get("steps") or []
                    step = steps[idx] if idx < len(steps) else {}
                    label, step_type = _describe_step(step)
                    error_text = _extract_error(step)

                    failed_steps.append(
                        FailedStepInfo(
                            job_file=job.get("file", ""),
                            runner_image=job.get("image", ""),
                            step_index=idx,
                            step_type=step_type,
                            step_label=label,
                            step_duration_sec=step.get("duration_sec"),
                            error_text=error_text,
                            detection_mode=mode,
                            tooling_artifact_suspected=_looks_like_tooling_artifact(error_text),
                            raw_keys=sorted(step.keys()),
                        )
                    )

                weakness_order = {
                    "per_step_error": 0,
                    "single_step_inferred": 1,
                    "job_level_fallback": 2,
                    "unknown_failure": 3,
                }
                if failed_jobs_count == 0:
                    run_detection_mode = "unknown_failure"
                else:
                    attributed_modes = [
                        m for m in per_job_modes if m != "unknown_failure"
                    ] or per_job_modes
                    run_detection_mode = max(
                        attributed_modes,
                        key=lambda m: weakness_order.get(m, 99),
                    )

        all_tooling = bool(failed_steps) and all(
            fs.tooling_artifact_suspected for fs in failed_steps
        )

        return RunEvent(
            source=self.source_name,
            run_id=raw_run.get("_id", ""),
            repo=raw_run.get("repository_name", ""),
            workflow=Path(raw_run.get("workflow_path", "")).name if raw_run.get("workflow_path") else "",
            run_number=raw_run.get("run_number", 0),
            attempt=raw_run.get("run_attempt", 0),
            event=meta.get("event", ""),
            branch=branch,
            is_protected_branch=_is_protected(branch),
            actor=(actor.get("login", "") if isinstance(actor, dict) else ""),
            commit_sha=str(head_commit.get("id", ""))[:12],
            commit_title=commit_title[:200],
            commit_message=commit_msg,
            commit_author=(author.get("name", "") if isinstance(author, dict) else ""),
            conclusion=conclusion,
            started_at=meta.get("run_started_at", ""),
            duration_sec=_duration(meta.get("run_started_at"), meta.get("updated_at")),
            n_jobs=len(log_insights),
            failed_jobs_count=failed_jobs_count,
            failed_steps=failed_steps,
            failure_detection=run_detection_mode,
            all_failures_are_tooling_artifacts=all_tooling,
            has_log_insights=len(log_insights) > 0,
            available_signals=self.available_signals,
        )

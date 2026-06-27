"""
ci_monitor.py
─────────────────────────────────────────────────────────────────────────────
Live CI run monitor — accumulates streaming events from ci_stream_simulator
and maintains a running failure-risk score.

This is the "lite" stateful object that sits between the stream simulator and
the APA triage agent.  It is intentionally kept small and dependency-free so it
can run entirely offline.

Risk scoring model
──────────────────
Each event type can nudge the risk score upward.  The score is clamped [0, 1]
and once it passes TRIAGE_THRESHOLD (default 0.75), the caller is expected to
trigger APA triage.

Signal weights (additive, applied once per matching event):

  ##[error] seen in log chunk        +0.40
  exit code / command failed phrase  +0.30
  step_failed event received         +0.50
  error_seen event received          +0.35
  many error lines accumulated       +0.15  (fires when seen ≥ 5)
  dependency install error keywords  +0.20
  test failure keywords              +0.20
  runner / tool version error        +0.20
  job_failed event                   +0.55
  run_failed event                   +1.00  (clamps to 1.0)

Signals are one-shot: each named signal can only fire once to prevent runaway
inflation from repeated log chunks carrying the same keyword.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# -- Windows UTF-8 output fix ------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ----------------------------------------------------------------------------


# ─── tuning ───────────────────────────────────────────────────────────────────

TRIAGE_THRESHOLD: float = 0.75   # trigger APA when risk >= this

# Text patterns → (signal_name, weight)
# Checked against the lower-cased text of each log_chunk.
_TEXT_SIGNALS: List[Tuple[str, str, float]] = [
    # pattern                        signal_name                 weight
    (r"##\[error\]",                "error_marker",              0.40),
    (r"exit code [1-9]",            "exit_code_nonzero",         0.30),
    (r"process completed with exit code [1-9]",
                                    "process_exit_nonzero",      0.30),
    (r"command failed",             "command_failed",            0.30),
    (r"npm err!",                   "npm_error",                 0.20),
    (r"pip.*error|error.*pip",      "pip_error",                 0.20),
    (r"could not resolve|module not found|cannot find module",
                                    "dependency_resolve",        0.20),
    (r"dependency conflict|version conflict|conflicting",
                                    "dep_conflict",              0.20),
    (r"test.*fail|fail.*test|assertion.*error|assertionerror",
                                    "test_failure",              0.20),
    (r"pytest.*failed|jest.*failed|cargo test.*failed",
                                    "test_runner_failure",       0.20),
    (r"runner.*version|version.*mismatch|unsupported.*version",
                                    "version_mismatch",          0.20),
    (r"permission denied|operation not permitted",
                                    "permission_error",          0.15),
    (r"connection refused|connection reset|timeout",
                                    "network_error",             0.10),
    (r"warning:|warn:|npm warn|warning\s", "warnings",           0.05),
]

_COMPILED_TEXT_SIGNALS: List[Tuple[re.Pattern, str, float]] = [
    (re.compile(pat, re.IGNORECASE), name, weight)
    for pat, name, weight in _TEXT_SIGNALS
]

# How many accumulated error lines trigger the "many errors" signal
MANY_ERRORS_THRESHOLD = 5


# ─── monitor state ────────────────────────────────────────────────────────────

@dataclass
class MonitorState:
    """
    Tracks the live state of one CI run as events arrive.

    Attributes
    ──────────
    run_id              — the GHA run identifier
    failure_risk        — float [0, 1] — current probability estimate
    triage_triggered    — True once APA was called
    confirmed_failed    — True once run_failed event arrives
    confirmed_success   — True once run_succeeded event arrives

    seen_log_chunks     — all log text chunks received so far
    current_error_lines — lines containing error-related keywords
    jobs_seen           — job files seen so far
    steps_seen          — (job_file, step_label) pairs seen
    failed_steps        — steps that emitted step_failed events
    error_events        — raw error_seen events received

    signals_fired       — set of signal names already applied (prevents double-counting)
    risk_log            — [(event_type, signal, delta, new_risk)] trace
    triage_result       — result dict returned by APA (filled after triage)
    event_count         — total events processed
    """

    run_id: str = ""
    failure_risk: float = 0.0
    triage_triggered: bool = False
    confirmed_failed: bool = False
    confirmed_success: bool = False

    # Accumulated evidence
    seen_log_chunks: List[str] = field(default_factory=list)
    current_error_lines: List[str] = field(default_factory=list)
    jobs_seen: List[str] = field(default_factory=list)
    steps_seen: List[Tuple[str, str]] = field(default_factory=list)
    failed_steps: List[Tuple[str, str]] = field(default_factory=list)
    error_events: List[dict] = field(default_factory=list)

    # Risk tracking
    signals_fired: Set[str] = field(default_factory=set)
    risk_log: List[dict] = field(default_factory=list)
    triage_result: Optional[dict] = None
    event_count: int = 0

    # Metadata (filled on run_started)
    repo: str = ""
    workflow: str = ""
    branch: str = ""
    commit_sha: str = ""
    commit_title: str = ""
    event_trigger: str = ""

    # Beliefs from APA (filled post-triage for comparison)
    current_beliefs: Dict[str, float] = field(default_factory=dict)

    def _apply_signal(self, signal_name: str, delta: float, source: str = "") -> None:
        """
        Apply a one-shot risk delta. If the signal has already fired, do nothing.
        Clamps failure_risk to [0, 1].
        """
        if signal_name in self.signals_fired:
            return
        self.signals_fired.add(signal_name)
        old_risk = self.failure_risk
        self.failure_risk = min(1.0, self.failure_risk + delta)
        self.risk_log.append({
            "event_count": self.event_count,
            "source": source,
            "signal": signal_name,
            "delta": round(delta, 3),
            "old_risk": round(old_risk, 3),
            "new_risk": round(self.failure_risk, 3),
        })

    def process_event(self, event: dict) -> List[str]:
        """
        Ingest one streaming event and update state + risk score.

        Returns a list of signal names that fired this call (may be empty).
        Callers can use this to decide whether to print a risk update.
        """
        self.event_count += 1
        etype = event.get("type", "")
        fired: List[str] = []

        def fire(signal: str, delta: float) -> None:
            old = self.failure_risk
            self._apply_signal(signal, delta, source=etype)
            if signal not in [entry["signal"] for entry in self.risk_log[:-1]]:
                # Signal was new — record it
                pass
            if self.failure_risk != old or signal not in self.signals_fired - {signal}:
                fired.append(signal)

        # ── run_started ──────────────────────────────────────────────────────
        if etype == "run_started":
            self.run_id = event.get("run_id", self.run_id)
            self.repo = event.get("repo", "")
            self.workflow = event.get("workflow", "")
            self.branch = event.get("branch", "")
            self.commit_sha = event.get("commit_sha", "")
            self.commit_title = event.get("commit_title", "")
            self.event_trigger = event.get("event_trigger", "")

        # ── job_started ──────────────────────────────────────────────────────
        elif etype == "job_started":
            jf = event.get("job_file", "")
            if jf and jf not in self.jobs_seen:
                self.jobs_seen.append(jf)

        # ── step_started ─────────────────────────────────────────────────────
        elif etype == "step_started":
            pair = (event.get("job_file", ""), event.get("step_label", ""))
            if pair not in self.steps_seen:
                self.steps_seen.append(pair)

        # ── log_chunk ────────────────────────────────────────────────────────
        elif etype == "log_chunk":
            text = event.get("text", "")
            self.seen_log_chunks.append(text)

            # Extract error-flavoured lines
            for line in text.splitlines():
                ll = line.lower()
                if any(kw in ll for kw in (
                    "error", "failed", "failure", "exception",
                    "fatal", "traceback", "permission denied",
                    "##[error]",
                )):
                    if line.strip() and line not in self.current_error_lines:
                        self.current_error_lines.append(line.strip())

            # Text-based risk signals
            text_lower = text.lower()
            for pattern, signal_name, weight in _COMPILED_TEXT_SIGNALS:
                if pattern.search(text_lower):
                    fire(signal_name, weight)

            # "Many errors" threshold signal
            if (
                len(self.current_error_lines) >= MANY_ERRORS_THRESHOLD
                and "many_error_lines" not in self.signals_fired
            ):
                fire("many_error_lines", 0.15)

        # ── error_seen ───────────────────────────────────────────────────────
        elif etype == "error_seen":
            self.error_events.append(event)
            # Apply a dedicated signal (slightly lower than the log_chunk path
            # to avoid double-counting when both fire)
            fire("error_event_received", 0.35)

        # ── step_failed ──────────────────────────────────────────────────────
        elif etype == "step_failed":
            pair = (event.get("job_file", ""), event.get("step_label", ""))
            if pair not in self.failed_steps:
                self.failed_steps.append(pair)
            fire("step_failed", 0.50)

        # ── job_failed ───────────────────────────────────────────────────────
        elif etype == "job_failed":
            fire("job_failed", 0.55)

        # ── run_failed ───────────────────────────────────────────────────────
        elif etype == "run_failed":
            self.confirmed_failed = True
            fire("run_confirmed_failed", 1.00)  # clamps to 1.0

        # ── run_succeeded ────────────────────────────────────────────────────
        elif etype == "run_succeeded":
            self.confirmed_success = True
            # Decrease risk dramatically (though success is rare in our dataset)
            self.failure_risk = max(0.0, self.failure_risk - 0.5)

        return fired

    # ── convenience accessors ─────────────────────────────────────────────────

    @property
    def should_triage(self) -> bool:
        """True when risk is high enough to trigger APA (and not yet triaged)."""
        return (
            not self.triage_triggered
            and (self.failure_risk >= TRIAGE_THRESHOLD or self.confirmed_failed)
        )

    @property
    def risk_bar(self, width: int = 20) -> str:
        """ASCII progress bar for the current failure_risk."""
        filled = round(self.failure_risk * width)
        bar = "#" * filled + "." * (width - filled)
        pct = self.failure_risk * 100
        return f"[{bar}] {pct:5.1f}%"

    def summary(self) -> str:
        """Human-readable one-line status summary."""
        status = (
            "FAILED" if self.confirmed_failed
            else "SUCCESS" if self.confirmed_success
            else "IN PROGRESS"
        )
        triage = " [APA TRIGGERED]" if self.triage_triggered else ""
        return (
            f"run={self.run_id[:40]}  "
            f"risk={self.failure_risk:.2f}  {self.risk_bar}  "
            f"events={self.event_count}  status={status}{triage}"
        )

    def build_partial_raw_run(self, case: dict) -> dict:
        """
        Construct a partial raw_run dict compatible with agent.run_agent().

        This bundles all evidence accumulated so far — even mid-run — into
        the precomputed_log_evidence format that _prepare_log_evidence() knows
        how to consume without touching the ZIP archive.

        Parameters
        ──────────
        case : the original targeted_cases entry (for metadata fallback)
        """
        intake = case.get("intake", {})

        # Gather error lines accumulated so far
        error_lines = list(self.current_error_lines[:50])

        # Add sample_error_lines from the original case that we haven't already
        for ln in case.get("extraction", {}).get("sample_error_lines", []):
            if ln not in error_lines:
                error_lines.append(ln)

        # Build a synthetic log excerpt text from seen chunks
        log_excerpt = "\n".join(self.seen_log_chunks)
        log_excerpt_texts = [log_excerpt[:8000]] if log_excerpt.strip() else []

        # Guess mentioned files from error lines (simple heuristic)
        mentioned_files = _extract_mentioned_files(error_lines)

        # Build a synthetic metadata dict matching what intake_parser.intake() reads
        metadata = {
            "head_branch": self.branch or intake.get("branch", ""),
            "conclusion": "in_progress" if not self.confirmed_failed else "failure",
            "event": self.event_trigger or intake.get("event", "push"),
            "run_started_at": "",
            "head_commit": {
                "id": self.commit_sha or intake.get("commit_sha", ""),
                "message": self.commit_title or intake.get("commit_title", ""),
                "author": {"name": ""},
            },
            "actor": {"login": ""},
        }

        # Build synthetic log_insights (mimics GHALogs structure)
        log_insights = []
        for job_file in self.jobs_seen:
            steps = []
            for jf, sl in self.steps_seen:
                if jf == job_file:
                    step: dict = {
                        "type": "run",
                        "name": sl,
                    }
                    # Add error text for failed steps
                    for fjf, fsl in self.failed_steps:
                        if fjf == job_file and fsl == sl:
                            step["error"] = "\n".join(error_lines[:3])
                            break
                    # Add error from error_events
                    for ev in self.error_events:
                        if ev.get("job_file") == job_file and ev.get("step_label") == sl:
                            step["error"] = ev.get("error_text", "")
                            break
                    steps.append(step)
            log_insights.append({
                "file": job_file,
                "image": "ubuntu-latest",
                "steps": steps,
            })

        # If we haven't seen any jobs yet, create a placeholder
        if not log_insights:
            placeholder_steps = [{"type": "run", "name": "unknown step"}]
            if error_lines:
                placeholder_steps[0]["error"] = error_lines[0]
            log_insights = [{"file": "job_1.txt", "image": "ubuntu-latest", "steps": placeholder_steps}]

        partial_raw_run = {
            "_id": self.run_id or intake.get("run_id", ""),
            "repository_name": self.repo or intake.get("repo", ""),
            "workflow_path": f".github/workflows/{self.workflow or intake.get('workflow', 'ci.yml')}",
            "run_number": intake.get("run_number", 0),
            "run_attempt": intake.get("attempt", 1),
            "metadata": metadata,
            "log_insights": log_insights,
            # Pre-computed evidence so the agent doesn't need the ZIP archive
            "precomputed_log_evidence": {
                "error_lines": error_lines,
                "mentioned_files": mentioned_files,
                "log_excerpt_texts": log_excerpt_texts,
            },
            # Signal that this is a monitoring-triggered partial run
            "_monitoring_partial": True,
            "_monitor_risk": round(self.failure_risk, 3),
            "_monitor_event_count": self.event_count,
        }

        return partial_raw_run


def _extract_mentioned_files(error_lines: List[str]) -> List[dict]:
    """
    Simple heuristic: find file paths mentioned in error lines.
    Returns dicts compatible with agent.py mentioned_files format.
    """
    file_pattern = re.compile(
        r"""
        (?:^|[\s"'(])          # boundary
        (                       # capture group
            (?:[A-Za-z0-9._/-]+ # path segments
            \.                  # must have an extension
            [A-Za-z0-9]{1,10}   # extension
            )
            (?::\d+(?::\d+)?)?  # optional :line:col
        )
        (?:$|[\s"':,)])         # boundary
        """,
        re.VERBOSE,
    )

    seen: Set[str] = set()
    results: List[dict] = []

    for line in error_lines:
        for m in file_pattern.finditer(line):
            path_raw = m.group(1)
            path = path_raw.split(":")[0]
            if path in seen or len(path) < 3:
                continue
            # Filter out unlikely paths (timestamps, versions, etc.)
            if re.match(r"^\d", path):
                continue
            seen.add(path)
            # Try to extract line number
            line_num = None
            if ":" in path_raw[len(path):]:
                try:
                    line_num = int(path_raw[len(path) + 1:].split(":")[0])
                except ValueError:
                    pass
            results.append({
                "path": path,
                "line": line_num,
                "column": None,
                "context": line.strip()[:80],
            })

    return results[:12]


# ─── factory ──────────────────────────────────────────────────────────────────

def create_monitor(run_id: str = "") -> MonitorState:
    """Create a fresh MonitorState for a new run."""
    return MonitorState(run_id=run_id)

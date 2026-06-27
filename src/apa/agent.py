# agent.py
# ─────────────────────────────────────────────────────────────────────
# The Investigation Agent — a LangGraph-based multi-step reasoning
# agent for CI/CD failure triage.
#
# Architecture:
#   - Shared deterministic preprocessing runs before the graph:
#       intake -> log extraction -> file-path extraction -> initial beliefs
#   - APA-side free evidence is fetched once before the graph:
#       commit diff (changed-file families) + run history — zero LLM cost
#   - Mandatory evidence step (deep_log_analysis): ONE LLM read of a
#     run-context header (metadata, failed-step shape, run history,
#     changed-file families) + the full failure excerpt
#     (single joint Bayesian update — each evidence source enters the
#     posterior exactly once; the APA mirror of the RPA signal battery)
#   - Planner tools are reserved for deep, expensive, external, or
#     conditional evidence (manifests, workflow YAML, env, log diff,
#     retrieval, web, PR context)
#   - Planner node (LLM) decides which substantive tool to call next
#   - Tool nodes execute and update Bayesian beliefs
#   - Agent stops when confidence exceeds threshold or max steps hit
#   - Final classification combines Bayesian beliefs with LLM reasoning
#
# This is the APA system. The RPA equivalent is the deterministic
# preprocessing + Bayesian tracker alone (no LLM, no agent loop).
# ─────────────────────────────────────────────────────────────────────


import json
import os
import gzip
import math
import requests
from typing import TypedDict, List, Optional, Annotated
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

from src.apa.intake_parser import intake, RunEvent
from src.apa.file_path_extractor import MentionedFile, extract_from_excerpt_windows
from src.apa.bayesian_tracker import (
    BeliefState,
    CATEGORIES,
    signal_branch_type,
    signal_commit_message,
    signal_detection_mode,
    signal_error_text,
    signal_many_jobs_failed,
    signal_parent_commit_run,
    signal_step_duration,
)
import re
import base64
from collections import Counter
from src.apa.bayesian_tracker_dual import (
    format_observation_workflow_contents,
    llm_generate_likelihood,
)
from src.apa.llm_usage import record_usage, usage_kwargs, log_transcript
from src.apa.llm_config import make_client, get_api_key
from src.apa.semantic_diff import analyze_semantic_diff
from src.apa import semantic_refiner
from src.apa.tool_selection import rank_tools_by_eig, format_eig_for_prompt, pick_eig_tool
from src.apa.log_extractor import extract_log_excerpt, extract_error_summary_lines

load_dotenv()

# ─── configuration ───────────────────────────────────────────────────

ENTROPY_THRESHOLD = 1.0    # stop investigating when entropy drops below this (bits)
# Step cap sits BELOW the selectable tool count (6 general + 1 conditional PR
# probe) so the agent can never run every tool — it must prioritize, which is
# the EIG planner's purpose. Five tools cover every failure category (see the
# coverage analysis); the cap excludes only the two weakest probes.
MAX_STEPS = int(os.environ.get("CI_AGENT_MAX_STEPS", "5"))  # cap investigation steps (cost lever)

# Two excerpt tiers for the log payload sent to the LLM.
# "short" is used only for lightweight callers (e.g. search query generation).
# "full"  is used for the mandatory deep_log_analysis step — post-dedup logs
#         typically land under this limit (Cases 1–5: 6.7k–19.1k chars after
#         collapse), so the agent sees the whole excerpt in ONE read and the
#         log evidence updates the posterior exactly once (no double-counting).
EXCERPT_SHORT_MAX_CHARS = 6_000
EXCERPT_SHORT_MAX_LINES = 80
EXCERPT_FULL_MAX_CHARS  = 20_000
EXCERPT_FULL_MAX_LINES  = 350

# Legacy aliases kept so any external callers don't break.
DEEP_LOG_MAX_CHARS = EXCERPT_FULL_MAX_CHARS
DEEP_LOG_MAX_LINES = EXCERPT_FULL_MAX_LINES

ZIP_PATH = os.environ.get("CI_AGENT_ZIP_PATH", "/home/guc_alaa/github_run_logs.zip")
DEFAULT_MODEL = os.environ.get("CI_AGENT_MODEL", "deepseek/deepseek-chat")


# Fast-path gate: if preprocessing confidence exceeds this, skip the agent
# skip APA only if RPA is clearly strong
FAST_PATH_THRESHOLD = float(os.environ.get("CI_AGENT_FAST_PATH_THRESHOLD", "0.75"))




# Planner mode controls how the next tool is chosen each step:
#   hybrid (default) — EIG ranking injected into LLM prompt; LLM decides
#   eig              — always pick argmax(EIG), no LLM planner call
#   llm              — original behaviour: LLM decides without EIG context
PLANNER_MODE = os.environ.get("CI_AGENT_PLANNER_MODE", "hybrid").lower()

# Model tiering (FrugalGPT-style cascade): the cheap model (state["model"])
# handles the easy structured tasks — planner tool choice and per-tool
# likelihood generation — while the final classification stage (classify +
# devil's advocate + recommended action) can run on a stronger model.
# Empty → same model everywhere.
CLASSIFY_MODEL = os.environ.get("CI_AGENT_CLASSIFY_MODEL", "")
# Secondary reasoning calls (devil's advocate, recommended action) don't need
# the strong/expensive reasoning model — they refine or phrase an already-made
# decision. They default to the cheap model unless explicitly overridden, so
# only the classify call pays for the reasoner.
SECONDARY_MODEL = os.environ.get("CI_AGENT_SECONDARY_MODEL", "")


def _json_mode_kwargs(model: str) -> dict:
    """response_format=json_object, except for reasoning models that reject it."""
    if any(tag in (model or "").lower() for tag in ("reasoner", "r1", "thinking")):
        return {}
    return {"response_format": {"type": "json_object"}}


def _strip_to_json(content: str) -> str:
    """Remove <think> blocks and markdown fences so json.loads succeeds."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    return re.sub(r"```(?:json)?", "", content).strip()


# ─── agent state ─────────────────────────────────────────────────────

def merge_lists(left: list, right: list) -> list:
    #“When multiple nodes update the same field, combine them using merge_lists().”
    seen = set()
    result = []
    for item in left + right:
        key = item if isinstance(item, str) else json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _build_capped_deep_log_excerpt(
    state: "AgentState",
    mode: str = "full",
) -> tuple[str, bool]:
    """Assemble a bounded deep-log excerpt from pre-extracted log windows.

    mode="full"  — up to EXCERPT_FULL_MAX_CHARS / EXCERPT_FULL_MAX_LINES.
                   Used by deep_log_analysis. Post-dedup logs typically fit
                   without truncation (≤20k chars for the 5-case benchmark).
    mode="short" — up to EXCERPT_SHORT_MAX_CHARS / EXCERPT_SHORT_MAX_LINES.
                   Used by lightweight callers that only need a quick signal.

    Budget is divided evenly across all blocks so every failed step gets
    representation. The tail of each block is kept (errors cluster there).
    """
    if mode == "short":
        max_chars = EXCERPT_SHORT_MAX_CHARS
        max_lines = EXCERPT_SHORT_MAX_LINES
    else:
        max_chars = EXCERPT_FULL_MAX_CHARS
        max_lines = EXCERPT_FULL_MAX_LINES

    blocks = [b for b in state.get("log_excerpt_texts", []) if b]
    if not blocks:
        fallback_lines = (state.get("error_lines", []) or [])[:20]
        fallback = "\n".join(fallback_lines).strip() or "(no log excerpt available)"
        return fallback[:max_chars], len(fallback) > max_chars

    n = len(blocks)
    lines_per_block = max(30, max_lines // n)
    chars_per_block = max(2000, max_chars // n)

    sections: list[str] = []
    any_truncated = False

    for block in blocks:
        block_lines = block.splitlines()
        # Take the tail — that's where the actual error output lives.
        if len(block_lines) > lines_per_block:
            kept = block_lines[-lines_per_block:]
            any_truncated = True
        else:
            kept = block_lines
        kept_text = "\n".join(kept).strip()
        if len(kept_text) > chars_per_block:
            kept_text = kept_text[-chars_per_block:]
            any_truncated = True
        if kept_text:
            sections.append(kept_text)

    excerpt = "\n\n".join(sections).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
        any_truncated = True
    return excerpt, any_truncated



class AgentState(TypedDict):
    # Input (set once)
    run_event: dict
    raw_run: dict
    api_key: str
    model: str

    # Bayesian beliefs (overwritten each update)
    beliefs: dict
    belief_history: Annotated[list, merge_lists]
    confidence: float
    entropy: float

    # Investigation progress
    tools_available: list
    tools_called: Annotated[list, merge_lists]
    investigation_log: Annotated[list, merge_lists]
    current_step: int
    done: bool

    # Evidence
    error_lines: list
    mentioned_files: list
    log_excerpt_texts: list
    changed_files: list          # files touched by the triggering commit
    commit_diff: dict
    failed_step_context: dict
    dependency_changes: dict
    run_history: dict
    similar_failures: list
    workflow_contents: list      # parsed signals from changed workflow files
    runner_environment: dict
    pr_context: dict
    preprocessing_summary: dict
    semantic_diff_links: list    # version-change ↔ error-log cross-references
    # excerpts_collected removed — was always True and never read (dead flag).

    # Output
    classification: dict

    # Internal routing
    _next_tool: str

# ─── helper: rebuild BeliefState from dict ───────────────────────────

def _get_belief_state(state: AgentState) -> BeliefState:
    bs = BeliefState()
    bs.probabilities = dict(state["beliefs"])
    bs.history = list(state["belief_history"])
    return bs

def _save_belief_state(state: AgentState, bs: BeliefState) -> AgentState:
    state["beliefs"] = dict(bs.probabilities)
    state["belief_history"] = list(bs.history)
    state["confidence"] = bs.confidence()
    state["entropy"] = bs.entropy()
    return state


def _get_client(state: AgentState) -> OpenAI:
    return make_client(api_key=state["api_key"])


# Per-channel evidence framing prepended to each tool's observation before the
# shared likelihood call. Each frame states what THIS evidence channel can and
# cannot discriminate — i.e. it specifies the likelihood model P(o|c) for the
# channel. No pattern→category rules here (that would re-encode the RPA
# tables); only the epistemics of the channel itself.
EVIDENCE_FRAMES = {
    "dependency_changes": (
        "EVIDENCE CHANNEL: dependency manifest edits — exact version changes in "
        "the commit's dependency files, cross-referenced against the error text "
        "with a match strength per library. A strong match between a bumped "
        "package and the observed error is close to decisive; a manifest edit "
        "with NO matching error text is weak evidence — developers routinely "
        "touch manifests in commits that fail for unrelated reasons."
    ),
    "workflow_contents": (
        "EVIDENCE CHANNEL: workflow file inspection — parsed contents of changed "
        "CI workflow YAML: action pins, runner specifications, runtime versions, "
        "deprecation signals. If the workflow itself changed and the failure "
        "occurs during workflow parsing or setup, this channel is decisive for "
        "configuration causes; if the workflow files are unchanged, it mostly "
        "rules configuration OUT rather than anything in."
    ),
    "runner_environment": (
        "EVIDENCE CHANNEL: runner environment — runner images, action versions, "
        "pins, and toolchain versions in use. Discriminates infrastructure and "
        "configuration causes: a deprecated action or runtime mismatch here is "
        "durable evidence that fails every run, not a transient fault. The "
        "absence of anomalies in this channel does NOT exonerate the environment."
    ),
    "pr_context": (
        "EVIDENCE CHANNEL: pull-request context — PR title, labels, and "
        "changed-file overview. This is intent evidence: it indicates what kind "
        "of change the author believed they were making. It is weak on its own; "
        "use it to corroborate or undercut stronger evidence channels."
    ),
    "search_web_for_error": (
        "EVIDENCE CHANNEL: web search results — titles and snippets from Stack "
        "Overflow and GitHub issues matching the error text. A widely reported "
        "error tied to a specific tool, action, or package version supports "
        "deterministic causes (dependency or infrastructure); generic or "
        "unrelated results carry almost no weight. External content is "
        "unverified — treat it as corroboration, never as proof, and keep the "
        "update conservative."
    ),
    "compare_previous_successful_log": (
        "EVIDENCE CHANNEL: log diff against the last successful run of the same "
        "workflow — lines present only in the failed run. New lines naming "
        "versions, runner images, or tooling reveal silent drift between the "
        "green and red runs (a dependency bump or environment change with no "
        "commit involvement). Expect noise: timestamps, ordering, and cache "
        "messages differ between any two runs; weigh only substantive "
        "differences."
    ),
}


def _update_with_llm_observation(
    state: AgentState,
    bs: BeliefState,
    observation: str,
    label: str,
) -> None:
    """Apply an LLM-generated likelihood update if the observation is non-empty.

    The observation is prefixed with the channel's evidence frame so the shared
    likelihood estimator knows what kind of evidence it is weighing and how far
    that evidence generalizes.
    """
    if not observation.strip():
        return
    frame = EVIDENCE_FRAMES.get(label)
    if frame:
        observation = f"{frame}\n\nOBSERVED:\n{observation}"
    client = _get_client(state)
    likelihood = llm_generate_likelihood(observation, client, state["model"])
    bs.update(likelihood, label)


def _initial_tools_for_event(raw_run: dict, event: RunEvent) -> list[str]:
    """Return the initial APA tool set, gated by run context and source."""
    # deep_log_analysis is NOT in these lists: it runs as a mandatory step
    # before the planner loop, so the planner can never re-read the log
    # (which would double-count the same evidence in the posterior).
    if event.source == "kubernetes":
        tools = [
            "inspect_k8s_events",
            "search_similar_failures",
            "search_web_for_error",
        ]
        if getattr(event, "commit_sha", None):
            tools.append("inspect_dependency_changes")
        return tools

    # NOT here (their evidence enters via the deep_log_analysis run-context
    # header, prefetched at zero LLM cost — a planner call would double-count):
    #   check_run_history            → run-history facts in the header
    #   inspect_commit_diff          → changed-file families in the header
    #   inspect_failed_step_context  → failed-step shape in the header
    # Planner tools are reserved for deep, expensive, external, or
    # conditionally available evidence.
    tools = [
        "inspect_dependency_changes",
        "inspect_runner_environment",
        "inspect_workflow_file",
        "search_similar_failures",
        "search_web_for_error",
        "compare_previous_successful_log",
    ]

    prs = raw_run.get("pull_requests") or []
    is_pr_event = str(event.event or "").lower() in {
        "pull_request",
        "pull_request_target",
        "pull_request_review",
    }
    if is_pr_event or prs:
        tools.append("inspect_pr_context")

    return tools


# cache the deserialized RunEvent so each tool node does not re-parse the same dict on every invocation.
# Keyed on a stable content hash of run_event rather than id() to avoid the id-reuse hazard
_RUN_EVENT_CACHE: dict[str, "RunEvent"] = {}

def _run_event_cache_key(d: dict) -> str:
    return json.dumps(
        {k: d[k] for k in ("run_id", "repo", "run_number", "attempt") if k in d},
        sort_keys=True,
    )


def _get_event(state: AgentState) -> RunEvent:
    d = state["run_event"]
    cache_key = _run_event_cache_key(d)
    cached = _RUN_EVENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from src.apa.intake_parser import RunEvent, FailedStepInfo
    failed_steps = []
    for fs in d.get("failed_steps", []):
        failed_steps.append(FailedStepInfo(**{
            k: v for k, v in fs.items()
            if k in FailedStepInfo.__dataclass_fields__
        }))
    event = RunEvent(**{
        k: v for k, v in d.items()
        if k in RunEvent.__dataclass_fields__ and k != "failed_steps"
    })
    event.failed_steps = failed_steps
    _RUN_EVENT_CACHE[cache_key] = event
    return event

def _signal_from_mentioned_files(files: List[dict]) -> dict:
    """Deterministic likelihood from file paths mentioned in error text."""
    base = {cat: 0.08 for cat in CATEGORIES}
    if not files:
        return {cat: 1.0 / len(CATEGORIES) for cat in CATEGORIES}

    for item in files:
        path = (item.get("path") or "").lower()
        if not path:
            continue

        ext = os.path.splitext(path)[1]
        base_name = path.split("/")[-1]

        if ".github/workflows/" in path or base_name in {"action.yml", "action.yaml"}:
            base["CONFIG_ERROR"] += 0.20
            base["INFRA_INCOMPATIBILITY"] += 0.05
        elif any(
            token in path
            for token in (
                "pom.xml", "build.gradle", "build.gradle.kts", "cargo.toml",
                "package.json", "requirements.txt", "pyproject.toml",
                "go.mod", "go.sum", "gemfile",
            )
        ):
            base["DEPENDENCY_CONFLICT"] += 0.22
            base["INFRA_INCOMPATIBILITY"] += 0.03
        elif any(seg in path for seg in ("/test/", "/tests/", "/spec/", "_test.", "_spec.")):
            base["TEST_FLAKINESS"] += 0.10
            base["CODE_REGRESSION"] += 0.05
        elif ext in {
            ".py", ".java", ".kt", ".kts", ".go", ".rs", ".ts", ".tsx",
            ".js", ".jsx", ".c", ".cc", ".cpp", ".cxx", ".rb", ".swift", ".cs",
        }:
            # Source file mentioned in error — modest signal; could be compile/import error
            base["CODE_REGRESSION"] += 0.06
            base["DEPENDENCY_CONFLICT"] += 0.04
        elif ext in {".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env"}:
            base["CONFIG_ERROR"] += 0.08

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


def _prepare_log_evidence(raw_run: dict, event: RunEvent) -> tuple[list[str], list[dict], list[str]]:
    """Prepare log evidence deterministically before the agent loop."""
    precomputed = raw_run.get("precomputed_log_evidence")
    if isinstance(precomputed, dict):
        error_lines = precomputed.get("error_lines")
        mentioned_files = precomputed.get("mentioned_files")
        log_excerpt_texts = precomputed.get("log_excerpt_texts")

        if isinstance(error_lines, list) and isinstance(mentioned_files, list) and isinstance(log_excerpt_texts, list):
            cleaned_error_lines: list[str] = [str(x) for x in error_lines if isinstance(x, str) and x.strip()]
            cleaned_mentioned: list[dict] = [x for x in mentioned_files if isinstance(x, dict)]
            cleaned_excerpts: list[str] = [str(x) for x in log_excerpt_texts if isinstance(x, str) and x.strip()]
            return cleaned_error_lines[:50], cleaned_mentioned[:12], cleaned_excerpts[:3]

    tarball = (raw_run.get("logs_archive") or {}).get("path", "")
    if tarball.startswith("/data/"):
        tarball = tarball[len("/data/"):]

    error_lines: List[str] = []
    excerpt_texts: List[str] = []
    mentioned: List[MentionedFile] = []

    for fs in event.failed_steps[:3]:
        ex = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )
        if ex.error_windows:
            excerpt_texts.append(ex.as_prompt_text(header=True))
            mentioned.extend(extract_from_excerpt_windows(ex.error_windows))

        summary_lines = extract_error_summary_lines(
            windows=ex.error_windows,
            marker_lines=ex.error_marker_lines,
            max_lines=50,
        )

        for line in summary_lines:
            if line not in error_lines:
                error_lines.append(line)

    mentioned_dicts = []
    seen = set()
    for m in sorted(mentioned, key=lambda x: (x.line is None, x.path)):
        key = (m.path, m.line, m.column, m.context)
        if key in seen:
            continue
        seen.add(key)
        mentioned_dicts.append(
            {
                "path": m.path,
                "line": m.line,
                "column": m.column,
                "context": m.context,
            }
        )

    return error_lines[:50], mentioned_dicts[:12], excerpt_texts[:3]


def _build_preprocessing_state(event: RunEvent, raw_run: dict) -> dict:
    """Run the cheap deterministic layer shared by RPA and APA.

    All signals here are zero-LLM and run before either the RPA decision
    rules or the APA agent loop. Adding a signal here improves BOTH systems.

    Signals applied (in order):
      1. branch_type         — bot branch vs protected vs unprotected
      2. jobs_failed         — ratio of failed jobs
      3. detection_mode      — how intake identified the failing step
      4. commit_message      — keyword patterns in the commit message
      5. error_text          — keyword patterns in extracted error lines
      6. mentioned_files     — file-type families in the error window
      7. step_duration       — how late in the build the failure occurred
      8. parent_commit_run   — was the immediately preceding run passing?
         (single GitHub API call; strongest predictor per Rausch et al. 2017)
      9. tooling_artifact    — dataset noise override
    """
    error_lines, mentioned_files, log_excerpt_texts = _prepare_log_evidence(raw_run, event)
    bs = BeliefState()
    signals_applied: List[str] = []

    def apply(likelihood: dict, name: str) -> None:
        bs.update(likelihood, name)
        signals_applied.append(name)

    signals = event.available_signals

    if "branch_type" in signals and event.branch is not None:
        apply(signal_branch_type(bool(event.is_protected_branch), event.branch), "branch_type")

    if "many_jobs_failed" in signals:
        apply(signal_many_jobs_failed(event.failed_jobs_count, event.n_jobs), "jobs_failed")

    if "detection_mode" in signals:
        apply(signal_detection_mode(event.failure_detection), "detection_mode")

    if "commit_message" in signals:
        commit_message = event.commit_message or event.commit_title or ""
        if commit_message:
            apply(signal_commit_message(commit_message), "commit_message")

    if "error_text" in signals and error_lines:
        apply(signal_error_text(error_lines), "error_text")

    if "error_text" in signals and mentioned_files:
        apply(_signal_from_mentioned_files(mentioned_files), "mentioned_files")

    # Step-duration timing signal: early failures → config/dep/infra;
    # late failures → test/code.  Uses the first failed step's duration.
    if event.failed_steps:
        first_dur = event.failed_steps[0].step_duration_sec
        apply(
            signal_step_duration(first_dur, event.failed_jobs_count, event.n_jobs),
            "step_duration",
        )

    # Parent-commit run signal — strongest single predictor.
    # One GitHub API call; skipped if repo/branch/run_number unavailable.
    # The full history dict is kept and returned so the APA side can reuse
    # it (deep_log_analysis header) without a second call to the same endpoint.
    parent_conclusion: Optional[str] = None
    run_history: dict = {}
    if event.repo and event.branch and event.run_number:
        try:
            run_history = _fetch_run_history(
                repo=event.repo,
                branch=event.branch,
                workflow_path=event.workflow or "",
                current_run_number=event.run_number,
            )
            parent_conclusion = run_history.get("parent_conclusion")
        except Exception:
            pass
    apply(signal_parent_commit_run(parent_conclusion), "parent_commit_run")

    if event.all_failures_are_tooling_artifacts:
        artifact_likelihood = {cat: 0.05 for cat in CATEGORIES}
        artifact_likelihood["TOOLING_ARTIFACT"] = 0.55
        apply(artifact_likelihood, "tooling_artifact")

    top_cat, top_prob = bs.top_category()
    preprocessing_summary = {
        "repo": event.repo,
        "workflow": event.workflow,
        "branch": event.branch,
        "protected_branch": event.is_protected_branch,
        "dependabot_like": any(bot in (event.branch or "").lower() for bot in ("dependabot", "renovate")),
        "event": event.event,
        "failure_detection": event.failure_detection,
        "tooling_artifact": event.all_failures_are_tooling_artifacts,
        "jobs_failed": event.failed_jobs_count,
        "jobs_total": event.n_jobs,
        "error_lines": len(error_lines),
        "mentioned_files": len(mentioned_files),
        "parent_conclusion": parent_conclusion,
        "signals_applied": signals_applied,
        "top_category": top_cat,
        "top_probability": top_prob,
    }

    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": list(bs.history),
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "error_lines": error_lines,
        "mentioned_files": mentioned_files,
        "log_excerpt_texts": log_excerpt_texts,
        "run_history": run_history,
        "preprocessing_summary": preprocessing_summary,
    }


# ─── node: initialize ────────────────────────────────────────────────
def initialize(state: AgentState) -> dict:
    summary = state.get("preprocessing_summary") or {}
    init_log = [
        f"[init] Starting investigation of {state['run_event'].get('repo', '?')} "
        f"run #{state['run_event'].get('run_number', '?')}"
    ]
    if summary:
        init_log.append(
            f"[init] deterministic preprocessing: top={summary.get('top_category', '?')} "
            f"({summary.get('top_probability', 0):.0%}), "
            f"error_lines={summary.get('error_lines', 0)}, "
            f"mentioned_files={summary.get('mentioned_files', 0)}, "
            f"detection={summary.get('failure_detection', '?')}"
        )
    return {
        "current_step": 0,
        "done": False,
        "investigation_log": init_log,
    }


# ─── node: planner ───────────────────────────────────────────────────

PLANNER_PROMPT = """You are investigating a CI/CD failure. Your goal is to determine what kind of failure this is.

Cheap deterministic facts are already available in the initial beliefs and investigation log:
branch type, protected branch status, job counts, failure detection mode, commit title/message, extracted error lines, mentioned files, and the commit diff (changed-file families).
Do not spend a tool call on facts that are already known.

CURRENT STATE:
  Step: {step}/{max_steps}
  Entropy: {entropy:.2f} bits
  Top beliefs: {top_beliefs}
  Tools already used: {tools_used}
  Tools still available: {tools_available}

EXPECTED INFORMATION GAIN PER TOOL (bits of entropy reduction, given current beliefs):
  A higher number means that tool is expected to resolve more uncertainty
  for a run with THIS belief profile. Use this as a calibrated prior.
{eig_table}

NOTE: The run's static metadata, its failed-step shape, its recent run history (parent-run outcome, recent failure rate), its changed-file families, and the FULL failure log excerpt have ALREADY been analyzed together (deep_log_analysis) before planning began — their findings are in the investigation log above and in the current beliefs. Do NOT spend a call re-gathering any of these facts.

AVAILABLE TOOLS:
  inspect_dependency_changes — Focus only on dependency-manifest edits and version-bump evidence linked to the error text.
  inspect_runner_environment — Inspect runner images, workflow runtime pins, and toolchain/version mismatch clues.
  inspect_workflow_file  — Fetch and parse changed workflow YAML files. Best for action pins, workflow syntax, and CI config drift.
  inspect_pr_context     — If the run belongs to a PR, inspect PR title, labels, and changed files for release-note / labeling / scope clues.
  search_similar_failures — APA-only semantic retrieval over prior failures. The cheap token-overlap retrieval was already applied in preprocessing.
  search_web_for_error   — Search StackOverflow and GitHub issues for obscure framework errors or missing packages.
  compare_previous_successful_log — Download the raw log of the last successful run and diff it against the current failed run to spot silent dependency bumps or infrastructure drift.

INVESTIGATION SO FAR:
{investigation_log}

DECISION:
If entropy is below {threshold:.2f} bits OR you've used most tools, choose "classify" to make your final judgment.
Otherwise, choose the tool that would give you the most useful information you don't have yet.
The EIG table is a strong prior — override it only if you have a domain-specific reason (e.g. the error text explicitly names a GitHub Action).

RULES:
1. You CANNOT call a tool that is already in "Tools already used" — those are done.
2. Choose from "Tools still available" only.
3. If you feel confident enough, choose "classify".
4. If "Tools still available" is empty, you MUST choose "classify".

Respond with ONLY a JSON object:
{{"tool": "tool_name_or_classify", "reasoning": "one sentence why"}}"""


def planner(state: AgentState) -> dict:
    step = state["current_step"] + 1
    bs = _get_belief_state(state)
    top3 = bs.top_n(3)
    top_str = ", ".join(f"{cat} ({prob:.0%})" for cat, prob in top3)
    available = state["tools_available"]
    # Always compute EIG rankings — used by all three planner modes.
    rankings = rank_tools_by_eig(bs, available)
    eig_table = format_eig_for_prompt(rankings)
    eig_summary = ", ".join(f"{t}={e:.3f}" for t, e in rankings)

    # ── Determine next tool ──────────────────────────────────────────
    # (Diff, run history, metadata and the full log are all consumed by the
    # mandatory deep_log_analysis step before planning begins, so the planner
    # only chooses among deep/expensive/conditional probes.)

    if PLANNER_MODE == "eig" or not available:
        # Pure EIG: deterministically pick argmax(EIG), no LLM call.
        if not available:
            tool, reasoning = "classify", "No tools remaining."
        elif state["entropy"] <= ENTROPY_THRESHOLD:
            tool, reasoning = "classify", f"Entropy {state['entropy']:.2f} <= threshold."
        else:
            tool, best_eig, _ = pick_eig_tool(bs, available)
            reasoning = f"argmax EIG={best_eig:.3f} bits."
        mode_tag = "eig"

    elif PLANNER_MODE == "llm":
        # Original LLM-only mode: no EIG context injected.
        prompt_llm = PLANNER_PROMPT.format(
            step=step, max_steps=MAX_STEPS,
            entropy=state["entropy"], top_beliefs=top_str,
            tools_used=", ".join(state["tools_called"]) or "none",
            tools_available=", ".join(available) or "none",
            eig_table="  (not provided in llm mode)",
            investigation_log="\n".join(state["investigation_log"][-5:]),
            threshold=ENTROPY_THRESHOLD,
        )
        tool, reasoning = _call_planner_llm(state, prompt_llm)
        mode_tag = "llm"

    else:
        # Hybrid (default): EIG rankings injected into prompt; LLM decides.
        prompt_hybrid = PLANNER_PROMPT.format(
            step=step, max_steps=MAX_STEPS,
            entropy=state["entropy"], top_beliefs=top_str,
            tools_used=", ".join(state["tools_called"]) or "none",
            tools_available=", ".join(available) or "none",
            eig_table=eig_table,
            investigation_log="\n".join(state["investigation_log"][-5:]),
            threshold=ENTROPY_THRESHOLD,
        )
        tool, reasoning = _call_planner_llm(state, prompt_hybrid)
        mode_tag = "hybrid"

    # ── Determine done / override ────────────────────────────────────
    override_log = []
    done = False
    if tool == "classify":
        done = True
    elif step >= MAX_STEPS:
        done = True
        override_log.append(f"[planner] max steps ({MAX_STEPS}) reached, forcing classification")
    elif tool not in available:
        done = True
        override_log.append(f"[planner/{mode_tag}] tool '{tool}' not available, forcing classification")


    # EIG rank of the chosen tool (for calibration tracking)
    eig_rank_map = {t: e for t, e in rankings}
    chosen_eig = eig_rank_map.get(tool, 0.0)

    log_entry = (
        f"[planner/{mode_tag} step {step}] "
        f"EIG: {eig_summary} | "
        f"chose '{tool}' (EIG={chosen_eig:.3f}): {reasoning}"
    )

    return {
        "current_step": step,
        "_next_tool": tool,
        "done": done,
        "investigation_log": [log_entry] + override_log,
    }


def _call_planner_llm(state: AgentState, prompt: str) -> tuple[str, str]:
    """Call the LLM planner and return (tool_name, reasoning). Falls back to 'classify'."""
    client = _get_client(state)
    try:
        _msgs = [
            {"role": "system", "content": "You are an intelligent CI/CD failure investigator. Choose wisely."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=state["model"],
            messages=_msgs,
            response_format={"type": "json_object"},
            temperature=0.0,  # reproducibility: planner tool choice must be stable run-to-run
            **usage_kwargs(),
        )
        record_usage(response, state["model"], call_type="chat", label="agent.planner")
        log_transcript("agent.planner", state["model"], _msgs, response)
        import re
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'```(?:json)?', '', content).strip()
        data = json.loads(content)
        return data.get("tool", "classify"), data.get("reasoning", "")
    except Exception as e:
        return "classify", f"Planner error: {e}"

# ─── tool nodes ──────────────────────────────────────────────────────

# Intake detection modes are internal enums; the LLM can't reason about the
# raw tokens, so each gets a one-line natural-language gloss in the prompt.
_DETECTION_GLOSS = {
    "per_step_error": "an explicit error string was found in one step's output",
    "single_step_inferred": "no explicit error text; the failing step was inferred from job structure",
    "job_level_fallback": "no step-level signal at all; only the job conclusion marked the failure",
}


def _build_run_context_header(event: RunEvent, run_history: dict, commit_diff: dict) -> str:
    """Compact fact sheet prepended to the deep_log_analysis observation.

    Carries every evidence source that is compact, free, and always available:
    the facts the RPA signal battery reads (branch, trigger, actor, commit
    title, job fan-out, failed-step shape, run history) plus the prefetched
    changed-file families. RPA and APA interpret identical evidence — RPA with
    hand-coded per-signal tables, APA jointly with the log in one LLM
    likelihood call. Each fact enters the posterior exactly once, here.

    Still EXCLUDED (owned by planner tools — deep/expensive/conditional):
      - manifest version changes        → inspect_dependency_changes
      - workflow YAML contents          → inspect_workflow_file
      - runner image / environment      → inspect_runner_environment
    """
    facts: list[str] = []

    branch = event.branch or ""
    if branch:
        flags = []
        if event.is_protected_branch:
            flags.append("protected")
        if any(b in branch.lower() for b in ("dependabot", "renovate")):
            flags.append("bot-managed dependency-update branch")
        facts.append(f"branch: {branch}" + (f"  ({', '.join(flags)})" if flags else ""))

    if event.event:
        facts.append(f"trigger: {event.event}")

    # attempt > 1 means this run is itself a retry — and it failed again,
    # which is sharp evidence against any transient (retry-fixable) cause.
    if event.attempt and event.attempt > 1:
        facts.append(
            f"run attempt: {event.attempt} — a plain retry of this run "
            f"ALREADY FAILED (evidence against transient flakiness)"
        )

    actor = event.actor or ""
    if actor:
        is_bot = "[bot]" in actor or actor.lower().endswith("bot")
        facts.append(f"actor: {actor}" + ("  (automated account)" if is_bot else ""))

    title = (event.commit_title or event.commit_message or "").strip()
    if title:
        facts.append(f"commit title: {title[:120]!r}")

    facts.append(f"jobs failed: {event.failed_jobs_count} of {event.n_jobs}")

    if event.failed_steps:
        fs = event.failed_steps[0]
        bits = []
        if fs.step_label:
            bits.append(f"'{fs.step_label[:60]}'")
        if fs.step_index is not None:
            bits.append(f"step #{fs.step_index} in the job")
        if fs.step_duration_sec is not None:
            bits.append(f"ran {fs.step_duration_sec:.0f}s before failing")
        if bits:
            facts.append("first failed step: " + ", ".join(bits))
        gloss = _DETECTION_GLOSS.get(fs.detection_mode or "")
        if gloss:
            facts.append(f"failure detection: {gloss}")
        other_labels = [
            f"'{ofs.step_label[:50]}'" for ofs in event.failed_steps[1:3] if ofs.step_label
        ]
        if other_labels:
            facts.append("other failed steps: " + ", ".join(other_labels))

    if event.workflow:
        facts.append(f"workflow: {event.workflow}")

    # Changed-file families (diff prefetched in preprocessing, zero LLM).
    families = (commit_diff or {}).get("families") or {}
    if any(families.values()):
        summary = commit_diff.get('summary', 'no notable file families')
        # Warn the LLM not to over-index on dependency files if source files also changed:
        # developers routinely touch both in the same commit.
        has_source = bool(families.get("source") or families.get("test"))
        has_dep = bool(families.get("dependency"))
        if has_source and has_dep:
            facts.append(
                f"files changed by the triggering commit: {summary} — "
                f"NOTE: both source and dependency files changed; prefer CODE_REGRESSION "
                f"unless the error log explicitly names a version conflict or missing package"
            )
        else:
            facts.append(f"files changed by the triggering commit: {summary}")

    # Run history (prefetched in preprocessing — same API call RPA uses).
    runs = run_history.get("runs") or []
    if runs:
        n_failed = sum(1 for r in runs if r.get("conclusion") == "failure")
        facts.append(f"recent runs of this workflow on this branch: {n_failed} of {len(runs)} failed")
    parent = run_history.get("parent_conclusion")
    if parent == "success":
        facts.append(
            "parent run PASSED — this failure was INTRODUCED by the triggering "
            "commit (strong evidence against pre-existing flakiness)"
        )
    elif parent == "failure":
        facts.append(
            "parent run ALSO FAILED — the failure PREDATES this commit "
            "(points toward flakiness, infrastructure, or a pre-existing regression)"
        )

    return (
        "RUN CONTEXT (read the failure log below with these facts in mind — "
        "they are circumstantial on their own, but they disambiguate the log: "
        "e.g. a bot dependency branch raises DEPENDENCY_CONFLICT, most jobs "
        "failing at once suggests CASCADE_FAILURE or infrastructure, a step "
        "failing within seconds early in the job points away from "
        "TEST_FLAKINESS/CODE_REGRESSION):\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\nFAILURE LOG EXCERPT:\n"
    )


_UNINFORMATIVE_LOG_MARKERS = (
    "process completed with exit code",
    "the operation was canceled",
    "the operation was cancelled",
    "##[error]process completed",
    "exited with code",
)


def _is_uninformative_log(excerpt: str) -> bool:
    """True when the failure log carries no real error signal.

    Many CI failures only emit a generic terminal line ("Process completed with
    exit code 1", "The operation was canceled") with no compile error, test
    assertion, stack trace, or install failure. There is nothing to diagnose
    from such a log: any specific category the LLM picks is a guess from
    circumstantial signals (which file changed, how many jobs failed), which
    adds noise rather than signal. In that regime APA should defer to the
    base-rate prior, not invent a confident "specific" category.
    """
    text = (excerpt or "").strip().lower()
    if not text:
        return True
    # Strip the generic terminal markers; if almost nothing substantive remains,
    # the log is uninformative. We keep lines that look like real diagnostics
    # (contain a path, an exception name, "error:", "failed", an assertion, etc.).
    substantive = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if any(m in ln for m in _UNINFORMATIVE_LOG_MARKERS):
            continue
        # Lines that are just job/step scaffolding carry no diagnosis.
        if ln.startswith(("##[group]", "##[endgroup]", "##[section]", "shell:", "env:")):
            continue
        substantive.append(ln)
    joined = " ".join(substantive)
    # Real diagnostics almost always contain one of these tokens.
    diagnostic_tokens = (
        "error", "exception", "traceback", "assert", "failed", "failure",
        "not found", "cannot find", "no such", "undefined", "unresolved",
        "expected", "syntaxerror", "typeerror", "warning", "denied",
        "conflict", "incompatible", "version", "lint", "eslint", "pylint",
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", "test", "err!",
    )
    # If any real diagnostic token is present, the log IS informative.
    if any(tok in joined for tok in diagnostic_tokens):
        return False
    # No diagnostic tokens and very little substantive text → uninformative.
    return len(joined) < 200


def deep_log_analysis(state: AgentState) -> dict:
    """Mandatory evidence step: one LLM read of run context + full failure excerpt.

    Runs exactly once, before the planner loop (initialize → here → planner).
    The observation is a static run-context header (metadata + run history,
    both prefetched at zero LLM cost) followed by the full log excerpt, so
    the LLM interprets the log WITH its context — a bot dependency branch or
    a passing parent run changes what the same error lines mean. This is the
    APA mirror of the RPA signal battery: identical facts, but one joint LLM
    likelihood instead of hand-coded per-signal tables.

    Post-dedup excerpts fit the 20k budget, so the agent reads the whole log
    in one pass and this evidence updates the posterior exactly once —
    a second read would double-count dependent evidence and violate the
    conditional-independence assumption of the Bayesian update.

    The commit diff is already in state (fetched in preprocessing); its
    changed-file families are surfaced in the run-context header so the LLM
    can weigh them against the log without any hand-coded category rules.
    """
    bs = _get_belief_state(state)
    excerpt, was_truncated = _build_capped_deep_log_excerpt(state, mode="full")

    # Stage 2: APA-only semantic refinement. Runs ON the deterministic Stage-1
    # excerpt (markers + dedup), keeps the error region and adds the most
    # failure-relevant context. ON by default (CI_AGENT_SEMANTIC_REFINE=0 to
    # disable for an ablation), fires only on oversized excerpts, and falls back
    # to the Stage-1 excerpt on any problem so it is never worse.
    refine_note = ""
    if semantic_refiner.is_enabled():
        # The embedding model (text-embedding-3-small) is OpenAI-only, so the
        # refiner must embed with an OpenAI client, NOT the system's DeepSeek
        # client. EMBEDDING_PROVIDER (default openai) selects it; falls back to
        # the system client if that provider has no key.
        try:
            emb_client = make_client(provider=os.environ.get("EMBEDDING_PROVIDER", "openai"))
        except Exception:
            emb_client = _get_client(state)
        cand_lines = excerpt.splitlines()
        refined_lines, rinfo = semantic_refiner.refine_lines(
            cand_lines, emb_client,
            embedding_model=os.environ.get("CI_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        if rinfo.get("applied"):
            excerpt = "\n".join(refined_lines)
            refine_note = f" [semantic-refined {rinfo['kept']}/{rinfo['of']} lines, baseline={rinfo.get('baseline')}]"

    # No hand-coded pattern->category disambiguation is injected here. The model
    # reasons over the run-context header (which carries the changed-file
    # families) and the raw log; encoding "if pattern then category" rules would
    # reproduce the RPA mapping inside the APA evidence step.
    header = _build_run_context_header(
        _get_event(state),
        state.get("run_history") or {},
        state.get("commit_diff") or {},
    )
    observation = header + excerpt

    client = _get_client(state)
    likelihood = llm_generate_likelihood(observation, client, state["model"])

    # Uninformative-log guard. When the log has no real error signal (only a
    # generic "exit code 1" / "operation was canceled"), the LLM likelihood is a
    # guess from circumstantial context, not evidence. Flatten it toward uniform
    # so this step cannot confidently swing the posterior to a "specific" wrong
    # category — APA then defers to the deterministic prior (which already
    # carries the changed-file / commit-message signals) instead of inventing
    # TEST_FLAKINESS/CONFIG_ERROR from noise. This is the single biggest source
    # of APA regressions: ~88% of the eval's scorable cases have empty logs.
    uninformative = _is_uninformative_log(excerpt)
    if uninformative:
        n = len(CATEGORIES)
        uniform = {c: 1.0 / n for c in CATEGORIES}
        # 80% uniform / 20% LLM: keep a faint nudge from any weak signal the LLM
        # found, but make it near-impossible to override the prior on no evidence.
        likelihood = {c: 0.8 * uniform[c] + 0.2 * likelihood.get(c, 1.0 / n) for c in CATEGORIES}
        total = sum(likelihood.values())
        likelihood = {c: v / total for c, v in likelihood.items()}

    bs.update(likelihood, "deep_log_analysis")

    # Not in tools_available (mandatory step), so nothing to remove there.
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["deep_log_analysis"],
        "investigation_log": [
            f"[deep_log_analysis] run context + full excerpt reviewed"
            f"{' (truncated)' if was_truncated else ''}{refine_note}"
            f"{' [uninformative log — likelihood flattened toward prior]' if uninformative else ''}. "
            f"Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
        ],
    }


# inspect_failed_step_context and inspect_commit_diff were removed as planner
# tools. Both consumed evidence that is compact, free, and always available —
# failed-step metadata from the intake payload and the prefetched changed-file
# families — so their facts now enter the posterior exactly once, through the
# run-context header of the mandatory deep_log_analysis step. Planner tools
# are reserved for evidence that is deep, expensive, external, or conditional.


def _failed_step_summary(event: RunEvent) -> dict:
    """Deterministic failed-step dossier for the classify prompt (zero LLM)."""
    if not event.failed_steps:
        return {}
    return {
        "failure_detection": event.failure_detection,
        "failed_jobs_count": event.failed_jobs_count,
        "total_jobs": event.n_jobs,
        "runner_images": sorted({fs.runner_image for fs in event.failed_steps[:3] if fs.runner_image}),
        "detection_modes": sorted({fs.detection_mode for fs in event.failed_steps[:3] if fs.detection_mode}),
        "steps": [
            {
                "job_file": fs.job_file,
                "runner_image": fs.runner_image,
                "step_type": fs.step_type,
                "step_label": fs.step_label,
                "detection_mode": fs.detection_mode,
                "tooling_artifact_suspected": fs.tooling_artifact_suspected,
            }
            for fs in event.failed_steps[:3]
        ],
    }


# A commit whose PRIMARY purpose is a dependency update (dependabot/renovate
# branch, a "bump …"/"chore(deps): …" message, or a manifest-only change) that
# breaks CI is a DEPENDENCY_CONFLICT even when the generic log carries no explicit
# version-mismatch line — the dependency change is what introduced the failure.
# This is distinct from a commit that merely touches a manifest alongside source
# code (that stays a CODE_REGRESSION lean).
_DEPENDENCY_COMMIT_RE = re.compile(
    r"\bbump\b|\bchore\(deps|\bdeps?\)\s*:|\bdependabot\b|\brenovate\b"
    r"|\bupgrade\b|\bdowngrade\b"
    r"|update\s+(?:the\s+)?(?:dependenc|dep\b|deps\b|lockfile|package\.json|requirements|go\.mod)"
    r"|pin\s+(?:dependenc|version|the\b)",
    re.I,
)


def _is_primary_dependency_commit(event, all_files: list, dep_files: list) -> bool:
    branch = (getattr(event, "branch", "") or "").lower()
    if any(b in branch for b in ("dependabot", "renovate")):
        return True
    title = f"{getattr(event, 'commit_title', '') or ''} {getattr(event, 'commit_message', '') or ''}"
    if _DEPENDENCY_COMMIT_RE.search(title):
        return True
    # Manifest-only commit (deps changed, no source/test touched).
    fam = _categorize_files(all_files or [])
    n_src = len(fam.get("source") or []) + len(fam.get("test") or [])
    return bool(dep_files) and n_src == 0


def inspect_dependency_changes(state: AgentState) -> dict:
    """Fetch only dependency-related changes and semantically link them to the error."""
    event = _get_event(state)
    bs = _get_belief_state(state)

    diff = {"files": state.get("changed_files", [])}
    if not diff["files"] and event.repo and event.commit_sha:
        diff = _fetch_commit_diff(event.repo, event.commit_sha)

    summary = _summarize_patch_files(diff.get("files", []))

    dep_files = []
    for f in diff.get("files", []):
        families = _categorize_files([f])
        if families.get("dependency"):
            dep_files.append(f)

    sem = analyze_semantic_diff(dep_files, state.get("error_lines", [])) if dep_files else {
        "linked_evidence": [],
        "observation_text": "",
    }

    dep_file_names = [f.get("filename", "") for f in dep_files[:8] if f.get("filename")]
    dep_context = ""
    if dep_files:
        error_text_lower = " ".join(state.get("error_lines", [])).lower()
        has_version_mismatch = any(w in error_text_lower for w in (
            "no matching version", "version conflict", "incompatible version",
            "lockfile out of date", "could not find", "no compatible",
            "resolution failed", "dependency conflict", "loosen the range",
            "could not solve", "solve the dependency", "cannot satisfy",
            "pip install", "package not found", "module not found",
            "cannot find module", "failed to resolve",
        ))
        all_files = diff.get("files", [])
        if not has_version_mismatch:
            if _is_primary_dependency_commit(event, all_files, dep_files):
                # The commit's primary purpose IS a dependency update; a dep-update
                # commit that breaks CI is a DEPENDENCY_CONFLICT even with a generic log.
                dep_context = (
                    "NOTE: the triggering commit is itself a DEPENDENCY OPERATION "
                    f"(branch/message indicates a dependency bump or update: "
                    f"'{(event.commit_title or '')[:80]}'). A dependency-update commit that "
                    "breaks CI is a DEPENDENCY_CONFLICT even when the generic log shows no "
                    "explicit version-mismatch line, because the dependency change is what "
                    "introduced the failure. Favor DEPENDENCY_CONFLICT here."
                )
            else:
                # Check if source/test files also changed (common: dev touches dep file + source in one commit)
                source_families = _categorize_files(all_files)
                also_has_source = bool(source_families.get("source") or source_families.get("test"))
                if also_has_source:
                    dep_context = (
                        "NOTE: dependency manifests changed but source/test files also changed "
                        "and the logs show no version-mismatch or missing-package error. "
                        "This is likely a CODE_REGRESSION — developers often touch dep files in "
                        "the same commit as source code. Do NOT classify as DEPENDENCY_CONFLICT "
                        "unless the error log explicitly names a version conflict or install failure."
                    )
                else:
                    dep_context = (
                        "NOTE: dependency manifests changed but the logs show no direct "
                        "version-mismatch or missing-package error, so the manifest edit is "
                        "weak, circumstantial evidence rather than a confirmed conflict."
                    )

    observation_parts = []
    if dep_file_names:
        observation_parts.append("dependency_files:\n" + "\n".join(f"- {name}" for name in dep_file_names))
    if sem.get("observation_text"):
        observation_parts.append(sem["observation_text"])
    if dep_context:
        observation_parts.append(dep_context)
    _update_with_llm_observation(
        state,
        bs,
        "\n\n".join(observation_parts),
        "dependency_changes",
    )

    remaining = [t for t in state["tools_available"] if t != "inspect_dependency_changes"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["inspect_dependency_changes"],
        "tools_available": remaining,
        "changed_files": diff.get("files", []),
        "commit_diff": summary,
        "dependency_changes": {
            "files": dep_files,
            "summary": f"{len(dep_files)} dependency file(s) changed",
            "semantic_observation": sem.get("observation_text", ""),
        },
        "semantic_diff_links": sem.get("linked_evidence", []),
        "investigation_log": [
            f"[inspect_dependency_changes] {len(dep_files)} dependency file(s), "
            f"{len(sem.get('linked_evidence', []))} version<->error link(s). "
            f"Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
        ],
    }


# _apply_final_category_overrides was removed: it was a hand-coded
# pattern->category rule table re-applied as a hard post-override on the LLM's
# decision, which reproduced the RPA mapping at APA's decision stage and
# contradicted the rules-vs-reasoning contrast. The final category is now the
# LLM's choice (optionally revised by the devil's-advocate review), with no
# deterministic override.


# check_run_history was removed as a planner tool: the run history is now
# prefetched in preprocessing (the same single API call that feeds the RPA
# parent-commit signal) and enters the posterior through the run-context
# header of deep_log_analysis. A separate tool call would re-fetch the same
# endpoint and double-count the same evidence.


def search_web_for_error(state: AgentState) -> dict:
    """Search the web for the error message."""
    bs = _get_belief_state(state)
    
    log_excerpt = state.get("deep_log_excerpt", "")
    if not log_excerpt:
        log_excerpt, _ = _build_capped_deep_log_excerpt(state, mode="short")
        
    client = _get_client(state)
    model = state.get("model", DEFAULT_MODEL)
    query_prompt = f"Extract a short, precise 3-6 word search query from this error log to search StackOverflow or GitHub issues. Only output the query string, nothing else.\\n\\n{log_excerpt[:1000]}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query_prompt}],
            temperature=0.0,
            max_tokens=30
        )
        query = resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        query = "GitHub Actions failure"
        
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(f"{query} site:stackoverflow.com OR site:github.com", max_results=3)
        snippets = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        note = f"[search_web_for_error] Searched '{query}'. Top results: {snippets}"
    except Exception as e:
        snippets = ""
        note = f"[search_web_for_error] Search failed: {e}"

    # Results enter the posterior through the shared estimator, framed by the
    # web-search evidence channel. A failed/empty search applies no update.
    prev_history_len = len(bs.history)
    if snippets:
        observation = f"search query: {query}\nresults:\n{snippets}"
        _update_with_llm_observation(state, bs, observation, "search_web_for_error")

    remaining = [t for t in state["tools_available"] if t != "search_web_for_error"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": bs.history[prev_history_len:],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["search_web_for_error"],
        "tools_available": remaining,
        "investigation_log": [note]
    }

def compare_previous_successful_log(state: AgentState) -> dict:
    """Fetch the log of the last successful run and diff it against the current failed log."""
    bs = _get_belief_state(state)
    event = _get_event(state)
    
    history = state.get("run_history")
    if not history:
        history = _fetch_run_history(event.repo, event.branch, event.workflow, event.run_number)
        
    runs = history.get("runs", [])
    successful_runs = [r for r in runs if r.get("conclusion") == "success" and r.get("run_number") < event.run_number]
    
    note = ""
    diff_text = ""
    last_success: dict = {}
    if not successful_runs:
        note = "[compare_previous_successful_log] No previous successful runs found to diff against."
    else:
        last_success = successful_runs[0]
        success_run_id = last_success.get("id")
        
        token = os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
            
        try:
            import zipfile
            import io
            import difflib
            import requests
            
            url_success = f"https://api.github.com/repos/{event.repo}/actions/runs/{success_run_id}/logs"
            resp_s = requests.get(url_success, headers=headers, timeout=20)
            success_log_text = ""
            if resp_s.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp_s.content)) as z:
                    for filename in z.namelist():
                        if not "/" in filename:
                            success_log_text += z.read(filename).decode("utf-8", errors="replace") + "\\n"
                            
            raw_run = state.get("raw_run", {})
            failed_run_id = raw_run.get("id")
            failed_log_text = ""
            if failed_run_id:
                url_failed = f"https://api.github.com/repos/{event.repo}/actions/runs/{failed_run_id}/logs"
                resp_f = requests.get(url_failed, headers=headers, timeout=20)
                if resp_f.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp_f.content)) as z:
                        for filename in z.namelist():
                            if not "/" in filename:
                                failed_log_text += z.read(filename).decode("utf-8", errors="replace") + "\\n"
                                
            if success_log_text and failed_log_text:
                s_lines = success_log_text.splitlines()
                f_lines = failed_log_text.splitlines()
                
                # Take only the last 2000 lines
                s_lines = s_lines[-2000:]
                f_lines = f_lines[-2000:]
                
                diff = list(difflib.unified_diff(s_lines, f_lines, n=1))
                diff_text = "\n".join(diff[:150])

                if not diff:
                    note = "[compare_previous_successful_log] Logs fetched, but no differences found in the last 2000 lines."
                else:
                    note = f"[compare_previous_successful_log] Diff between successful run {last_success.get('run_number')} and this failed run:\n{diff_text}"
            else:
                note = "[compare_previous_successful_log] Could not fetch raw logs from GitHub API."
        except Exception as e:
            note = f"[compare_previous_successful_log] Failed to compare logs: {e}"

    # The diff enters the posterior through the shared estimator, framed by
    # the log-diff evidence channel. No diff retrieved → no update.
    prev_history_len = len(bs.history)
    if diff_text:
        observation = (
            f"diff of failed run against last successful run "
            f"#{last_success.get('run_number', '?')} (unified diff, capped):\n"
            f"{diff_text[:8000]}"
        )
        _update_with_llm_observation(state, bs, observation, "compare_previous_successful_log")

    remaining = [t for t in state["tools_available"] if t != "compare_previous_successful_log"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": bs.history[prev_history_len:],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["compare_previous_successful_log"],
        "tools_available": remaining,
        "investigation_log": [note]
    }

def _categorize_files(files: list) -> dict:
    """Group filenames into file families for the observation formatter."""
    families: dict = {
        "workflow": [], "dependency": [], "source": [],
        "test": [], "config": [], "docs": [], "other": [],
    }
    workflow_exts = {".yml", ".yaml"}
    dep_names = {
        "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
        "settings.gradle.kts", "cargo.toml", "cargo.lock",
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "requirements.txt", "requirements-dev.txt", "pyproject.toml",
        "setup.py", "setup.cfg", "poetry.lock",
        "go.mod", "go.sum",
        "gemfile", "gemfile.lock",
    }
    source_exts = {
        ".py", ".java", ".kt", ".kts", ".go", ".rs",
        ".ts", ".tsx", ".js", ".jsx",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
        ".rb", ".swift", ".cs", ".php", ".scala", ".clj",
    }
    doc_exts = {".md", ".rst", ".txt"}
    doc_names = {"readme", "changelog", "license", "contributing", "authors"}

    for f in files:
        fname = f.get("filename", "")
        lower = fname.lower()
        base = lower.split("/")[-1]
        _, ext = os.path.splitext(lower)

        if ".github/workflows/" in lower or lower.endswith("action.yml") or lower.endswith("action.yaml"):
            families["workflow"].append(fname)
        elif "dockerfile" in base or lower.endswith(".dockerignore"):
            families["workflow"].append(fname)
        elif base in dep_names:
            families["dependency"].append(fname)
        elif ext in source_exts:
            if any(seg in lower for seg in ("/test/", "/tests/", "/spec/", "_test.", "_spec.", "test_")):
                families["test"].append(fname)
            else:
                families["source"].append(fname)
        elif ext in doc_exts or any(base.startswith(d) for d in doc_names):
            families["docs"].append(fname)
        elif ext in {".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env"}:
            families["config"].append(fname)
        else:
            families["other"].append(fname)

    return {k: v for k, v in families.items() if v}


def _fetch_workflow_content(repo: str, path: str, sha: str) -> str:
    """Fetch raw text content of a file at a given commit SHA via GitHub API."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(url, headers=headers, params={"ref": sha}, timeout=15)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        content_b64 = data.get("content", "")
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_workflow_signals(text: str) -> dict:
    """Extract actionable signals from workflow YAML text."""
    action_versions = []
    runners = []
    pinned_versions = []
    deprecated_nodes = []

    for line in text.splitlines():
        stripped = line.strip()

        # uses: owner/action@version  (with or without leading "- ")
        m = re.match(r"(?:-\s+)?uses:\s+([^\s#]+)", stripped)
        if m:
            ref = m.group(1)
            action_versions.append(ref)
            # flag deprecated node12/node16 runtimes
            if re.search(r"@v[12]\b|node12|node16", ref, re.IGNORECASE):
                deprecated_nodes.append(ref)

        # runs-on: ubuntu-latest / windows-latest / macos-12 / etc.
        m = re.match(r"runs-on:\s+([^\s#\[]+)", stripped)
        if m:
            runners.append(m.group(1).strip("'\""))

        # env / matrix version-like values: KEY: "3.11" or java-version: 17
        m = re.match(r"[\w-]*[Vv]ersion[\w-]*:\s+['\"]?([\d][^\s'\"#,\]]+)['\"]?", stripped)
        if m:
            pinned_versions.append(f"{stripped.split(':')[0].strip()}: {m.group(1)}")

    # Deduplicate while preserving order
    def dedup(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    return {
        "action_versions": dedup(action_versions),
        "runners": dedup(runners),
        "pinned_versions": dedup(pinned_versions),
        "deprecated_nodes": dedup(deprecated_nodes),
    }


def _find_runtime_versions(text: str) -> dict[str, list[str]]:
    """Extract runtime/version clues from workflow YAML or log excerpts."""
    patterns = {
        "python": r"(?:python-version|python)\s*[:=]\s*['\"]?([0-9][0-9A-Za-z._-]*)",
        "node": r"(?:node-version|node)\s*[:=]\s*['\"]?([0-9][0-9A-Za-z._-]*)",
        "java": r"(?:java-version|java)\s*[:=]\s*['\"]?([0-9][0-9A-Za-z._-]*)",
        "go": r"(?:go-version|golang|go)\s*[:=]\s*['\"]?([0-9][0-9A-Za-z._-]*)",
        "ruby": r"(?:ruby-version|ruby)\s*[:=]\s*['\"]?([0-9][0-9A-Za-z._-]*)",
    }
    found: dict[str, list[str]] = {name: [] for name in patterns}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        deduped = []
        seen = set()
        for item in matches:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        found[name] = deduped[:4]
    return found


def _fetch_commit_diff(repo: str, sha: str) -> dict:
    """Fetch a commit's changed files and diff statistics from GitHub."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return {"additions": 0, "deletions": 0, "files": []}
        data = resp.json()
        stats = data.get("stats") or {}
        files = [
            {
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "changes": f.get("changes", 0),
                # 2000 chars: enough for most single-dep version bumps with context.
                # Semantic diff analysis works best with full context around changed lines.
                "patch_excerpt": (f.get("patch") or "")[:2000],
            }
            for f in (data.get("files") or [])
        ]
        return {
            "additions": stats.get("additions", 0),
            "deletions": stats.get("deletions", 0),
            "files": files,
        }
    except Exception:
        return {"additions": 0, "deletions": 0, "files": []}


def _summarize_patch_files(files: list) -> dict:
    families = _categorize_files(files)
    diff_summary = []
    if families.get("workflow"):
        diff_summary.append(f"workflow files: {', '.join(families['workflow'][:4])}")
    if families.get("dependency"):
        diff_summary.append(f"dependency manifests: {', '.join(families['dependency'][:4])}")
    if families.get("source"):
        diff_summary.append(f"{len(families['source'])} source file(s)")
    if families.get("test"):
        diff_summary.append(f"{len(families['test'])} test file(s)")
    if families.get("config"):
        diff_summary.append(f"config files: {', '.join(families['config'][:4])}")
    if families.get("docs"):
        diff_summary.append(f"docs files: {', '.join(families['docs'][:4])}")
    return {
        "files": files,
        "families": families,
        "summary": "; ".join(diff_summary) if diff_summary else "no notable file families",
    }


def _fetch_run_history(
    repo: str, branch: str, workflow_path: str, current_run_number: int
) -> dict:
    """
    Single GitHub API call that returns both:
      - recent matching runs (for aggregate failure-rate signal)
      - parent run conclusion (for the commit counterfactual signal)

    Replaces the previous pair of _fetch_recent_workflow_runs /
    _fetch_parent_run_status, which each made identical requests to the
    same endpoint with the same parameters.
    """
    empty = {"runs": [], "summary": "no run history retrieved", "parent_conclusion": None}
    if not repo or not branch:
        return empty
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{repo}/actions/runs"
    params = {"branch": branch, "per_page": 10}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code != 200:
            return empty
        data = resp.json() or {}
        all_runs = data.get("workflow_runs") or []

        matching_runs = []
        parent_candidates = []

        for run in all_runs:
            run_path = run.get("path", "")
            if workflow_path and run_path and workflow_path not in run_path and run_path not in workflow_path:
                continue
            rn = run.get("run_number", 0)
            conclusion = run.get("conclusion")

            # Collect recent runs for aggregate summary (up to 5)
            if len(matching_runs) < 5:
                matching_runs.append({
                    "id": run.get("id"),
                    "name": run.get("name"),
                    "conclusion": conclusion,
                    "status": run.get("status"),
                    "created_at": run.get("created_at"),
                    "html_url": run.get("html_url"),
                    "path": run_path,
                    "run_number": rn,
                })

            # Collect parent candidates (runs before this one)
            if current_run_number and rn < current_run_number:
                parent_candidates.append((rn, conclusion))

        n_total = len(matching_runs)
        n_failed = sum(1 for r in matching_runs if r.get("conclusion") == "failure")
        summary = (
            f"{n_failed}/{n_total} recent matching runs failed"
            if n_total else "no matching recent runs found"
        )

        parent_conclusion: Optional[str] = None
        if parent_candidates:
            parent_candidates.sort(key=lambda x: -x[0])  # highest run_number first
            parent_conclusion = parent_candidates[0][1]

        return {
            "runs": matching_runs,
            "summary": summary,
            "parent_conclusion": parent_conclusion,
        }
    except Exception:
        return empty


def _load_similarity_corpus() -> list:
    """Load previously evaluated cases for similarity search.

    Fix 2: @lru_cache(maxsize=1) with no arguments made the lifetime implicit
    and stale across batch-eval runs if files changed. Now called once at
    import time into _SIMILARITY_CORPUS so the lifetime is explicit.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = [
        os.path.join(base_dir, "data", "honest_eval_results.json"),
        os.path.join(base_dir, "data", "test_agent_batch_results.json"),
        "/home/guc_alaa/honest_eval_results.json",
        "/home/guc_alaa/test_agent_batch_results.json",
    ]
    corpus = []
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("cases", [])
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            if "case_label" in item:
                intake = item.get("intake", {})
                extraction = item.get("extraction", {})
                classification = item.get("classification", {})
                corpus.append( 
                    {
                        "label": item.get("case_label", ""),
                        "repo": intake.get("repo", ""),
                        "workflow": intake.get("workflow", ""),
                        "commit_title": intake.get("commit_title", ""),
                        "commit_message": intake.get("commit_message", ""),
                        "error_lines": extraction.get("error_lines", []),
                        "mentioned_files": extraction.get("mentioned_files", []),
                        "category": classification.get("category", ""),
                        "action": classification.get("action", ""),
                    }
                )
            elif "label" in item and "result" in item:
                classification = item.get("result", {}).get("classification", {})
                corpus.append(
                    {
                        "label": item.get("label", ""),
                        "repo": item.get("result", {}).get("repo", ""),
                        "workflow": item.get("result", {}).get("workflow", ""),
                        "commit_title": item.get("result", {}).get("commit_title", ""),
                        "commit_message": item.get("result", {}).get("commit_message", ""),
                        "error_lines": item.get("result", {}).get("error_lines", []),
                        "mentioned_files": item.get("result", {}).get("mentioned_files", []),
                        "category": classification.get("category", ""),
                        "action": classification.get("action", ""),
                    }
                )
    return corpus


# Fix 2 (cont.): load once at import time — lifetime is explicit.
_SIMILARITY_CORPUS: list = _load_similarity_corpus()



# “Open the .github/workflows/...yml file and check whether the CI configuration is the reason for the failure.”
def inspect_workflow_file(state: AgentState) -> dict:
    """Fetch and parse changed workflow YAML files to surface exact version/runner signals."""
    event = _get_event(state)
    repo = event.repo
    sha = event.commit_sha

    changed_files = state.get("changed_files", [])
    _fallback_fetched_files = False
    if not changed_files and repo and sha:
        # changed_files is normally prefetched in run_agent; this fallback
        # only fires when the state was built without the prefetch.
        _fallback_fetched_files = True
        diff = _fetch_commit_diff(repo, sha)
        changed_files = diff.get("files", [])

    # Only look at workflow files from the changed_files list.
    wf_files = [
        f["filename"] for f in changed_files
        if ".github/workflows/" in f.get("filename", "").lower()
        or f.get("filename", "").lower().endswith(("action.yml", "action.yaml"))
    ]

    all_signals: dict = {
        "action_versions": [], "runners": [],
        "pinned_versions": [], "deprecated_nodes": [],
    }
    files_fetched = []

    for wf_path in wf_files[:3]:   # cap at 3 files to stay within budget
        text = _fetch_workflow_content(repo, wf_path, sha) if repo and sha else ""
        if not text:
            continue
        signals = _parse_workflow_signals(text)
        files_fetched.append(wf_path)
        for k in all_signals:
            all_signals[k].extend(signals.get(k, []))

    # Deduplicate across files
    def dedup(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]
    for k in all_signals:
        all_signals[k] = dedup(all_signals[k])

    bs = _get_belief_state(state)
    if any(all_signals.values()):
        obs = format_observation_workflow_contents(all_signals)
        _update_with_llm_observation(state, bs, obs, "workflow_contents")

    n_fetched = len(files_fetched)
    n_actions = len(all_signals["action_versions"])
    n_deprecated = len(all_signals["deprecated_nodes"])
    summary = (
        f"{n_fetched} file(s) read, {n_actions} action pins"
        + (f", {n_deprecated} deprecated" if n_deprecated else "")
    )

    remaining = [t for t in state["tools_available"] if t != "inspect_workflow_file"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["inspect_workflow_file"],
        "tools_available": remaining,
        "changed_files": changed_files,
        "workflow_contents": [{"file": f, **all_signals} for f in files_fetched] if files_fetched else [],
        "investigation_log": [
            (
                "[inspect_workflow_file] "
                + ("[fallback: changed_files was not prefetched — fetched from GitHub here] " if _fallback_fetched_files else "")
                + f"{summary}. Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
            )
        ],
    }


def inspect_runner_environment(state: AgentState) -> dict:
    """Inspect runner image and toolchain/version clues for environment mismatch."""
    event = _get_event(state)
    bs = _get_belief_state(state)

    changed_files = state.get("changed_files", [])
    if not changed_files and event.repo and event.commit_sha:
        diff = _fetch_commit_diff(event.repo, event.commit_sha)
        changed_files = diff.get("files", [])

    wf_files = [
        f["filename"] for f in changed_files
        if ".github/workflows/" in f.get("filename", "").lower()
        or f.get("filename", "").lower().endswith(("action.yml", "action.yaml"))
    ]

    workflow_texts = []
    workflow_signals = []
    for wf_path in wf_files[:3]:
        text = _fetch_workflow_content(event.repo, wf_path, event.commit_sha) if event.repo and event.commit_sha else ""
        if not text:
            continue
        workflow_texts.append(text)
        workflow_signals.append({"file": wf_path, **_parse_workflow_signals(text)})

    combined_workflow_text = "\n".join(workflow_texts)
    runtime_versions = _find_runtime_versions(
        "\n".join(state.get("log_excerpt_texts", [])[:2]) + "\n" + combined_workflow_text
    )
    runner_images = sorted({fs.runner_image for fs in event.failed_steps if fs.runner_image})
    action_versions = []
    pinned_versions = []
    parsed_runners = []
    deprecated_nodes = []
    for item in workflow_signals:
        action_versions.extend(item.get("action_versions", []))
        pinned_versions.extend(item.get("pinned_versions", []))
        parsed_runners.extend(item.get("runners", []))
        deprecated_nodes.extend(item.get("deprecated_nodes", []))

    observation_lines = [
        f"failed_step_runners: {', '.join(runner_images) or 'unknown'}",
        f"workflow_runners: {', '.join(parsed_runners[:6]) or 'none'}",
        f"deprecated_actions: {', '.join(deprecated_nodes[:6]) or 'none'}",
        f"version_pins: {', '.join(pinned_versions[:6]) or 'none'}",
    ]
    for name, values in runtime_versions.items():
        if values:
            observation_lines.append(f"{name}_versions: {', '.join(values)}")
    _update_with_llm_observation(
        state,
        bs,
        "\n".join(observation_lines),
        "runner_environment",
    )

    summary = {
        "runner_images": runner_images,
        "workflow_runners": parsed_runners[:6],
        "action_versions": action_versions[:8],
        "pinned_versions": pinned_versions[:8],
        "deprecated_nodes": deprecated_nodes[:8],
        "runtime_versions": {k: v for k, v in runtime_versions.items() if v},
    }
    remaining = [t for t in state["tools_available"] if t != "inspect_runner_environment"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["inspect_runner_environment"],
        "tools_available": remaining,
        "changed_files": changed_files,
        "runner_environment": summary,
        "investigation_log": [
            f"[inspect_runner_environment] runners={', '.join(runner_images) or 'unknown'}, "
            f"runtime_families={len(summary['runtime_versions'])}. "
            f"Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
        ],
    }


def _fetch_pr_context(repo: str, pr_number: int) -> dict:
    """Fetch PR title, labels, and changed files from GitHub."""
    empty = {
        "number": pr_number,
        "title": "",
        "labels": [],
        "files": [],
        "summary": "no PR context retrieved",
    }
    if not repo or not pr_number:
        return empty

    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        pr_resp = requests.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
            headers=headers,
            timeout=20,
        )
        if pr_resp.status_code != 200:
            return empty
        pr_data = pr_resp.json() or {}

        files_resp = requests.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files",
            headers=headers,
            params={"per_page": 30},
            timeout=20,
        )
        files_data = files_resp.json() if files_resp.status_code == 200 else []
        files = [
            {
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "changes": f.get("changes", 0),
            }
            for f in files_data[:20]
        ] if isinstance(files_data, list) else []

        labels = [lbl.get("name", "") for lbl in pr_data.get("labels", []) if lbl.get("name")]
        summary = f"PR #{pr_number}: {len(files)} file(s), labels={', '.join(labels[:5]) or 'none'}"
        return {
            "number": pr_number,
            "title": pr_data.get("title", ""),
            "labels": labels,
            "files": files,
            "summary": summary,
        }
    except Exception:
        return empty


def inspect_pr_context(state: AgentState) -> dict:
    """Inspect pull-request metadata when the run is PR-triggered or commit-linked."""
    event = _get_event(state)
    bs = _get_belief_state(state)

    raw_run = state.get("raw_run") or {}
    pr_number = None
    prs = raw_run.get("pull_requests") or []
    if prs and isinstance(prs, list):
        pr_number = prs[0].get("number")

    pr_context = _fetch_pr_context(event.repo, pr_number) if pr_number else {
        "number": None,
        "title": "",
        "labels": [],
        "files": [],
        "summary": "run is not linked to a PR",
    }

    observation_parts = [f"event={event.event}", pr_context.get("summary", "")]
    if pr_context.get("title"):
        observation_parts.append(f"pr_title={pr_context['title']}")
    if pr_context.get("labels"):
        observation_parts.append("labels=" + ", ".join(pr_context["labels"][:8]))
    if pr_context.get("files"):
        observation_parts.append(
            "files=\n" + "\n".join(f"- {f['filename']}" for f in pr_context["files"][:10])
        )
    _update_with_llm_observation(state, bs, "\n".join(observation_parts), "pr_context")

    remaining = [t for t in state["tools_available"] if t != "inspect_pr_context"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["inspect_pr_context"],
        "tools_available": remaining,
        "pr_context": pr_context,
        "investigation_log": [
            f"[inspect_pr_context] {pr_context.get('summary', 'no PR context')}. "
            f"Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
        ],
    }


def _semantic_similarity_search(
    state: AgentState,
    event: RunEvent,
    top_k: int = 5,
) -> list[dict]:
    """APA semantic retrieval using embeddings over the local failure corpus.

    Routes to ChromaDB (APA-v) when USE_CHROMA=1.
    Falls back to the original brute-force cosine search otherwise.
    The token-overlap preprocessing path is never affected by this flag.
    """
    _USE_CHROMA = os.environ.get("USE_CHROMA", "0") == "1"

    if _USE_CHROMA:
        # ── ChromaDB ANN path (APA-v) ─────────────────────────────────
        try:
            from src.apa.chroma_case_store import get_chroma_store, CHROMA_PATH
            store = get_chroma_store(path=CHROMA_PATH, client=_get_client(state))
            records = store.find_similar_case_records(
                commit_title=event.commit_title or "",
                error_lines=state.get("error_lines", []),
                mentioned_files=state.get("mentioned_files", []),
                k=top_k,
            )
            # Normalise shape: add semantic_score alias so the caller
            # (search_similar_failures node) sees a consistent field name.
            for r in records:
                r.setdefault("semantic_score", r.get("similarity", 0.0))
            return records
        except Exception as exc:
            # ChromaDB unavailable — fall through to brute-force path
            print(f"  [chroma] retrieval failed, using fallback: {exc}")

    # ── Original brute-force cosine path (APA baseline) ───────────────
    if not _SIMILARITY_CORPUS:
        return []

    query_text = " ".join([
        event.commit_title or "",
        event.commit_message or "",
        " ".join(state.get("error_lines", [])[:8]),
    ]).strip()
    if not query_text:
        return []

    docs = []
    entries = []
    for entry in _SIMILARITY_CORPUS[:200]:
        if not entry.get("category"):
            continue
        doc = " ".join([
            entry.get("commit_title", ""),
            entry.get("commit_message", ""),
            " ".join(entry.get("error_lines", [])[:8]),
        ]).strip()
        if not doc:
            continue
        docs.append(doc[:4000])
        entries.append(entry)
    if not docs:
        return []

    client = _get_client(state)
    model = os.environ.get("CI_AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        response = client.embeddings.create(model=model, input=[query_text[:4000]] + docs)
        vectors = [item.embedding for item in response.data]
        query_vec, doc_vecs = vectors[0], vectors[1:]
    except Exception:
        return []

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if not na or not nb:
            return 0.0
        return dot / (na * nb)

    scored = []
    for entry, vec in zip(entries, doc_vecs):
        score = cosine(query_vec, vec)
        scored.append((score, entry))
    scored.sort(key=lambda item: -item[0])

    results = []
    for score, entry in scored[:top_k]:
        item = dict(entry)
        item["semantic_score"] = round(score, 4)
        results.append(item)
    return results



def search_similar_failures(state: AgentState) -> dict:
    """APA-only semantic retrieval over prior failures using embeddings/API."""
    event = _get_event(state)
    bs = _get_belief_state(state)

    similar = _semantic_similarity_search(state, event)
    if similar:
        counts = Counter(item.get("category") for item in similar if item.get("category"))
        if counts:
            sim_base = {cat: 0.05 for cat in CATEGORIES}
            for cat, count in counts.items():
                if cat in sim_base:
                    sim_base[cat] += 0.18 * count
            sim_total = sum(sim_base.values())
            bs.update({k: v / sim_total for k, v in sim_base.items()}, "semantic_similar_failures")

    remaining = [t for t in state["tools_available"] if t != "search_similar_failures"]
    return {
        "beliefs": dict(bs.probabilities),
        "belief_history": [bs.history[-1]] if bs.history else [],
        "confidence": bs.confidence(),
        "entropy": bs.entropy(),
        "tools_called": ["search_similar_failures"],
        "tools_available": remaining,
        "similar_failures": similar,
        "investigation_log": [
            f"[search_similar_failures] semantic retrieval returned {len(similar)} case(s). "
            f"Top: {bs.top_category()[0]} ({bs.top_category()[1]:.0%})"
        ],
    }


# ─── node: classify ──────────────────────────────────────────────────

# Error markers that indicate the REAL failure region in a (possibly huge) log excerpt.
# Weak signal: any error-ish token (used only as a last resort).
_ERROR_FOCUS_RE = re.compile(
    r"={3,}\s*FAILURES|={3,}\s*ERRORS|\bTraceback \(most recent call last\)|"
    r"\bError:|\berror:|\bERROR\b|\b\w*Error\b|\bFAILED\b|\d+\s+failed|"
    r"\bassert\b|AssertionError|\bpanic:|\bfatal:|compilation error|"
    r"cannot find|could not|not found|undefined|npm ERR!|exit code [1-9]",
    re.I,
)
# STRONG, specific terminal-failure signatures. The real failure clusters at the END of a
# CI log, and benign "error"-ish tokens (gcc banners, "Error! not enough time") appear early.
# So we anchor on the LAST strong signature, not the first weak one.
_STRONG_ERROR_RE = re.compile(
    r"={3,}\s*FAILURES|={3,}\s*ERRORS|Traceback \(most recent call last\)|"
    r"AddressSanitizer|\bSegmentation fault\b|\bpanic:|\bfatal error\b|"
    r"\bFAILED\b|\d+\s+(?:tests?\s+)?failed\b|error\[[A-Z]?\d+\]|error:\s|"
    r"npm ERR!|AssertionError|\w+Error:|\w+Exception:|compilation error|"
    r"undefined reference|cannot find|Process completed with exit code [1-9]",
    re.I,
)

def _focus_log_on_error(excerpt_texts, max_chars=2200):
    """Return a window centered on the failure. Anchors on the LAST STRONG error signature
    (failures cluster at the end of a CI log; early "error"-ish tokens are usually benign
    banners). Falls back to the last weak-error match, then to the TAIL of the log (not the
    head), since exit-code/error annotations sit at the end.
    """
    blob = "\n".join(t for t in (excerpt_texts or [])[:3] if t)
    if not blob:
        return ""
    if len(blob) <= max_chars:
        return blob
    # Prefer the LAST strong signature; else the LAST weak one.
    anchor = None
    for rgx in (_STRONG_ERROR_RE, _ERROR_FOCUS_RE):
        matches = list(rgx.finditer(blob))
        if matches:
            anchor = matches[-1]
            break
    if anchor is None:
        # no error token at all -> keep the TAIL (errors/exit codes live at the end)
        return "…(earlier output omitted)…\n" + blob[-max_chars:]
    # window: most context BEFORE the anchor (the diagnostic lines precede the marker),
    # a little after, biased so the anchor sits ~3/4 of the way through the window.
    start = max(0, anchor.start() - (max_chars * 3 // 4))
    window = blob[start:start + max_chars]
    prefix = "…(earlier output omitted)…\n" if start > 0 else ""
    return prefix + window

# Few-shot examples for the 3-shot ablation (CI_AGENT_FEWSHOT=1). These are SYNTHETIC,
# hand-written, and deliberately NOT drawn from the evaluation corpus (no leakage): made-up
# repos/SHAs covering CODE_REGRESSION, DEPENDENCY_CONFLICT, CONFIG_ERROR. They teach the
# expected reasoning (connect the log error to the changed file) and output format.
CLASSIFY_FEWSHOT = """WORKED EXAMPLES (study the reasoning, then classify the real case below):

Example 1 — error log: "FAIL src/utils/date.test.ts ✕ formats ISO date / Expected 2020-01-01 received Invalid Date".
Changed files: src/utils/date.ts (modified). The test fails on code that was just changed in the
triggering commit. → category: CODE_REGRESSION ("the developer fixed the regression in date.ts").

Example 2 — error log: "npm ERR! ERESOLVE could not resolve / peer react@^18 from @testing-library/react@14".
Changed files: package.json (react bumped 17→18), package-lock.json. The failure is a dependency
version conflict introduced by the bump. → category: DEPENDENCY_CONFLICT ("the developer pinned/aligned
the dependency versions").

Example 3 — error log: "Error: Missing download info for actions/setup-node@v1" / "node: command not found".
Changed files: .github/workflows/ci.yml (modified). The failure is a deprecated/misconfigured workflow
action, not application code. → category: CONFIG_ERROR ("the developer fixed the workflow file").

Now classify the REAL case below using the same reasoning.
---
"""

CLASSIFY_PROMPT = """Based on your investigation, classify this CI/CD failure.

IMPORTANT: You are predicting the DEVELOPER'S FIX TYPE, not just the failure cause.
Ask yourself: "What did the developer have to change to fix this?" — then pick the category.

DETERMINISTIC PREPROCESSING (RPA baseline — these hand-coded signals are strong priors):
{preprocessing_summary}

INVESTIGATION SUMMARY:
{investigation_log}

BAYESIAN BELIEFS (posterior after all evidence — treat the top category as a strong prior;
only deviate from it if you have concrete log evidence that contradicts it):
{beliefs}

ERROR EVIDENCE:
{error_lines}

CHANGED FILES IN TRIGGERING COMMIT:
{changed_files}

COMMIT DIFF SUMMARY:
{commit_diff}
{raw_evidence}
FAILED STEP CONTEXT:
{failed_step_context}

DEPENDENCY CHANGE CONTEXT:
{dependency_changes}

RUNNER / ENVIRONMENT CONTEXT:
{runner_environment}

PR CONTEXT:
{pr_context}

WORKFLOW FILE SIGNALS:
{workflow_signals}

SEMANTIC DIFF LINKS (version bump ↔ error log — highest confidence evidence):
{semantic_links}

RECENT RUN HISTORY:
{run_history}

SIMILAR FAILURE RETRIEVAL:
{similar_failures}

FAILURE CONTEXT:
  Repo: {repo}
  Branch: {branch} (protected: {protected})
  Commit: {commit}
  Failed jobs: {failed}/{total}

Choose the MOST SPECIFIC category supported by the evidence.

TIEBREAKER RULE (apply before the discrimination guide below):
  If the Bayesian top category is CODE_REGRESSION with probability ≥ 0.35 AND
  source/test code files were changed in the triggering commit AND
  the error log shows a compile error, test failure, assertion, or runtime exception —
  choose CODE_REGRESSION UNLESS the log shows an explicit "module not found" / version
  conflict / workflow YAML error. Do not let a dependency manifest also being present
  in the commit override this: developers often touch package files in the same commit.

CATEGORY DISCRIMINATION GUIDE:
  CODE_REGRESSION — a test or build failed because recently-changed source code introduced a bug or regression.
    DEFAULT to CODE_REGRESSION when source code files (.py, .java, .ts, .go, etc.) OR build scripts (CMakeLists.txt, setup.py, Makefile, .sh scripts) were changed in the triggering commit.
    ALSO CODE_REGRESSION if: the commit is a revert of a previous commit (revert in title/message) — reverts fix code regressions.
    NOT CODE_REGRESSION if: the error is "module not found", "version conflict", "deprecated action", "YAML parse error".
    NOT CODE_REGRESSION if: the error is a generic cancellation ("The operation was canceled") AND no source/build files were changed — then use CONFIG_ERROR.
    NOT CODE_REGRESSION if: the failure is a linter/static-analysis rejection — use QUALITY_VIOLATION instead.

  DEPENDENCY_CONFLICT — module-not-found, unresolved dependency, version incompatibility, pip/npm/cargo install failures.
    ALSO DEPENDENCY_CONFLICT if: a GitHub Action version is deprecated/unsupported, or a tool version is EOL.
    NOT DEPENDENCY_CONFLICT if: the error is a test or compile failure — a dependency file in the commit does NOT make this DEPENDENCY_CONFLICT if the error itself is a runtime/test failure.
    NOT DEPENDENCY_CONFLICT if: the dependency file change is the TRIGGER commit but the developer's fix involved changing CI scripts — use CONFIG_ERROR.

  CONFIG_ERROR (= developer performed WORKFLOW_FIX) — the developer fixed a CI/workflow file (.yml, .github/, .circleci/).
    USE CONFIG_ERROR when: all jobs fail with "The operation was canceled" AND no source code or build scripts were changed in the triggering commit.
    USE CONFIG_ERROR when: ONLY workflow/CI pipeline files changed in the triggering commit AND the error is generic.
    USE CONFIG_ERROR when: the error is structural to the CI pipeline (wrong runner, bad YAML, missing secrets) AND there is no source code error trace.
    CRITICAL: Do NOT use CONFIG_ERROR for build scripts (CMakeLists.txt, setup.py, .sh). Those are CODE_REGRESSION.
    NOT CONFIG_ERROR if: the failure shows a specific code compile error or test failure — that is CODE_REGRESSION.

  QUALITY_VIOLATION — a linter or static-analysis tool (pylint, flake8, ESLint, checkstyle, rubocop, shellcheck, etc.) explicitly rejected the code.
    USE QUALITY_VIOLATION when: error output names a lint tool, style checker, or static analyser, and the violation count is nonzero.
    NOT QUALITY_VIOLATION if: the only error is a compile or runtime failure with no linter output.

  TEST_FLAKINESS — test was already non-deterministic before this commit; no code or dependency change caused it.
    NOT TEST_FLAKINESS if: the triggering commit modifies source code — if code changed, lean toward CODE_REGRESSION.

  INFRA_INCOMPATIBILITY — CI tooling or runner image is deterministically incompatible with the project (fails every time until a file is changed).
    USE INFRA_INCOMPATIBILITY when: action/tool version is unsupported by the current runner image, missing system library, glibc mismatch.
    NOT INFRA_INCOMPATIBILITY if: a retry would be likely to fix it — use ENV_FLAKINESS instead.

  ENV_FLAKINESS — transient runner, network, or CI infrastructure problem where a retry is expected to succeed.
    Includes: network timeouts, rate-limit errors, ephemeral runner outages, flaky dependency mirrors.
    NOT ENV_FLAKINESS if: source code or dependency files changed in the triggering commit.
    NOT ENV_FLAKINESS if: the failure is deterministic (every run fails the same way) — use INFRA_INCOMPATIBILITY.

FIX-TYPE MAPPING (use this as your PRIMARY decision rule — override category guide above if needed):
  Developer fixed source code files (.py, .js, .java, etc.) → CODE_REGRESSION
  Developer fixed a workflow file (.yml, .github/) → CONFIG_ERROR
  Developer pinned/upgraded a dependency (requirements.txt, package.json) → DEPENDENCY_CONFLICT
  Developer fixed lint/style violations → QUALITY_VIOLATION
  Developer reverted the triggering commit → CODE_REGRESSION (a revert is always fixing a code regression)
  Commit title contains "revert" or "Revert" → CODE_REGRESSION
  Developer fixed a test → TEST_FLAKINESS or CODE_REGRESSION

CATEGORIES: {categories}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-3 sentences citing specific evidence. State what type of fix you believe the developer applied.",
  "category": "one category",
  "severity": "CRITICAL|HIGH|MODERATE|LOW",
  "confidence": 0.0-1.0
}}"""



DEVIL_ADVOCATE_PROMPT = """You are a peer reviewer challenging a CI/CD failure diagnosis.

Original diagnosis: {category} (confidence: {confidence:.0%})
Reasoning given: {reasoning}

Investigation evidence:
{investigation_log}

Bayesian beliefs (posterior):
{beliefs}

Error lines: {error_lines}
Changed files: {changed_files}
Run history: {run_history}

Your job: Play devil's advocate. Ask: "What if the diagnosis is WRONG?"
- What evidence *contradicts* or *is unexplained by* the original diagnosis?
- Is there a simpler or more specific alternative category that better fits the evidence?
- If the run history was NOT checked before diagnosing {category}, is this diagnosis reliable?

If you find strong contradicting evidence, provide an alternative category.
If the original diagnosis is well-supported, confirm it.

Respond with ONLY a JSON object:
{{
  "upheld": true/false,
  "alternative_category": "category name if upheld=false, else null",
  "critique": "1-2 sentences on what evidence supports or contradicts the original diagnosis"
}}"""


# 3-shot examples for the remediation ablation (CI_AGENT_FEWSHOT=1). SYNTHETIC, leakage-free.
# They model the grounded reasoning (cite the error, point at the changed file) and a concrete,
# multi-part fix — exactly the behaviour the rules below ask for.
ACTION_FEWSHOT = """EXAMPLES OF GOOD FIX RECOMMENDATIONS (match this grounding and concreteness):

Example A — log: "FAIL date.test.ts ✕ Expected 2020-01-01 received Invalid Date"; hunk shows
src/utils/date.ts changed `new Date(s)` to `Date.parse(s)`. GOOD FIX: "In src/utils/date.ts, the
new `Date.parse(s)` returns a number, not a Date, so the formatter receives Invalid Date. Revert that
line to `new Date(s)` (or wrap it: `new Date(Date.parse(s))`), and re-run date.test.ts to confirm."

Example B — log: "npm ERR! ERESOLVE peer react@^18 from @testing-library/react@14"; hunk shows
package.json bumped react 17→18. GOOD FIX: "In package.json, the react 17→18 bump conflicts with
@testing-library/react@14 which needs react@^17. Either pin react back to ^17.0.2, or bump
@testing-library/react to ^15 which supports react 18, then regenerate package-lock.json."

Example C — log: "Error: Missing download info for actions/setup-node@v1"; hunk shows
.github/workflows/ci.yml uses setup-node@v1. GOOD FIX: "In .github/workflows/ci.yml, update
actions/setup-node@v1 to @v4 (v1 is deprecated and no longer resolvable); also bump actions/checkout
to @v4 in the same file if present."

---
"""

# General CI-failure -> typical-fix domain knowledge (NOT case-specific; no leakage of the
# test case's answer). Injected into the recommender when CI_AGENT_DOMAIN=1, mirroring the
# "domain knowledge" Bui et al. inject. Helps the model map an error class to its canonical fix.
ACTION_DOMAIN_KNOWLEDGE = """
DOMAIN KNOWLEDGE (typical CI failure classes and their canonical fixes; apply only the one that matches the evidence):
  - Dependency conflict (ERESOLVE, "no matching version", version/GLIBC/peer mismatch): pin or align the
    offending package to a compatible version and regenerate the lockfile (package-lock.json / yarn.lock /
    Cargo.lock / go.sum). Do NOT edit source.
  - Missing dependency (ModuleNotFoundError, ImportError, "cannot find package", exit code 2 on import):
    add the missing package to the manifest (requirements.txt / package.json / go.mod) or fix the import path.
  - Code regression (test assertion fails, compile/type error, runtime exception in changed source): fix the
    specific function/line in the changed source file, or revert the breaking change in that file.
  - Deprecated API (AttributeError on a removed attribute, "X is deprecated"): update the call site to the new
    API, or pin the library to the version that still has it.
  - Workflow/config error (deprecated GitHub Action, "missing input/secret", malformed YAML, runner image):
    update the action version, add the missing workflow input/secret, or fix the YAML in .github/workflows.
  - Infra/toolchain incompatibility (runner OS/Node/Go version, "command not found"): pin the toolchain/runner
    version in the workflow to a compatible one.
"""

ACTION_PROMPT = """You are a senior CI/CD engineer writing the fix a developer would actually apply to make this failed pipeline green again. You have the full evidence the investigation gathered below.
{domain_knowledge}
DIAGNOSIS: {category} (confidence: {confidence:.0%})
Reasoning: {reasoning}

EVIDENCE
  Repo: {repo} | Triggering commit: {commit}
  Error lines from the failing log:
{error_lines}

  Failing-log excerpt (read this — it tells you WHAT actually broke):
{log_excerpt}

  Actual code changes in the triggering commit (read the hunks — locate the fix HERE):
{diff_hunks}

  Diff summary: {commit_diff}
  Changed files: {changed_files}
  Dependency changes: {dependency_changes}
  Flakiness check: {flaky_note}
{similar_fixes}
Before writing the fix, reason in this exact order (do NOT skip a step):
  STEP 1 — What failed: quote the SPECIFIC error/failed-assertion from the log excerpt. If the
    log shows ONLY a generic "exit code 1" / "process completed" with no real error content,
    say so explicitly — do NOT invent a cause that is not in the evidence.
  STEP 2 — Where: the developer's fix is almost always IN ONE OF THE CHANGED FILES listed above,
    because this commit is what turned the build red. Choose the fix file FROM the "Changed
    files" list — pick the changed file most consistent with the error in STEP 1. ONLY name a
    file outside the changed-files list if the error trace EXPLICITLY points to a specific other
    file (then name that exact file). Never recommend changes to a file that is neither in the
    changed-files list nor named in the error trace — that is the single most common mistake.
  STEP 3 — Fix: the concrete change to make in that file.
Then output ONLY the fix (the recommended_action), applying these rules:
- NEVER assert a cause that is not supported by the log excerpt or the diff hunks. Do not
  free-associate from general knowledge (e.g. do not blame "WebSocket events" or "a memory
  leak" unless the evidence shows it). A confident wrong guess is worse than an honest "the
  failure is in <file named in the error>, which is not in this commit's diff — fix it there."
- FIRST read the failing-log excerpt and the diff hunks above. The fix is almost always in a
  file that appears in the hunks. Name that file and the concrete line/change to make.
- A real fix is OFTEN MULTI-PART. If the evidence shows the failure needs more than one change
  (e.g. update the source AND adjust a test, or bump a version AND update its lockfile, or fix
  the workflow AND a dependency), state ALL the concrete changes you can justify from the
  evidence — lead with the primary one, then the secondary ones. Do not artificially collapse a
  multi-file fix into a single action.
- PREFER A FORWARD FIX over a revert. If the hunks and error let you localize the broken
  code/config, fix it there (correct the failing assertion in test X, add the missing import
  in file Y, fix the YAML key in workflow Z, pin package P to a compatible version). Reverting
  throws away the developer's intended work and is rarely what they actually did.
- Recommend a revert ONLY when there is NO inspectable diff (a bare merge) AND the failure
  clearly began with this commit. Say so explicitly.
- Do NOT claim "test flakiness" or recommend a bare re-run unless the Flakiness check above
  permits it. A real error in the log is NOT flakiness.
- If only docs/README changed but tests fail, the real cause is elsewhere — say which file in
  the hunks or error lines is the likely culprit; do NOT give up or guess "regression somewhere".
- The triggering commit IS the change that immediately preceded this failure. Do NOT dismiss its
  changes as "cosmetic" or "cannot cause the failure" and then give up — if the build went red
  right after this commit, one of its changes is the most likely culprit. Name the single most
  likely file/line from the diff and the concrete fix there, even if the change looks harmless.
- ALWAYS commit to a concrete, actionable fix. Never answer with "re-run", "it's flaky", "monitor",
  or "investigate". If you are unsure, still name the single most probable file from the diff and
  the specific change to make there.
- Ground every claim in the evidence. Do not invent files or versions not present above.

Respond with ONLY a JSON object:
{{
  "recommended_action": "the 2-3 sentence concrete fix"
}}"""


def classify(state: AgentState) -> dict:
    event = _get_event(state)
    bs = _get_belief_state(state)
    preprocessing_summary_str = json.dumps(state.get("preprocessing_summary") or {}, indent=2, sort_keys=True)

    beliefs_str = "\n".join(
        f"  {cat}: {prob:.3f}"
        for cat, prob in sorted(bs.probabilities.items(), key=lambda x: -x[1])
        if prob > 0.01
    )

    error_str = "\n".join(state["error_lines"][:8]) if state["error_lines"] else "(no error text collected)"

    changed_files_str = "\n".join(
        f"  [{f.get('status', '?')}] {f.get('filename', '?')}"
        for f in state.get("changed_files", [])[:20]
    ) or "(not retrieved)"

    commit_diff = state.get("commit_diff") or {}
    if commit_diff:
        commit_diff_str = (
            f"files changed: {len(commit_diff.get('files', []))}\n"
            f"summary: {commit_diff.get('summary', 'n/a')}\n"
            f"families: {json.dumps(commit_diff.get('families', {}), sort_keys=True)}"
        )
    else:
        commit_diff_str = "(not retrieved)"

    # ── Shared raw-evidence block (used by BOTH classify and recommended_action) ──
    # Both stages benefit from seeing the ACTUAL changed code and the raw failing log,
    # not just summaries. Built once here so the recommender inherits exactly what the
    # classifier reasoned over.
    _diff_files = (commit_diff or {}).get("files", []) if isinstance(commit_diff, dict) else []
    _hunk_parts = []
    for _f in _diff_files[:6]:
        _patch = (_f.get("patch_excerpt") or _f.get("patch") or "").strip()
        if not _patch:
            continue
        _hunk_parts.append(f"--- {_f.get('filename','?')} [{_f.get('status','?')}]\n{_patch[:900]}")
    diff_hunks_str = "\n\n".join(_hunk_parts)[:3500] or "(no inspectable patch — likely a merge commit)"
    log_excerpt_str = _focus_log_on_error(state.get("log_excerpt_texts", []), max_chars=2200) or "(no log excerpt collected)"
    _rh_shared = state.get("run_history") or {}
    _flaky_ok = bool(_rh_shared.get("intermittent") or _rh_shared.get("recently_passed"))
    flaky_note_str = ("Run history shows intermittent pass/fail — flakiness is plausible."
                      if _flaky_ok else
                      "Run history does NOT show intermittent behaviour — do NOT diagnose flakiness or recommend a bare re-run.")

    failed_step_context = state.get("failed_step_context") or {}
    failed_step_context_str = json.dumps(failed_step_context, indent=2) if failed_step_context else "(not retrieved)"

    dependency_changes = state.get("dependency_changes") or {}
    dependency_changes_str = json.dumps(dependency_changes, indent=2) if dependency_changes else "(not retrieved)"

    runner_environment = state.get("runner_environment") or {}
    runner_environment_str = json.dumps(runner_environment, indent=2) if runner_environment else "(not retrieved)"

    pr_context = state.get("pr_context") or {}
    pr_context_str = json.dumps(pr_context, indent=2) if pr_context else "(not retrieved)"

    run_history = state.get("run_history") or {}
    if run_history:
        run_history_str = (
            f"summary: {run_history.get('summary', 'n/a')}\n"
            f"runs: {json.dumps(run_history.get('runs', [])[:5], indent=2)}"
        )
    else:
        run_history_str = "(not retrieved)"

    similar_failures = state.get("similar_failures") or []
    if similar_failures:
        similar_failures_str = json.dumps(similar_failures[:5], indent=2)
    else:
        similar_failures_str = "(not retrieved)"

    # Semantic diff links: version bumps cross-referenced to error log
    sem_links = state.get("semantic_diff_links") or []
    if sem_links:
        sem_parts = []
        for e in sem_links[:5]:
            arrow = f"{e.get('old_version', '')} → {e.get('new_version', '')}" if e.get('old_version') else e.get('new_version', '?')
            sem_parts.append(
                f"  {e.get('library', '?')} bumped {arrow} in "
                f"{e.get('file', '?')} [{e.get('ecosystem', '?')}] "
                f"(match_strength={e.get('match_strength', 0):.2f})"
            )
            for el in (e.get('matching_error_lines') or [])[:2]:
                sem_parts.append(f"    ↳ error: {el[:120]}")
        semantic_links_str = "\n".join(sem_parts)
    else:
        semantic_links_str = "(inspect_dependency_changes not called or no version changes found)"

    wc = state.get("workflow_contents") or []
    if wc:
        wc_parts = []
        for entry in wc:
            fname = entry.get("file", "?")
            actions = ", ".join(entry.get("action_versions", [])[:6]) or "none"
            runners = ", ".join(entry.get("runners", [])[:4]) or "none"
            deprecated = ", ".join(entry.get("deprecated_nodes", [])) or "none"
            pins = ", ".join(entry.get("pinned_versions", [])[:4]) or "none"
            wc_parts.append(
                f"  {fname}:\n"
                f"    actions: {actions}\n"
                f"    runners: {runners}\n"
                f"    deprecated: {deprecated}\n"
                f"    version pins: {pins}"
            )
        workflow_signals_str = "\n".join(wc_parts)
    else:
        workflow_signals_str = "(not retrieved — inspect_workflow_file was not called)"

    # The classify prompt receives only the gathered evidence and the posterior.
    # Hand-coded pattern->category disambiguation rules were removed here so the
    # final decision is LLM reasoning over evidence, not a rule table re-applied
    # at the decision stage (which would reproduce the RPA mapping inside APA).

    # Fix 6: cap investigation_log to last 15 entries to avoid blowing the
    # context window on verbose runs with many tool steps.
    # NOTE: an experiment that fed raw diff hunks + log excerpt into classify was tested
    # and REJECTED — it dropped coarse accuracy (~70%->52%) by biasing the model toward
    # whatever file changed (e.g. a workflow YAML tweak). Curated/summarized evidence
    # classifies better, so classify keeps its existing evidence set. (The raw hunks ARE
    # used downstream by recommended_action, where localizing a forward fix needs them.)
    raw_evidence_block = ""

    # Probe (CI_AGENT_CLASSIFY_FOCUS=1): give classify the error-FOCUSED log window (the
    # real failure region) instead of only the thin sampled error_lines. This is the same
    # honest fix that helped remediation; it is NOT the diff-hunks experiment (which hurt).
    classify_error_lines = error_str
    if os.environ.get("CI_AGENT_CLASSIFY_FOCUS") == "1":
        _focused = _focus_log_on_error(state.get("log_excerpt_texts", []), max_chars=1800)
        if _focused.strip():
            classify_error_lines = (error_str + "\n\nFAILING-LOG ERROR REGION:\n" + _focused)[:2600]

    prompt = CLASSIFY_PROMPT.format(
        preprocessing_summary=preprocessing_summary_str,
        investigation_log="\n".join(state["investigation_log"][-15:]),
        beliefs=beliefs_str,
        error_lines=classify_error_lines,
        changed_files=changed_files_str,
        commit_diff=commit_diff_str,
        raw_evidence=raw_evidence_block,
        failed_step_context=failed_step_context_str,
        dependency_changes=dependency_changes_str,
        runner_environment=runner_environment_str,
        pr_context=pr_context_str,
        workflow_signals=workflow_signals_str,
        semantic_links=semantic_links_str,
        run_history=run_history_str,
        similar_failures=similar_failures_str,
        repo=event.repo,
        branch=event.branch,
        protected=event.is_protected_branch,
        commit=event.commit_title,
        failed=event.failed_jobs_count,
        total=event.n_jobs,
        categories=", ".join(CATEGORIES),
    )

    # 3-shot ablation (CI_AGENT_FEWSHOT=1): prepend leakage-free worked examples.
    # Default OFF so the locked 0-shot numbers are unchanged unless we opt in.
    if os.environ.get("CI_AGENT_FEWSHOT") == "1":
        prompt = CLASSIFY_FEWSHOT + prompt

    # Uninformative-log directive. When the log shows no real error (just a
    # generic exit code / cancellation), there is nothing to diagnose: picking a
    # "specific" category like CONFIG_ERROR or TEST_FLAKINESS from circumstantial
    # signals (only docs changed, N jobs failed) is a guess that empirically
    # hurts accuracy. In that case TRUST THE BAYESIAN POSTERIOR (which already
    # encodes the changed-file and commit-message priors) and prefer its top
    # category — usually CODE_REGRESSION, the base rate — over a speculative one.
    excerpt_for_check, _ = _build_capped_deep_log_excerpt(state, mode="full")
    if _is_uninformative_log(excerpt_for_check):
        prompt = (
            "IMPORTANT: The error log for this run is UNINFORMATIVE — it shows only a "
            "generic terminal failure (e.g. 'Process completed with exit code 1' or "
            "'The operation was canceled') with no compile error, test assertion, stack "
            "trace, or install failure. You therefore have NO direct evidence of the "
            "failure type. Do NOT infer a specific category (CONFIG_ERROR, TEST_FLAKINESS, "
            "DEPENDENCY_CONFLICT, etc.) from circumstantial signals like which file the "
            "commit touched or how many jobs failed. Instead DEFER TO THE BAYESIAN "
            "POSTERIOR below and choose its highest-probability category.\n\n"
            + prompt
        )

    client = _get_client(state)
    # Model tiering: only the classify call runs on the strong/reasoning model
    # (CI_AGENT_CLASSIFY_MODEL). The secondary calls below (devil's advocate,
    # recommended action) use the cheaper SECONDARY_MODEL, falling back to the
    # base model. Planner and per-tool likelihoods stay on the base model.
    clf_model = CLASSIFY_MODEL or state["model"]
    secondary_model = SECONDARY_MODEL or state["model"]
    try:
        response = client.chat.completions.create(
            model=clf_model,
            messages=[
                # Fix 7: system = short role/persona; user = evidence-rich task.
                # This is the correct ordering for instruction-tuned models.
                {"role": "system", "content": (
    "You are a precise CI/CD failure classifier. Your goal is to predict the TYPE OF FIX "
    "the developer applied, not just to diagnose the failure cause. Reason over the evidence "
    "and the posterior you are given, choose the single best-supported category, and do not "
    "use UNKNOWN. Make your best evidence-grounded judgment."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # reproducibility: the final category must be stable run-to-run
            **_json_mode_kwargs(clf_model),
            **usage_kwargs(),
        )
        record_usage(response, clf_model, call_type="chat", label="agent.classify")
        log_transcript("agent.classify", clf_model,
                       [{"role": "user", "content": prompt}], response)
        data = json.loads(_strip_to_json(response.choices[0].message.content))
    except Exception as e:
        top_cat, top_prob = bs.top_category()
        data = {
            "category": top_cat,
            "severity": "MODERATE",
            "confidence": bs.confidence(),
            "bayesian_probability": top_prob,
            "reasoning": f"Classification error: {e}",
        }

    category = data.get("category", "CODE_REGRESSION")

    classification = {
        "category": category,
        "severity": data.get("severity", "MODERATE"),
        "confidence": float(data.get("confidence", 0.0)),
        "reasoning": data.get("reasoning", ""),
    }

    # Devil's-advocate review was REMOVED. Evaluation showed it revised ~half
    # of diagnoses and net-hurt accuracy: it flipped correct high-posterior
    # calls to wrong ones (e.g. CODE_REGRESSION@0.97 -> DEPENDENCY_CONFLICT just
    # because a lockfile changed). It was also a second reasoning-model call per
    # case. The classify call's answer now stands as the final category.
    da_critique = ""

    # ── Recommended Action: concrete fix suggestion ─────────────────────
    # Skippable (CI_AGENT_SKIP_ACTION=1): the eval scores only the category, so
    # this extra LLM call is pure cost there. Disabled by default in production.
    recommended_action = ""
    if os.environ.get("CI_AGENT_SKIP_ACTION") != "1":
      try:
        dep_changes_list = state.get("dependency_changes", {})
        dep_str_da = json.dumps(dep_changes_list, ensure_ascii=False)[:400] if dep_changes_list else "(not retrieved)"

        # L3 RAG: if retrieval populated similar past failures (with their known fixes),
        # show them to the recommender so it can reuse a proven fix — the mechanism LogSage
        # and Bui et al. credit for high remediation accuracy. Empty when retrieval is off.
        sims = state.get("similar_failures") or []
        sim_lines = []
        for s in sims[:3]:
            if not isinstance(s, dict):
                continue
            cat = s.get("category", "?"); fix = (s.get("fix_reasoning") or s.get("reasoning") or "").strip()
            repo_s = s.get("repo", "")
            if fix:
                sim_lines.append(f"  - [{cat}] {repo_s}: {fix[:160]}")
        similar_fixes_str = ("\n  SIMILAR PAST FAILURES AND HOW THEY WERE FIXED (reuse if the\n"
                             "  current error matches one of these patterns):\n" + "\n".join(sim_lines)) if sim_lines else ""

        action_prompt = ACTION_PROMPT.format(
            category=classification["category"],
            confidence=classification["confidence"],
            reasoning=classification["reasoning"],
            repo=event.repo,
            commit=event.commit_title[:120],
            error_lines=error_str[:900],
            log_excerpt=log_excerpt_str,
            diff_hunks=diff_hunks_str,
            commit_diff=commit_diff_str[:600],
            changed_files=changed_files_str[:400],
            dependency_changes=dep_str_da,
            flaky_note=flaky_note_str,
            similar_fixes=similar_fixes_str,
            domain_knowledge=(ACTION_DOMAIN_KNOWLEDGE if os.environ.get("CI_AGENT_DOMAIN") == "1" else ""),
        )
        # Separate flag so the action few-shot can be enabled WITHOUT changing classification
        # (CI_AGENT_FEWSHOT also affects classify; CI_AGENT_ACTION_FEWSHOT is action-only).
        if os.environ.get("CI_AGENT_FEWSHOT") == "1" or os.environ.get("CI_AGENT_ACTION_FEWSHOT") == "1":
            action_prompt = ACTION_FEWSHOT + action_prompt
        # Optional: use the stronger reasoner for the fix recommendation (CI_AGENT_ACTION_MODEL).
        action_model = os.environ.get("CI_AGENT_ACTION_MODEL") or secondary_model
        action_response = client.chat.completions.create(
            model=action_model,
            messages=[
                {"role": "system", "content": "You are a CI/CD expert. Output ONLY valid JSON with a recommended_action field."},
                {"role": "user", "content": action_prompt},
            ],
            temperature=0.1,
            # reasoner needs headroom for hidden reasoning tokens before the answer.
            max_tokens=(1500 if any(t in action_model.lower() for t in ("reasoner","r1","thinking")) else 400),
            **_json_mode_kwargs(action_model),
            **usage_kwargs(),
        )
        record_usage(action_response, action_model, call_type="chat", label="agent.recommended_action")
        log_transcript("agent.recommended_action", action_model,
                       [{"role": "user", "content": action_prompt}], action_response)
        action_data = json.loads(_strip_to_json(action_response.choices[0].message.content))
        recommended_action = action_data.get("recommended_action", "")
      except Exception as e:
        recommended_action = f"(action generation skipped: {e})"

    classification["recommended_action"] = recommended_action

    return {
        "classification": classification,
        "investigation_log": [
            f"[classify] → {classification['category']} "
            f"(conf={classification['confidence']:.0%}, "
            f"bayes={bs.top_category()[0]} @ {bs.confidence():.0%})"
            + (f" | peer-review: {da_critique[:80]}" if da_critique else "")
        ],
    }


# ─── routing logic ───────────────────────────────────────────────────

def route_after_planner(state: AgentState) -> str:
    """Route to the next node based on planner's decision."""
    if state.get("done", False):
        return "classify"

    tool = state.get("_next_tool", "classify")
    if tool in state.get("tools_available", []):
        return tool
    return "classify"


# ─── build the graph ─────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Construct the LangGraph investigation agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("initialize", initialize)
    graph.add_node("deep_log_analysis", deep_log_analysis)   # mandatory, pre-planner
    graph.add_node("planner", planner)
    graph.add_node("inspect_dependency_changes", inspect_dependency_changes)
    graph.add_node("inspect_workflow_file", inspect_workflow_file)
    graph.add_node("inspect_runner_environment", inspect_runner_environment)
    graph.add_node("inspect_pr_context", inspect_pr_context)
    graph.add_node("search_similar_failures", search_similar_failures)
    graph.add_node("search_web_for_error", search_web_for_error)
    graph.add_node("compare_previous_successful_log", compare_previous_successful_log)
    graph.add_node("classify", classify)

    # Entry point
    graph.set_entry_point("initialize")

    # initialize → deep_log_analysis (mandatory: run-context header + full
    # log excerpt, one joint likelihood update) → planner
    graph.add_edge("initialize", "deep_log_analysis")
    graph.add_edge("deep_log_analysis", "planner")

    # planner → conditional routing (deep_log_analysis is NOT routable —
    # it already ran, and a re-read would double-count the same evidence)
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "inspect_dependency_changes": "inspect_dependency_changes",
            "inspect_workflow_file": "inspect_workflow_file",
            "inspect_runner_environment": "inspect_runner_environment",
            "inspect_pr_context": "inspect_pr_context",
            "search_similar_failures": "search_similar_failures",
            "search_web_for_error": "search_web_for_error",
            "compare_previous_successful_log": "compare_previous_successful_log",
            "classify": "classify",
        },
    )

    # All tools → back to planner
    graph.add_edge("inspect_dependency_changes", "planner")
    graph.add_edge("inspect_workflow_file", "planner")
    graph.add_edge("inspect_runner_environment", "planner")
    graph.add_edge("inspect_pr_context", "planner")
    graph.add_edge("search_similar_failures", "planner")
    graph.add_edge("search_web_for_error", "planner")
    graph.add_edge("compare_previous_successful_log", "planner")

    # classify → END
    graph.add_edge("classify", END)

    return graph.compile()


# ─── public API ──────────────────────────────────────────────────────

def run_agent(raw_run: dict, api_key: str = None, model: str = DEFAULT_MODEL) -> dict:
    """
    Run the investigation agent on a raw GHALogs run.
    Returns the full agent state including classification and trace.
    """
    if api_key is None:
        api_key = get_api_key() or ""
    if not model:
        model = DEFAULT_MODEL

    event = intake(raw_run)
    event_dict = asdict(event)
    preprocessing = _build_preprocessing_state(event, raw_run)

    preprocessing_log = [
        (
            f"[preprocess] deterministic signals applied: "
            f"{', '.join(preprocessing['preprocessing_summary'].get('signals_applied', [])) or 'none'}; "
            f"top={preprocessing['preprocessing_summary'].get('top_category', '?')} "
            f"({preprocessing['preprocessing_summary'].get('top_probability', 0):.0%})"
        )
    ]

    # Commit diff is must-do evidence (the old planner step-1 guard forced it
    # on every run anyway), so fetch it once here — one GitHub API call, zero
    # LLM cost — and let every downstream consumer reuse it from state: the
    # mandatory deep_log_analysis disambiguation, inspect_* tools, and the
    # classify prompt. APA-side only; the shared RPA signal battery is untouched.
    changed_files: list = []
    commit_diff_summary: dict = {}
    if event.repo and event.commit_sha:
        changed_files = _fetch_commit_diff(event.repo, event.commit_sha).get("files", [])
        commit_diff_summary = _summarize_patch_files(changed_files)
        preprocessing_log.append(
            f"[preprocess] commit diff fetched: {len(changed_files)} file(s); "
            f"{commit_diff_summary.get('summary', 'n/a')}"
        )

    # APA starts from the shared informed prior — the same Dirichlet-smoothed
    # empirical Bayes prior that seeds RPA preprocessing (INFORMED_PRIOR).
    # The RPA *posterior* (after deterministic signals) lives in
    # preprocessing_summary for logging/comparison but does NOT seed the agent
    # loop; doing so would bias every LLM step toward RPA's conclusion before
    # the agent has seen any evidence. So both systems share the same starting
    # category distribution, but APA re-derives its own posterior from scratch.
    prior_bs = BeliefState()   # seeded from INFORMED_PRIOR (not uniform)
    initial_state: AgentState = {
        "run_event": event_dict,
        "raw_run": raw_run,
        "beliefs": dict(prior_bs.probabilities),
        "belief_history": list(prior_bs.history),
        "confidence": prior_bs.confidence(),
        "entropy": prior_bs.entropy(),
        "tools_available": [
            *_initial_tools_for_event(raw_run, event),
        ],
        "tools_called": [],
        "investigation_log": preprocessing_log,
        "current_step": 0,
        "done": False,
        "error_lines": preprocessing["error_lines"],
        "mentioned_files": preprocessing["mentioned_files"],
        "log_excerpt_texts": preprocessing["log_excerpt_texts"],
        "changed_files": changed_files,
        "commit_diff": commit_diff_summary,
        "failed_step_context": _failed_step_summary(event),
        "dependency_changes": {},
        "run_history": preprocessing.get("run_history", {}),
        "similar_failures": [],
        "workflow_contents": [],
        "runner_environment": {},
        "pr_context": {},
        "semantic_diff_links": [],
        "preprocessing_summary": preprocessing["preprocessing_summary"],
        # Fix 8: excerpts_collected removed — was always True and never read.
        "classification": {},
        "api_key": api_key,
        "model": model,
        "_next_tool": "",
    }

    agent = build_agent_graph()
    final_state = agent.invoke(initial_state)

    result = {
        "classification": final_state["classification"],
        "investigation_log": final_state["investigation_log"],
        "beliefs": final_state["beliefs"],
        "belief_history": final_state["belief_history"],
        "tools_used": final_state["tools_called"],
        "steps_taken": final_state["current_step"],
        "error_lines": final_state["error_lines"],
        "mentioned_files": final_state["mentioned_files"],
        "changed_files": final_state.get("changed_files", []),
        "commit_diff": final_state.get("commit_diff", {}),
        "failed_step_context": final_state.get("failed_step_context", {}),
        "dependency_changes": final_state.get("dependency_changes", {}),
        "run_history": final_state.get("run_history", {}),
        "similar_failures": final_state.get("similar_failures", []),
        "workflow_contents": final_state.get("workflow_contents", []),
        "runner_environment": final_state.get("runner_environment", {}),
        "pr_context": final_state.get("pr_context", {}),
        "semantic_diff_links": final_state.get("semantic_diff_links", []),
        "preprocessing_summary": final_state.get("preprocessing_summary", {}),
        "fast_path": False,
    }

    return result


# ─── pretty printer ─────────────────────────────────────────────────

def print_agent_result(result: dict) -> None:
    fast_path = result.get("fast_path", False)
    banner = "  FAST PATH (preprocessing confidence gate)" if fast_path else ""

    print("\n" + "=" * 70)
    print("INVESTIGATION TRACE" + banner)
    print("=" * 70)
    for entry in result["investigation_log"]:
        print(f"  {entry}")

    cl = result["classification"]
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)
    print(f"  category:    {cl.get('category')}")
    print(f"  severity:    {cl.get('severity')}")
    print(f"  confidence:  {cl.get('confidence', 0):.0%}")
    print(f"  reasoning:   {cl.get('reasoning')}")
    mentioned_files = result.get("mentioned_files") or []
    if mentioned_files:
        print(f"  mentioned:   {', '.join(m.get('path', '?') for m in mentioned_files[:5])}")
    beliefs = result.get("beliefs") or {}
    if beliefs:
        bayes_top, bayes_prob = max(beliefs.items(), key=lambda kv: kv[1])
        print(f"  bayes top:   {bayes_top} @ {bayes_prob:.0%}")
    print(f"  steps taken: {result.get('steps_taken')}")
    print(f"  tools used:  {', '.join(result.get('tools_used', []) or [])}")
    print(f"  fast_path:   {fast_path}")



# ─── self-test on bcrypt ─────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
    RUN_ID = "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1"

    print(f"Finding {RUN_ID}...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run.get("_id") == RUN_ID:
                break
        else:
            raise SystemExit("Run not found.")

    print(f"Running investigation agent on pyca/bcrypt...\n")
    result = run_agent(run)
    print_agent_result(result)

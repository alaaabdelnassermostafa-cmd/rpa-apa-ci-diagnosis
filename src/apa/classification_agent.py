# classification_agent.py
# ─────────────────────────────────────────────────────────────────────
# The Classification Agent.
#
# Reads a RunEvent (from intake_parser.py) and optionally log excerpts
# (from log_extractor.py), then uses an LLM to classify the failure.
#
# Usage:
#   result = classify(event, client)              # metadata only
#   result = classify(event, client, log_excerpts) # metadata + logs
#
# Output is a typed ClassificationResult dataclass.
# ─────────────────────────────────────────────────────────────────────

import os
import json
import gzip
import textwrap
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from openai import OpenAI

from src.apa.intake_parser import intake, RunEvent
from src.apa.log_extractor import LogExcerpt
from src.apa.file_path_extractor import MentionedFile, extract_from_excerpt_windows
from dotenv import load_dotenv
from src.apa.llm_usage import record_usage

load_dotenv()


# ─── typed output schema ──────────────────────────────────────────────

@dataclass
class FixSuggestion:
    file: str
    change: str
    reason: str


@dataclass
class ClassificationResult:
    category: str            # CODE_REGRESSION | DEPENDENCY_CONFLICT | ...
    severity: str            # CRITICAL | HIGH | MODERATE | LOW
    confidence: float        # 0.0 – 1.0
    action: str              # BLOCK_DEPLOY | RETRY | FIX_WORKFLOW | ...
    reasoning: str           # the LLM's own explanation
    evidence: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    mentioned_files: List[MentionedFile] = field(default_factory=list)
    fix_suggestion: Optional[FixSuggestion] = None


# ─── taxonomy — this IS the agent's world model ───────────────────────

FAILURE_CATEGORIES = {
    "CODE_REGRESSION":       "The commit introduced a logic, syntax, type, or build error in source code. Use this as the default when the commit touches source files and the error is a compile/test/runtime failure with no clearer explanation. Also use when auth errors or missing-resource errors are triggered by code that calls external services differently.",
    "DEPENDENCY_CONFLICT":   "A package, library, or runtime version is explicitly incompatible — the error message names a specific package and version mismatch (e.g. 'requires X>=2.0 but 1.8 installed', 'lockfile out of date', 'no matching version'). Do NOT use just because the commit touches dependency manifests (Cargo.toml, pom.xml, package.json); the error text must show a version conflict.",
    "CONFIG_ERROR":          "The workflow YAML file itself has a structural problem: wrong syntax, incorrect or deprecated action reference (e.g. actions/checkout@v1), missing required 'with' input, or a workflow-level environment variable that is undefined. Do NOT use for auth/token errors unless the failing step is a workflow-configured authentication step. Do NOT use for missing cloud resources (S3 bucket, Azure blob) unless the resource name is hardcoded in the YAML. Do NOT use when the commit only touches source files.",
    "QUALITY_VIOLATION":     "A static-analysis or linting tool (checkstyle, pylint, flake8, ESLint, rubocop, shellcheck, etc.) rejected the code. The developer must fix style or correctness violations before the pipeline can proceed — retry will NOT help.",
    "TEST_FLAKINESS":        "An intermittent test failure unrelated to the code change — the same test passes on retry without any code modification.",
    "INFRA_INCOMPATIBILITY": "CI tooling version doesn't match project requirements in a way that fails deterministically — e.g. action version needs newer glibc than the runner image provides, or a required tool is absent from the runner. Retry will NOT help; a configuration or pinning change is needed.",
    "ENV_FLAKINESS":         "Transient runner, network, or CI infrastructure problem — retry likely succeeds. Includes network blips during dependency downloads, ephemeral runner outages, and rate-limit timeouts. Distinguished from INFRA_INCOMPATIBILITY by the fact that a plain retry is expected to fix it.",
    "CASCADE_FAILURE":       "Job failed because a sibling job failed, not because of its own problem. Usually marked by 'operation was canceled' or 'needs' dependency failure.",
    "TOOLING_ARTIFACT":      "Not a real GHA failure — e.g. a log-parser bug or dataset extraction noise (bash-command-extractor, Converting circular structure to JSON, BashWord parser exceptions).",
}

SEVERITY_LEVELS = {
    "CRITICAL": "Protected/release branch + core workflow + widespread failure. Production at risk.",
    "HIGH":     "Protected branch OR release workflow with partial failure.",
    "MODERATE": "Feature branch with broad or repeated failure.",
    "LOW":      "Feature branch with isolated failure, or tooling noise.",
}

ACTIONS = {
    "BLOCK_DEPLOY":       "Do not promote or merge until the failure is resolved.",
    "RETRY":              "Retry the run once because the failure appears transient.",
    "FIX_WORKFLOW":       "Change workflow, build, runtime, or environment configuration before rerunning.",
    "FIX_CODE":           "Change application, library, or test code before rerunning.",
    "PIN_VERSION":        "Pin or roll back a tool, action, runtime, or dependency version.",
    "INVESTIGATE_FAILURE":"Gather more evidence and manually inspect the failing path because the cause is still unclear.",
    "IGNORE_INFRA_BUG":   "Dataset/tooling artifact, not a real failure — no action needed.",
}


# ─── prompts ──────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = f"""You are the Classification Agent in a multi-agent system for autonomous CI/CD incident response.

Your job is to analyze a failed GitHub Actions workflow run and produce a structured classification that downstream agents (Validation, Planner, Execution) will act on.

You must classify along three axes: failure category, severity, and recommended action.

FAILURE CATEGORIES:
{chr(10).join(f"  {k}: {v}" for k, v in FAILURE_CATEGORIES.items())}

SEVERITY LEVELS:
{chr(10).join(f"  {k}: {v}" for k, v in SEVERITY_LEVELS.items())}

RECOMMENDED ACTIONS:
{chr(10).join(f"  {k}: {v}" for k, v in ACTIONS.items())}

REASONING GUIDELINES:
- Read the commit message carefully — it often reveals the intent of the change and hints at what might have broken.
- A protected branch (main, master, release/*) raises severity automatically.
- If failed steps share identical suspiciously-formatted "error" fields — especially bash parser exceptions, "Converting circular structure to JSON", or HTML-encoded tracebacks — this is almost certainly a dataset tooling artifact, NOT a real GHA failure. Classify as TOOLING_ARTIFACT with action IGNORE_INFRA_BUG.
- Prefer the narrowest concrete action supported by the evidence. Use FIX_WORKFLOW, FIX_CODE, PIN_VERSION, RETRY, or BLOCK_DEPLOY when the logs support them.
- Use INVESTIGATE_FAILURE only when the evidence is genuinely too weak or contradictory to recommend a narrower action.
- If you cannot determine the real cause from the available fields, say so in 'unknowns' rather than guessing.
- Confidence must reflect real uncertainty. 0.9+ only when the evidence is unambiguous.

DISAMBIGUATION RULES (apply these before finalising the category):
- LINTER/STATIC-ANALYSIS ERRORS (checkstyle, pylint, flake8, ESLint, rubocop, shellcheck, tslint, prettier): → QUALITY_VIOLATION. The developer must fix the style or correctness violation. Do NOT classify as CODE_REGRESSION unless there is also a runtime/compile error beyond the linter report.
- AUTH/TOKEN errors ('could not read Username', 'token or opts.auth is required', '401 Unauthorized'): these are NOT automatic CONFIG_ERROR. If the commit changes source files, prefer CODE_REGRESSION — code changes can alter how external calls are authenticated. Only use CONFIG_ERROR if the failing step is explicitly a workflow-level auth setup step (e.g. configure-aws-credentials, google-auth).
- MISSING RESOURCE errors ('NoSuchBucket', 'No such file or directory', 'Unable to find artifacts'): prefer CODE_REGRESSION if the commit changes source or build scripts; prefer CONFIG_ERROR only if the missing resource name is literally hardcoded in the workflow YAML.
- SDK/BUILD TOOL CONFLICTS ('Found multiple publish output files', 'NETSDK1152', 'MSB' errors, 'duplicate class'): these are CODE_REGRESSION — they indicate a code-level project structure problem, not a package version conflict. Do NOT classify as DEPENDENCY_CONFLICT.
- TOUCHING DEPENDENCY FILES (Cargo.toml, pom.xml, package.json) WITHOUT a version-mismatch error: classify as CODE_REGRESSION, not DEPENDENCY_CONFLICT. The fact that a dependency file was edited does not mean the failure is a dependency conflict.
- NO ERROR LOGS AVAILABLE: infer from the commit diff only. Source files changed → CODE_REGRESSION. Only workflow YAML changed → CONFIG_ERROR. Only dependency manifest changed WITH a lockfile → DEPENDENCY_CONFLICT. Mixed or unclear → CODE_REGRESSION.
- DEPENDABOT PUSH READ-ONLY ACCESS ('Workflows triggered by Dependabot on the push event run with read-only access'): → DEPENDENCY_CONFLICT. This is a known GitHub limitation: Dependabot push events cannot upload Code Scanning results. The root cause is the version bump, NOT a workflow misconfiguration. Do NOT classify as CONFIG_ERROR.
- 'FATAL: NOT A GIT REPOSITORY' error with data/source/CSV file changes → CODE_REGRESSION. A malformed file in the commit (e.g. CSV with unescaped backslashes) can break git tooling in CI. Only classify as CONFIG_ERROR if exclusively workflow files were changed and no other files.
- GENERIC EXIT CODES OR CANCELLATIONS ('exit code 1', 'operation was canceled', 'exit code 143') with no stack trace: Do NOT default to CONFIG_ERROR. If source files were changed, classify as CODE_REGRESSION. If only dependency files were changed, classify as DEPENDENCY_CONFLICT. Only classify as CONFIG_ERROR if exclusively workflow files were changed.
- ENV_FLAKINESS vs INFRA_INCOMPATIBILITY: use ENV_FLAKINESS if retry is expected to fix the issue (network timeout, runner rate-limit). Use INFRA_INCOMPATIBILITY if the failure is deterministic — it will fail every time until a tool version or runner image is changed.

OUTPUT FORMAT:
Respond with a JSON object containing exactly these keys:
  category    (one of the failure categories listed above)
  severity    (one of the severity levels listed above)
  confidence  (float 0.0 to 1.0)
  action      (one of the actions listed above)
  reasoning   (2-4 sentences explaining your decision, referencing specific fields from the input)
  evidence    (array of short strings naming the specific input fields you relied on)
  unknowns    (array of short strings naming information you wish you had but didn't)

Return ONLY the JSON object. No prose before or after."""

LOG_EVIDENCE_GUIDANCE = """

When LOG EVIDENCE is provided, treat it as the highest-quality signal:
- The actual error text often reveals the true cause and may contradict
  what the commit message or step labels suggested.
- Distinguish "the operation was canceled" markers (which usually mean a
  SIBLING job's failure caused this one to be aborted — i.e. cascade,
  not root cause) from real error markers like compile errors, test
  failures, or "Process completed with exit code 1" with substantive
  output above it.
- If the log evidence reveals a transient problem (network reset,
  timeout, "Connection reset", runner crashes), prefer ENV_FLAKINESS +
  RETRY even if the error category superficially looks like something
  else.
- If multiple jobs failed but only some show real root-cause errors
  while others show "operation was canceled", the root cause is in the
  jobs with substantive errors. Severity should reflect the root cause,
  not the cascade count.
"""

FIX_SUGGESTION_SYSTEM_PROMPT = """You are the remediation suggester in a CI/CD failure triage system.

You will receive:
- the run context
- the extracted log evidence
- the classification result that was already chosen

Your job is to propose the most specific fix supported by that evidence.

RULES:
- Prefer an exact file path when the evidence points to one.
- Name the exact change to make, not a vague direction.
- Explain why that change follows from the evidence.
- Be conservative. If the evidence does not name or strongly imply a concrete repository file, do not guess one.
- Do not invent generic workflow files like ".github/workflows/ci.yml" or "pipeline.yml" unless the evidence explicitly mentions a workflow file, GitHub Action reference, or workflow YAML problem.
- Only name a repository file when at least one of these is true:
  1. the logs mention a file path,
  2. the commit message strongly identifies a known config/dependency file,
  3. the failing step clearly points to a workflow or dependency manifest family.
- If the best action is RETRY and no code or config edit is needed, set file to "(none)" and say to rerun once with no repository changes.
- If the evidence is too weak to name a concrete file, use file "(unclear)" and make the change field explicitly say what must be inspected next.
- Do not invent repository files unless the evidence strongly implies them.

Return ONLY a JSON object with exactly these keys:
  file
  change
  reason
"""


# ─── prompt builders ─────────────────────────────────────────────────

# Lines of log content to include per failed step in the prompt.
PROMPT_LINES_PER_EXCERPT = 80


def build_user_message(event: RunEvent) -> str:
    """Build user prompt from metadata only."""
    failed_steps_summary = []
    for i, fs in enumerate(event.failed_steps, 1):
        block = (
            f"  [{i}] job={fs.job_file}\n"
            f"      runner={fs.runner_image}\n"
            f"      step_type={fs.step_type}  duration={fs.step_duration_sec}s"
        )
        if fs.error_text:
            block += f"\n      error_text={fs.error_text[:400]}"
        failed_steps_summary.append(block)

    steps_block = "\n".join(failed_steps_summary) if failed_steps_summary else "  (none extracted)"

    return f"""Analyze this failed GitHub Actions run:

RUN
  repo:        {event.repo}
  workflow:    {event.workflow}
  run:         #{event.run_number}  attempt {event.attempt}
  event:       {event.event}
  branch:      {event.branch}
  protected:   {event.is_protected_branch}
  actor:       {event.actor}
  duration:    {event.duration_sec}s

COMMIT
  sha:         {event.commit_sha}
  author:      {event.commit_author}
  title:       {event.commit_title}
  message:     {event.commit_message}

OUTCOME
  conclusion:  {event.conclusion}
  jobs:        {event.n_jobs} total, {event.failed_jobs_count} failed

FAILED STEPS
{steps_block}

Classify this run."""


def _build_log_section(log_excerpts: List[LogExcerpt]) -> str:
    """Format log excerpts into a LOG EVIDENCE block for the prompt."""
    log_blocks = []
    for i, excerpt in enumerate(log_excerpts, 1):
        all_lines = []
        for window in excerpt.error_windows:
            all_lines.extend(window)
        tail = all_lines[-PROMPT_LINES_PER_EXCERPT:] if all_lines else []
        truncated = len(all_lines) > PROMPT_LINES_PER_EXCERPT

        block = (
            f"--- step [{i}] {excerpt.job_file} ---\n"
            f"strategy: {excerpt.strategy_used}, "
            f"##[error] markers: {len(excerpt.error_marker_lines)}\n"
        )
        if truncated:
            block += (
                f"(showing last {PROMPT_LINES_PER_EXCERPT} of "
                f"{len(all_lines)} extracted lines)\n"
            )
        block += "\n".join(tail) if tail else "(no log content extracted)"
        log_blocks.append(block)

    return "\n\nLOG EVIDENCE\n" + "\n\n".join(log_blocks)


def _extract_mentioned_files(log_excerpts: Optional[List[LogExcerpt]]) -> List[MentionedFile]:
    if not log_excerpts:
        return []
    dedup = {}
    for excerpt in log_excerpts:
        for mentioned in extract_from_excerpt_windows(excerpt.error_windows):
            existing = dedup.get(mentioned.path)
            if existing is None:
                dedup[mentioned.path] = mentioned
                continue
            if existing.line is None and mentioned.line is not None:
                existing.line = mentioned.line
            if existing.column is None and mentioned.column is not None:
                existing.column = mentioned.column
            if len(mentioned.context) > len(existing.context):
                existing.context = mentioned.context
    return sorted(dedup.values(), key=lambda f: (f.line is None, f.path))


def _format_mentioned_files(mentioned_files: List[MentionedFile]) -> str:
    if not mentioned_files:
        return ""
    lines = ["", "MENTIONED FILES"]
    for mentioned in mentioned_files[:8]:
        loc = mentioned.path
        if mentioned.line is not None:
            loc += f":{mentioned.line}"
        if mentioned.column is not None:
            loc += f":{mentioned.column}"
        entry = f"- {loc}"
        if mentioned.context:
            entry += f" | {mentioned.context[:120]}"
        lines.append(entry)
    omitted = len(mentioned_files) - len(mentioned_files[:8])
    if omitted > 0:
        lines.append(f"- ... {omitted} more mentioned files omitted")
    return "\n".join(lines)


def _build_fix_suggestion_message(
    event: RunEvent,
    classification_data: dict,
    log_excerpts: Optional[List[LogExcerpt]] = None,
    mentioned_files: Optional[List[MentionedFile]] = None,
) -> str:
    parts = [
        build_user_message(event),
        "",
        "CLASSIFICATION",
        f"  category:   {classification_data.get('category', 'CODE_REGRESSION')}",
        f"  severity:   {classification_data.get('severity', 'LOW')}",
        f"  confidence: {classification_data.get('confidence', 0.0)}",
        f"  action:     {classification_data.get('action', 'INVESTIGATE_FAILURE')}",
        f"  reasoning:  {classification_data.get('reasoning', '')}",
    ]

    evidence = classification_data.get("evidence") or []
    unknowns = classification_data.get("unknowns") or []
    if evidence:
        parts.append(f"  evidence:   {json.dumps(evidence, ensure_ascii=False)}")
    if unknowns:
        parts.append(f"  unknowns:   {json.dumps(unknowns, ensure_ascii=False)}")
    if log_excerpts:
        parts.append(_build_log_section(log_excerpts))
    if mentioned_files:
        parts.append(_format_mentioned_files(mentioned_files))

    parts.append("")
    parts.append(
        'Based on the evidence you collected, what is the specific fix? '
        'Name the exact file, the exact change, and why.'
    )
    return "\n".join(parts)


def _normalized_fix_suggestion(
    fix_data: dict,
    event: RunEvent,
    classification_data: dict,
    log_excerpts: Optional[List[LogExcerpt]] = None,
) -> FixSuggestion:
    file_value = (fix_data.get("file") or "(unclear)").strip() or "(unclear)"
    change_value = (fix_data.get("change") or "").strip()
    reason_value = (fix_data.get("reason") or "").strip()

    evidence_text_parts = [
        event.commit_title or "",
        event.commit_message or "",
        classification_data.get("reasoning") or "",
        " ".join(classification_data.get("evidence") or []),
        " ".join(classification_data.get("unknowns") or []),
    ]
    if log_excerpts:
        for excerpt in log_excerpts[:3]:
            evidence_text_parts.extend(excerpt.error_marker_lines[:3])
            for window in excerpt.error_windows[:2]:
                evidence_text_parts.extend(window[-12:])
    evidence_text = "\n".join(part for part in evidence_text_parts if part).lower()

    generic_workflow_guesses = {
        ".github/workflows/ci.yml",
        ".github/workflows/ci.yaml",
        ".github/workflows/pipeline.yml",
        ".github/workflows/pipeline.yaml",
        ".github/workflows/build.yml",
        ".github/workflows/build.yaml",
    }
    concrete_workflow_evidence = any(
        token in evidence_text for token in (
            ".github/workflows/",
            "action.yml",
            "action.yaml",
            "dockerfile",
            "actions/",
        )
    )
    file_path_mentioned = file_value.lower() in evidence_text if file_value not in {"(unclear)", "(none)"} else False
    dependency_manifest_names = (
        "pom.xml", "build.gradle", "build.gradle.kts", "cargo.toml",
        "package.json", "package-lock.json", "pnpm-lock.yaml", "poetry.lock",
        "pyproject.toml", "requirements.txt", "go.mod", "go.sum",
        ".github/dependabot.yml", ".github/dependabot.yaml",
    )
    file_family_implied = any(name in file_value.lower() for name in dependency_manifest_names) and any(
        token in evidence_text for token in (
            "version", "dependency", "gradle", "maven", "cargo", "pip",
            "checkout", "actions/", "glibc", "node", "java", "go ",
        )
    )

    should_fallback_unclear = False
    if file_value.lower() in generic_workflow_guesses and not concrete_workflow_evidence:
        should_fallback_unclear = True
    elif file_value.startswith(".github/workflows/") and not concrete_workflow_evidence:
        should_fallback_unclear = True
    elif file_value not in {"(unclear)", "(none)"} and not (file_path_mentioned or file_family_implied or concrete_workflow_evidence):
        if len(file_value.split("/")) > 1:
            should_fallback_unclear = True

    if should_fallback_unclear:
        file_value = "(unclear)"
        if not change_value:
            change_value = "Inspect the failing step output and update the specific workflow, dependency manifest, or source file identified there."
        else:
            change_value = (
                "Inspect the failing step output and apply the suggested change in the specific file named by the logs or manifest family once confirmed."
            )
        if not reason_value:
            reason_value = "The evidence supports the remediation direction, but it does not support naming an exact repository file safely."
        else:
            reason_value = (
                "The evidence supports the remediation direction, but not the exact repository file that was suggested."
            )

    return FixSuggestion(
        file=file_value,
        change=change_value,
        reason=reason_value,
    )


# ─── the agent ────────────────────────────────────────────────────────

def classify(
    event: RunEvent,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    log_excerpts: Optional[List[LogExcerpt]] = None,
) -> ClassificationResult:
    """
    Classify a failed run.

    Args:
        event:        RunEvent from intake_parser.
        client:       OpenAI client pointed at OpenRouter.
        model:        Model name.
        log_excerpts: Optional list of LogExcerpts from log_extractor.
                      When provided, log content is added to the prompt
                      and the LLM receives additional reasoning guidance.
                      When None, behaves as metadata-only classification.

    Returns:
        ClassificationResult with category, severity, confidence,
        action, reasoning, evidence, and unknowns.
    """
    if log_excerpts:
        system_prompt = BASE_SYSTEM_PROMPT + LOG_EVIDENCE_GUIDANCE
        user_message = build_user_message(event) + _build_log_section(log_excerpts)
    else:
        system_prompt = BASE_SYSTEM_PROMPT
        user_message = build_user_message(event)

    mentioned_files = _extract_mentioned_files(log_excerpts)
    if mentioned_files:
        user_message += _format_mentioned_files(mentioned_files)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    record_usage(response, model, call_type="chat", label="classification_agent.classify")

    data = json.loads(response.choices[0].message.content)

    fix_suggestion = None
    try:
        fix_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FIX_SUGGESTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_fix_suggestion_message(
                        event,
                        data,
                        log_excerpts=log_excerpts,
                        mentioned_files=mentioned_files,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        record_usage(
            fix_response,
            model,
            call_type="chat",
            label="classification_agent.fix_suggestion",
        )
        fix_data = json.loads(fix_response.choices[0].message.content)
        fix_suggestion = _normalized_fix_suggestion(
            fix_data,
            event,
            data,
            log_excerpts=log_excerpts,
        )
    except Exception as e:
        fix_suggestion = FixSuggestion(
            file="(unclear)",
            change="Inspect the failing step and update the workflow or code path identified by the logs.",
            reason=f"Fix suggestion generation failed: {type(e).__name__}: {e}",
        )

    return ClassificationResult(
        category=data.get("category", "CODE_REGRESSION"),
        severity=data.get("severity", "LOW"),
        confidence=float(data.get("confidence", 0.0)),
        action=data.get("action", "INVESTIGATE_FAILURE"),
        reasoning=data.get("reasoning", ""),
        evidence=data.get("evidence") or [],
        unknowns=data.get("unknowns") or [],
        mentioned_files=mentioned_files,
        fix_suggestion=fix_suggestion,
    )


# ─── pretty printer ───────────────────────────────────────────────────

def print_result(result: ClassificationResult) -> None:
    print(f"  CLASSIFICATION")
    print(f"    category    {result.category}")
    print(f"    severity    {result.severity}")
    print(f"    confidence  {result.confidence:.2f}")
    print(f"    action      {result.action}")
    if result.fix_suggestion:
        print(f"    fix_file    {result.fix_suggestion.file}")
    print()
    print(f"  REASONING")
    for line in textwrap.wrap(result.reasoning, width=68):
        print(f"    {line}")
    if result.evidence:
        print()
        print(f"  EVIDENCE    {', '.join(result.evidence)}")
    if result.unknowns:
        print(f"  UNKNOWNS    {', '.join(result.unknowns)}")
    if result.mentioned_files:
        print()
        print("  MENTIONED FILES")
        for mentioned in result.mentioned_files[:8]:
            loc = mentioned.path
            if mentioned.line is not None:
                loc += f":{mentioned.line}"
            if mentioned.column is not None:
                loc += f":{mentioned.column}"
            print(f"    {loc}")
    if result.fix_suggestion:
        print()
        print("  FIX SUGGESTION")
        print(f"    file      {result.fix_suggestion.file}")
        change_lines = textwrap.wrap(result.fix_suggestion.change, width=68) or [""]
        print(f"    change    {change_lines[0]}")
        for line in change_lines[1:]:
            print(f"              {line}")
        reason_lines = textwrap.wrap(result.fix_suggestion.reason, width=68) or [""]
        print(f"    reason    {reason_lines[0]}")
        for line in reason_lines[1:]:
            print(f"              {line}")


# ─── convenience: load diverse failed runs ────────────────────────────

def load_diverse_failed_runs(path: str, n: int = 5):
    seen_repos = set()
    results = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = run.get("metadata") or {}
            if meta.get("conclusion") != "failure":
                continue
            if not run.get("log_insights"):
                continue
            repo = run.get("repository_name", "")
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            results.append(run)
            if len(results) >= n:
                break
    return results


# ─── main (quick test) ────────────────────────────────────────────────

if __name__ == "__main__":
    from llm_config import make_client
    try:
        client = make_client()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        exit(1)

    print("Loading 3 failed runs from different repos...\n")
    raw_runs = load_diverse_failed_runs("/home/guc_alaa/runs.json.gz", n=3)
    print(f"Loaded {len(raw_runs)} runs.\n")

    for i, raw in enumerate(raw_runs, 1):
        event = intake(raw)
        print("=" * 72)
        print(f"[{i}]  {event.repo}  #{event.run_number}  ({event.workflow})")
        print("=" * 72)
        print(f"  commit   {event.commit_title}")
        print(f"  branch   {event.branch}  [protected={event.is_protected_branch}]")
        print(f"  failed   {event.failed_jobs_count}/{event.n_jobs} jobs")
        print()

        try:
            result = classify(event, client)
            print_result(result)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}")

        print()

    print("Done.")

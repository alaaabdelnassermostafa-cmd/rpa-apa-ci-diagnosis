# log_extractor.py
# ─────────────────────────────────────────────────────────────────────
# Step 2 of the APA pipeline: the Log Extractor.
#
# Input:  a run's tarball path inside the GHALogs zip archive, plus
#         information about which step failed (from intake_parser).
# Output: a cleaned, bounded slice of log text ready to hand to the
#         classification agent.
#
# This module is deterministic. No LLM call here.
#
# ─── How it finds the relevant slice ─────────────────────────────────
# GitHub Actions emits ##[error] markers at the EXACT line where a
# failure was detected. The actual error text (stack trace, compiler
# message, test failure) lives in the lines immediately BEFORE the
# marker. So our primary strategy is:
#
#   1. Strip timestamps and ANSI escapes.
#   2. Find every ##[error] marker.
#   3. For each marker, take the N lines BEFORE it as the error window.
#
# Fallback: if a log has no ##[error] marker (rare — happens for
# action-gate failures and the like), we fall back to "find the
# failing step's ##[group]Run block, plus the lines that follow it
# until the next group" — which captures both the declaration and
# the actual output.
# ─────────────────────────────────────────────────────────────────────

import io
import os
import re
import tarfile
import zipfile

# When ZIP_URL is set, tarballs are pulled from a REMOTE zip via HTTP range
# requests (zip64) instead of a local file — lets us extract from the 142 GB
# Zenodo archive without downloading it. The RemoteZip (and its 100 MB central
# directory) is read ONCE and cached for the whole process.
_REMOTE_ZIP = None


def _read_tarball_bytes(zip_path: str, tarball_name: str) -> bytes:
    url = os.environ.get("ZIP_URL")
    if url:
        global _REMOTE_ZIP
        if _REMOTE_ZIP is None:
            from remotezip import RemoteZip
            _REMOTE_ZIP = RemoteZip(url)
        return _REMOTE_ZIP.read(tarball_name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(tarball_name)
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ─── run-length deduplication ────────────────────────────────────────
# Collapses repetitive lines that carry only a changing leaf token.
# Three strategies cover the observed patterns in CI logs:
#
#   1. Template match  — mask numbers/versions/paths, compare remainder.
#      Catches: Java stacktraces (same class, different line number).
#
#   2. Fixed-prefix match — lines sharing the same leading keyword word.
#      Catches: "   Compiling <crate> <ver>", "   Downloading …" etc.
#      The whole payload changes but the verb is identical.
#
#   3. Fixed-suffix match — lines ending with the same word/phrase.
#      Catches: "[INFO] … SKIPPED", "[INFO] … SUCCESS", "… done" etc.
#
# Each run of N≥4 matching lines collapses to:
#   <first line>
#   ... [N-2 similar lines collapsed] ...
#   <last line>

_LEAF_RE = re.compile(
    r"(?:"
    r"\d[\d.]*"                          # version numbers / plain integers
    r"|0x[0-9a-fA-F]+"                  # hex addresses
    r"|/[^\s]+"                          # path segments
    r"|[a-zA-Z0-9_$]+\.[a-zA-Z0-9_$]+"  # dotted identifiers (java class refs)
    r")"
)

# Prefixes that mark "progress" lines with no diagnostic value.
_PROGRESS_PREFIXES = (
    "compiling ",
    "downloading ",
    "downloaded ",
    "fetching ",
    "verifying ",
    "checking ",
    "installing ",
    "unpacking ",
    "extracting ",
    "resolving ",
    "running ",
)

# Suffixes that mark cascade/progress lines.
_PROGRESS_SUFFIXES = (
    " skipped",
    " success",
    " failure",    # maven module-level; distinct from the root [ERROR]
    " done",
    " cached",
)


def _line_template(line: str) -> str:
    """Mask leaf tokens so structurally similar lines hash the same."""
    return _LEAF_RE.sub("<X>", line).strip()


def _line_key(line: str) -> str:
    """
    Return a grouping key for run-detection.
    Priority: prefix-verb > suffix-word > template.
    """
    stripped = line.strip().lower()
    for pfx in _PROGRESS_PREFIXES:
        if stripped.startswith(pfx):
            return "__prefix__" + pfx
    for sfx in _PROGRESS_SUFFIXES:
        if stripped.endswith(sfx):
            return "__suffix__" + sfx
    return _line_template(line)


def _collapse_runs(lines: List[str], min_run: int = 4) -> List[str]:
    """Collapse runs of ≥min_run structurally equivalent lines."""
    if not lines:
        return lines

    out: List[str] = []
    i = 0
    while i < len(lines):
        key = _line_key(lines[i])
        j = i + 1
        while j < len(lines) and _line_key(lines[j]) == key:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            out.append(lines[i])
            out.append(f"    ... [{run_len - 2} similar lines collapsed] ...")
            out.append(lines[j - 1])
        else:
            out.extend(lines[i:j])
        i = j
    return out


# ─── tuning constants ────────────────────────────────────────────────

# Lines BEFORE each ##[error] marker to include in the excerpt.
LINES_BEFORE_ERROR = 150

# Lines AFTER each ##[error] marker to include (small — usually just
# the next step starting, but occasionally there's a hint).
LINES_AFTER_ERROR = 5

# Cap on total lines kept across all error windows. Prevents
# pathological logs (many ##[error] markers) from blowing up.
MAX_TOTAL_LINES = 3000

# Length of the GitHub Actions timestamp prefix on every log line.
TIMESTAMP_PREFIX_LEN = 28

# ANSI escape sequences (colored output noise).
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


# ─── typed output schema ─────────────────────────────────────────────

@dataclass
class LogExcerpt:
    job_file: str
    step_label: str

    # The actual content
    error_windows: List[List[str]] = field(default_factory=list)
    # Just the ##[error] marker lines themselves, for quick reference
    error_marker_lines: List[str] = field(default_factory=list)

    # Diagnostics
    total_lines_in_file: int = 0
    strategy_used: str = "none"     # "error_marker" | "group_fallback" | "none"
    truncated: bool = False
    extraction_note: str = ""

    def as_prompt_text(self, header: bool = True) -> str:
        """Render the excerpt as clean text ready for an LLM prompt."""
        parts = []
        if header:
            parts.append(f"=== Log excerpt: {self.job_file} ===")
            parts.append(f"Failing step: {self.step_label}")
            parts.append(f"Extraction strategy: {self.strategy_used}")
            if self.truncated:
                parts.append(f"(output truncated to fit {MAX_TOTAL_LINES} line cap)")
            parts.append("")

        if not self.error_windows:
            parts.append("(no error context could be extracted)")
            return "\n".join(parts)

        for i, window in enumerate(self.error_windows, 1):
            if len(self.error_windows) > 1:
                parts.append(f"--- error window {i}/{len(self.error_windows)} ---")
            else:
                parts.append("--- error context ---")
            parts.extend(_collapse_runs(window))
            parts.append("")

        return "\n".join(parts)


# ─── helpers ─────────────────────────────────────────────────────────

def _strip_timestamp(line: str) -> str:
    if len(line) > TIMESTAMP_PREFIX_LEN + 1:
        return line[TIMESTAMP_PREFIX_LEN + 1:]
    return line


def _strip_ansi(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line)


def _clean_line(line: str) -> str:
    return _strip_ansi(_strip_timestamp(line))


def _read_log_file_from_tarball(
    tarball_bytes: bytes,
    job_file: str,
) -> Optional[str]:
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tf:
        try:
            member = tf.getmember(job_file)
        except KeyError:
            return None
        if not member.isfile():
            return None
        return tf.extractfile(member).read().decode("utf-8", errors="replace")


# GitHub emits "##[error]Process completed with exit code 1." at the END of every
# failed step. Anchoring the excerpt window on THIS generic marker captures
# whatever ran right before it (passing tests, checkout, artifact upload) instead
# of the real error — the dominant cause of "uninformative" excerpts. So we
# classify markers and prefer substantive ones; if only generic markers exist we
# search the body for real error CONTENT lines (npm ERR!, traceback, etc.).
_GENERIC_MARKER_RE = re.compile(
    r"process completed with exit code|the operation was cancell?ed|"
    r"exited with code|##\[error\]\s*$",
    re.I,
)

# Real, diagnosable error CONTENT (not a GitHub ##[...] command line). Patterns
# are deliberately specific — bare tool names ("eslint") or loose "error" match
# filenames/paths and produce false anchors, so we require error-shaped context.
_REAL_ERROR_RE = re.compile(
    r"error:\s|error\]|\d+\s+errors?\b|errors?\s+(?:found|occurred|encountered)|"
    r"error compiling|compilation error|build failed|"
    r"\btests?\s+failed\b|test failure|^FAILED\b|\bFAILED\s+\w|\d+\s+failed\b|"
    r"traceback \(most recent call last\)|assertionerror|"
    r"\bexception\b|\bpanic:|\bfatal:\s|fatal error|segmentation fault|"
    r"syntaxerror|typeerror|referenceerror|nameerror|importerror|modulenotfounderror|"
    r"cannot find|could not find|command not found|no such file|could not resolve|"
    r"no matching version|version conflict|npm err!|"
    r"\b[EWF]\d{3,4}\s|"            # flake8/lint codes with trailing space (E302 ...)
    r"(?<![_a-z])unauthorized\b|permission denied|access denied|"
    r"invalid workflow|yaml.{0,20}error|timed out\b|connection refused",
    re.I,
)


def _is_generic_marker(line: str) -> bool:
    return bool(_GENERIC_MARKER_RE.search(line))


def _find_real_error_lines(lines: List[str], limit: int = 8) -> List[int]:
    """Indices of real error CONTENT lines (excluding GitHub ##[...] markers)."""
    out: List[int] = []
    for i, ln in enumerate(lines):
        if "##[" in ln:            # GitHub command/marker lines handled separately
            continue
        if _REAL_ERROR_RE.search(ln):
            out.append(i)
            if len(out) >= limit:
                break
    return out


def _find_error_markers(lines: List[str]) -> List[int]:
    """Return line indices of ##[error] markers."""
    return [i for i, ln in enumerate(lines) if "##[error]" in ln]


def _build_error_windows(
    lines: List[str],
    marker_indices: List[int],
    after: int = LINES_AFTER_ERROR,
) -> List[List[str]]:
    """
    For each anchor line, return a window of lines around it:
    LINES_BEFORE_ERROR before, the anchor itself, `after` lines after.
    Adjacent/overlapping windows are merged. (`after` is larger for real
    error CONTENT anchors, where the message continues below the line.)
    """
    spans: List[Tuple[int, int]] = []
    for idx in marker_indices:
        start = max(0, idx - LINES_BEFORE_ERROR)
        end = min(len(lines) - 1, idx + after)
        spans.append((start, end))

    # Merge overlapping spans
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return [lines[s:e + 1] for s, e in merged]


def _fallback_group_block(
    lines: List[str],
    step_label: str,
) -> Optional[List[str]]:
    """
    Fallback when no ##[error] marker exists.

    Find the ##[group]Run block matching step_label, then return:
      - the group declaration block, PLUS
      - the lines that follow ##[endgroup] until the next ##[group].
    The "after endgroup" portion is where the step's actual output
    lives, which is what we actually want for diagnosis.
    """
    if not lines:
        return None

    step_label_lower = (step_label or "").lower().strip()

    # Find all group starts
    group_starts: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if "##[group]Run " in ln:
            idx = ln.index("##[group]Run ") + len("##[group]Run ")
            header = ln[idx:].strip()
            group_starts.append((i, header))

    if not group_starts:
        return None

    # Try to match by step label
    matched = None
    if step_label_lower:
        for line_idx, header in group_starts:
            if step_label_lower in header.lower():
                matched = line_idx
                break
        if matched is None:
            first_word = step_label_lower.split()[0] if step_label_lower else ""
            if first_word and len(first_word) > 2:
                for line_idx, header in group_starts:
                    if first_word in header.lower():
                        matched = line_idx
                        break

    if matched is None:
        # Fall back to the LAST group (closest to where execution stopped)
        matched = group_starts[-1][0]

    # End of this section = the line before the next ##[group] header,
    # or end of file if this was the last group.
    next_group_start = None
    for line_idx, _ in group_starts:
        if line_idx > matched:
            next_group_start = line_idx
            break
    end = next_group_start - 1 if next_group_start is not None else len(lines) - 1

    return lines[matched:end + 1]


# ─── the extractor ───────────────────────────────────────────────────

def extract_log_excerpt(
    zip_path: str,
    tarball_name: str,
    job_file: str,
    step_label: str,
) -> LogExcerpt:
    """
    Main entry point.

    Given:
      zip_path      — path to github_run_logs.zip on disk
      tarball_name  — path inside the zip
      job_file      — file inside the tarball
      step_label    — label of the failing step (used for fallback only)

    Returns:
      a LogExcerpt with cleaned lines ready to feed to the classifier.
    """
    result = LogExcerpt(job_file=job_file, step_label=step_label)

    # 1. Pluck the tarball out of the big zip (local file OR remote via range).
    try:
        tarball_bytes = _read_tarball_bytes(zip_path, tarball_name)
    except KeyError:
        result.extraction_note = f"tarball not found in zip: {tarball_name}"
        return result
    except (zipfile.BadZipFile, OSError) as e:
        result.extraction_note = f"zip read error: {e}"
        return result

    # 2. Read the specific job_file.
    try:
        raw_text = _read_log_file_from_tarball(tarball_bytes, job_file)
    except tarfile.TarError as e:
        result.extraction_note = f"tarball read error: {e}"
        return result

    if raw_text is None:
        result.extraction_note = f"job_file not found in tarball: {job_file}"
        return result

    # 3. Clean every line.
    raw_lines = raw_text.splitlines()
    cleaned = [_clean_line(ln) for ln in raw_lines]
    result.total_lines_in_file = len(cleaned)

    # 4. Primary strategy: anchor the window on the REAL error.
    #    Prefer substantive ##[error] markers; if only the generic
    #    "exit code 1"/"canceled" markers exist, anchor on real error CONTENT
    #    lines (npm ERR!, traceback, compile error, ...); only fall back to the
    #    generic markers if nothing better is found. This avoids capturing the
    #    passing/checkout output that sits before the generic end-of-step marker.
    all_markers = _find_error_markers(cleaned)
    result.error_marker_lines = [cleaned[i] for i in all_markers]
    substantive = [i for i in all_markers if not _is_generic_marker(cleaned[i])]
    real_errors = _find_real_error_lines(cleaned)

    if substantive:
        anchors, strat, after = substantive, "error_marker", LINES_AFTER_ERROR
    elif real_errors:
        anchors, strat, after = real_errors, "error_pattern", 40
    elif all_markers:
        anchors, strat, after = all_markers, "error_marker_generic", LINES_AFTER_ERROR
    else:
        anchors, strat, after = [], "none", LINES_AFTER_ERROR

    if anchors:
        windows = _build_error_windows(cleaned, anchors, after=after)

        # Enforce the global cap by trimming windows from the FRONT
        # (errors are at the back of each window — we keep those).
        total_lines = sum(len(w) for w in windows)
        if total_lines > MAX_TOTAL_LINES:
            result.truncated = True
            # Trim oldest content first
            keep_budget = MAX_TOTAL_LINES
            trimmed: List[List[str]] = []
            for window in reversed(windows):
                if len(window) <= keep_budget:
                    trimmed.insert(0, window)
                    keep_budget -= len(window)
                else:
                    trimmed.insert(0, window[-keep_budget:])
                    keep_budget = 0
                    break
            windows = trimmed

        result.error_windows = windows
        result.strategy_used = strat
        return result

    # 5. Fallback: no ##[error] markers — use the group-block strategy.
    fallback = _fallback_group_block(cleaned, step_label)
    if fallback:
        if len(fallback) > MAX_TOTAL_LINES:
            fallback = fallback[-MAX_TOTAL_LINES:]
            result.truncated = True
        result.error_windows = [fallback]
        result.strategy_used = "group_fallback"
    else:
        result.strategy_used = "none"
        result.extraction_note = "no ##[error] markers and no matching group"

    return result

#for agent.py preprocessing
# keeps lines around these keywords: context_before = 2, context_after = 2 
# used in build_preprocessing_state(), Bayesian signal: apply(signal_error_text(error_lines), 
#"error_text"), inspect_commit_diff(), classify()
def extract_error_summary_lines(
    windows: list[list[str]],
    marker_lines: list[str],
    max_lines: int = 50,
    context_before: int = 2,
    context_after: int = 2,
    max_chars_per_line: int = 240,
) -> list[str]:
    keywords = (
        "error", "errors", "failed", "failure", "fatal",
        "exception", "traceback", "assert",
        "module not found", "cannot find module", "could not resolve",
        "dependency conflict", "version conflict",
        "undefined reference",
        "segmentation fault", "core dumped",
        "typeerror", "valueerror", "syntaxerror", "importerror",
        "connection refused", "connection reset", "timeout",
        "permission denied", "authentication failed", "unauthorized",
        "forbidden", "access denied",
        "no space left on device",
        "process completed with exit code",
        "deprecated",
        "test failed", "tests failed", "assertionerror",
        "npm err!", "pip", "cargo", "gradle", "maven",
    )

    lines = []
    for window in windows:
        lines.extend(window)

    lines.extend(marker_lines)
    lines = [(ln or "").strip() for ln in lines if (ln or "").strip()]

    keep_indices = set()
    for i, line in enumerate(lines):
        lowered = line.lower()
        if any(kw in lowered for kw in keywords):
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after + 1)
            keep_indices.update(range(start, end))

    selected = []
    seen = set()
    for i in sorted(keep_indices):
        text = lines[i][:max_chars_per_line]
        if text not in seen:
            seen.add(text)
            selected.append(text)
        if len(selected) >= max_lines:
            break

    return selected

# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test on the hibernate failure we just diagnosed.
    ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
    TARBALL = "logs/hibernate/hibernate-search/Hibernate_Search_simple_build_e96d/66-1.tar.gz"
    JOB_FILE = "1_Build and test on Java 11.txt"
    STEP_LABEL = "./ci/docker-cleanup.sh"  # what intake guessed (wrong)

    print(f"Extracting log excerpt...")
    print(f"  zip       {ZIP_PATH}")
    print(f"  tarball   {TARBALL}")
    print(f"  job_file  {JOB_FILE}")
    print(f"  step      {STEP_LABEL}")
    print()

    excerpt = extract_log_excerpt(
        zip_path=ZIP_PATH,
        tarball_name=TARBALL,
        job_file=JOB_FILE,
        step_label=STEP_LABEL,
    )

    print("-" * 70)
    print("EXTRACTION SUMMARY")
    print("-" * 70)
    print(f"  total lines in file       {excerpt.total_lines_in_file:,}")
    print(f"  strategy used             {excerpt.strategy_used}")
    print(f"  ##[error] markers found   {len(excerpt.error_marker_lines)}")
    print(f"  error windows             {len(excerpt.error_windows)}")
    print(f"  total lines kept          {sum(len(w) for w in excerpt.error_windows):,}")
    print(f"  truncated                 {excerpt.truncated}")
    if excerpt.extraction_note:
        print(f"  note                      {excerpt.extraction_note}")
    print()

    if excerpt.error_marker_lines:
        print("-" * 70)
        print("ERROR MARKER LINES")
        print("-" * 70)
        for ln in excerpt.error_marker_lines:
            print(f"  {ln}")
        print()

    for i, window in enumerate(excerpt.error_windows, 1):
        print("-" * 70)
        print(f"WINDOW {i}: LAST 40 LINES (where the error usually is)")
        print("-" * 70)
        for ln in window[-40:]:
            print(f"  {ln}")
        print()

    out_path = Path("/home/guc_alaa/hibernate_excerpt_v2.txt")
    out_path.write_text(excerpt.as_prompt_text(), encoding="utf-8")
    print(f"Full prompt text written to {out_path}")
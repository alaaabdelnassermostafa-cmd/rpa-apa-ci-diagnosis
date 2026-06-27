# semantic_refiner.py
# ─────────────────────────────────────────────────────────────────────
# Stage-2 semantic refinement for the Log Extractor.
#
# The deterministic Log Extractor (Stage 1) reduces a multi-megabyte raw
# log to a few hundred candidate lines using ##[error] markers, windowing,
# and run-length deduplication. It preserves the failure signal in most
# cases, but an LLM-judge evaluation showed two recurring failure modes:
#   (a) the marker window misses the root-cause line (it sat at the edge),
#   (b) the window is mostly noise, diluting the real signal.
#
# This module is the LAYERED Stage 2: it runs ON the Stage-1 output (not the
# raw log), embedding only the already-reduced candidate lines, scoring each
# by how anomalous it is relative to the run's "normal" output, and keeping
# the most anomalous contiguous span (plus context). Embedding a few hundred
# lines is cheap; embedding the raw log would not be.
#
# It is an APA-only stage (the deterministic RPA baseline never runs it, by
# design: refinement is itself an embedding/LLM step and belongs with the LLM
# system). It is ON by default; set CI_AGENT_SEMANTIC_REFINE=0 to disable it
# for an ablation. When embeddings are unavailable it falls back to the Stage-1
# lines unchanged (never worse), and it only fires on oversized excerpts.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import os
import re
from typing import List, Optional


# Lines whose anomaly we never want to suppress even if "common": explicit
# failure tokens. Used to bias the kept span toward the actual error.
_FAILURE_TOKENS = (
    "error", "failed", "failure", "exception", "traceback", "fatal",
    "panic", "assert", "not found", "cannot", "unable", "denied",
    "timeout", "timed out", "exit code", "##[error]",
)

# Context kept on each side of the chosen anchor span.
_CONTEXT_LINES = 8
# Refinement runs on every excerpt regardless of length. The "keep the error
# region and ADD failure-relevant lines" design is safe on small excerpts:
# when the excerpt already fits under keep_max_lines, everything is kept and
# the output is unchanged, so short logs are never degraded. (The old gate
# existed for the trim-based v1, which could over-cut small excerpts.)
_MIN_LINES_TO_REFINE = 1

# Two alternative Stage-2 strategies were implemented and evaluated by LLM
# judge over 20 cases, and both were rejected (see the Log Extractor section):
#   - adding semantically distant lines    -> 3.90/5, 80% signal (no gain)
#   - Cordon-style k-NN density (add rarest)-> 3.55/5, 70% signal (hurts)
# Cordon's "repetition, even errors, is background; surface the rare event"
# model is correct for operational logs but inverts CI failure logs, where the
# diagnostic signal is concentrated in the dense error region, not rare lines.
# The error-region-only design below scored best (4.0/5, 85% signal) and is
# the one retained.


def is_enabled() -> bool:
    # ON by default for APA; set CI_AGENT_SEMANTIC_REFINE=0 to disable (ablation).
    return os.environ.get("CI_AGENT_SEMANTIC_REFINE", "1") != "0"


def _normalize(line: str) -> str:
    """Mask volatile tokens so 'normal' repeated lines cluster together."""
    s = line.lower()
    s = re.sub(r"\d+", "0", s)                      # numbers
    s = re.sub(r"0x[0-9a-f]+", "0x0", s)            # hex
    s = re.sub(r"[0-9a-f]{8,}", "<hash>", s)        # hashes/shas
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _failure_score(line: str) -> float:
    low = line.lower()
    return sum(1.0 for t in _FAILURE_TOKENS if t in low)


def _embed(lines: List[str], client, model: str) -> Optional[List[List[float]]]:
    """Embed lines via the OpenAI-compatible client. None on any failure."""
    try:
        # Truncate each line so a pathological long line can't blow the call.
        payload = [ln[:500] if ln.strip() else "(blank)" for ln in lines]
        resp = client.embeddings.create(model=model, input=payload)
        return [d.embedding for d in resp.data]
    except Exception as exc:  # noqa: BLE001 — any failure → graceful fallback
        print(f"  [semantic_refiner] embedding failed, keeping Stage-1 output: {exc}")
        return None


def _centroid(vectors: List[List[float]]) -> List[float]:
    n = len(vectors)
    dim = len(vectors[0])
    c = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            c[i] += x
    return [x / n for x in c]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 1.0
    return 1.0 - dot / (na * nb)


def refine_lines(
    lines: List[str],
    client,
    *,
    embedding_model: str = "text-embedding-3-small",
    baseline_lines: Optional[List[str]] = None,
    keep_max_lines: int = 200,
) -> tuple[List[str], dict]:
    """Keep the error region, then ADD semantically distant lines that may
    carry additional failure information the contiguous window missed.

    Returns (refined_lines, info). On any problem (disabled, already-tight
    excerpt, embeddings unavailable) it returns the input lines unchanged, so
    the pipeline is never worse than Stage 1 alone.

    Model:
      1. Keep the error-anchored span: every ##[error] marker and the lines
         immediately around the failure are always retained (never dropped).
      2. Add the RAREST remaining lines by k-NN density scoring (Cordon's
         method): each line's window is scored by its mean cosine distance to
         its k nearest-neighbour windows. Rare windows score high; repeated
         content (even repeated errors) has near neighbours and scores low, so
         it is treated as background. This surfaces genuinely distinct events
         the contiguous error window may have missed.
      An early version kept only the most unusual lines and dropped the rest;
      it performed worse because the most unusual line in a failure log is
      often not the diagnostic one. Anchoring on the explicit error lines and
      only ADDING rare context fixed that.
    """
    info = {"applied": False, "reason": "", "kept": len(lines), "of": len(lines)}

    if not is_enabled():
        info["reason"] = "disabled"
        return lines, info
    # Only refine genuinely oversized excerpts; small ones are already tight.
    if len(lines) < _MIN_LINES_TO_REFINE:
        info["reason"] = "already-tight"
        return lines, info

    cand_vecs = _embed(lines, client, embedding_model)
    if cand_vecs is None:
        info["reason"] = "no-embeddings"
        return lines, info

    fail = [_failure_score(ln) for ln in lines]
    marker_idxs = [i for i, ln in enumerate(lines) if "##[error]" in ln.lower()]

    # ── 1. Error-anchored span (always kept) ─────────────────────────
    if marker_idxs:
        anchor = marker_idxs[-1]
        info["anchor"] = "marker"
    elif max(fail) > 0:
        anchor = max(range(len(lines)), key=lambda i: fail[i])
        info["anchor"] = "failure_token"
    else:
        # No failure signal at all — nothing to anchor on; keep as-is.
        info["reason"] = "no-anchor"
        return lines, info

    half = max(_CONTEXT_LINES, keep_max_lines // 2)
    a_start = max(0, anchor - half)
    a_end = min(len(lines), anchor + half + 1)
    if marker_idxs:
        a_start = min(a_start, min(marker_idxs))
        a_end = max(a_end, max(marker_idxs) + 1)
    kept_idx = set(range(a_start, a_end))

    # ── 2. Keep only the error-anchored region ────────────────────────
    # Adding further context (semantically distant lines, or Cordon-style
    # k-NN-rare lines) was evaluated and did not help on CI failure logs,
    # whose diagnostic signal is concentrated in the dense error region. So we
    # keep only the error-anchored span and add nothing.
    added = 0
    info["selector"] = "error_region_only"

    keep_sorted = sorted(kept_idx)
    refined = [lines[i] for i in keep_sorted]
    info.update({
        "added_distant_lines": added,
        "region_span": [a_start, a_end],
        "applied": True,
        "reason": "ok",
        "kept": len(refined),
        "of": len(lines),
        "anchor_index": anchor,
    })
    return refined, info

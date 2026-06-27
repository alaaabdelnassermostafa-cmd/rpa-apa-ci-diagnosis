# bayesian_tracker_dual.py
# ─────────────────────────────────────────────────────────────────────
# Dual-mode Bayesian belief tracker.
#
# RPA mode: hand-coded signal functions generate likelihood vectors.
# APA mode: LLM reads evidence and generates likelihood vectors.
#
# Same Bayesian math underneath. Same entropy, same information gain.
# The ONLY difference is how likelihoods are produced.
#
# This is the core of the thesis comparison: rules vs reasoning,
# measured on identical mathematical ground.
# ─────────────────────────────────────────────────────────────────────

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Import the hand-coded signals from the original tracker
from src.apa.bayesian_tracker import (
    CATEGORIES,
    N_CATEGORIES,
    BeliefState,
    print_beliefs,
    signal_many_jobs_failed,
    signal_error_text,
    signal_branch_type,
    signal_commit_message,
    signal_previous_runs,
    signal_detection_mode,
)
from src.apa.llm_usage import record_usage


# ─── LLM-based signal generator (APA mode) ──────────────────────────

LLM_SIGNAL_PROMPT = """You are a Bayesian signal generator for CI/CD failure classification.

Given an observation about a failed CI/CD run, estimate how likely this observation would be under each failure category. Return a probability distribution that sums to 1.0.

FAILURE CATEGORIES:
  CODE_REGRESSION — The commit introduced a logic, syntax, or build error in source code. The developer must change application or test code. ALSO: if the commit title starts with "Revert", this is CODE_REGRESSION — a revert is always fixing a prior code regression.
  DEPENDENCY_CONFLICT — A package version is explicitly incompatible or missing (lockfile mismatch, install failure, named version conflict). The developer must pin or update a manifest. IMPORTANT: the mere presence of a dependency file in the commit does NOT make this DEPENDENCY_CONFLICT — the error log must actually show a version conflict, missing package, or install failure.
  CONFIG_ERROR — The CI workflow YAML itself has a structural problem: wrong action pin, missing required input, deprecated runner, or undefined secret. The developer must edit a workflow file.
  QUALITY_VIOLATION — A linter or static-analysis tool rejected the code. Retry will NOT help; the developer must fix the violations.
  TEST_FLAKINESS — Intermittent test failure unrelated to any code change; retry fixes it.
  INFRA_INCOMPATIBILITY — CI tooling or runner image is deterministically incompatible with the project (fails every run until a config/version change is made). Retry will NOT help.
  ENV_FLAKINESS — Transient CI infrastructure problem (network timeout, rate limit, ephemeral runner outage). A plain retry is expected to fix it.
  CASCADE_FAILURE — Job failed because a sibling job it depends on already failed.
  TOOLING_ARTIFACT — Dataset/log-parser bug, not a real failure.

OBSERVATION:
{observation}

Think step by step:
1. Which categories would MOST likely produce this observation?
2. Which categories would LEAST likely produce it?
3. Assign probabilities reflecting these judgments. They must sum to 1.0.

Respond with ONLY a JSON object mapping each category to its probability:
{{
  "reasoning": "step by step thoughts on which categories are most and least likely based on the observation",
  "CODE_REGRESSION": 0.xx,
  "DEPENDENCY_CONFLICT": 0.xx,
  ...
}}"""


def llm_generate_likelihood(
    observation: str,
    client,
    model: str = "gpt-4.1-mini",
) -> Dict[str, float]:
    """
    Use the LLM to generate a likelihood vector for an observation.
    This is the APA replacement for hand-coded signal functions.
    """
    prompt = LLM_SIGNAL_PROMPT.format(observation=observation)

    try:
        from src.apa.llm_usage import usage_kwargs
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise probabilistic reasoner. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            **usage_kwargs(),
        )
        record_usage(response, model, call_type="chat", label="bayesian_tracker_dual.likelihood")
        from src.apa.llm_usage import log_transcript
        log_transcript("likelihood", model,
                       [{"role": "user", "content": prompt}], response)
        import re
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'```(?:json)?', '', content).strip()
        data = json.loads(content)

        # Validate and normalize
        result = {}
        for cat in CATEGORIES:
            val = data.get(cat, 0.01)
            result[cat] = max(float(val), 0.001)

        total = sum(result.values())
        return {k: v / total for k, v in result.items()}

    except Exception as e:
        print(f"    LLM signal error: {e}")
        return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}


# ─── observation formatters ──────────────────────────────────────────
# These turn raw data into human-readable observation strings for the
# LLM. The RPA version doesn't need these — it uses the raw data
# directly in its signal functions.

def format_observation_jobs(n_failed: int, n_total: int) -> str:
    return (
        f"In this CI run, {n_failed} out of {n_total} jobs failed. "
        f"{'All' if n_failed == n_total else 'Some'} jobs in the "
        f"workflow reported failure."
    )


def format_observation_error(error_lines: List[str]) -> str:
    errors = "\n".join(error_lines[:10])
    return f"The following error text was found in the CI logs:\n{errors}"


def format_observation_branch(is_protected: bool, branch: str) -> str:
    prot = "a protected" if is_protected else "an unprotected"
    return (
        f"The failure occurred on {prot} branch named '{branch}'."
    )


def format_observation_commit(message: str) -> str:
    return f"The commit message that triggered this CI run was:\n{message}"


def format_observation_history(n_failures: int, n_total: int) -> str:
    if n_total == 0:
        return "No recent run history is available for this workflow."
    rate = n_failures / n_total
    return (
        f"Looking at the last {n_total} runs of this workflow, "
        f"{n_failures} failed ({rate:.0%} failure rate)."
    )


def format_observation_detection(mode: str) -> str:
    descriptions = {
        "per_step_error": "A specific step in the job had a clear error payload.",
        "single_step_inferred": "The job had only one step, and the run failed, so that step is the failure by elimination.",
        "job_level_fallback": "No step had a clear error. The failure was attributed to the last step that started running.",
        "unknown_failure": "The failure could not be attributed to any specific step.",
    }
    desc = descriptions.get(mode, f"Detection mode: {mode}")
    return f"Failure detection: {desc}"


def format_observation_workflow_contents(signals: dict) -> str:
    """
    signals keys:
      action_versions   — list of "action@version" strings found in uses: lines
      runners           — list of runs-on values
      pinned_versions   — list of explicit version env vars / matrix values
      deprecated_nodes  — list of actions using deprecated node versions (node12/node16)
    """
    parts = []
    if signals.get("deprecated_nodes"):
        parts.append(
            f"workflow uses deprecated Node.js runtimes: {', '.join(signals['deprecated_nodes'][:5])}"
        )
    if signals.get("action_versions"):
        parts.append(
            f"action version pins in workflow: {', '.join(signals['action_versions'][:8])}"
        )
    if signals.get("runners"):
        parts.append(
            f"runner environments: {', '.join(set(signals['runners'][:6]))}"
        )
    if signals.get("pinned_versions"):
        parts.append(
            f"version pins in env/matrix: {', '.join(signals['pinned_versions'][:6])}"
        )
    if not parts:
        return "Workflow file retrieved but contained no actionable version/runner signals."
    return "Workflow file inspection reveals:\n" + "\n".join(f"  - {p}" for p in parts)


def format_observation_changed_files(file_families: dict) -> str:
    """
    file_families keys: workflow, dependency, source, test, config, docs, other
    Values are lists of filenames.
    """
    parts = []
    if file_families.get("workflow"):
        parts.append(
            f"workflow/CI config files changed: {', '.join(file_families['workflow'][:5])}"
        )
    if file_families.get("dependency"):
        parts.append(
            f"dependency manifest files changed: {', '.join(file_families['dependency'][:5])}"
        )
    if file_families.get("source"):
        n = len(file_families["source"])
        sample = ", ".join(file_families["source"][:3])
        parts.append(f"{n} source file(s) changed (e.g. {sample})")
    if file_families.get("test"):
        n = len(file_families["test"])
        sample = ", ".join(file_families["test"][:3])
        parts.append(f"{n} test file(s) changed (e.g. {sample})")
    if file_families.get("config"):
        parts.append(
            f"non-workflow config files changed: {', '.join(file_families['config'][:3])}"
        )
    if file_families.get("docs"):
        parts.append(
            f"docs/README/changelog files changed: {', '.join(file_families['docs'][:3])}"
        )
    if not parts:
        return "No changed files could be retrieved for this commit."
    return "Files changed in the triggering commit:\n" + "\n".join(f"  - {p}" for p in parts)


# ─── dual-mode tracker ───────────────────────────────────────────────

@dataclass
class DualTracker:
    """
    Wraps BeliefState with two modes of operation:
      mode="rpa" → uses hand-coded signal functions
      mode="apa" → uses LLM to generate likelihood vectors
    """
    mode: str  # "rpa" or "apa"
    state: BeliefState = field(default_factory=BeliefState)
    client: object = None  # OpenAI client, only needed for APA mode
    model: str = "gpt-4.1-mini"
    api_calls: int = 0

    def observe_jobs(self, n_failed: int, n_total: int) -> None:
        if self.mode == "rpa":
            likelihood = signal_many_jobs_failed(n_failed, n_total)
        else:
            obs = format_observation_jobs(n_failed, n_total)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "jobs_failed")

    def observe_errors(self, error_lines: List[str]) -> None:
        if not error_lines:
            return
        if self.mode == "rpa":
            likelihood = signal_error_text(error_lines)
        else:
            obs = format_observation_error(error_lines)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "error_text")

    def observe_branch(self, is_protected: bool, branch: str) -> None:
        if self.mode == "rpa":
            likelihood = signal_branch_type(is_protected, branch)
        else:
            obs = format_observation_branch(is_protected, branch)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "branch_type")

    def observe_commit(self, message: str) -> None:
        if self.mode == "rpa":
            likelihood = signal_commit_message(message)
        else:
            obs = format_observation_commit(message)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "commit_message")

    def observe_history(self, n_failures: int, n_total: int) -> None:
        if self.mode == "rpa":
            likelihood = signal_previous_runs(n_failures, n_total)
        else:
            obs = format_observation_history(n_failures, n_total)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "previous_runs")

    def observe_detection(self, mode: str) -> None:
        if self.mode == "rpa":
            likelihood = signal_detection_mode(mode)
        else:
            obs = format_observation_detection(mode)
            likelihood = llm_generate_likelihood(obs, self.client, self.model)
            self.api_calls += 1
        self.state.update(likelihood, "detection_mode")

    def result(self) -> dict:
        top_cat, top_prob = self.state.top_category()
        return {
            "category": top_cat,
            "probability": top_prob,
            "confidence": self.state.confidence(),
            "entropy": self.state.entropy(),
            "all_probabilities": dict(self.state.probabilities),
            "trace": self.state.history,
            "mode": self.mode,
            "api_calls": self.api_calls,
        }


# ─── self-test: compare RPA vs APA on bcrypt ────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-MODE BAYESIAN TRACKER — bcrypt case")
    print("=" * 70)

    # ── RPA mode ────────────────────────────────────────────────────
    print("\n>>> RPA MODE (hand-coded signals)\n")
    rpa = DualTracker(mode="rpa")

    rpa.observe_branch(False, "dependabot/github_actions/actions/checkout-4.1.0")
    rpa.observe_jobs(8, 8)
    rpa.observe_errors(["GLIBC_2.27 not found", "operation was canceled"])
    rpa.observe_commit("Bump actions/checkout from 3.6.0 to 4.1.0")
    rpa.observe_history(0, 5)

    print_beliefs(rpa.state)
    rpa_result = rpa.result()
    print(f"\n  RESULT: {rpa_result['category']} "
          f"(p={rpa_result['probability']:.3f}, "
          f"conf={rpa_result['confidence']:.1%})")
    print(f"  API calls: {rpa_result['api_calls']}")

    # ── APA mode ────────────────────────────────────────────────────
    from llm_config import make_client
    try:
        client = make_client()
    except RuntimeError:
        print("\n(skipping APA mode — no LLM API key configured)")
        client = None

        print("\n\n>>> APA MODE (LLM-generated signals)\n")
        apa = DualTracker(mode="apa", client=client)

        apa.observe_branch(False, "dependabot/github_actions/actions/checkout-4.1.0")
        apa.observe_jobs(8, 8)
        apa.observe_errors(["GLIBC_2.27 not found", "operation was canceled"])
        apa.observe_commit("Bump actions/checkout from 3.6.0 to 4.1.0")
        apa.observe_history(0, 5)

        print_beliefs(apa.state)
        apa_result = apa.result()
        print(f"\n  RESULT: {apa_result['category']} "
              f"(p={apa_result['probability']:.3f}, "
              f"conf={apa_result['confidence']:.1%})")
        print(f"  API calls: {apa_result['api_calls']}")

        # ── comparison ──────────────────────────────────────────────
        print("\n\n>>> COMPARISON")
        print(f"  RPA: {rpa_result['category']} "
              f"(p={rpa_result['probability']:.3f}, "
              f"conf={rpa_result['confidence']:.1%}, "
              f"calls={rpa_result['api_calls']})")
        print(f"  APA: {apa_result['category']} "
              f"(p={apa_result['probability']:.3f}, "
              f"conf={apa_result['confidence']:.1%}, "
              f"calls={apa_result['api_calls']})")
        if rpa_result['category'] == apa_result['category']:
            print("  → Same category. APA confidence "
                  f"{'higher' if apa_result['confidence'] > rpa_result['confidence'] else 'lower'}.")
        else:
            print(f"  → DIFFERENT categories! This is where the approaches diverge.")
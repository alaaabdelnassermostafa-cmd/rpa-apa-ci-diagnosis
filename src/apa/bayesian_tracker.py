# bayesian_tracker.py
# ─────────────────────────────────────────────────────────────────────
# Bayesian belief tracker for CI/CD failure classification.
#
# Maintains a probability distribution over failure categories.
# Updates beliefs as new evidence arrives from tool calls.
# Computes entropy and information gain for tool selection.
#
# No LLM calls. Pure math. ~150 lines.
# ─────────────────────────────────────────────────────────────────────

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── failure categories (must match classification_agent.py) ─────────

CATEGORIES = [
    "CODE_REGRESSION",       # source/test file changed; compile or test error
    "DEPENDENCY_CONFLICT",   # package version incompatibility or resolution failure
    "CONFIG_ERROR",          # CI workflow YAML structural problem
    "QUALITY_VIOLATION",     # linter / static-analysis threshold not met
    "TEST_FLAKINESS",        # non-deterministic test; no code change caused it
    "INFRA_INCOMPATIBILITY", # toolchain/runner version mismatch — deterministic, not transient
    "ENV_FLAKINESS",         # transient runner/network failure — retry will fix it
    "CASCADE_FAILURE",       # job failed because a dependency job failed
    "TOOLING_ARTIFACT",      # dataset noise; not a real CI failure
]

N_CATEGORIES = len(CATEGORIES)

# ─── empirical Bayes prior (with online Dirichlet update) ────────────
#
# The prior is the MAP estimate of a Dirichlet whose pseudo-count vector
# has two parts:
#
#   α_base[c]      — fixed seed counts derived from Rausch et al. (MSR 2017),
#                    Table 2, mapped onto the 9-category taxonomy and combined
#                    with the category frequencies observed in the evaluation
#                    corpus. This part never changes at runtime.
#
#   n_obs[c]       — online observation counts. Every time a ground-truth
#                    label is confirmed for a case, that category's count is
#                    incremented by 1 and persisted to disk. This is the
#                    standard Dirichlet–multinomial conjugate update: each
#                    observed label adds one pseudo-count.
#
#   α[c]  = α_base[c] + n_obs[c]
#   P0(c) = α[c] / Σ_c' α[c']
#
# The "test failures" Rausch bucket (~41%) is split 75/25 into
# CODE_REGRESSION / TEST_FLAKINESS (regressions dominate; genuinely
# intermittent tests are a minority). "build config errors" (~24%) split
# into CONFIG_ERROR / INFRA_INCOMPATIBILITY; "dependency issues" (~16%)
# split into DEPENDENCY_CONFLICT / ENV_FLAKINESS. QUALITY_VIOLATION,
# CASCADE_FAILURE, TOOLING_ARTIFACT take conservative floor values.

_ALPHA_BASE: Dict[str, float] = {
    "CODE_REGRESSION":      189.0,
    "CONFIG_ERROR":          62.0,
    "DEPENDENCY_CONFLICT":   51.0,
    "TEST_FLAKINESS":        16.0,
    "ENV_FLAKINESS":         10.0,
    "INFRA_INCOMPATIBILITY":  8.0,
    "QUALITY_VIOLATION":      5.0,
    "CASCADE_FAILURE":        3.0,
    "TOOLING_ARTIFACT":       1.0,
}

# Where the running ground-truth observation counts are persisted.
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRIOR_COUNTS_PATH = Path(
    os.environ.get("PRIOR_COUNTS_PATH", str(_BASE_DIR / "data" / "prior_counts.json"))
)


def _load_observed_counts() -> Dict[str, float]:
    """Load persisted online observation counts; zeros if none exist yet."""
    counts = {cat: 0.0 for cat in CATEGORIES}
    try:
        with open(PRIOR_COUNTS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for cat in CATEGORIES:
            counts[cat] = float(saved.get(cat, 0.0))
    except (FileNotFoundError, ValueError, OSError):
        pass
    return counts


def _save_observed_counts(counts: Dict[str, float]) -> None:
    """Persist the online observation counts to disk."""
    try:
        PRIOR_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PRIOR_COUNTS_PATH, "w", encoding="utf-8") as f:
            json.dump({cat: counts.get(cat, 0.0) for cat in CATEGORIES}, f, indent=2)
    except OSError:
        pass


# Live observation-count vector, mutated by update_prior_with_label().
_OBSERVED_COUNTS: Dict[str, float] = _load_observed_counts()


def build_informed_prior() -> Dict[str, float]:
    """
    Dirichlet-smoothed empirical Bayes prior, including online observations.

    alpha[c] = alpha_base[c] + n_obs[c]
    prior[c] = alpha[c] / sum(alpha)
    """
    alpha = {
        cat: _ALPHA_BASE.get(cat, 1.0) + _OBSERVED_COUNTS.get(cat, 0.0)
        for cat in CATEGORIES
    }
    total = sum(alpha.values())
    return {cat: alpha[cat] / total for cat in CATEGORIES}


INFORMED_PRIOR: Dict[str, float] = build_informed_prior()


def get_informed_prior() -> Dict[str, float]:
    """Return the current informed prior (reflects all online updates so far)."""
    return INFORMED_PRIOR


def update_prior_with_label(category: str, weight: float = 1.0, persist: bool = True) -> Dict[str, float]:
    """Online Dirichlet update: fold one confirmed ground-truth label into the prior.

    Increments n_obs[category] by `weight`, persists the updated count vector,
    and recomputes the module-level INFORMED_PRIOR so subsequent BeliefState
    initializations start from the adjusted prior.

    Returns the new prior distribution. No-op for unknown categories.
    """
    global INFORMED_PRIOR
    if category not in _OBSERVED_COUNTS:
        return INFORMED_PRIOR
    _OBSERVED_COUNTS[category] += weight
    if persist:
        _save_observed_counts(_OBSERVED_COUNTS)
    INFORMED_PRIOR = build_informed_prior()
    return INFORMED_PRIOR

# ─── signal definitions ─────────────────────────────────────────────
# v2: rebalanced based on evaluation results.
# Key changes from v1:
#   - jobs_failed signal weakened (was too dominant)
#   - error_text signal expanded with many more patterns
#   - new signals: step_types, workflow_name
#   - all signals return closer-to-uniform when input is uninformative

def signal_many_jobs_failed(n_failed: int, n_total: int) -> Dict[str, float]:
    """Signal: how many jobs failed out of total?"""
    base = {cat: 0.10 for cat in CATEGORIES}
    ratio = n_failed / max(n_total, 1)

    if ratio > 0.8 and n_total >= 4:
        base["CASCADE_FAILURE"] += 0.05
        base["INFRA_INCOMPATIBILITY"] += 0.03
        base["ENV_FLAKINESS"] += 0.02
    elif ratio < 0.3 and n_total >= 3:
        base["CODE_REGRESSION"] += 0.04
        base["TEST_FLAKINESS"] += 0.03

    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def signal_error_text(error_keywords: List[str]) -> Dict[str, float]:
    """Signal: what patterns appear in the error text?"""
    if not error_keywords:
        return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

    base = {cat: 0.05 for cat in CATEGORIES}
    text = " ".join(error_keywords).lower()

    # Infrastructure / toolchain version mismatch (deterministic — retry won't fix)
    if any(w in text for w in ("glibc", "libc.so", "libstdc", "libm.so",
                                "unsupported class file major version",
                                "java.lang.unsupportedclassversionerror")):
        base["INFRA_INCOMPATIBILITY"] += 0.35
    if any(w in text for w in ("node20", "node18", "node16", "node12",
                                "deprecated action", "deprecated node")):
        base["INFRA_INCOMPATIBILITY"] += 0.20

    # Transient environment / network failures (retry will fix)
    if any(w in text for w in ("connection reset", "connection refused",
                                "timeout", "timed out", "etimedout",
                                "econnreset", "econnrefused",
                                "socket hang up", "network error",
                                "could not resolve host", "ssl_error",
                                "certificate", "no space left", "disk full",
                                "out of memory", "oom", "killed",
                                "resource exhausted", "quota exceeded")):
        base["ENV_FLAKINESS"] += 0.35

    # Dependency resolution failures
    if any(w in text for w in ("no module named", "modulenotfounderror",
                                "importerror", "cannot find module",
                                "could not resolve dependencies",
                                "peer dependency", "npm err",
                                "pip install failed", "cargo error",
                                "unresolved dependency", "version conflict",
                                "incompatible", "requires python",
                                "requires java", "lockfile out of date",
                                "no matching version")):
        base["DEPENDENCY_CONFLICT"] += 0.25

    # Quality / linter violations
    if any(w in text for w in ("checkstyle", "pylint", "flake8", "rubocop",
                                "eslint", "tslint", "shellcheck",
                                "findbugs", "spotbugs", "pmd",
                                "sonar", "code style", "lint error",
                                "quality gate", "coverage threshold")):
        base["QUALITY_VIOLATION"] += 0.40

    # Test failures
    if any(w in text for w in ("assert", "assertion", "expect",
                                "test failed", "tests failed",
                                "pytest", "junit", "rspec",
                                "mocha", "jest", "test result")):
        base["CODE_REGRESSION"] += 0.10
        base["TEST_FLAKINESS"] += 0.12
    elif any(w in text for w in ("fail ", "failed ", "failures")):
        base["CODE_REGRESSION"] += 0.03
        base["DEPENDENCY_CONFLICT"] += 0.03
        base["CONFIG_ERROR"] += 0.03

    # Compile / build errors
    if any(w in text for w in ("syntax error", "syntaxerror",
                                "compile error", "compilation failed",
                                "undefined reference", "linker error",
                                "unexpected token", "parse error")):
        base["CODE_REGRESSION"] += 0.25
    if any(w in text for w in ("build failed", "build failure")):
        base["CODE_REGRESSION"] += 0.08
        base["DEPENDENCY_CONFLICT"] += 0.06
        base["CONFIG_ERROR"] += 0.04

    # SDK/project-structure conflicts
    if any(w in text for w in ("netsdk", "found multiple publish output",
                                "duplicate class", "duplicate file",
                                "conflicting files", "multiple artifacts")):
        base["CODE_REGRESSION"] += 0.25

    # CI workflow structural errors
    if any(w in text for w in ("invalid workflow", "missing input",
                                "unexpected value", "workflow.*invalid")):
        base["CONFIG_ERROR"] += 0.20
    if any(w in text for w in ("permission denied", "not authorized",
                                "env variable", "environment variable")):
        base["CONFIG_ERROR"] += 0.08
        base["CODE_REGRESSION"] += 0.04

    # Dependabot read-only access → DEPENDENCY_CONFLICT not CONFIG_ERROR
    if any(w in text for w in ("dependabot", "read-only access",
                                "workflows triggered by dependabot")):
        base["DEPENDENCY_CONFLICT"] += 0.30
        base["CONFIG_ERROR"] -= 0.10

    # "not a git repository" with non-workflow changes → CODE_REGRESSION
    if any(w in text for w in ("not a git repository", "fatal: not a git")):
        base["CODE_REGRESSION"] += 0.15
        base["CONFIG_ERROR"] -= 0.10

    # Cancellation signals cascade
    if any(w in text for w in ("canceled", "cancelled", "operation was canceled")):
        base["CASCADE_FAILURE"] += 0.15

    # Dataset tooling noise
    if any(w in text for w in ("bashword", "circular structure",
                                "parser exception", "bash-command-extractor")):
        base["TOOLING_ARTIFACT"] += 0.40

    # Generic exit code — very weak
    if "exit code 1" in text and len(error_keywords) <= 2:
        base["CODE_REGRESSION"] += 0.05
        base["CONFIG_ERROR"] += 0.05

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


def signal_branch_type(is_protected: bool, branch: str) -> Dict[str, float]:
    """Signal: what kind of branch is this?"""
    base = {cat: 0.10 for cat in CATEGORIES}
    branch_lower = branch.lower()

    if any(bot in branch_lower for bot in ("dependabot", "renovate")):
        base["DEPENDENCY_CONFLICT"] += 0.15
        base["INFRA_INCOMPATIBILITY"] += 0.05
    elif not is_protected:
        base["CODE_REGRESSION"] += 0.03

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


def signal_commit_message(message: str) -> Dict[str, float]:
    """Signal: what does the commit message suggest?"""
    base = {cat: 0.09 for cat in CATEGORIES}
    msg = message.lower()

    if any(w in msg for w in ("upgrade", "bump", "update dep",
                               "update version", "update dependency",
                               "chore(deps", "deps):", "deps:", "dependabot",
                               "renovate", "pin version", "pin dep")):
        base["DEPENDENCY_CONFLICT"] += 0.15
        base["INFRA_INCOMPATIBILITY"] += 0.05

    if any(w in msg for w in ("fix ", "fix:", "fixed ", "hotfix",
                               "patch ", "repair ", "resolve ",
                               "bug ", "bugfix")):
        base["CODE_REGRESSION"] += 0.05
        base["CONFIG_ERROR"] += 0.03
        base["DEPENDENCY_CONFLICT"] += 0.03

    if any(w in msg for w in ("refactor", "rename", "move ",
                               "reorganize", "cleanup", "clean up")):
        base["CODE_REGRESSION"] += 0.04
        base["CONFIG_ERROR"] += 0.04

    if any(w in msg for w in ("ci ", "ci:", "workflow", "action",
                               "yaml", "yml", "pipeline", "github action")):
        base["CONFIG_ERROR"] += 0.15

    if any(w in msg for w in ("lint", "checkstyle", "pylint", "flake8",
                               "rubocop", "eslint", "shellcheck", "quality")):
        base["QUALITY_VIOLATION"] += 0.20

    if any(w in msg for w in ("test ", "test:", "spec ", "coverage",
                               "add test", "fix test")):
        base["TEST_FLAKINESS"] += 0.08
        base["CODE_REGRESSION"] += 0.05

    if any(w in msg for w in ("docker", "container", "image ", "dockerfile")):
        base["INFRA_INCOMPATIBILITY"] += 0.10
        base["CONFIG_ERROR"] += 0.05

    if any(w in msg for w in ("merge ", "merge:")):
        base["CODE_REGRESSION"] += 0.05
    # A commit titled "Revert ..." is definitive: it was pushed to undo a prior
    # code change. The developer's fix IS a revert, which maps to CODE_REGRESSION.
    if msg.startswith("revert ") or "revert:" in msg:
        base["CODE_REGRESSION"] += 0.30

    if any(w in msg for w in ("wip", "tmp", "temp", "draft")):
        base["CODE_REGRESSION"] += 0.08

    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def signal_previous_runs(n_recent_failures: int, n_recent_total: int) -> Dict[str, float]:
    """Signal: how often has this workflow been failing recently?

    High failure rate → pre-existing flaky/infra condition.
    Low failure rate → isolated regression introduced by this commit.
    Validated by Rausch et al. (2017): build climate is a top predictor.
    """
    base = {cat: 0.10 for cat in CATEGORIES}

    if n_recent_total == 0:
        return base

    failure_rate = n_recent_failures / n_recent_total
    if failure_rate > 0.5:
        base["ENV_FLAKINESS"] += 0.12
        base["TEST_FLAKINESS"] += 0.08
        base["INFRA_INCOMPATIBILITY"] += 0.06
        base["CONFIG_ERROR"] += 0.05
    elif failure_rate > 0.2:
        base["TEST_FLAKINESS"] += 0.05
        base["ENV_FLAKINESS"] += 0.03
    else:
        base["CODE_REGRESSION"] += 0.08
        base["DEPENDENCY_CONFLICT"] += 0.05

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


def signal_parent_commit_run(parent_conclusion: Optional[str]) -> Dict[str, float]:
    """Signal: what was the conclusion of the run on the immediately preceding commit?

    The strongest single predictor of build outcome (Rausch et al. 2017:
    50-80% of failed builds follow a previous failure).

      parent passed → THIS commit introduced the failure.
        Strong evidence for: CODE_REGRESSION, DEPENDENCY_CONFLICT, CONFIG_ERROR,
        QUALITY_VIOLATION (all require a file change to manifest).
        Weak evidence for: TEST_FLAKINESS, ENV_FLAKINESS, INFRA_INCOMPATIBILITY
        (those are commit-independent).

      parent also failed → pre-existing condition, not a regression.
        Strong evidence for: TEST_FLAKINESS, ENV_FLAKINESS, INFRA_INCOMPATIBILITY.
        Weak evidence for: CODE_REGRESSION, DEPENDENCY_CONFLICT.

      parent unknown → uniform; no information.
    """
    def _norm(raw: Dict[str, float]) -> Dict[str, float]:
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    if parent_conclusion == "success":
        return _norm({
            "CODE_REGRESSION":       0.50,
            "DEPENDENCY_CONFLICT":   0.28,
            "CONFIG_ERROR":          0.18,
            "QUALITY_VIOLATION":     0.12,
            "TEST_FLAKINESS":        0.03,
            "INFRA_INCOMPATIBILITY": 0.03,
            "ENV_FLAKINESS":         0.03,
            "CASCADE_FAILURE":       0.06,
            "TOOLING_ARTIFACT":      0.06,
        })

    elif parent_conclusion == "failure":
        return _norm({
            "CODE_REGRESSION":       0.03,
            "DEPENDENCY_CONFLICT":   0.05,
            "CONFIG_ERROR":          0.06,
            "QUALITY_VIOLATION":     0.04,
            "TEST_FLAKINESS":        0.38,
            "INFRA_INCOMPATIBILITY": 0.25,
            "ENV_FLAKINESS":         0.28,
            "CASCADE_FAILURE":       0.15,
            "TOOLING_ARTIFACT":      0.15,
        })

    return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}


def signal_step_duration(
    step_duration_sec: Optional[float],
    n_failed: int,
    n_total: int,
) -> Dict[str, float]:
    """Signal: when in the build did the failure occur?

    Rausch et al. (2017) Fig. 2: git/dependency/buildconfig errors occur in
    the first half of the build; test failures dominate the second half.
    Mapping: early + short → CONFIG/DEPENDENCY/INFRA; late + long → TEST/CODE.
    """
    base = {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}
    if step_duration_sec is None:
        return base

    ratio = n_failed / max(n_total, 1)

    if step_duration_sec < 30:
        # Very fast failure — almost certainly config, dependency, or infra
        base["CONFIG_ERROR"] += 0.18
        base["DEPENDENCY_CONFLICT"] += 0.15
        base["INFRA_INCOMPATIBILITY"] += 0.12
        base["CODE_REGRESSION"] -= 0.05
        base["TEST_FLAKINESS"] -= 0.05
    elif step_duration_sec < 120:
        # Early failure — likely compile, dependency, or infra
        base["CODE_REGRESSION"] += 0.08
        base["DEPENDENCY_CONFLICT"] += 0.08
        base["CONFIG_ERROR"] += 0.06
    else:
        # Late failure — test suite ran, so build compiled successfully
        base["TEST_FLAKINESS"] += 0.12
        base["CODE_REGRESSION"] += 0.10
        base["CONFIG_ERROR"] -= 0.05
        base["DEPENDENCY_CONFLICT"] -= 0.03

    # Clamp negatives before normalising
    base = {k: max(v, 0.001) for k, v in base.items()}
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def signal_detection_mode(mode: str) -> Dict[str, float]:
    """Signal: how was the failure detected by intake?"""
    base = {cat: 0.10 for cat in CATEGORIES}

    if mode == "per_step_error":
        base["CODE_REGRESSION"] += 0.05
        base["DEPENDENCY_CONFLICT"] += 0.03
    elif mode == "single_step_inferred":
        base["CONFIG_ERROR"] += 0.05
    elif mode == "job_level_fallback":
        base["CASCADE_FAILURE"] += 0.03
        base["ENV_FLAKINESS"] += 0.02

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}
# ─── the tracker ─────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """Current probability distribution over failure categories."""
    probabilities: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.probabilities:
            self.probabilities = dict(INFORMED_PRIOR)

    def entropy(self) -> float:
        """Shannon entropy of current beliefs. Lower = more certain."""
        return -sum(
            p * math.log2(p) for p in self.probabilities.values()
            if p > 0
        )

    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        return math.log2(N_CATEGORIES)

    def confidence(self) -> float:
        """Confidence = 1 - (entropy / max_entropy). Range 0-1."""
        return 1.0 - (self.entropy() / self.max_entropy())

    def top_category(self) -> Tuple[str, float]:
        """Most likely category and its probability."""
        best = max(self.probabilities, key=self.probabilities.get)
        return best, self.probabilities[best]

    def top_n(self, n: int = 3) -> List[Tuple[str, float]]:
        """Top N categories by probability."""
        sorted_cats = sorted(
            self.probabilities.items(), key=lambda x: -x[1]
        )
        return sorted_cats[:n]

    def update(self, likelihood: Dict[str, float], signal_name: str = "") -> None:
        """
        Bayesian update: multiply current probabilities by likelihood
        vector and renormalize.
        """
        # Multiply
        new_probs = {}
        for cat in CATEGORIES:
            new_probs[cat] = self.probabilities[cat] * likelihood.get(cat, 0.1)

        # Normalize
        total = sum(new_probs.values())
        if total > 0:
            new_probs = {k: v / total for k, v in new_probs.items()}
        else:
            new_probs = {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

        # Record history
        old_entropy = self.entropy()
        self.probabilities = new_probs
        new_entropy = self.entropy()

        self.history.append({
            "signal": signal_name,
            "top_3": self.top_n(3),
            "entropy": new_entropy,
            "information_gain": old_entropy - new_entropy,
            "confidence": self.confidence(),
        })

    def expected_information_gain(self, possible_likelihoods: List[Dict[str, float]]) -> float:
        """
        Expected information gain if we were to observe one of the
        given possible likelihood vectors (with equal probability).
        Used for tool selection — pick the tool with highest EIG.
        """
        current_entropy = self.entropy()
        expected_posterior_entropy = 0.0

        for likelihood in possible_likelihoods:
            # Simulate the update
            new_probs = {}
            for cat in CATEGORIES:
                new_probs[cat] = self.probabilities[cat] * likelihood.get(cat, 0.1)
            total = sum(new_probs.values())
            if total > 0:
                new_probs = {k: v / total for k, v in new_probs.items()}
                entropy = -sum(p * math.log2(p) for p in new_probs.values() if p > 0)
            else:
                entropy = current_entropy
            expected_posterior_entropy += entropy

        expected_posterior_entropy /= len(possible_likelihoods)
        return current_entropy - expected_posterior_entropy


# ─── pretty printer ──────────────────────────────────────────────────

def print_beliefs(state: BeliefState) -> None:
    print(f"\n  BELIEFS  (entropy={state.entropy():.3f}, confidence={state.confidence():.1%})")
    for cat, prob in sorted(state.probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"    {cat:<25} {prob:.3f}  {bar}")

    if state.history:
        print(f"\n  INVESTIGATION TRACE:")
        for i, step in enumerate(state.history):
            top = step["top_3"][0]
            print(f"    step {i+1}: {step['signal']:<25} "
                  f"→ {top[0]} ({top[1]:.2f})  "
                  f"IG={step['information_gain']:.3f}  "
                  f"conf={step['confidence']:.1%}")


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Bayesian Belief Tracker — self-test ===\n")
    print("Simulating the bcrypt case step by step:\n")

    state = BeliefState()
    print("Initial beliefs (informed prior):")
    print_beliefs(state)

    # Step 1: branch is dependabot
    print("\n--- Signal: branch type (dependabot) ---")
    state.update(
        signal_branch_type(False, "dependabot/github_actions/actions/checkout-4.1.0"),
        "branch_type",
    )
    print_beliefs(state)

    # Step 2: 8 out of 8 jobs failed
    print("\n--- Signal: 8/8 jobs failed ---")
    state.update(
        signal_many_jobs_failed(8, 8),
        "jobs_failed",
    )
    print_beliefs(state)

    # Step 3: error text contains GLIBC
    print("\n--- Signal: error text has 'GLIBC_2.27 not found' ---")
    state.update(
        signal_error_text(["GLIBC_2.27 not found", "operation was canceled"]),
        "error_text",
    )
    print_beliefs(state)

    # Step 4: commit message is a version bump
    print("\n--- Signal: commit message 'Bump actions/checkout' ---")
    state.update(
        signal_commit_message("Bump actions/checkout from 3.6.0 to 4.1.0"),
        "commit_message",
    )
    print_beliefs(state)

    # Step 5: parent commit passed
    print("\n--- Signal: parent commit passed ---")
    state.update(
        signal_parent_commit_run("success"),
        "parent_commit_run",
    )
    print_beliefs(state)

    # Step 6: previous runs all passed
    print("\n--- Signal: last 5 runs all passed ---")
    state.update(
        signal_previous_runs(0, 5),
        "previous_runs",
    )
    print_beliefs(state)

    print("\n=== FINAL ===")
    top_cat, top_prob = state.top_category()
    print(f"  Category:   {top_cat}")
    print(f"  Probability: {top_prob:.3f}")
    print(f"  Confidence:  {state.confidence():.1%}")
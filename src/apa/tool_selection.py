# tool_selection.py
# ─────────────────────────────────────────────────────────────────────
# Uncertainty-Directed Tool Selection
#
# Replaces the LLM planner's implicit "which tool looks useful?" with an
# explicit expected information gain (EIG) computation backed by calibrated
# per-tool outcome scenarios.
#
# Theory
# ──────
# For each available tool T, EIG(T) estimates how many bits of belief
# entropy the tool call is expected to remove:
#
#   EIG(T) = H(beliefs_now) − E[H(beliefs_after_T)]
#
# The expectation is over OUTCOME_SCENARIOS[T] — a small set of likelihood
# vectors representing the distinct kinds of evidence the tool could discover.
# The tool with the highest EIG is the one that, on average, resolves the most
# uncertainty for runs with the current belief profile.
#
# This computation is fully deterministic (no LLM call). It personalises
# the tool ranking to every run's unique uncertainty profile: the same tool
# may rank first for a run dominated by CODE_REGRESSION uncertainty but third
# for a run split between TEST_FLAKINESS and ENV_FLAKINESS.
#
# Calibration
# ───────────
# Scenarios are derived from domain knowledge about what each tool discovers.
# They can be validated post-hoc by comparing `predicted_eig` (from the
# ranking) against `actual_ig` (entropy_before − entropy_after from
# belief_history) for each tool call in a batch eval run.
#
# Usage
# ─────
#   from tool_selection import rank_tools_by_eig, format_eig_for_prompt
#
#   rankings = rank_tools_by_eig(bs, available_tools)
#   prompt_section = format_eig_for_prompt(rankings)   # inject into planner
#
# Public API
# ──────────
#   OUTCOME_SCENARIOS          dict[str, list[dict[str,float]]]
#   compute_tool_eig(bs, tool) -> float
#   rank_tools_by_eig(bs, tools) -> list[tuple[str, float]]
#   format_eig_for_prompt(rankings) -> str
# ─────────────────────────────────────────────────────────────────────

from src.apa.bayesian_tracker import CATEGORIES, BeliefState

# ─── outcome scenario table ──────────────────────────────────────────
#
# Each entry is a list of possible likelihood vectors for one tool.
# Each likelihood vector is a dict {category: weight} that will be
# normalised before being used in BeliefState.expected_information_gain().
#
# Design principle: prefer fewer, meaningfully distinct scenarios over
# many fine-grained ones.  4–5 per tool is enough for calibrated EIG.
#
# Weights are additive bumps on top of an unnormalised base of 0.05 per
# category (10 categories × 0.05 = 0.50 before bumps).  The dict is always
# normalised to a proper probability distribution before use, so the absolute
# scale of the base does not matter — only the relative weights do.

def _scenario(*bumps: tuple[str, float]) -> dict[str, float]:
    """Build a normalised likelihood vector from category bumps."""
    base = {cat: 0.05 for cat in CATEGORIES}
    for cat, weight in bumps:
        if cat in base:
            base[cat] += weight
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


OUTCOME_SCENARIOS: dict[str, list[dict[str, float]]] = {

    # ── deep_log_analysis ───────────────────────────────────────────
    # Reads the full failure excerpt and lets the LLM reason over raw error
    # text.  Can surface nearly any category, but most likely to produce a
    # strong signal when error text contains actionable patterns.
    "deep_log_analysis": [
        _scenario(("CODE_REGRESSION",       0.55), ("TEST_FLAKINESS",      0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.55), ("CONFIG_ERROR",        0.10)),
        _scenario(("INFRA_INCOMPATIBILITY", 0.50), ("ENV_FLAKINESS",       0.10)),
        _scenario(("TEST_FLAKINESS",        0.45), ("ENV_FLAKINESS",       0.15)),
        _scenario(("QUALITY_VIOLATION",     0.55), ("CODE_REGRESSION",     0.15)),
        _scenario(("ENV_FLAKINESS",         0.50), ("INFRA_INCOMPATIBILITY", 0.15)),
        # inconclusive — log too sparse or generic exit-code only
        _scenario(),  # near-uniform
    ],

    # ── inspect_commit_diff ─────────────────────────────────────────
    # Fetches files changed in the triggering commit.  Tells us *what* was
    # touched, enabling the semantic diff linker.  Most informative when
    # the category depends on what was changed.
    "inspect_commit_diff": [
        _scenario(("DEPENDENCY_CONFLICT",   0.45), ("CONFIG_ERROR",        0.10)),
        _scenario(("CONFIG_ERROR",          0.50), ("INFRA_INCOMPATIBILITY", 0.15)),
        _scenario(("CODE_REGRESSION",       0.40), ("TEST_FLAKINESS",      0.10)),
        _scenario(("INFRA_INCOMPATIBILITY", 0.40), ("CONFIG_ERROR",        0.10)),
        # nothing notable changed — low signal
        _scenario(),  # near-uniform
    ],

    # ── check_run_history ───────────────────────────────────────────
    # Fetches the parent commit's run conclusion — the sharpest regression
    # indicator.  Always produces a clear signal in either direction;
    # the main uncertainty is which direction.
    "check_run_history": [
        # parent passed  → this commit introduced it
        _scenario(("CODE_REGRESSION",       0.45), ("DEPENDENCY_CONFLICT", 0.20),
                  ("CONFIG_ERROR",          0.15)),
        # parent also failed → pre-existing condition
        _scenario(("TEST_FLAKINESS",        0.40), ("ENV_FLAKINESS",       0.25),
                  ("INFRA_INCOMPATIBILITY", 0.15)),
        # parent cancelled / skipped — ambiguous
        _scenario(("CASCADE_FAILURE",       0.25), ("ENV_FLAKINESS",       0.15)),
        # no run history available
        _scenario(),
    ],

    # ── inspect_workflow_file ───────────────────────────────────────
    # Parses changed workflow YAML.  Highly specific — only useful when
    # a workflow file actually changed; noisy otherwise.
    "inspect_workflow_file": [
        # deprecated action runtime (node12/node16)
        _scenario(("INFRA_INCOMPATIBILITY", 0.55), ("CONFIG_ERROR",        0.10)),
        # version pin or runner change
        _scenario(("CONFIG_ERROR",          0.45), ("INFRA_INCOMPATIBILITY", 0.15)),
        # new/changed runner
        _scenario(("ENV_FLAKINESS",         0.35), ("INFRA_INCOMPATIBILITY", 0.20)),
        # workflow file unchanged / not parsed
        _scenario(),  # near-uniform
    ],

    # ── inspect_dependency_changes ───────────────────────────────────
    # Focuses on dependency manifests / lockfiles and version bump evidence.
    # Most informative when failures are caused by missing/incompatible deps.
    "inspect_dependency_changes": [
        _scenario(("DEPENDENCY_CONFLICT",   0.65), ("CONFIG_ERROR",        0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.55), ("CODE_REGRESSION",      0.10)),
        # tool/runtime version bumps can look like infra incompatibility
        _scenario(("INFRA_INCOMPATIBILITY", 0.45), ("DEPENDENCY_CONFLICT",  0.25)),
        _scenario(),
    ],

    # ── inspect_runner_environment ───────────────────────────────────
    # Inspect runner images, pinned runtimes, and action/runtime deprecations.
    "inspect_runner_environment": [
        _scenario(("INFRA_INCOMPATIBILITY", 0.65), ("CONFIG_ERROR",         0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.45), ("INFRA_INCOMPATIBILITY", 0.25)),
        _scenario(("ENV_FLAKINESS",         0.30), ("INFRA_INCOMPATIBILITY", 0.20)),
        _scenario(),
    ],

    # ── inspect_k8s_events ───────────────────────────────────────────
    # Fetches recent Kubernetes warning events for the pod/deployment.
    # K8s-specific symptoms map into the shared taxonomy:
    #   OOM kill / pod crash    → INFRA_INCOMPATIBILITY (resource limit mismatch)
    #   image pull failure      → DEPENDENCY_CONFLICT or CONFIG_ERROR
    #   scheduling failure      → ENV_FLAKINESS
    "inspect_k8s_events": [
        _scenario(("INFRA_INCOMPATIBILITY", 0.60), ("CONFIG_ERROR",       0.15)),
        _scenario(("DEPENDENCY_CONFLICT",   0.55), ("CONFIG_ERROR",       0.20)),
        _scenario(("ENV_FLAKINESS",         0.50), ("INFRA_INCOMPATIBILITY", 0.20)),
        _scenario(("CODE_REGRESSION",       0.40), ("INFRA_INCOMPATIBILITY", 0.25)),
        _scenario(),  # near-uniform
    ],

}


# ─── public functions ────────────────────────────────────────────────

def compute_tool_eig(bs: BeliefState, tool: str) -> float:
    """
    Expected information gain for calling `tool` given current belief state.

    Uses BeliefState.expected_information_gain() with the pre-defined outcome
    scenarios for the tool.  Returns 0.0 for unknown tools or when the tool
    would increase entropy (clamped — a tool can't have negative value).
    """
    scenarios = OUTCOME_SCENARIOS.get(tool)
    if not scenarios:
        return 0.0
    return max(0.0, bs.expected_information_gain(scenarios))


def rank_tools_by_eig(
    bs: BeliefState,
    available_tools: list[str],
) -> list[tuple[str, float]]:
    """
    Rank available tools by expected information gain (highest first).

    Returns a list of (tool_name, eig_bits) tuples sorted descending by EIG.
    EIG is computed from the current belief state, so the ranking is
    personalised to this run's uncertainty profile.
    """
    ranked = [(tool, compute_tool_eig(bs, tool)) for tool in available_tools]
    ranked.sort(key=lambda x: -x[1])

    # Context-sensitive boost: if the top belief is ENV_FLAKINESS or TEST_FLAKINESS
    # and check_run_history hasn't been called yet, boost its EIG significantly.
    # Rationale: flakiness by definition requires temporal evidence (run history).
    # Without it, any flakiness diagnosis is essentially a guess. This is not a
    # rule — it ensures the EIG math reflects the epistemic dependency.
    top_cat = bs.top_category()[0]
    if top_cat in ("ENV_FLAKINESS", "TEST_FLAKINESS"):
        ranked = [
            (tool, eig * 2.5 if tool == "check_run_history" else eig)
            for tool, eig in ranked
        ]
        ranked.sort(key=lambda x: -x[1])

    return ranked


def format_eig_for_prompt(rankings: list[tuple[str, float]]) -> str:
    """
    Format EIG rankings as a concise table for injection into the planner prompt.

    Example output:
      deep_log_analysis      0.421 bits  ← highest expected gain
      inspect_commit_diff    0.283 bits
      check_run_history      0.194 bits
      inspect_workflow_file  0.038 bits  ← lowest expected gain
    """
    if not rankings:
        return "(no tools available)"
    max_eig = rankings[0][1] if rankings else 1.0
    lines = []
    for i, (tool, eig) in enumerate(rankings):
        annotation = "  <- highest expected gain" if i == 0 else ""
        if i == len(rankings) - 1 and len(rankings) > 1:
            annotation = "  <- lowest expected gain"
        lines.append(f"  {tool:<26} {eig:.3f} bits{annotation}")
    return "\n".join(lines)


def pick_eig_tool(
    bs: BeliefState,
    available_tools: list[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Deterministically select the tool with highest EIG (pure-EIG mode).

    Returns (selected_tool, eig_of_selected, full_ranking).
    Falls back to 'classify' if no tools are available.
    """
    if not available_tools:
        return "classify", 0.0, []
    rankings = rank_tools_by_eig(bs, available_tools)
    best_tool, best_eig = rankings[0]
    return best_tool, best_eig, rankings


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from bayesian_tracker import signal_branch_type, signal_error_text

    print("=== Tool Selection EIG Self-Test ===\n")

    # Scenario A: Dependabot branch, version conflict error
    # Expect: inspect_commit_diff or check_run_history ranked highest
    bs_a = BeliefState()
    bs_a.update(signal_branch_type(False, "dependabot/pip/requests-3.0.0"), "branch")
    bs_a.update(signal_error_text(["no module named requests", "version conflict"]), "error")
    print("Scenario A: dependabot branch + version conflict error")
    print(f"  Belief top-3: {[(c, f'{p:.0%}') for c,p in bs_a.top_n(3)]}")
    rankings_a = rank_tools_by_eig(bs_a, list(OUTCOME_SCENARIOS.keys()))
    print(f"  Tool EIG ranking:\n{format_eig_for_prompt(rankings_a)}\n")

    # Scenario B: Protected branch, GLIBC not found
    # Expect: deep_log_analysis or check_run_history ranked highest
    bs_b = BeliefState()
    bs_b.update(signal_branch_type(True, "main"), "branch")
    bs_b.update(signal_error_text(["GLIBC_2.27 not found", "operation was canceled"]), "error")
    print("Scenario B: protected branch + GLIBC error")
    print(f"  Belief top-3: {[(c, f'{p:.0%}') for c,p in bs_b.top_n(3)]}")
    rankings_b = rank_tools_by_eig(bs_b, list(OUTCOME_SCENARIOS.keys()))
    print(f"  Tool EIG ranking:\n{format_eig_for_prompt(rankings_b)}\n")

    # Scenario C: Highly uniform beliefs (maximally uncertain)
    # Expect: deep_log_analysis ranked highest (most scenarios, broadest coverage)
    bs_c = BeliefState()  # uniform prior
    print("Scenario C: uniform prior (no signals yet)")
    print(f"  Belief top-3: {[(c, f'{p:.0%}') for c,p in bs_c.top_n(3)]}")
    rankings_c = rank_tools_by_eig(bs_c, list(OUTCOME_SCENARIOS.keys()))
    print(f"  Tool EIG ranking:\n{format_eig_for_prompt(rankings_c)}\n")

    print("=== Verify: EIG changes as beliefs concentrate ===")
    print("After check_run_history says parent PASSED (CODE_REGRESSION family):")
    from bayesian_tracker import signal_parent_commit_run
    bs_d = BeliefState()
    bs_d.update(signal_parent_commit_run("success"), "parent_run")
    rankings_d = rank_tools_by_eig(bs_d, list(OUTCOME_SCENARIOS.keys()))
    print(f"  Tool EIG ranking:\n{format_eig_for_prompt(rankings_d)}")

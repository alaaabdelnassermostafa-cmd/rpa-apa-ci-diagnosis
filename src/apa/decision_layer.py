# decision_layer.py
# ─────────────────────────────────────────────────────────────────────
# Decision layer: transforms agent classification into actionable
# recommendations with trust tiers, policy guardrails, and audit logs.
#
# This is a thin post-processing layer on top of run_agent() output.
# No changes to agent.py are needed.
#
# Design: all outputs use a stable schema so that downstream consumers
# (GitHub Actions webhook, Kubernetes operator, dashboard) can parse
# the decision without knowing about the agent internals.
#
# Future integration points:
#   - GitHub: dispatch() → POST to Checks API / PR comment
#   - Kubernetes: dispatch() → patch rollout / scale canary
#   - Dashboard: audit record → event stream
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─── action vocabulary ───────────────────────────────────────────────
# Covers CI triage, canary, and feature-flag decision points.

class Action(str, Enum):
    # CI triage actions
    BLOCK_MERGE   = "BLOCK_MERGE"     # Fail the PR check, block the merge
    RETRY         = "RETRY"           # Re-run the job (up to N times)
    QUARANTINE    = "QUARANTINE"      # Mark as flaky, don't block CI
    INVESTIGATE   = "INVESTIGATE"     # Surface to human for review
    IGNORE        = "IGNORE"          # Tooling noise, suppress

    # Canary / deployment actions (for future K8s integration)
    PROMOTE       = "PROMOTE"         # Promote canary to full rollout
    PAUSE         = "PAUSE"           # Pause rollout, wait for signal
    ROLLBACK      = "ROLLBACK"        # Roll back to previous version

    # Feature flag actions
    KEEP_ENABLED  = "KEEP_ENABLED"    # Feature flag stays on
    DISABLE_FLAG  = "DISABLE_FLAG"    # Kill switch: disable risky feature


class TrustTier(str, Enum):
    T0 = "T0"  # observe only — surface to human, no autonomous action
    T1 = "T1"  # recommend — suggest action, require human approval
    T2 = "T2"  # act — agent executes autonomously


class PolicyOutcome(str, Enum):
    ALLOW    = "ALLOW"     # original action stands
    OVERRIDE = "OVERRIDE"  # policy changed the action
    BLOCK    = "BLOCK"     # policy blocked autonomous execution


# ─── decision record ─────────────────────────────────────────────────

@dataclass
class Decision:
    """Stable output schema for downstream consumers (GitHub, K8s, dashboard)."""

    # Classification (from agent)
    category: str = ""
    severity: str = "MODERATE"
    confidence: float = 0.0
    reasoning: str = ""

    # Decision layer
    action: str = "INVESTIGATE"
    trust_tier: str = "T0"
    policy_outcome: str = "ALLOW"
    policy_rules_triggered: List[str] = field(default_factory=list)

    # Deployment context (for canary/flag decisions)
    deployment_action: Optional[str] = None
    flag_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AuditRecord:
    """Full audit trail for a single decision. Immutable after creation."""

    id: str = ""
    timestamp: str = ""
    run_id: str = ""
    repo: str = ""

    # Classification
    category: str = ""
    severity: str = "MODERATE"
    confidence: float = 0.0
    reasoning: str = ""

    # Decision
    action: str = "INVESTIGATE"
    trust_tier: str = "T0"
    policy_outcome: str = "ALLOW"
    policy_rules_triggered: List[str] = field(default_factory=list)
    human_override: bool = False

    # Deployment
    deployment_action: Optional[str] = None
    flag_action: Optional[str] = None

    # Trace
    tools_used: List[str] = field(default_factory=list)
    steps_taken: int = 0
    fast_path: bool = False
    bayesian_top: str = ""
    bayesian_confidence: float = 0.0

    # System
    agent_mode: str = "APA"       # "APA" or "RPA"
    planner_mode: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ─── test triage ─────────────────────────────────────────────────────

# Mapping: (category, min_confidence, max_confidence) → action
# Evaluated top-to-bottom; first match wins.
TRIAGE_RULES: List[Tuple[str, float, float, str]] = [
    # Flaky tests: quarantine if confident, retry if unsure
    ("TEST_FLAKINESS",       0.70, 1.01, "QUARANTINE"),
    ("TEST_FLAKINESS",       0.50, 0.70, "RETRY"),
    ("TEST_FLAKINESS",       0.00, 0.50, "INVESTIGATE"),

    # Code regression: always block (real bug)
    ("CODE_REGRESSION",      0.50, 1.01, "BLOCK_MERGE"),
    ("CODE_REGRESSION",      0.00, 0.50, "INVESTIGATE"),

    # Dependency conflict: block if clear, investigate if ambiguous
    ("DEPENDENCY_CONFLICT",  0.70, 1.01, "BLOCK_MERGE"),
    ("DEPENDENCY_CONFLICT",  0.50, 0.70, "INVESTIGATE"),
    ("DEPENDENCY_CONFLICT",  0.00, 0.50, "RETRY"),

    # Config error: block if clear
    ("CONFIG_ERROR",         0.60, 1.01, "BLOCK_MERGE"),
    ("CONFIG_ERROR",         0.00, 0.60, "INVESTIGATE"),

    # Quality violation (lint/static-analysis): block — retry won't help
    ("QUALITY_VIOLATION",    0.60, 1.01, "BLOCK_MERGE"),
    ("QUALITY_VIOLATION",    0.00, 0.60, "INVESTIGATE"),

    # Env flakiness (transient): retry
    ("ENV_FLAKINESS",        0.50, 1.01, "RETRY"),
    ("ENV_FLAKINESS",        0.00, 0.50, "INVESTIGATE"),

    # Cascade failure: investigate (multi-job, needs human)
    ("CASCADE_FAILURE",      0.00, 1.01, "INVESTIGATE"),

    # Infra incompatibility (deterministic): block — needs tool/runner change
    ("INFRA_INCOMPATIBILITY", 0.60, 1.01, "BLOCK_MERGE"),
    ("INFRA_INCOMPATIBILITY", 0.00, 0.60, "INVESTIGATE"),

    # Tooling artifact: ignore if confident
    ("TOOLING_ARTIFACT",     0.60, 1.01, "IGNORE"),
    ("TOOLING_ARTIFACT",     0.00, 0.60, "INVESTIGATE"),
]


def recommend_action(category: str, confidence: float) -> str:
    """Map (category, confidence) → recommended CI triage action."""
    for rule_cat, lo, hi, action in TRIAGE_RULES:
        if rule_cat == category and lo <= confidence < hi:
            return action
    return Action.INVESTIGATE.value


# ─── canary decision (for future K8s integration) ────────────────────

def recommend_canary_action(
    category: str,
    confidence: float,
    error_rate_delta: float = 0.0,
) -> str:
    """
    Canary deployment decision based on CI triage + live error rate.

    For now this uses static rules. When connected to a K8s operator,
    error_rate_delta would come from Prometheus/Datadog metrics.
    """
    # Critical regression with high confidence → rollback
    if category == "CODE_REGRESSION" and confidence >= 0.70:
        return Action.ROLLBACK.value

    # Dependency/config issue → pause and investigate
    if category in ("DEPENDENCY_CONFLICT", "CONFIG_ERROR") and confidence >= 0.60:
        return Action.PAUSE.value

    # Error rate spike (future: from live metrics)
    if error_rate_delta > 0.05:
        return Action.PAUSE.value

    # Flaky/transient → promote (canary is fine, issue is environmental)
    if category in ("TEST_FLAKINESS", "ENV_FLAKINESS"):
        return Action.PROMOTE.value

    # Default: pause and let human decide
    return Action.PAUSE.value


# ─── feature flag decision ───────────────────────────────────────────

def recommend_flag_action(category: str, confidence: float) -> str:
    """Feature flag kill-switch decision."""
    if category == "CODE_REGRESSION" and confidence >= 0.70:
        return Action.DISABLE_FLAG.value
    if category in ("DEPENDENCY_CONFLICT", "CONFIG_ERROR") and confidence >= 0.80:
        return Action.DISABLE_FLAG.value
    return Action.KEEP_ENABLED.value


# ─── trust tiers ─────────────────────────────────────────────────────

def assign_trust_tier(confidence: float, category: str = "") -> str:
    """
    Map confidence to a trust tier.

    T2 (>=0.80): agent acts autonomously
    T1 (>=0.60): agent recommends, human approves
    T0 (<0.60):  agent observes, human decides

    The tier directly maps to the RPA→APA autonomy spectrum:
      T0 = RPA-level (deterministic signals only, no autonomous action)
      T1 = APA with guardrails (LLM reasoning, human approval gate)
      T2 = full APA (autonomous execution)
    """
    if confidence >= 0.80:
        return TrustTier.T2.value
    elif confidence >= 0.60:
        return TrustTier.T1.value
    return TrustTier.T0.value


# ─── policy guardrails ───────────────────────────────────────────────
# Formalized version of the hardcoded overrides in
# agent.py:_apply_final_category_overrides.
#
# Each rule is a dict with:
#   id        — unique identifier for audit trail
#   condition — predicate on the decision dict
#   override  — fields to change if condition matches

POLICY_RULES = [
    {
        "id": "low_confidence_block_requires_human",
        "description": "Low-confidence blocks require human approval",
        "condition": lambda d: d.get("confidence", 0) < 0.60
                               and d.get("action") == Action.BLOCK_MERGE.value,
        "override": {
            "action": Action.INVESTIGATE.value,
            "trust_tier": TrustTier.T0.value,
        },
    },
    {
        "id": "critical_regression_must_block",
        "description": "Critical regressions always block regardless of confidence",
        "condition": lambda d: d.get("category") == "CODE_REGRESSION"
                               and d.get("severity") == "CRITICAL"
                               and d.get("confidence", 0) >= 0.70,
        "override": {
            "action": Action.BLOCK_MERGE.value,
            "trust_tier": TrustTier.T2.value,
        },
    },
    {
        "id": "protected_branch_low_conf_no_auto",
        "description": "Protected branch decisions require high confidence",
        "condition": lambda d: d.get("protected_branch", False)
                               and d.get("confidence", 0) < 0.50,
        "override": {
            "action": Action.INVESTIGATE.value,
            "trust_tier": TrustTier.T0.value,
        },
    },
    {
        "id": "never_auto_rollback_low_confidence",
        "description": "Canary rollback requires high confidence",
        "condition": lambda d: d.get("deployment_action") == Action.ROLLBACK.value
                               and d.get("confidence", 0) < 0.70,
        "override": {
            "deployment_action": Action.PAUSE.value,
            "trust_tier": TrustTier.T1.value,
        },
    },
]


def apply_policy_guardrails(decision: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Apply policy rules to a decision dict (mutates in place).

    Returns (policy_outcome, list_of_triggered_rule_ids).
    """
    triggered: List[str] = []
    for rule in POLICY_RULES:
        try:
            if rule["condition"](decision):
                for key, val in rule["override"].items():
                    decision[key] = val
                triggered.append(rule["id"])
        except Exception:
            continue

    outcome = PolicyOutcome.OVERRIDE.value if triggered else PolicyOutcome.ALLOW.value
    return outcome, triggered


# ─── audit record builder ────────────────────────────────────────────

def build_audit_record(
    agent_result: Dict[str, Any],
    context: Dict[str, Any] = None,
    decision: Decision = None,
) -> AuditRecord:
    """Build a complete audit record from agent output + decision."""
    ctx = context or {}
    cl = agent_result.get("classification", {})
    beliefs = agent_result.get("beliefs", {})

    bayes_top = ""
    bayes_conf = 0.0
    if beliefs:
        bayes_top = max(beliefs, key=beliefs.get)
        bayes_conf = beliefs[bayes_top]

    prep = agent_result.get("preprocessing_summary", {})

    return AuditRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=ctx.get("run_id", prep.get("repo", "") + "/" + str(prep.get("workflow", ""))),
        repo=ctx.get("repo", prep.get("repo", "")),
        category=cl.get("category", "CODE_REGRESSION"),
        severity=cl.get("severity", "MODERATE"),
        confidence=float(cl.get("confidence", 0.0)),
        reasoning=cl.get("reasoning", ""),
        action=decision.action if decision else "INVESTIGATE",
        trust_tier=decision.trust_tier if decision else "T0",
        policy_outcome=decision.policy_outcome if decision else "ALLOW",
        policy_rules_triggered=decision.policy_rules_triggered if decision else [],
        deployment_action=decision.deployment_action if decision else None,
        flag_action=decision.flag_action if decision else None,
        tools_used=agent_result.get("tools_used", []),
        steps_taken=agent_result.get("steps_taken", 0),
        fast_path=agent_result.get("fast_path", False),
        bayesian_top=bayes_top,
        bayesian_confidence=bayes_conf,
        agent_mode=ctx.get("agent_mode", "APA"),
        planner_mode=ctx.get("planner_mode", "hybrid"),
    )


# ─── main enrichment function ────────────────────────────────────────

def enrich_result(
    agent_result: Dict[str, Any],
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Take a run_agent() result and add decision layer fields.

    This is the primary entry point. Downstream consumers (GitHub webhook,
    K8s controller, dashboard) read agent_result["decision"] and
    agent_result["audit"].

    Parameters
    ----------
    agent_result : dict
        Output of run_agent() — must contain "classification".
    context : dict, optional
        Extra context: run_id, repo, protected_branch, agent_mode, etc.

    Returns
    -------
    dict
        The same agent_result dict, enriched with "decision" and "audit" keys.
    """
    ctx = context or {}
    cl = agent_result.get("classification", {})
    category = cl.get("category", "CODE_REGRESSION")
    confidence = float(cl.get("confidence", 0.0))
    severity = cl.get("severity", "MODERATE")

    # 1. Triage action
    action = recommend_action(category, confidence)

    # 2. Trust tier
    trust_tier = assign_trust_tier(confidence, category)

    # 3. Canary decision (for future K8s integration)
    deployment_action = recommend_canary_action(
        category, confidence,
        error_rate_delta=ctx.get("error_rate_delta", 0.0),
    )

    # 4. Feature flag decision
    flag_action = recommend_flag_action(category, confidence)

    # 5. Policy guardrails (may override action / trust tier)
    decision_input: Dict[str, Any] = {
        "category": category,
        "confidence": confidence,
        "severity": severity,
        "action": action,
        "trust_tier": trust_tier,
        "deployment_action": deployment_action,
        "flag_action": flag_action,
        "protected_branch": ctx.get("protected_branch", False),
    }
    policy_outcome, triggered_rules = apply_policy_guardrails(decision_input)

    decision = Decision(
        category=category,
        severity=severity,
        confidence=confidence,
        reasoning=cl.get("reasoning", ""),
        action=decision_input["action"],
        trust_tier=decision_input["trust_tier"],
        policy_outcome=policy_outcome,
        policy_rules_triggered=triggered_rules,
        deployment_action=decision_input["deployment_action"],
        flag_action=decision_input["flag_action"],
    )

    audit = build_audit_record(agent_result, ctx, decision)

    agent_result["decision"] = decision.to_dict()
    agent_result["audit"] = audit.to_dict()

    return agent_result


# ─── dispatch interface (future: GitHub / K8s) ──────────────────────
# Currently a no-op. When connected to GitHub Actions or a K8s operator,
# this will POST to the appropriate API.

class Dispatcher:
    """
    Abstract dispatch interface for executing decisions.

    Subclass this to integrate with:
      - GitHub Checks API (post check run, PR comment)
      - Kubernetes (patch rollout, scale canary)
      - Slack/PagerDuty (alert on INVESTIGATE actions)
    """

    def dispatch(self, decision: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decision. Returns a receipt dict.

        Override in subclasses:
          GitHubDispatcher.dispatch() → POST /repos/{repo}/check-runs
          K8sDispatcher.dispatch()    → PATCH /apis/apps/v1/deployments/{name}
        """
        return {
            "dispatched": False,
            "reason": "no dispatcher configured (dry-run mode)",
            "action": decision.get("action"),
            "trust_tier": decision.get("trust_tier"),
        }


class GitHubDispatcher(Dispatcher):
    """
    GitHub integration: Check Runs, PR comments, and labels.

    Requires a GitHub token with `checks:write` and `pull_requests:write`
    permissions (a GitHub App installation token or a fine-grained PAT).

    Usage:
        dispatcher = GitHubDispatcher(token=os.environ["GITHUB_TOKEN"], repo="owner/repo")
        receipt = dispatcher.dispatch(decision, audit)
    """

    API_BASE = "https://api.github.com"

    # Map triage action -> GitHub Check Run conclusion
    ACTION_TO_CONCLUSION = {
        Action.BLOCK_MERGE.value:  "failure",
        Action.RETRY.value:        "neutral",
        Action.QUARANTINE.value:   "neutral",
        Action.INVESTIGATE.value:  "action_required",
        Action.IGNORE.value:       "success",
    }

    # Map category -> PR label
    CATEGORY_LABELS = {
        "CODE_REGRESSION":       "triage:regression",
        "DEPENDENCY_CONFLICT":   "triage:dependency",
        "CONFIG_ERROR":          "triage:config",
        "QUALITY_VIOLATION":     "triage:quality",
        "TEST_FLAKINESS":        "triage:flaky",
        "ENV_FLAKINESS":         "triage:env-flaky",
        "CASCADE_FAILURE":       "triage:cascade",
        "INFRA_INCOMPATIBILITY": "triage:infra",
        "TOOLING_ARTIFACT":      "triage:tooling",
    }

    def __init__(self, token: str = "", repo: str = "", dry_run: bool = False):
        import os
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.repo = repo
        self.dry_run = dry_run
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests as _requests
            self._session = _requests.Session()
            self._session.headers.update({
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            if self.token:
                self._session.headers["Authorization"] = f"token {self.token}"
        return self._session

    def _post(self, path: str, payload: dict) -> dict:
        """POST to GitHub API. Returns response JSON or error dict."""
        url = f"{self.API_BASE}{path}"
        if self.dry_run:
            return {"dry_run": True, "url": url, "payload": payload}
        try:
            resp = self._get_session().post(url, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "url": url}

    def create_check_run(
        self, head_sha: str, decision: Dict[str, Any], audit: Dict[str, Any],
    ) -> dict:
        """Create a GitHub Check Run reflecting the triage decision."""
        action = decision.get("action", "INVESTIGATE")
        conclusion = self.ACTION_TO_CONCLUSION.get(action, "action_required")
        category = decision.get("category", "CODE_REGRESSION")
        confidence = decision.get("confidence", 0)
        tier = decision.get("trust_tier", "T0")

        title = f"{category} - {action} (T{tier[-1]})"
        summary_lines = [
            f"**Category:** {category}",
            f"**Action:** {action}",
            f"**Confidence:** {confidence:.0%}",
            f"**Trust Tier:** {tier}",
            "",
            f"**Reasoning:** {decision.get('reasoning', 'N/A')}",
        ]
        if decision.get("deployment_action"):
            summary_lines.append(f"**Canary:** {decision['deployment_action']}")
        if decision.get("policy_outcome") != "ALLOW":
            summary_lines.append(f"**Policy:** {decision['policy_outcome']}")
            for rule in decision.get("policy_rules_triggered", []):
                summary_lines.append(f"  - {rule}")

        return self._post(f"/repos/{self.repo}/check-runs", {
            "name": "CI Triage Agent",
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": title,
                "summary": "\n".join(summary_lines),
            },
        })

    def post_pr_comment(
        self, pr_number: int, decision: Dict[str, Any], audit: Dict[str, Any],
        remediation_markdown: str = "",
    ) -> dict:
        """Post a triage summary as a PR comment, with optional remediation plan."""
        action = decision.get("action", "INVESTIGATE")
        category = decision.get("category", "CODE_REGRESSION")
        confidence = decision.get("confidence", 0)
        tier = decision.get("trust_tier", "T0")

        action_emoji = {
            "BLOCK_MERGE": "🚫", "RETRY": "🔄",
            "QUARANTINE": "🧪", "INVESTIGATE": "🔍",
            "IGNORE": "✅",
        }
        emoji = action_emoji.get(action, "⚙️")

        body = (
            f"## {emoji} CI Triage: {action}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Category | `{category}` |\n"
            f"| Action | **{action}** |\n"
            f"| Confidence | {confidence:.0%} |\n"
            f"| Trust Tier | {tier} |\n"
            f"| Canary | {decision.get('deployment_action', 'N/A')} |\n"
            f"| Feature Flag | {decision.get('flag_action', 'N/A')} |\n\n"
            f"**Reasoning:** {decision.get('reasoning', 'N/A')}\n\n"
        )

        # Include remediation plan if available
        if remediation_markdown:
            body += f"---\n\n{remediation_markdown}\n\n---\n\n"

        body += (
            f"<details><summary>Audit Trail</summary>\n\n"
            f"```json\n{json.dumps(audit, indent=2)}\n```\n</details>\n\n"
            f"*Automated by CI Triage Agent ({audit.get('agent_mode', 'APA')} mode)*"
        )

        return self._post(f"/repos/{self.repo}/issues/{pr_number}/comments", {
            "body": body,
        })

    def add_labels(self, pr_number: int, category: str) -> dict:
        """Add a triage label to the PR based on failure category."""
        label = self.CATEGORY_LABELS.get(category)
        if not label:
            return {"skipped": True, "reason": f"no label for {category}"}
        return self._post(f"/repos/{self.repo}/issues/{pr_number}/labels", {
            "labels": [label],
        })

    def dispatch(
        self, decision: Dict[str, Any], audit: Dict[str, Any],
        remediation_markdown: str = "",
    ) -> Dict[str, Any]:
        """Full dispatch: check run + PR comment (with remediation) + labels."""
        head_sha = audit.get("commit_sha", decision.get("commit_sha", ""))
        pr_number = audit.get("pr_number", decision.get("pr_number"))
        category = decision.get("category", "CODE_REGRESSION")

        receipt: Dict[str, Any] = {
            "dispatched": bool(self.token and self.repo),
            "target": f"github:{self.repo}",
        }

        if not self.token:
            receipt["dispatched"] = False
            receipt["reason"] = "no GITHUB_TOKEN configured"
            return receipt

        if not self.repo:
            receipt["dispatched"] = False
            receipt["reason"] = "no repo configured"
            return receipt

        # 1. Check Run (requires head_sha)
        if head_sha:
            receipt["check_run"] = self.create_check_run(head_sha, decision, audit)
        else:
            receipt["check_run"] = {"skipped": True, "reason": "no head_sha"}

        # 2. PR comment + labels (requires pr_number)
        if pr_number:
            receipt["pr_comment"] = self.post_pr_comment(
                pr_number, decision, audit, remediation_markdown,
            )
            receipt["labels"] = self.add_labels(pr_number, category)
        else:
            receipt["pr_comment"] = {"skipped": True, "reason": "no pr_number"}
            receipt["labels"] = {"skipped": True, "reason": "no pr_number"}

        return receipt


class K8sDispatcher(Dispatcher):
    """
    Kubernetes integration: canary rollout control.

    Supports three deployment actions:
      - PROMOTE:  resume rollout (unpause), let it complete
      - PAUSE:    pause the rollout, wait for human or further signal
      - ROLLBACK: undo the last rollout to the previous revision

    Requires the `kubernetes` Python package. Auth is resolved in order:
      1. In-cluster config (when running as a pod)
      2. Kubeconfig file (~/.kube/config or KUBECONFIG env)

    Usage:
        dispatcher = K8sDispatcher(namespace="production", deployment="my-app")
        receipt = dispatcher.dispatch(decision, audit)
    """

    def __init__(
        self,
        namespace: str = "default",
        deployment: str = "",
        dry_run: bool = False,
    ):
        self.namespace = namespace
        self.deployment = deployment
        self.dry_run = dry_run
        self._client = None
        self._available = None

    def _ensure_client(self) -> bool:
        """Lazy-init the K8s client. Returns True if available."""
        if self._available is not None:
            return self._available
        try:
            from kubernetes import client, config as k8s_config
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
            self._client = client.AppsV1Api()
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def _patch_deployment(self, body: dict) -> dict:
        """Patch the target deployment. Returns the API response as dict."""
        if self.dry_run:
            return {"dry_run": True, "patch": body}
        if not self._ensure_client():
            return {"error": "kubernetes client not available"}
        try:
            resp = self._client.patch_namespaced_deployment(
                name=self.deployment,
                namespace=self.namespace,
                body=body,
            )
            return {
                "name": resp.metadata.name,
                "namespace": resp.metadata.namespace,
                "generation": resp.metadata.generation,
                "replicas": resp.spec.replicas,
                "paused": resp.spec.paused,
            }
        except Exception as e:
            return {"error": str(e)}

    def promote(self) -> dict:
        """Resume (unpause) the rollout and let it complete."""
        return self._patch_deployment({
            "spec": {"paused": False},
        })

    def pause(self) -> dict:
        """Pause the current rollout."""
        return self._patch_deployment({
            "spec": {"paused": True},
        })

    def rollback(self) -> dict:
        """
        Roll back to previous revision.

        Uses the rollback annotation trick: set the revision annotation
        to 0 which triggers a rollback to the last known-good state.
        For newer K8s (>=1.18), this patches the deployment to match
        the previous ReplicaSet template.
        """
        if self.dry_run:
            return {"dry_run": True, "action": "rollback"}
        if not self._ensure_client():
            return {"error": "kubernetes client not available"}
        try:
            # Get current deployment
            dep = self._client.read_namespaced_deployment(
                name=self.deployment,
                namespace=self.namespace,
            )
            # Find the previous ReplicaSet
            from kubernetes import client
            apps = client.AppsV1Api()
            rs_list = apps.list_namespaced_replica_set(
                namespace=self.namespace,
                label_selector=",".join(
                    f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items()
                ),
            )
            # Sort by revision annotation, pick the second-newest
            def _revision(rs):
                ann = rs.metadata.annotations or {}
                try:
                    return int(ann.get("deployment.kubernetes.io/revision", "0"))
                except ValueError:
                    return 0

            sorted_rs = sorted(rs_list.items, key=_revision, reverse=True)
            if len(sorted_rs) < 2:
                return {"error": "no previous revision to roll back to"}

            prev_rs = sorted_rs[1]
            # Patch the deployment template to match previous RS
            patch_body = {
                "spec": {
                    "template": prev_rs.spec.template.to_dict(),
                },
            }
            resp = self._client.patch_namespaced_deployment(
                name=self.deployment,
                namespace=self.namespace,
                body=patch_body,
            )
            return {
                "rolled_back": True,
                "to_revision": _revision(prev_rs),
                "generation": resp.metadata.generation,
            }
        except Exception as e:
            return {"error": str(e)}

    def dispatch(self, decision: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the canary deployment action."""
        dep_action = decision.get("deployment_action", "PAUSE")
        receipt: Dict[str, Any] = {
            "dispatched": bool(self.deployment),
            "target": f"k8s:{self.namespace}/{self.deployment}",
            "deployment_action": dep_action,
        }

        if not self.deployment:
            receipt["dispatched"] = False
            receipt["reason"] = "no deployment configured"
            return receipt

        if dep_action == Action.PROMOTE.value:
            receipt["result"] = self.promote()
        elif dep_action == Action.ROLLBACK.value:
            receipt["result"] = self.rollback()
        else:
            # PAUSE is the safe default
            receipt["result"] = self.pause()

        return receipt


# ─── pretty printer ─────────────────────────────────────────────────

def print_decision(result: Dict[str, Any]) -> None:
    """Print decision layer output for terminal debugging."""
    dec = result.get("decision", {})
    aud = result.get("audit", {})

    print("\n" + "=" * 70)
    print("DECISION LAYER")
    print("=" * 70)

    # Classification
    print(f"  category:     {dec.get('category', '?')}")
    print(f"  severity:     {dec.get('severity', '?')}")
    print(f"  confidence:   {dec.get('confidence', 0):.0%}")

    # Triage
    action = dec.get("action", "?")
    tier = dec.get("trust_tier", "?")
    action_colors = {
        "BLOCK_MERGE": "\033[91m",    # red
        "RETRY": "\033[93m",          # yellow
        "QUARANTINE": "\033[96m",     # cyan
        "INVESTIGATE": "\033[95m",    # magenta
        "IGNORE": "\033[90m",         # gray
    }
    color = action_colors.get(action, "")
    reset = "\033[0m" if color else ""
    print(f"  action:       {color}{action}{reset}")
    print(f"  trust tier:   {tier}")

    # Deployment
    if dec.get("deployment_action"):
        print(f"  canary:       {dec['deployment_action']}")
    if dec.get("flag_action"):
        print(f"  feature flag: {dec['flag_action']}")

    # Policy
    policy = dec.get("policy_outcome", "ALLOW")
    if policy != "ALLOW":
        print(f"  policy:       {policy}")
        for rule_id in dec.get("policy_rules_triggered", []):
            print(f"    → triggered: {rule_id}")

    # Audit
    print(f"  audit id:     {aud.get('id', '?')[:12]}...")
    print(f"  agent mode:   {aud.get('agent_mode', '?')}")
    print(f"  bayes top:    {aud.get('bayesian_top', '?')} @ {aud.get('bayesian_confidence', 0):.0%}")
    print("=" * 70)


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Decision Layer — self-test ===\n")

    # Simulate a run_agent() result
    fake_result = {
        "classification": {
            "category": "TEST_FLAKINESS",
            "severity": "MODERATE",
            "confidence": 0.78,
            "reasoning": "Intermittent timeout in network test, no code changes.",
        },
        "beliefs": {
            "CODE_REGRESSION": 0.05,
            "DEPENDENCY_CONFLICT": 0.03,
            "CONFIG_ERROR": 0.02,
            "QUALITY_VIOLATION": 0.01,
            "ENV_FLAKINESS": 0.10,
            "TEST_FLAKINESS": 0.72,
            "TOOLING_ARTIFACT": 0.01,
            "CASCADE_FAILURE": 0.01,
            "INFRA_INCOMPATIBILITY": 0.05,
        },
        "tools_used": ["inspect_commit_diff", "deep_log_analysis", "check_run_history"],
        "steps_taken": 3,
        "fast_path": False,
        "preprocessing_summary": {"repo": "example/repo", "workflow": "ci.yml"},
    }

    enriched = enrich_result(fake_result, context={
        "repo": "example/repo",
        "run_id": "example/repo_ci.yml_42_1",
        "protected_branch": False,
        "agent_mode": "APA",
    })

    print_decision(enriched)

    print("\n--- Audit record (JSON) ---")
    print(json.dumps(enriched["audit"], indent=2))

    # Test a few triage mappings
    print("\n--- Triage rule checks ---")
    tests = [
        ("CODE_REGRESSION",    0.85, "BLOCK_MERGE"),
        ("CODE_REGRESSION",    0.40, "INVESTIGATE"),
        ("TEST_FLAKINESS",     0.75, "QUARANTINE"),
        ("TEST_FLAKINESS",     0.55, "RETRY"),
        ("QUALITY_VIOLATION",  0.70, "BLOCK_MERGE"),
        ("ENV_FLAKINESS",      0.60, "RETRY"),
        ("TOOLING_ARTIFACT",   0.90, "IGNORE"),
        ("INFRA_INCOMPATIBILITY", 0.75, "BLOCK_MERGE"),
    ]
    all_pass = True
    for cat, conf, expected in tests:
        actual = recommend_action(cat, conf)
        ok = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_pass = False
        print(f"  {ok} {cat} @ {conf:.0%} -> {actual} (expected {expected})")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")

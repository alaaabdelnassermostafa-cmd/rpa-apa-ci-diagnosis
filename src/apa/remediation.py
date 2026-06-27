# remediation.py
# ─────────────────────────────────────────────────────────────────────
# Layer 5: Remediation Plan Generator
#
# After the agent diagnoses a CI/CD failure (classification) and the
# decision layer recommends an action, this module generates a concrete
# remediation plan: what files to change, what the fix looks like, and
# step-by-step instructions.
#
# This is NOT a Copilot-style inline suggestion. It leverages the full
# diagnostic context (Bayesian beliefs, commit diffs, error logs,
# workflow analysis) that only the triage agent has access to.
#
# Integration points:
#   - GitHubDispatcher: post the remediation as a PR comment
#   - K8sDispatcher: include in rollback/pause decision context
#   - Dashboard: render as an actionable card
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI
from src.apa.llm_config import make_client
from src.apa.llm_usage import record_usage


# ─── remediation data model ─────────────────────────────────────────

@dataclass
class RemediationStep:
    """A single actionable step in the remediation plan."""
    order: int = 0
    description: str = ""
    file: str = ""                   # file to modify (if applicable)
    action: str = ""                 # "modify", "add", "delete", "run_command"
    details: str = ""               # specific change or command

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RemediationPlan:
    """Complete remediation plan generated from diagnostic context."""

    # Root cause summary (human-readable)
    root_cause: str = ""
    
    # Category-specific fix type
    fix_type: str = ""               # "workflow_yaml", "dependency_pin",
                                     # "code_change", "retry", "config_update"
    
    # Affected files
    affected_files: List[str] = field(default_factory=list)
    
    # Suggested fix (diff format when possible)
    suggested_fix: str = ""
    
    # Step-by-step instructions
    steps: List[RemediationStep] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0          # how confident the system is in this fix
    auto_applicable: bool = False    # can this be applied without human review?
    requires_human_review: bool = True
    estimated_fix_time: str = ""     # e.g. "< 5 minutes"
    
    # References
    evidence_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() if isinstance(s, RemediationStep) else s for s in self.steps]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render as a GitHub-friendly markdown comment."""
        lines = [
            "## 🔧 Remediation Plan",
            "",
            f"**Root Cause:** {self.root_cause}",
            f"**Fix Type:** `{self.fix_type}`",
            f"**Confidence:** {self.confidence:.0%}",
            f"**Estimated Fix Time:** {self.estimated_fix_time}",
            "",
        ]

        if self.affected_files:
            lines.append("### Affected Files")
            for f in self.affected_files:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.suggested_fix:
            lines.append("### Suggested Fix")
            lines.append("```diff")
            lines.append(self.suggested_fix)
            lines.append("```")
            lines.append("")

        if self.steps:
            lines.append("### Steps")
            for step in self.steps:
                s = step if isinstance(step, dict) else step.to_dict()
                lines.append(f"{s['order']}. **{s['description']}**")
                if s.get("file"):
                    lines.append(f"   - File: `{s['file']}`")
                if s.get("details"):
                    lines.append(f"   - {s['details']}")
            lines.append("")

        if self.requires_human_review:
            lines.append("> ⚠️ **Human review required** before applying this fix.")
        else:
            lines.append("> ✅ This fix can be auto-applied with high confidence.")

        if self.evidence_sources:
            lines.append("")
            lines.append("<details><summary>Evidence Sources</summary>")
            lines.append("")
            for src in self.evidence_sources:
                lines.append(f"- {src}")
            lines.append("")
            lines.append("</details>")

        return "\n".join(lines)


# ─── prompt template ────────────────────────────────────────────────

REMEDIATION_PROMPT = """You are a CI/CD remediation expert. Based on the diagnosis below, generate a concrete fix plan.

DIAGNOSIS:
- Category: {category}
- Severity: {severity}
- Confidence: {confidence:.0%}
- Reasoning: {reasoning}

ERROR LOGS:
{error_lines}

COMMIT DIFF:
{commit_diff}

CHANGED FILES:
{changed_files}

DEPENDENCY CHANGES:
{dependency_changes}

WORKFLOW ANALYSIS:
{workflow_contents}

SEMANTIC DIFF LINKS (version bumps matched to errors):
{semantic_links}

Generate a JSON remediation plan. Be SPECIFIC — reference actual file names, version numbers, and error messages from the evidence above.

Rules:
1. For DEPENDENCY_CONFLICT: suggest exact version pins or range updates
2. For CONFIG_ERROR: suggest exact YAML changes to workflow files
3. For CODE_REGRESSION: identify the breaking change and suggest a revert or fix
4. For INFRA_INCOMPATIBILITY: suggest runner/tool version updates
5. For ENV_FLAKINESS: suggest retry strategies and caching fixes
6. For TEST_FLAKINESS: suggest quarantine annotations or retry directives

Respond with ONLY this JSON:
{{
  "root_cause": "1-2 sentence plain-English explanation of what went wrong",
  "fix_type": "workflow_yaml|dependency_pin|code_change|retry_strategy|config_update|runner_update",
  "affected_files": ["path/to/file1", "path/to/file2"],
  "suggested_fix": "diff-style fix showing exact changes (use - for removals, + for additions)",
  "steps": [
    {{"order": 1, "description": "What to do", "file": "path/to/file", "action": "modify|add|delete|run_command", "details": "Specific change"}},
  ],
  "confidence": 0.85,
  "auto_applicable": false,
  "requires_human_review": true,
  "estimated_fix_time": "< 5 minutes",
  "evidence_sources": ["error log line X", "version bump in Y"]
}}"""


# ─── remediation generator ──────────────────────────────────────────

def _format_evidence(agent_state: Dict[str, Any]) -> Dict[str, str]:
    """Extract and format all diagnostic evidence from agent state."""
    
    error_lines = agent_state.get("error_lines", [])
    error_str = "\n".join(error_lines[:10]) if error_lines else "(no errors extracted)"

    commit_diff = agent_state.get("commit_diff", {})
    if commit_diff:
        files = commit_diff.get("files", [])
        diff_parts = [f"Summary: {commit_diff.get('summary', 'n/a')}"]
        for f in files[:8]:
            fname = f.get("filename", "?")
            status = f.get("status", "?")
            patch = f.get("patch_excerpt", "")[:300]
            diff_parts.append(f"  [{status}] {fname}")
            if patch:
                diff_parts.append(f"    {patch}")
        commit_diff_str = "\n".join(diff_parts)
    else:
        commit_diff_str = "(not retrieved)"

    changed_files = agent_state.get("changed_files", [])
    changed_str = "\n".join(
        f"  [{f.get('status', '?')}] {f.get('filename', '?')}"
        for f in changed_files[:15]
    ) or "(not retrieved)"

    dep_changes = agent_state.get("dependency_changes", {})
    dep_str = json.dumps(dep_changes, indent=2) if dep_changes else "(not retrieved)"

    workflow = agent_state.get("workflow_contents", [])
    if workflow:
        wc_parts = []
        for entry in workflow:
            fname = entry.get("file", "?")
            actions = ", ".join(entry.get("action_versions", [])[:6]) or "none"
            runners = ", ".join(entry.get("runners", [])[:4]) or "none"
            deprecated = ", ".join(entry.get("deprecated_nodes", [])) or "none"
            wc_parts.append(f"  {fname}: actions=[{actions}] runners=[{runners}] deprecated=[{deprecated}]")
        wc_str = "\n".join(wc_parts)
    else:
        wc_str = "(not retrieved)"

    sem_links = agent_state.get("semantic_diff_links", [])
    if sem_links:
        sem_parts = []
        for e in sem_links[:5]:
            arrow = f"{e.get('old_version', '')} → {e.get('new_version', '')}"
            sem_parts.append(f"  {e.get('library', '?')} bumped {arrow} in {e.get('file', '?')}")
            for el in (e.get("matching_error_lines") or [])[:2]:
                sem_parts.append(f"    ↳ error: {el[:120]}")
        sem_str = "\n".join(sem_parts)
    else:
        sem_str = "(no version-error cross-references)"

    return {
        "error_lines": error_str,
        "commit_diff": commit_diff_str,
        "changed_files": changed_str,
        "dependency_changes": dep_str,
        "workflow_contents": wc_str,
        "semantic_links": sem_str,
    }


def generate_remediation(
    classification: Dict[str, Any],
    agent_state: Dict[str, Any],
    client: OpenAI = None,
    model: str = None,
) -> RemediationPlan:
    """
    Generate a remediation plan from the agent's classification and diagnostic state.

    Parameters
    ----------
    classification : dict
        The agent's classification output (category, severity, confidence, reasoning).
    agent_state : dict
        The full agent state after investigation (contains all evidence).
    client : OpenAI, optional
        LLM client. If None, creates one from env.
    model : str
        Model to use for generation.

    Returns
    -------
    RemediationPlan
        A structured remediation plan with affected files, fix diff, and steps.
    """
    client = client or make_client()
    import os
    model = model or os.environ.get("CI_AGENT_MODEL", "deepseek-v4-pro")

    category = classification.get("category", "CODE_REGRESSION")
    confidence = float(classification.get("confidence", 0.0))

    # Don't generate remediation for low-confidence diagnoses
    if confidence < 0.40:
        return RemediationPlan(
            root_cause="Diagnosis confidence too low to generate a reliable remediation plan.",
            fix_type="manual_investigation",
            confidence=confidence,
            requires_human_review=True,
            steps=[RemediationStep(
                order=1,
                description="Manual investigation required",
                action="investigate",
                details="The agent could not confidently diagnose the root cause. Review the CI logs manually.",
            )],
        )

    evidence = _format_evidence(agent_state)

    prompt = REMEDIATION_PROMPT.format(
        category=category,
        severity=classification.get("severity", "MODERATE"),
        confidence=confidence,
        reasoning=classification.get("reasoning", ""),
        **evidence,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior DevOps engineer who specializes in CI/CD pipeline fixes. "
                        "Generate precise, actionable remediation plans based on the diagnostic evidence. "
                        "Always reference specific file names, version numbers, and error messages."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
        )
        record_usage(response, model, call_type="chat", label="remediation.generate")
        data = json.loads(response.choices[0].message.content)

        steps = []
        for s in data.get("steps", []):
            steps.append(RemediationStep(
                order=s.get("order", 0),
                description=s.get("description", ""),
                file=s.get("file", ""),
                action=s.get("action", ""),
                details=s.get("details", ""),
            ))

        return RemediationPlan(
            root_cause=data.get("root_cause", ""),
            fix_type=data.get("fix_type", ""),
            affected_files=data.get("affected_files", []),
            suggested_fix=data.get("suggested_fix", ""),
            steps=steps,
            confidence=float(data.get("confidence", confidence)),
            auto_applicable=data.get("auto_applicable", False),
            requires_human_review=data.get("requires_human_review", True),
            estimated_fix_time=data.get("estimated_fix_time", ""),
            evidence_sources=data.get("evidence_sources", []),
        )

    except Exception as e:
        return RemediationPlan(
            root_cause=f"Remediation generation failed: {e}",
            fix_type="error",
            confidence=0.0,
            requires_human_review=True,
        )


# ─── integration helper ─────────────────────────────────────────────

def enrich_with_remediation(
    agent_result: Dict[str, Any],
    agent_state: Dict[str, Any],
    client: OpenAI = None,
    model: str = None,
) -> Dict[str, Any]:
    """
    Add a remediation plan to an enriched agent result.

    Call this AFTER decision_layer.enrich_result() to add the
    "remediation" key to the result dict.

    Parameters
    ----------
    agent_result : dict
        Output of enrich_result() — must contain "classification".
    agent_state : dict
        The full agent state after investigation.

    Returns
    -------
    dict
        The same agent_result dict with a "remediation" key added.
    """
    classification = agent_result.get("classification", {})
    plan = generate_remediation(classification, agent_state, client, model)
    agent_result["remediation"] = plan.to_dict()
    agent_result["remediation_markdown"] = plan.to_markdown()
    return agent_result

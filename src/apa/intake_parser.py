import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.apa.adapters import GitHubAdapter, RunEvent, FailedStepInfo

# ─── the agent ────────────────────────────────────────────────────────

def intake(raw_run: dict) -> RunEvent:
    """Wrapper around GitHubAdapter for backward compatibility."""
    adapter = GitHubAdapter()
    return adapter.parse(raw_run)


# ─── pretty printer ───────────────────────────────────────────────────

def pretty_print(event: RunEvent, raw_run: Optional[dict] = None) -> None:
    print("=" * 70)
    print(f"RUN EVENT  |  {event.run_id} ({event.source})")
    print("=" * 70)
    print(f"  repo         {event.repo}")
    print(f"  workflow     {event.workflow}")
    print(f"  run          #{event.run_number}  (attempt {event.attempt})")
    print(f"  event        {event.event}")
    print(f"  branch       {event.branch}   [protected={event.is_protected_branch}]")
    print(f"  actor        {event.actor}")
    print(f"  started      {event.started_at}")
    dur = f"{event.duration_sec:.1f}s" if event.duration_sec else "unknown"
    print(f"  duration     {dur}")
    print()
    print(f"  CONCLUSION   {event.conclusion.upper()}")
    print(f"  detection    {event.failure_detection}")
    if event.all_failures_are_tooling_artifacts:
        print(f"  ⚠ all detected failures look like parsing artifacts")
    if event.commit_sha:
        print(f"  commit       {event.commit_sha}  by {event.commit_author}")
        print(f"  title        {event.commit_title}")
    print()
    print(f"  jobs         {event.n_jobs}")
    print(f"  failed jobs  {event.failed_jobs_count}")

    if event.failed_steps:
        print()
        print("  FAILED STEPS:")
        for i, fs in enumerate(event.failed_steps, 1):
            print(f"    [{i}] {fs.job_file}")
            print(f"         runner    {fs.runner_image}")
            print(f"         step      #{fs.step_index}  ({fs.step_type})")
            print(f"         label     {fs.step_label}")
            print(f"         dur       {fs.step_duration_sec}")
            print(f"         detection {fs.detection_mode}")
            if fs.tooling_artifact_suspected:
                print(f"         ⚠ tooling artifact suspected")
            if fs.error_text:
                print(f"         error     {fs.error_text[:200]}")
            else:
                print(f"         error     <none — inferred by {fs.detection_mode}>")

    elif event.conclusion == "failure":
        print()
        print("  ⚠ failure with no failed step detected — schema diagnostic:")
        if raw_run:
            insights = raw_run.get("log_insights") or []
            if insights and insights[0].get("steps"):
                first_step = insights[0]["steps"][0]
                print(f"     first step keys: {sorted(first_step.keys())}")
                print(f"     first step sample: {json.dumps(first_step, default=str)[:400]}")


# ─── run on both samples ──────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        ("gha_data/sample_failed_run.json", "gha_data/sample_failed_event.json"),
        ("gha_data/sample_case5_run.json",  "gha_data/sample_case5_event.json"),
    ]

    for in_path, out_path in samples:
        p = Path(in_path)
        if not p.exists():
            print(f"(skipping missing sample: {in_path})\n")
            continue

        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        event = intake(raw)
        pretty_print(event, raw_run=raw)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(event), f, indent=2, default=str)
        print(f"\n✓ Clean event saved → {out_path}\n")

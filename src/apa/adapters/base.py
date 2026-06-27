from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class FailedStepInfo:
    job_file: str                       # log file / pod name
    runner_image: str                   # macos-12 / container image
    step_index: Optional[int]           # position within the job / init container index
    step_type: Optional[str]            # "action" / "run" / "container"
    step_label: Optional[str]           # action name or command summary
    step_duration_sec: Optional[float]
    error_text: Optional[str]           # actual error string if found
    detection_mode: str = "per_step_error"   # how we decided this step failed
    tooling_artifact_suspected: bool = False  # parser crash, not real failure
    raw_keys: List[str] = field(default_factory=list)  # diagnostic


@dataclass
class RunEvent:
    # ─── Non-default fields (must come first) ───
    source: str
    run_id: str
    repo: str
    event: str
    conclusion: str
    started_at: str
    n_jobs: int
    failed_jobs_count: int
    duration_sec: Optional[float]

    # ─── Default fields ───
    workflow: Optional[str] = None
    run_number: Optional[int] = None
    attempt: Optional[int] = None
    branch: Optional[str] = None
    is_protected_branch: Optional[bool] = None
    actor: Optional[str] = None

    commit_sha: Optional[str] = None
    commit_title: Optional[str] = None
    commit_message: Optional[str] = None
    commit_author: Optional[str] = None

    failed_steps: List[FailedStepInfo] = field(default_factory=list)
    failure_detection: str = "not_a_failure"
    all_failures_are_tooling_artifacts: bool = False
    has_log_insights: bool = False
    available_signals: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class FailureAdapter:
    """Base class for all failure intake adapters."""
    
    @property
    def source_name(self) -> str:
        """Return the identifier for this source (e.g. 'github', 'kubernetes')."""
        raise NotImplementedError

    @property
    def available_signals(self) -> List[str]:
        """
        Return the list of Bayesian signals this adapter can provide.
        Used by the Signal Registry to selectively apply Bayesian updates.
        Example: ['error_text', 'commit_message', 'branch_type']
        """
        raise NotImplementedError

    def parse(self, raw_payload: Any) -> RunEvent:
        """Parse raw incoming data into a standard RunEvent."""
        raise NotImplementedError

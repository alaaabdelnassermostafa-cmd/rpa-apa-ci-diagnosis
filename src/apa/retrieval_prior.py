# retrieval_prior.py
# ─────────────────────────────────────────────────────────────────────
# Layer 3: Retrieval-Augmented Priors
#
# Given a new failed run, find similar past evaluated cases and use
# their outcomes to build a smarter Bayesian prior instead of
# starting uniform.
#
# Steps:
#   1. Embed each past case as a text vector (commit + branch + errors)
#   2. Embed the new case the same way
#   3. Find k most similar past cases by cosine similarity
#   4. Weight their category outcomes → prior distribution
#
# No new LLM reasoning call here — pure math on top of past results.
# ─────────────────────────────────────────────────────────────────────

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from src.apa.llm_config import make_client
from src.apa.llm_usage import record_usage

from src.apa.bayesian_tracker import CATEGORIES, BeliefState

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_PATH = _BASE_DIR / "data" / "honest_eval_results.json"
if not EVAL_PATH.exists():
    EVAL_PATH = Path("/home/guc_alaa/honest_eval_results.json")

CACHE_PATH = _BASE_DIR / "data" / "retrieval_cache.json"
if not CACHE_PATH.exists():
    CACHE_PATH = Path("/home/guc_alaa/retrieval_cache.json")

DB_PATH = Path(os.environ.get("CASE_MEMORY_DB_PATH", _BASE_DIR / "data" / "case_memory.sqlite3"))

K_NEIGHBORS = 5          # how many past cases to use
MIN_SIMILARITY = 0.72    # ignore past cases below this similarity
PRIOR_WEIGHT = 0.35      # how much the prior shifts beliefs (0=ignore, 1=replace)

# ─── ChromaDB feature flag ───────────────────────────────────────────
# Set USE_CHROMA=1 to route search_similar_cases() through the
# ChromaDB vector store (APA-v level) instead of the SQLite+cosine path.
# The token-overlap path in agent.py preprocessing is never affected.
USE_CHROMA: bool = os.environ.get("USE_CHROMA", "0") == "1"
CHROMA_PATH: str = os.environ.get("CHROMA_PATH", str(_BASE_DIR / "data" / "chroma"))


# ─── text representation of a case ──────────────────────────────────

def case_to_text(
    repo: str,
    branch: str,
    commit_title: str,
    error_lines: List[str],
    failure_detection: str,
    n_failed: int,
    n_total: int,
) -> str:
    """
    Convert a case's key features into a short text for embedding.
    Keep it focused — only the signals that matter for category.
    """
    errors = " | ".join(error_lines[:4]) if error_lines else "no error text"
    return (
        f"branch: {branch} "
        f"commit: {commit_title} "
        f"errors: {errors} "
        f"jobs: {n_failed}/{n_total} failed "
        f"detection: {failure_detection}"
    )


# ─── embedding ───────────────────────────────────────────────────────
def embed_text(text: str, client=None) -> List[float]:
    """Get embedding vector using OpenRouter."""
    oa_client = client or make_client()
    response = oa_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text[:2000],
    )
    record_usage(
        response,
        "openai/text-embedding-3-small",
        call_type="embedding",
        label="retrieval_prior.embed_text",
    )
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── memory: past evaluated cases ───────────────────────────────────

class CaseMemory:
    """
    Holds embeddings of past evaluated cases.
    Loads from honest_eval_results.json and caches embeddings to disk
    so we don't re-embed on every run.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self.cases: List[dict] = []
        self.db_path = DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._load_or_build()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_memory (
                run_id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                branch TEXT NOT NULL,
                workflow TEXT NOT NULL,
                commit_title TEXT NOT NULL,
                failure_detection TEXT NOT NULL,
                n_failed INTEGER NOT NULL,
                n_total INTEGER NOT NULL,
                category TEXT NOT NULL,
                gt_verdict TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _load_or_build(self):
        row = self.conn.execute("SELECT COUNT(*) AS n FROM case_memory").fetchone()
        if row and row["n"] > 0:
            self._load_cases_from_db()
            return

        self._bootstrap_from_eval_file()
        self._load_cases_from_db()

    def _bootstrap_from_eval_file(self) -> None:
        if not EVAL_PATH.exists():
            print("  [retrieval] no past cases found, using uniform prior")
            return

        with EVAL_PATH.open("r", encoding="utf-8") as handle:
            eval_results = json.load(handle)

        cache = {}
        if CACHE_PATH.exists():
            try:
                cache = json.load(open(CACHE_PATH))
            except Exception:
                cache = {}

        rebuilt_cache = False
        inserted = 0

        for result in eval_results:
            intake = result.get("intake", {})
            cl = result.get("classification", {})
            gt = result.get("ground_truth", {})
            category = cl.get("category", "")
            run_id = intake.get("run_id", "")

            if not category or not run_id:
                continue

            text = case_to_text(
                repo=intake.get("repo", ""),
                branch=intake.get("branch", ""),
                commit_title=intake.get("commit_title", ""),
                error_lines=result.get("extraction", {}).get("sample_error_lines", []),
                failure_detection=intake.get("failure_detection", ""),
                n_failed=intake.get("failed_jobs_count", 0),
                n_total=intake.get("n_jobs", 1),
            )

            if run_id in cache:
                embedding = cache[run_id]
            else:
                try:
                    embedding = embed_text(text, self.client)
                    cache[run_id] = embedding
                    rebuilt_cache = True
                except Exception as e:
                    print(f"  [retrieval] embed failed for {run_id}: {e}")
                    continue

            self.conn.execute(
                """
                INSERT OR REPLACE INTO case_memory (
                    run_id, repo, branch, workflow, commit_title,
                    failure_detection, n_failed, n_total,
                    category, gt_verdict, text, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    intake.get("repo", ""),
                    intake.get("branch", ""),
                    intake.get("workflow", ""),
                    intake.get("commit_title", ""),
                    intake.get("failure_detection", ""),
                    intake.get("failed_jobs_count", 0),
                    intake.get("n_jobs", 1),
                    category,
                    gt.get("match_verdict", "NO_DATA"),
                    text,
                    json.dumps(embedding),
                ),
            )
            inserted += 1

        self.conn.commit()

        if rebuilt_cache:
            CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        print(f"  [retrieval] bootstrapped {inserted} cases into {self.db_path}")

    def _load_cases_from_db(self) -> None:
        self.cases = []
        rows = self.conn.execute(
            """
            SELECT run_id, repo, branch, workflow, commit_title,
                   failure_detection, n_failed, n_total,
                   category, gt_verdict, text, embedding_json
            FROM case_memory
            """
        ).fetchall()
        for row in rows:
            try:
                embedding = json.loads(row["embedding_json"])
            except Exception:
                continue
            self.cases.append(
                {
                    "run_id": row["run_id"],
                    "repo": row["repo"],
                    "branch": row["branch"],
                    "workflow": row["workflow"],
                    "commit_title": row["commit_title"],
                    "failure_detection": row["failure_detection"],
                    "n_failed": row["n_failed"],
                    "n_total": row["n_total"],
                    "text": row["text"],
                    "embedding": embedding,
                    "category": row["category"],
                    "gt_verdict": row["gt_verdict"],
                }
            )
        print(f"  [retrieval] loaded {len(self.cases)} past cases from {self.db_path}")

    def find_similar(
        self,
        query_text: str,
        k: int = K_NEIGHBORS,
        min_sim: float = MIN_SIMILARITY,
    ) -> List[Tuple[dict, float]]:
        """
        Find k most similar past cases to the query.
        Returns list of (case, similarity) sorted by similarity desc.
        """
        if not self.cases:
            return []

        try:
            query_embedding = embed_text(query_text, self.client)
        except Exception as e:
            print(f"  [retrieval] query embed failed: {e}")
            return []

        similarities = []
        for case in self.cases:
            sim = cosine_similarity(query_embedding, case["embedding"])
            if sim >= min_sim:
                similarities.append((case, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def find_similar_case_records(
        self,
        repo: str,
        workflow: str,
        branch: str,
        commit_title: str,
        commit_message: str,
        error_lines: List[str],
        mentioned_files: List[dict],
        failure_detection: str,
        n_failed: int,
        n_total: int,
        k: int = K_NEIGHBORS,
        min_sim: float = MIN_SIMILARITY,
    ) -> List[dict]:
        query_parts = [
            case_to_text(
                repo=repo,
                branch=branch,
                commit_title=commit_title,
                error_lines=error_lines,
                failure_detection=failure_detection,
                n_failed=n_failed,
                n_total=n_total,
            ),
            workflow or "",
            commit_message or "",
            " ".join(
                item.get("path", "") if isinstance(item, dict) else str(item)
                for item in mentioned_files[:8]
            ),
        ]
        similar = self.find_similar(" ".join(part for part in query_parts if part), k=k, min_sim=min_sim)
        return [
            {
                "run_id": case["run_id"],
                "repo": case["repo"],
                "workflow": case.get("workflow", ""),
                "category": case["category"],
                "similarity": round(score, 3),
                "gt_verdict": case.get("gt_verdict", "NO_DATA"),
            }
            for case, score in similar
        ]


# ─── prior computation ───────────────────────────────────────────────

def compute_retrieval_prior(
    similar_cases: List[Tuple[dict, float]],
    weight: float = PRIOR_WEIGHT,
) -> Dict[str, float]:
    """
    Build a weighted prior from similar past cases.
    Cases with MATCH ground truth verdict are weighted 2x.
    Cases with MISMATCH are weighted 0.5x (still included but less trusted).
    """
    uniform = 1.0 / len(CATEGORIES)

    if not similar_cases:
        return {cat: uniform for cat in CATEGORIES}

    verdict_weights = {
        "MATCH": 2.0,
        "PARTIAL": 1.2,
        "NO_DATA": 0.8,
        "NOT_SCORABLE": 0.8,
        "MISMATCH": 0.4,
    }

    category_weights = {cat: 0.0 for cat in CATEGORIES}
    total_weight = 0.0

    for case, sim in similar_cases:
        verdict = case.get("gt_verdict", "NO_DATA")
        trust = verdict_weights.get(verdict, 0.8)
        effective_weight = sim * trust
        cat = case["category"]
        if cat in category_weights:
            category_weights[cat] += effective_weight
        total_weight += effective_weight


    if total_weight > 0:
        for cat in category_weights:
            category_weights[cat] /= total_weight

    prior = {}
    for cat in CATEGORIES:
        prior[cat] = (1 - weight) * uniform + weight * category_weights[cat]

    total = sum(prior.values())
    return {k: v / total for k, v in prior.items()}

# ─── public API ──────────────────────────────────────────────────────

def get_retrieval_prior(
    repo: str,
    branch: str,
    commit_title: str,
    error_lines: List[str],
    failure_detection: str,
    n_failed: int,
    n_total: int,
    memory: CaseMemory,
    verbose: bool = False,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Main entry point. Returns (prior_distribution, similar_cases_info).

    prior_distribution: dict mapping category → probability (sums to 1)
    similar_cases_info: list of dicts with repo, category, similarity
    """
    query = case_to_text(
        repo=repo,
        branch=branch,
        commit_title=commit_title,
        error_lines=error_lines,
        failure_detection=failure_detection,
        n_failed=n_failed,
        n_total=n_total,
    )

    similar = memory.find_similar(query)

    if verbose and similar:
        print(f"  [retrieval] found {len(similar)} similar past cases:")
        for case, sim in similar:
            print(f"    {case['repo']:<35} {case['category']:<25} sim={sim:.3f}")

    prior = compute_retrieval_prior(similar)

    similar_info = [
        {
            "repo": c["repo"],
            "category": c["category"],
            "similarity": round(sim, 3),
            "gt_verdict": c.get("gt_verdict", "?"),
        }
        for c, sim in similar
    ]

    return prior, similar_info


def search_similar_cases(
    repo: str,
    workflow: str,
    branch: str,
    commit_title: str,
    commit_message: str,
    error_lines: List[str],
    mentioned_files: List[dict],
    failure_detection: str,
    n_failed: int,
    n_total: int,
    memory: CaseMemory,
    verbose: bool = False,
) -> List[dict]:
    """
    Find similar past failures.

    Routes to ChromaDB (APA-v) when USE_CHROMA=1, otherwise uses the
    existing SQLite+cosine CaseMemory path (APA baseline).
    """
    if USE_CHROMA:
        from src.apa.chroma_case_store import get_chroma_store
        store = get_chroma_store(path=CHROMA_PATH, client=memory.client)
        similar = store.find_similar_case_records(
            commit_title=commit_title,
            error_lines=error_lines,
            mentioned_files=mentioned_files,
        )
        if verbose and similar:
            print(f"  [retrieval/chroma] found {len(similar)} similar cases")
        return similar

    # ── Existing SQLite+cosine path (unchanged) ──────────────────────
    similar = memory.find_similar_case_records(
        repo=repo,
        workflow=workflow,
        branch=branch,
        commit_title=commit_title,
        commit_message=commit_message,
        error_lines=error_lines,
        mentioned_files=mentioned_files,
        failure_detection=failure_detection,
        n_failed=n_failed,
        n_total=n_total,
    )
    if verbose and similar:
        print(f"  [retrieval] found {len(similar)} similar cases from database")
    return similar


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import gzip

    RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")

    client = make_client()

    print("Loading case memory...")
    memory = CaseMemory(client)

    # Test on bcrypt — a case that should find similar infra failures
    print("\nQuerying for bcrypt-like case:")
    prior, similar = get_retrieval_prior(
        repo="pyca/bcrypt",
        branch="dependabot/github_actions/actions/checkout-4.1.0",
        commit_title="Bump actions/checkout from 3.6.0 to 4.1.0",
        error_lines=["GLIBC_2.27 not found", "operation was canceled"],
        failure_detection="job_level_fallback",
        n_failed=8,
        n_total=8,
        memory=memory,
        verbose=True,
    )

    print("\nResulting prior (vs uniform 0.100):")
    for cat, prob in sorted(prior.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 50)
        delta = prob - (1.0 / len(CATEGORIES))
        marker = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "  ")
        print(f"  {cat:<25} {prob:.3f}{marker}  {bar}")

    # Test on a case that looks like a code regression
    print("\n\nQuerying for code-regression-like case:")
    prior2, similar2 = get_retrieval_prior(
        repo="some/repo",
        branch="main",
        commit_title="Add new feature to payment processing",
        error_lines=["AssertionError: expected 200 got 500",
                     "FAILED tests/test_payment.py::test_checkout"],
        failure_detection="per_step_error",
        n_failed=1,
        n_total=3,
        memory=memory,
        verbose=True,
    )

    print("\nResulting prior:")
    for cat, prob in sorted(prior2.items(), key=lambda x: -x[1]):
        delta = prob - (1.0 / len(CATEGORIES))
        marker = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "  ")
        print(f"  {cat:<25} {prob:.3f}{marker}")

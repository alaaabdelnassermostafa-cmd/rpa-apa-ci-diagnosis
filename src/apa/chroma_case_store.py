# chroma_case_store.py
# ─────────────────────────────────────────────────────────────────────
# ChromaDB-backed vector store for CI/CD failure cases.
#
# Uses OpenAI text-embedding-3-small (1536-dim) for embeddings.
# Requires OPEN_AI_KEY in environment.
#
# Thesis architecture levels:
#   L1 — RPA:    no retrieval, uniform prior
#   L2 — APA:    LangGraph + EIG, uniform prior
#   L3 — APA+R:  APA + retrieval-augmented prior from this store
#
# The feedback loop:
#   diagnose case → judge verdict → upsert_case() → next case gets
#   a smarter prior because similar past cases are now in the store.
# ─────────────────────────────────────────────────────────────────────

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.apa.bayesian_tracker import CATEGORIES, get_informed_prior, update_prior_with_label

# ─── paths & constants ───────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_CHROMA_PATH = str(
    Path(os.environ.get("CHROMA_PATH", str(_BASE_DIR / "data" / "chroma")))
)

_COLLECTION_NAME = "ci_failure_cases"
_EMBED_MODEL = "text-embedding-3-small"

# Minimum cosine similarity to count as a useful neighbour
MIN_SIMILARITY = float(os.environ.get("CHROMA_MIN_SIMILARITY", "0.50"))

# How strongly the retrieval prior pulls beliefs away from uniform (0=ignore, 1=replace)
PRIOR_WEIGHT = 0.35

# Trust multipliers per judge verdict when computing the weighted prior
_VERDICT_TRUST = {
    "CORRECT": 2.0,
    "PARTIAL":  1.2,
    "NO_DATA":  0.8,
    "NOT_SCORABLE": 0.6,
    "WRONG":    0.3,
}


# ─── text representation ─────────────────────────────────────────────

def case_to_text(
    commit_title: str,
    error_lines: List[str],
    mentioned_files: Optional[List] = None,
    reasoning: str = "",
) -> str:
    """
    Build the short text string embedded for each failure case.

    Priority order for discriminative signal:
      1. APA reasoning  (rich, category-specific language — best signal)
      2. error lines    (what actually broke)
      3. commit title   (intent of the change)
      4. file basenames (ecosystem / manifest type)

    Reasoning text (when available) dominates because it already contains
    category-specific language produced by the LLM: e.g. "three .rs files
    modified, exit code 101, indicating code regression". This makes
    retrieval far more discriminative than shallow error codes alone.
    """
    title  = (commit_title or "").strip() or "no commit title"
    errors = " | ".join(line.strip() for line in error_lines[:5] if line.strip())
    errors = errors or "no error text"

    if reasoning and "Classification error" not in reasoning:
        # Reasoning is the richest signal — put it first
        text = f"reasoning: {reasoning.strip()[:400]}  commit: {title}  errors: {errors}"
    else:
        text = f"commit: {title}  errors: {errors}"

    if mentioned_files:
        tokens: List[str] = []
        for item in mentioned_files[:8]:
            raw = item.get("path", "") if isinstance(item, dict) else str(item)
            basename = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if basename and len(basename) > 1:
                tokens.append(basename)
        if tokens:
            seen: set = set()
            unique = [t for t in tokens if not (t in seen or seen.add(t))]  # type: ignore
            text += "  files: " + " ".join(unique)

    return text


# ─── core store ──────────────────────────────────────────────────────

class ChromaCaseStore:
    """
    ChromaDB-backed store of past CI/CD failure cases.

    Embeddings use all-MiniLM-L6-v2 locally via sentence-transformers.
    No external API calls needed — fully offline after first model download.

    Each stored document represents one diagnosed failure case with:
      - document text: case_to_text(commit_title, error_lines, files)
      - metadata: run_id, repo, workflow, category, gt_verdict, n_failed, n_total

    Public interface:
      find_similar()        → List[(metadata_dict, similarity_float)]
      compute_prior()       → Dict[category, probability]  (Bayesian prior)
      upsert_case()         → add/update one case (the feedback loop entry point)
      bootstrap_from_benchmark() → bulk-ingest from benchmark_final_eval.json
      count()               → int
    """

    def __init__(self, path: str = DEFAULT_CHROMA_PATH):
        api_key = os.environ.get("OPEN_AI_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI key not found. Set OPEN_AI_KEY in .env")
        self._embed_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=_EMBED_MODEL,
        )
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── basic ops ─────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def upsert_case(
        self,
        run_id: str,
        commit_title: str,
        error_lines: List[str],
        category: str,
        gt_verdict: str,
        repo: str = "",
        workflow: str = "",
        n_failed: int = 0,
        n_total: int = 1,
        mentioned_files: Optional[List] = None,
        reasoning: str = "",
    ) -> None:
        """
        Add or update one case in the store.

        Call this after the judge scores a case to close the feedback loop:
        future cases will find this one as a neighbour and get a better prior.

        gt_verdict should be one of: CORRECT, PARTIAL, WRONG, NOT_SCORABLE.
        Cases with WRONG verdict are stored but given low trust weight (0.3×)
        in compute_prior(), so they weakly push beliefs away from their category.

        reasoning: APA's own explanation text — the richest embedding signal.
        """
        if not run_id or not category:
            return

        text = case_to_text(commit_title, error_lines, mentioned_files, reasoning)
        self._collection.upsert(
            ids=[run_id],
            documents=[text],
            metadatas=[{
                "run_id":     run_id,
                "repo":       repo,
                "workflow":   workflow,
                "category":   category,
                "gt_verdict": gt_verdict,
                "n_failed":   n_failed,
                "n_total":    n_total,
            }],
        )

        # Online Dirichlet update of the global prior. Only CORRECT verdicts
        # (confirmed to match ground truth) are folded in, so the prior tracks
        # the true marginal category distribution rather than the model's guesses.
        if gt_verdict == "CORRECT":
            update_prior_with_label(category)

    # ── retrieval ─────────────────────────────────────────────────────

    def find_similar(
        self,
        commit_title: str,
        error_lines: List[str],
        mentioned_files: Optional[List] = None,
        k: int = 5,
        min_sim: float = MIN_SIMILARITY,
    ) -> List[Tuple[dict, float]]:
        """
        Find up to k most similar past cases above min_sim threshold.

        Returns list of (metadata_dict, similarity) sorted by similarity desc.
        similarity is cosine similarity in [0, 1].
        """
        if self._collection.count() == 0:
            return []

        query = case_to_text(commit_title, error_lines, mentioned_files)
        if not query.strip():
            return []

        # Over-fetch then filter — ChromaDB returns cosine distance (lower=closer)
        n_fetch = min(k * 3, self._collection.count())
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_fetch,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            print(f"  [chroma] query failed: {e}")
            return []

        out: List[Tuple[dict, float]] = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sim = 1.0 - dist  # cosine distance → cosine similarity
            if sim < min_sim:
                continue
            out.append((meta, round(sim, 4)))
            if len(out) >= k:
                break

        return out

    # ── prior computation ─────────────────────────────────────────────

    def compute_prior(
        self,
        commit_title: str,
        error_lines: List[str],
        mentioned_files: Optional[List] = None,
        rpa_prior: Optional[Dict[str, float]] = None,
        k: int = 5,
        min_sim: float = MIN_SIMILARITY,
        weight: float = PRIOR_WEIGHT,
        verbose: bool = False,
    ) -> Tuple[Dict[str, float], List[dict]]:
        """
        Build a retrieval-augmented Bayesian prior for a new case.

        Algorithm:
          1. Find k most similar past cases (cosine ANN over local embeddings)
          2. Weight each neighbour by: similarity × verdict_trust
             (CORRECT cases contribute 2×, WRONG cases contribute 0.3×)
          3. Blend with RPA prior (falls back to uniform if rpa_prior not given):
             prior[c] = (1 - weight) × rpa_prior[c] + weight × neighbour_votes[c]

        Returns:
          prior_dict: category → probability (sums to 1.0)
          similar_cases: list of dicts for logging/display
        """
        informed_prior = get_informed_prior()
        base = rpa_prior if rpa_prior else dict(informed_prior)

        neighbours = self.find_similar(
            commit_title, error_lines, mentioned_files, k=k, min_sim=min_sim
        )

        if verbose and neighbours:
            print(f"  [chroma] {len(neighbours)} neighbours:")
            for m, s in neighbours:
                print(f"    {m['repo']:<35} {m['category']:<25} sim={s:.3f} verdict={m['gt_verdict']}")

        if not neighbours:
            return {cat: base.get(cat, informed_prior[cat]) for cat in CATEGORIES}, []

        # Weighted vote per category
        cat_weights = {cat: 0.0 for cat in CATEGORIES}
        total_w = 0.0
        for meta, sim in neighbours:
            trust = _VERDICT_TRUST.get(meta.get("gt_verdict", "NO_DATA"), 0.8)
            w = sim * trust
            cat = meta.get("category", "")
            if cat in cat_weights:
                cat_weights[cat] += w
            total_w += w

        if total_w > 0:
            cat_weights = {c: v / total_w for c, v in cat_weights.items()}

        # Blend retrieval signal with base prior
        prior = {
            cat: (1.0 - weight) * base.get(cat, informed_prior[cat]) + weight * cat_weights[cat]
            for cat in CATEGORIES
        }
        total = sum(prior.values())
        prior = {c: v / total for c, v in prior.items()}

        similar_info = [
            {
                "run_id":     m.get("run_id", ""),
                "repo":       m.get("repo", ""),
                "category":   m.get("category", ""),
                "similarity": s,
                "gt_verdict": m.get("gt_verdict", "NO_DATA"),
            }
            for m, s in neighbours
        ]

        return prior, similar_info

    # ── bulk bootstrap ────────────────────────────────────────────────

    def bootstrap_from_benchmark(
        self,
        benchmark_path: Optional[Path] = None,
        verdicts: Tuple[str, ...] = ("CORRECT", "PARTIAL"),
        verbose: bool = True,
    ) -> int:
        """
        Ingest cases from benchmark_final_eval.json into the store.

        Only inserts cases where the APA judge verdict is in `verdicts`.
        Idempotent — already-present run_ids are skipped.

        Returns number of newly inserted cases.
        """
        bp = benchmark_path or (_BASE_DIR / "data" / "benchmark_final_eval.json")
        if not Path(bp).exists():
            if verbose:
                print(f"  [chroma] benchmark file not found: {bp}")
            return 0

        with open(bp, "r", encoding="utf-8") as f:
            data = json.load(f)

        existing_ids: set = set()
        if self._collection.count() > 0:
            existing_ids = set(self._collection.get(include=[])["ids"])

        inserted = 0
        skipped  = 0

        for rec in data:
            apa = rec.get("apa_eig", {})
            pred = apa.get("prediction") or {}
            judge = apa.get("judge", {})

            verdict  = judge.get("verdict", "")
            category = pred.get("category", "")
            run_id   = rec.get("run_id", "")

            if verdict not in verdicts or not category or not run_id:
                skipped += 1
                continue
            if run_id in existing_ids:
                skipped += 1
                continue

            reasoning = pred.get("reasoning", "") or ""
            self.upsert_case(
                run_id=run_id,
                commit_title=rec.get("commit", ""),
                error_lines=rec.get("error_lines", []),
                category=category,
                gt_verdict=verdict,
                repo=rec.get("repo", ""),
                workflow="",
                n_failed=0,
                n_total=1,
                reasoning=reasoning,
            )
            existing_ids.add(run_id)
            inserted += 1

        if verbose:
            print(f"  [chroma/benchmark] inserted={inserted} skipped={skipped} "
                  f"total={self._collection.count()}")
        return inserted

    def bootstrap_from_eval_file(
        self,
        eval_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> int:
        """
        Ingest cases from honest_eval_results.json (legacy format).
        Kept for backwards compatibility with the old bootstrap_chroma.py.
        """
        ep = eval_path or (_BASE_DIR / "data" / "honest_eval_results.json")
        if not Path(ep).exists():
            if verbose:
                print(f"  [chroma] eval file not found: {ep}")
            return 0

        with open(ep, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

        existing_ids: set = set()
        if self._collection.count() > 0:
            existing_ids = set(self._collection.get(include=[])["ids"])

        inserted = 0
        skipped  = 0

        for result in eval_results:
            intake = result.get("intake", {})
            cl     = result.get("classification", {})
            gt     = result.get("ground_truth", {})

            run_id   = str(intake.get("run_id", ""))
            category = cl.get("category", "")

            if not run_id or not category:
                skipped += 1
                continue
            if run_id in existing_ids:
                skipped += 1
                continue

            extraction = result.get("extraction", {})
            error_lines = extraction.get("sample_error_lines", [])
            mentioned   = extraction.get("mentioned_files", [])

            self.upsert_case(
                run_id=run_id,
                commit_title=intake.get("commit_title", ""),
                error_lines=error_lines,
                category=category,
                gt_verdict=gt.get("match_verdict", "NO_DATA"),
                repo=intake.get("repo", ""),
                workflow=intake.get("workflow", ""),
                n_failed=intake.get("failed_jobs_count", 0),
                n_total=intake.get("n_jobs", 1),
                mentioned_files=mentioned,
            )
            existing_ids.add(run_id)
            inserted += 1

        if verbose:
            print(f"  [chroma/eval] inserted={inserted} skipped={skipped} "
                  f"total={self._collection.count()}")
        return inserted


# ─── process-level singleton ─────────────────────────────────────────

_store: Optional[ChromaCaseStore] = None


def get_chroma_store(path: str = DEFAULT_CHROMA_PATH, **_kwargs) -> ChromaCaseStore:
    """
    Return the process-global ChromaCaseStore, initialising on first call.
    Extra kwargs are accepted and ignored for backwards compatibility.
    """
    global _store
    if _store is None:
        _store = ChromaCaseStore(path=path or DEFAULT_CHROMA_PATH)
    return _store

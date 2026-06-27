# APA: Agentic CI-Failure Diagnosis Framework

APA turns a single failed CI run into a structured diagnosis: a failure **category**, a
**confidence**, the **reasoning**, and a concrete **fix recommendation**, plus the
investigation trace (which tools were called and how the belief evolved). It is the
deployable form of the three-level system evaluated in the thesis.

## Pipeline

```
raw failed run ─▶ intake (normalise) ─▶ deterministic preprocessing
                                          (log extractor, file-path extractor)
                ─▶ Bayesian belief tracker (9 categories)
                ─▶ APA agent loop  (EIG planner ▸ tools ▸ likelihood update)
                ─▶ classification + fix recommendation
```

## Install / configure

```bash
pip install -r requirements.txt
# LLM key for the diagnostic models (DeepSeek by default):
export DEEPSEEK_API_KEY=sk-...
# optional: stronger classifier (matches the thesis), and live evidence gathering
export CI_AGENT_CLASSIFY_MODEL=deepseek-reasoner
export GITHUB_TOKEN=ghp_...
```

## Python API

```python
from apa.framework import diagnose, load_case, diagnose_raw

# (a) a stored, curated case — for demos and reproducing the thesis
case = load_case("brooooooklyn/image")
dx = diagnose(case)
print(dx.pretty())
print(dx.category, dx.confidence, dx.recommended_action)

# (b) a brand-new failure (a raw GitHub Actions / GHALogs record)
dx = diagnose_raw(raw_github_actions_record)
```

`diagnose()` returns a `Diagnosis` with: `category, confidence, severity, reasoning,
recommended_action, error_lines, implicated_files, tools_used, steps, beliefs`, plus
`to_dict()` and `pretty()`.

## Command line

```bash
python -m apa list                              # show some available stored cases
python -m apa diagnose "brooooooklyn/image"     # diagnose a stored case
python -m apa diagnose "<repo or run_id>" --json
python -m apa diagnose --file run.json          # diagnose a raw record
python -m apa diagnose "<query>" --no-fix       # classification only
```

(Run with `PYTHONPATH=".;src"` on Windows / `PYTHONPATH=.:src` on Unix, or install the
package, so both `apa` and the top-level helpers resolve.)

## Levels

- **L1 (RPA)** — deterministic signal battery only (baseline).
- **L2 (APA)** — the agentic loop above; this is what `diagnose()` runs.
- **L3 (APA + retrieval)** — adds a prior from a ChromaDB store of past diagnoses
  (`apa.chroma_case_store.ChromaCaseStore`); enable by populating the store and passing
  retrieved neighbours into the loop.

## Cost

A full diagnosis is ~5 model calls and ~10k tokens, about **USD 0.006 per case**, because
the deterministic extractor keeps the (million-character) raw log out of the token budget.

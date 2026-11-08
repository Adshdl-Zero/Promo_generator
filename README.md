# Kuku promo generation pipeline

## Modules and commands

All module-specific command entrypoints live in `tests/cmd/`; the root `main.py` dispatches to them. Run commands from the repository root.

1. **Data ingestion and screenplay parsing**

	```powershell
	python .\main.py ingest
	```

	Reads screenplay files from `data/` and writes `outputs/episodes/parsed_episodes.json`. To choose another source or output, run `python .\tests\cmd\ingest_episodes.py --help`.

2. **Narrative understanding and dramatic moment detection**

	```powershell
	python .\main.py analyze-scenes
	```

	Reads `outputs/episodes/parsed_episodes.json`; writes `outputs/scenes/scene_analysis.json`.

3. **Continuous promo candidate generation**

	```powershell
	python .\main.py generate-candidates
	```

	Reads the parsed episodes and scene analysis; writes `outputs/candidates/promo_candidates.json`. Module 2 should finish first. By default, up to 25 candidates are evaluated; use `--max-candidates N` to override.

4. **Promo ranking and selection**

	```powershell
	python .\main.py rank
	```

	Writes ranked JSON/CSV and score cache under `outputs/promo_ranking_and_selection/`.

5. **Quality evaluation and validation**

	```powershell
	python .\main.py quality
	```

	Writes quality reports/cache under `outputs/quality_evaluation_and_validation/`.

6. **Multi-segment promo generation**

	```powershell
	python .\main.py multi-segment --no-cache
	```

	Writes `outputs/candidates/multi_segment_candidates.json`, `outputs/multi_segment_promo_generation/multi_segment_promos.csv`, and `outputs/multi_segment_promo_generation/run_metadata.json`.

7. **Streamlit user interface**

	```powershell
	python .\main.py ui
	```

	This launches the app at `tests/cmd/app.py`. You can also run `streamlit run .\tests\cmd\app.py`.

	From the **Run pipeline** page, upload one or more `.txt`, `.pdf`, or `.docx` screenplay files and choose **Save input and run all modules**. Inputs are retained under `data/uploads/<run-id>/`; each run's intermediate files and reports are isolated under `outputs/runs/<run-id>/`. The upload flow runs Modules 1–6 and 8 against only those uploaded files; Module 7 is the UI itself.

8. **Human evaluation and quality assurance**

	```powershell
	python .\main.py evaluate
	```

	Writes reports under `outputs/evaluation/`. Module 8 does not call an LLM; human/AI agreement metrics require actual reviewer submissions.

## Configuration

Set provider settings in `.env` or edit `config.yaml`. When `.env` is present, its values override inherited process environment values so a long-lived terminal cannot keep using a stale API key.

```env
LLM_PROVIDER=mock
LLM_MODEL=mock-model
LLM_API_KEY=
LLM_TEMPERATURE=0.2
LLM_MAX_RETRIES=3
LLM_TIMEOUT=60
```

Supported providers are `mock`, `gemini`, `openai`, and `openai_compatible`.

The provider configured in `.env` or `config.yaml` is used when `--provider` is omitted. Use `--provider mock` only for offline validation and tests; for a real API-backed run, omit that override so the configured provider and `LLM_API_KEY` from `.env` are used.

# Promo ranking and quality evaluation (Modules 4 and 5)

Module 4 consumes Module 3 candidates and adds weighted raw ranking plus diversity-aware selection. The defaults in `config.yaml` use the assignment weights, select at most 20 promos, allow at most 3 per episode, and use configurable overlap and score thresholds. If required Module 4 scores are absent, it uses the configured provider and stores results in a persistent cache.

Module 5 independently validates technical correctness and evaluates only selected promos (up to `quality.maximum_narrative_evaluations`). It writes technical and narrative findings separately. Its concurrency defaults to 1 and narrative API attempts default to 1 per promo to avoid duplicate calls on malformed responses; increase `quality.max_narrative_attempts` only if retries are desired. Transient non-rate-limit retries use bounded exponential backoff and provider `Retry-After` when supplied. When `outputs/candidates/promo_candidates.json` is empty, the reports correctly contain zero records and no API calls are made.

Run Modules 4–6 in order using the project CLI:

```powershell
python .\main.py run-all
```

You can pass provider overrides such as `--provider mock` to module commands for offline testing. The normal provider is selected from `.env` or `config.yaml`.

Module 8 evaluates only final promos that Module 4 marked `selected` and Module 5 accepted (`PASS` or `PASS_WITH_WARNINGS`, with no technical failure). It reads the ranked output, Module 5 quality report, parsed episodes, saved human reviews, and Module 6 run metadata; unranked Module 6 candidates are not added as finalists. Human reviews and Module 6 metadata are optional when absent. It writes metrics JSON, evaluation CSV, error analysis JSON, and a Markdown summary under `outputs/evaluation/`.

## Project architecture overview

1. Module 1: Data ingestion and screenplay parsing (`src/ingestion.py`)
2. Module 2: Narrative understanding and dramatic moment detection (`src/analysis/`)
3. Module 3: Continuous promo candidate generation (`src/generation/`)
4. Module 4: Promo ranking and selection (`src/ranking/`)
5. Module 5: Quality evaluation and validation (`src/quality/`)
6. Module 6: Multi-segment promo generation (`src/generation/multi_segment_generator.py`)
7. Module 7: Streamlit user interface (`tests/cmd/app.py`)
8. Module 8: Human evaluation and quality assurance (`src/evaluation/system_evaluation.py`)

Generated and processed episode, scene, candidate, and evaluation data is stored under `outputs/` (`outputs/episodes/`, `outputs/scenes/`, `outputs/candidates/`, and `outputs/evaluation/`). Module reports are grouped in `outputs/promo_ranking_and_selection/`, `outputs/quality_evaluation_and_validation/`, and `outputs/multi_segment_promo_generation/`.

## Installation and environment notes

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For real LLM calls, set `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` in `.env` (or configure them in `config.yaml`) without committing secrets. Omit `--provider` to use that configured provider; use `--provider mock` only when intentionally running offline.

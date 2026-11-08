"""Module 7: Streamlit user interface for the Kuku promo workflow."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from io import BytesIO
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = ROOT / "data" / "uploads"
RUNS_ROOT = ROOT / "outputs" / "runs"
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".pdf", ".docx"}
PIPELINE_MODULE_ORDER = (1, 2, 3, 6, 4, 5, 8)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime dependency is optional until installed.
    st = None

from src.config import get_settings
from src.evaluation.system_evaluation import compute_system_evaluation, load_human_reviews, save_human_review

def _read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_list(path: str | Path, default: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    value = _read_json(path, default)
    if value is None:
        return default or []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("reviews", []) if path.name.endswith("reviews.json") else []
    return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_upload(uploaded_file: Any) -> tuple[str, bytes]:
    """Validate the uploaded screenplay's extension, MIME type, and file structure."""
    original_name = str(getattr(uploaded_file, "name", "")).replace("\\", "/")
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", original_name.rsplit("/", 1)[-1]).strip(" .")
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported upload {filename or '(unnamed)'!r}; accepted types: {allowed}.")
    content = uploaded_file.getvalue()
    if not content or not content.strip():
        raise ValueError(f"Uploaded file {filename!r} is empty.")

    provided_mime = str(getattr(uploaded_file, "type", "") or "").split(";", 1)[0].lower()
    accepted_mimes = {
        ".txt": {"text/plain", "application/octet-stream"},
        ".pdf": {"application/pdf", "application/octet-stream"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
        },
    }
    if provided_mime and provided_mime not in accepted_mimes[suffix]:
        raise ValueError(f"Uploaded file {filename!r} has an incompatible MIME type ({provided_mime}).")

    if suffix == ".txt":
        if b"\x00" in content:
            raise ValueError(f"Uploaded TXT file {filename!r} is not readable text.")
        decoded = False
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                content.decode(encoding)
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            raise ValueError(f"Uploaded TXT file {filename!r} is not readable text.")
    elif suffix == ".pdf":
        if not content.lstrip().startswith(b"%PDF-"):
            raise ValueError(f"Uploaded PDF file {filename!r} is invalid or corrupted.")
        try:
            import fitz
            with fitz.open(stream=content, filetype="pdf") as document:
                if document.is_closed or document.page_count < 1:
                    raise ValueError("PDF contains no readable pages.")
                document[0].get_text("text")
        except Exception as exc:
            raise ValueError(f"Uploaded PDF file {filename!r} is invalid or corrupted: {exc}") from exc
    else:
        try:
            with zipfile.ZipFile(BytesIO(content)) as document:
                if "word/document.xml" not in document.namelist():
                    raise ValueError("DOCX document body is missing.")
                if document.testzip() is not None:
                    raise ValueError("DOCX archive failed its integrity check.")
        except Exception as exc:
            raise ValueError(f"Uploaded DOCX file {filename!r} is invalid or corrupted: {exc}") from exc
    return filename, content


@st.cache_data(show_spinner=False) if st is not None else (lambda func: func)
def _cached_read_json(path: str | Path, default: Any = None) -> Any:
    return _read_json(path, default)


@st.cache_data(show_spinner=False) if st is not None else (lambda func: func)
def _load_dashboard_stats() -> dict[str, Any]:
    episodes = _load_list(ROOT / "outputs/episodes/parsed_episodes.json")
    continuous = _load_list(ROOT / "outputs/candidates/promo_candidates.json")
    multi_segment = _load_list(ROOT / "outputs/candidates/multi_segment_candidates.json")
    ranked = _load_list(ROOT / "outputs/promo_ranking_and_selection/ranked_promos.json")
    quality_report = _read_json(ROOT / "outputs/quality_evaluation_and_validation/quality_report.json", {})
    failures = _read_json(ROOT / "outputs/quality_evaluation_and_validation/failure_analysis.json", {})
    human_reviews = load_human_reviews(ROOT / "outputs/evaluation/human_reviews.json")
    module6_metadata = _read_json(ROOT / "outputs/multi_segment_promo_generation/run_metadata.json", {})

    def count_selected(items: list[dict[str, Any]]) -> int:
        return sum(1 for item in items if str(item.get("selection_status", "")).lower() == "selected")

    quality_rows = quality_report.get("candidate_results", []) if isinstance(quality_report, dict) else []
    pass_count = sum(1 for item in quality_rows if str(item.get("quality_decision", "")).upper() == "PASS")
    warning_count = sum(1 for item in quality_rows if str(item.get("quality_decision", "")).upper() == "PASS_WITH_WARNINGS")
    fail_count = sum(1 for item in quality_rows if str(item.get("quality_decision", "")).upper() == "FAIL")
    score_values: list[float] = []
    for item in ranked:
        try:
            score_values.append(float(item.get("raw_score", 0) or 0))
        except (TypeError, ValueError):
            pass
    avg_score = round(sum(score_values) / len(score_values), 2) if score_values else None

    return {
        "episode_count": len(episodes),
        "continuous_count": len(continuous),
        "multi_segment_count": len(multi_segment),
        "ranked_count": len(ranked),
        "selected_count": count_selected(ranked),
        "pass_count": pass_count,
        "warning_count": warning_count,
        "fail_count": fail_count,
        "average_score": avg_score,
        "processing_status": "ready" if quality_report or ranked or multi_segment else "missing_data",
        "quality_rows": quality_rows,
        "failures": failures,
        "human_reviews": human_reviews,
        "module6_metadata": module6_metadata,
    }


def _build_quality_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in rows:
        label = str(item.get("quality_decision", "UNKNOWN")).strip() or "UNKNOWN"
        counts[label] += 1
    return dict(counts)


def _build_score_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    bucket_counts = Counter()
    for item in rows:
        score = item.get("raw_score")
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue
        bucket = "0-49" if score_val < 50 else "50-69" if score_val < 70 else "70-84" if score_val < 85 else "85+"
        bucket_counts[bucket] += 1
    return dict(bucket_counts)


def _run_pipeline_command(command: str, provider: str | None) -> tuple[int, str]:
    settings = get_settings()
    base = [sys.executable]
    if command == "Rank/quality/multi-segment (existing data)":
        active_provider = provider or settings.llm.provider
        command_sets = [
            [
                str(ROOT / "tests/cmd/run_promo_generation.py"),
                "--provider", active_provider,
                "--input", "outputs/episodes/parsed_episodes.json",
                "--analysis", "outputs/scenes/scene_analysis.json",
                "--output", "outputs/candidates/promo_candidates.json",
            ],
            [
                str(ROOT / "tests/cmd/run_module6_multi_segment.py"),
                "--provider", active_provider,
                "--episodes", "outputs/episodes/parsed_episodes.json",
                "--analysis", "outputs/scenes/scene_analysis.json",
                "--candidates", "outputs/candidates/promo_candidates.json",
                "--ranked", "outputs/candidates/promo_candidates.json",
                "--output", "outputs/candidates/multi_segment_candidates.json",
                "--csv", "outputs/multi_segment_promo_generation/multi_segment_promos.csv",
                "--metadata", "outputs/multi_segment_promo_generation/run_metadata.json",
            ],
            [
                str(ROOT / "tests/cmd/run_module4_rank.py"),
                "--provider", active_provider,
                "--multi-segment", "outputs/candidates/multi_segment_candidates.json",
                "--output", "outputs/promo_ranking_and_selection/ranked_promos.json",
                "--csv", "outputs/promo_ranking_and_selection/ranked_promos.csv",
            ],
            [
                str(ROOT / "tests/cmd/run_module5_quality.py"),
                "--provider", active_provider,
                "--ranked", "outputs/promo_ranking_and_selection/ranked_promos.json",
                "--multi-segment", "outputs/candidates/multi_segment_candidates.json",
                "--output", "outputs/quality_evaluation_and_validation/quality_report.json",
                "--csv", "outputs/quality_evaluation_and_validation/quality_report.csv",
                "--failures", "outputs/quality_evaluation_and_validation/failure_analysis.json",
            ],
        ]
        outputs: list[str] = []
        for stage_args in command_sets:
            exit_code, output = _run_script(base + stage_args)
            outputs.append(output)
            if exit_code != 0:
                return exit_code, "\n".join(outputs)
        return 0, "\n".join(outputs)
    if command == "Rank only":
        return _run_script(base + [str(ROOT / "tests/cmd/run_module4_rank.py"), "--provider", provider or settings.llm.provider])
    if command == "Quality only":
        return _run_script(base + [str(ROOT / "tests/cmd/run_module5_quality.py"), "--provider", provider or settings.llm.provider])
    if command == "Multi-segment only":
        return _run_script(base + [str(ROOT / "tests/cmd/run_module6_multi_segment.py"), "--provider", provider or settings.llm.provider])
    return 1, "Unsupported command selection."


def _run_script(argv: list[str]) -> tuple[int, str]:
    child_env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _store_uploaded_files(
    uploaded_files: list[Any],
    upload_root: str | Path = UPLOAD_ROOT,
) -> tuple[str, Path]:
    """Store validated screenplay uploads in a unique directory under data/uploads."""
    if not uploaded_files:
        raise ValueError("Upload at least one TXT, PDF, or DOCX screenplay file.")

    prepared: list[tuple[str, bytes]] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename, content = _validate_upload(uploaded_file)
        prepared.append((f"{index:03d}_{filename}", content))

    run_id = uuid4().hex[:12]
    destination = Path(upload_root) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for filename, content in prepared:
        (destination / filename).write_bytes(content)
    return run_id, destination


def _build_uploaded_pipeline_commands(
    input_directory: str | Path,
    run_id: str,
    provider: str | None,
) -> tuple[Path, list[tuple[str, list[str]]]]:
    """Create isolated commands for one run in dependency order: 1, 2, 3, 6, 4, 5, 8."""
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise ValueError("Run ID contains unsupported characters.")

    output_root = RUNS_ROOT / run_id
    episodes = output_root / "episodes" / "parsed_episodes.json"
    analysis = output_root / "scenes" / "scene_analysis.json"
    continuous = output_root / "candidates" / "promo_candidates.json"
    multi_segment = output_root / "candidates" / "multi_segment_candidates.json"
    ranked = output_root / "promo_ranking_and_selection" / "ranked_promos.json"
    ranked_csv = output_root / "promo_ranking_and_selection" / "ranked_promos.csv"
    ranking_cache = output_root / "promo_ranking_and_selection" / "ranking_score_cache.json"
    quality = output_root / "quality_evaluation_and_validation" / "quality_report.json"
    quality_csv = output_root / "quality_evaluation_and_validation" / "quality_report.csv"
    failures = output_root / "quality_evaluation_and_validation" / "failure_analysis.json"
    quality_cache = output_root / "quality_evaluation_and_validation" / "quality_evaluation_cache.json"
    multi_dir = output_root / "multi_segment_promo_generation"
    evaluation_dir = output_root / "evaluation"
    provider_args = ["--provider", provider or get_settings().llm.provider]
    python = sys.executable
    command_dir = ROOT / "tests" / "cmd"

    commands = [
        (
            "Module 1 — Data ingestion and screenplay parsing",
            [python, str(command_dir / "ingest_episodes.py"), str(Path(input_directory)), "--output", str(episodes)],
        ),
        (
            "Module 2 — Narrative understanding and dramatic moment detection",
            [python, str(command_dir / "run_scene_analysis.py"), "--input", str(episodes), "--output", str(analysis), *provider_args],
        ),
        (
            "Module 3 — Continuous promo candidate generation",
            [python, str(command_dir / "run_promo_generation.py"), "--input", str(episodes), "--analysis", str(analysis), "--output", str(continuous), *provider_args],
        ),
        (
            "Module 6 — Multi-segment promo generation",
            [
                python, str(command_dir / "run_module6_multi_segment.py"),
                "--episodes", str(episodes), "--analysis", str(analysis), "--candidates", str(continuous),
                "--ranked", str(ranked), "--output", str(multi_segment),
                "--csv", str(multi_dir / "multi_segment_promos.csv"),
                "--metadata", str(multi_dir / "run_metadata.json"),
                "--cache", str(multi_dir / "evaluation_cache.json"),
                "--checkpoint", str(multi_dir / "checkpoint.json"), *provider_args,
            ],
        ),
        (
            "Module 4 — Promo ranking and selection",
            [
                python, str(command_dir / "run_module4_rank.py"), "--input", str(continuous),
                "--multi-segment", str(multi_segment), "--episodes", str(episodes), "--analysis", str(analysis),
                "--output", str(ranked), "--csv", str(ranked_csv), "--cache", str(ranking_cache), *provider_args,
            ],
        ),
        (
            "Module 5 — Quality evaluation and validation",
            [
                python, str(command_dir / "run_module5_quality.py"), "--ranked", str(ranked),
                "--candidates", str(continuous), "--multi-segment", str(multi_segment), "--episodes", str(episodes),
                "--output", str(quality), "--csv", str(quality_csv), "--failures", str(failures),
                "--cache", str(quality_cache), *provider_args,
            ],
        ),
        (
            "Module 8 — Human evaluation and quality assurance",
            [
                python, str(command_dir / "run_module8_evaluation.py"), "--ranked", str(ranked),
                "--multi-segment", str(multi_segment), "--quality", str(quality), "--episodes", str(episodes),
                "--reviews", str(evaluation_dir / "human_reviews.json"),
                "--module6-metadata", str(multi_dir / "run_metadata.json"),
                "--metrics-output", str(evaluation_dir / "evaluation_metrics.json"),
                "--report-output", str(evaluation_dir / "evaluation_report.csv"),
                "--errors-output", str(evaluation_dir / "error_analysis.json"),
                "--summary-output", str(evaluation_dir / "evaluation_summary.md"),
            ],
        ),
    ]
    return output_root, commands


def _execute_uploaded_pipeline(
    input_directory: str | Path,
    run_id: str,
    provider: str | None,
    *,
    command_runner: Any = None,
    on_step: Any = None,
    start_at_module: int = 1,
) -> tuple[int, str, Path]:
    output_root, commands = _build_uploaded_pipeline_commands(input_directory, run_id, provider)
    start_index = next(
        (index for index, (label, _) in enumerate(commands) if label.startswith(f"Module {start_at_module} ")),
        None,
    )
    if start_index is None:
        raise ValueError(f"Module {start_at_module} is not part of the uploaded pipeline.")
    runner = command_runner or _run_script
    logs: list[str] = []
    for label, command in commands[start_index:]:
        if on_step:
            on_step(label, "running")
        try:
            exit_code, output = runner(command)
        except Exception as exc:
            exit_code, output = 1, str(exc)
        logs.append(f"{label}\n{output}".rstrip())
        if exit_code != 0:
            if on_step:
                on_step(label, "error")
            return exit_code, "\n\n".join(logs), output_root
        if on_step:
            on_step(label, "complete")
    return 0, "\n\n".join(logs), output_root


def _render_dashboard() -> None:
    stats = _load_dashboard_stats()
    st.title("Dashboard")
    cols = st.columns(6)
    metrics = [
        ("Episodes processed", stats["episode_count"]),
        ("Continuous candidates", stats["continuous_count"]),
        ("Multi-segment candidates", stats["multi_segment_count"]),
        ("Ranked promos", stats["ranked_count"]),
        ("Selected promos", stats["selected_count"]),
        ("Average promo score", f"{stats['average_score']:.2f}" if stats["average_score"] is not None else "N/A"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    st.subheader("Quality summary")
    q_cols = st.columns(4)
    q_cols[0].metric("PASS", stats["pass_count"])
    q_cols[1].metric("PASS_WITH_WARNINGS", stats["warning_count"])
    q_cols[2].metric("FAIL", stats["fail_count"])
    q_cols[3].metric("Processing status", stats["processing_status"])

    ranked_rows = _load_list(ROOT / "outputs/promo_ranking_and_selection/ranked_promos.json")
    if ranked_rows:
        score_chart = _build_score_distribution(ranked_rows)
        if score_chart:
            st.subheader("Promo score distribution")
            st.bar_chart(score_chart)

        counts_by_episode = Counter()
        for item in ranked_rows:
            episode_ids = item.get("episode_ids") or []
            for episode_id in episode_ids:
                counts_by_episode[str(episode_id)] += 1
        if counts_by_episode:
            st.subheader("Candidate counts by episode")
            st.bar_chart(dict(counts_by_episode))

    quality_rows = stats["quality_rows"]
    if quality_rows:
        dist = _build_quality_distribution(quality_rows)
        st.subheader("Quality categories")
        st.bar_chart(dist)

    failure_items = stats["failures"].get("failure_frequency_by_category", {}) if isinstance(stats["failures"], dict) else {}
    if failure_items:
        st.subheader("Failure category distribution")
        st.bar_chart(failure_items)
    elif not quality_rows:
        st.info("No quality output file exists yet. Run the ranking or quality pipeline to populate this dashboard.")


def _render_pipeline_page() -> None:
    st.title("Run pipeline")
    settings = get_settings()
    provider = settings.llm.provider
    st.subheader("Add screenplay input and run the pipeline")
    st.caption("Upload TXT, PDF, or DOCX scripts. Files are stored under data/uploads/; each run writes isolated artifacts under outputs/runs/.")
    st.caption("Execution order: " + " → ".join(str(module) for module in PIPELINE_MODULE_ORDER))
    uploaded_files = st.file_uploader(
        "Screenplay files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="pipeline_screenplay_uploads",
    )
    if st.button("Start processing", type="primary", disabled=not uploaded_files, key="run_uploaded_pipeline"):
        try:
            run_id, input_directory = _store_uploaded_files(uploaded_files)
        except (OSError, ValueError) as exc:
            st.error(f"Could not save uploaded input: {exc}")
        else:
            st.write(f"Saved {len(uploaded_files)} file(s) to {input_directory.relative_to(ROOT)}")
            with st.status("Starting the uploaded-data pipeline…", expanded=True) as progress:
                def update_step(label: str, state: str) -> None:
                    if state == "running":
                        progress.update(label=label, state="running")
                    elif state == "complete":
                        progress.update(label=f"{label} complete", state="running")
                    else:
                        progress.update(label=f"{label} failed", state="error")

                exit_code, pipeline_output, run_output = _execute_uploaded_pipeline(
                    input_directory,
                    run_id,
                    provider,
                    on_step=update_step,
                )
                progress.update(
                    label="Uploaded-data pipeline complete" if exit_code == 0 else "Uploaded-data pipeline stopped on error",
                    state="complete" if exit_code == 0 else "error",
                )
            with st.expander("Pipeline logs", expanded=exit_code != 0):
                st.code(pipeline_output or "No console output returned.")
            if exit_code == 0:
                st.success(
                    "Modules "
                    + " → ".join(str(module) for module in PIPELINE_MODULE_ORDER)
                    + " completed for the uploaded input."
                )
                st.write(f"Run outputs: {run_output.relative_to(ROOT)}")
                metrics = _read_json(run_output / "evaluation" / "evaluation_metrics.json", {})
                if metrics:
                    st.metric("Candidates evaluated", metrics.get("generated_candidate_count", 0))
                    st.metric("Human-reviewed promos", metrics.get("reviewed_promo_count", 0))
            else:
                st.error(f"Pipeline stopped with exit code {exit_code}. Uploaded files and completed stage outputs were retained.")

    st.divider()
    st.subheader("Run existing outputs")
    command = st.selectbox("Action", ["Rank/quality/multi-segment (existing data)", "Rank only", "Quality only", "Multi-segment only"])
    if st.button("Start"):
        with st.spinner("Running pipeline..."):
            code, output = _run_pipeline_command(command, provider)
        st.code(output or "No console output returned.")
        if code == 0:
            st.success("Pipeline run completed successfully.")
        else:
            st.error(f"Pipeline exited with code {code}.")

    st.caption("Provider and model are read from the current configuration.")


def _render_ranked_promos() -> None:
    st.title("Ranked promos")
    rows = _load_list(ROOT / "outputs/promo_ranking_and_selection/ranked_promos.json")
    if not rows:
        st.info("No ranked promos exist yet. Run the ranking pipeline to generate them.")
        return

    episode_filter = st.selectbox("Episode", ["All", *sorted({str(value) for row in rows for value in (row.get("episode_ids") or [])})])
    type_filter = st.selectbox("Promo type", ["All", "continuous", "multi_segment"])
    quality_filter = st.selectbox("Quality category", ["All", *sorted({str(row.get("quality_category", "")) for row in rows if row.get("quality_category")})])
    score_range = st.slider("Minimum score", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    filtered = []
    for row in rows:
        episode_ids = row.get("episode_ids") or []
        if episode_filter != "All" and str(episode_filter) not in {str(item) for item in episode_ids}:
            continue
        if type_filter != "All" and str(row.get("promo_type", "continuous")) != type_filter:
            continue
        if quality_filter != "All" and str(row.get("quality_category", "")) != quality_filter:
            continue
        try:
            raw_score = float(row.get("raw_score", 0) or 0)
        except (TypeError, ValueError):
            raw_score = 0.0
        if raw_score < score_range:
            continue
        filtered.append(row)

    filtered.sort(key=lambda row: float(row.get("raw_score", 0) or 0), reverse=True)
    if not filtered:
        st.warning("No promos matched the selected filters.")
        return

    for row in filtered:
        with st.expander(f"#{row.get('rank', '?')} • {row.get('promo_id', 'unknown')} • {row.get('quality_category', 'unknown')} ({row.get('raw_score', 'n/a')})"):
            st.write(f"Promo type: {row.get('promo_type', 'continuous')}")
            st.write(f"Duration: {row.get('total_duration', 'unknown')}s")
            st.write(f"Hook: {row.get('hook', {}).get('description', '')}")
            st.write(f"Cliffhanger: {row.get('cliffhanger', {}).get('description', '')}")
            st.write(f"Summary: {row.get('overall_summary') or row.get('combined_summary') or 'No summary available.'}")


def _render_promo_detail() -> None:
    st.title("Promo details")
    ranked_rows = _load_list(ROOT / "outputs/promo_ranking_and_selection/ranked_promos.json")
    if not ranked_rows:
        st.info("No ranked promos are available for inspection.")
        return
    promo_ids = [str(item.get("promo_id", "")) for item in ranked_rows if item.get("promo_id")]
    target_id = st.selectbox("Select promo", promo_ids)
    candidate = next((row for row in ranked_rows if str(row.get("promo_id", "")) == target_id), ranked_rows[0])

    st.subheader(f"Promo {target_id}")
    st.write(candidate.get("overall_summary") or candidate.get("combined_summary") or "No summary found.")
    segments = candidate.get("segments", [])
    if segments:
        for index, segment in enumerate(segments, start=1):
            st.markdown(f"### Segment {index}")
            st.write(f"Episode: {segment.get('episode_id')} • Scenes: {segment.get('scene_ids', [])}")
            st.write(f"Time: {segment.get('start_time')} to {segment.get('end_time')} ({segment.get('segment_duration')}s)")
    else:
        st.write("No source segments stored for this promo.")

    evaluation = candidate.get("evaluation") or {}
    if evaluation:
        st.subheader("Evaluation")
        st.write(evaluation)

    quality_report = _read_json(ROOT / "outputs/quality_evaluation_and_validation/quality_report.json", {})
    candidate_quality = next((item for item in quality_report.get("candidate_results", []) if str(item.get("promo_id", "")) == target_id), {})
    if candidate_quality:
        st.subheader("Quality report")
        st.write(candidate_quality)

    reviews = load_human_reviews(ROOT / "outputs/evaluation/human_reviews.json")
    reviewer_form = st.form("human_review")
    reviewer_form.subheader("Human review")
    reviewer = reviewer_form.text_input("Reviewer ID", value="anonymous")
    label = reviewer_form.selectbox("Quality label", ["Strong", "Average", "Poor"])
    hook = reviewer_form.slider("Opening hook", 1, 5, 3)
    cliffhanger = reviewer_form.slider("Ending cliffhanger", 1, 5, 3)
    standalone = reviewer_form.slider("Standalone understandability", 1, 5, 3)
    coherence = reviewer_form.slider("Narrative coherence", 1, 5, 3)
    emotional = reviewer_form.slider("Emotional impact", 1, 5, 3)
    comments = reviewer_form.text_area("Comments")
    failure_categories = reviewer_form.text_input("Failure categories (comma separated)")
    submitted = reviewer_form.form_submit_button("Save review")
    if submitted:
        review_payload = {
            "promo_id": target_id,
            "reviewer_id": reviewer,
            "reviewed_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "overall_label": label,
            "hook_rating": hook,
            "cliffhanger_rating": cliffhanger,
            "standalone_rating": standalone,
            "coherence_rating": coherence,
            "emotional_impact_rating": emotional,
            "promotional_potential_rating": int(round((hook + cliffhanger + standalone + coherence + emotional) / 5)),
            "comments": comments,
            "failure_categories": [item.strip() for item in failure_categories.split(",") if item.strip()],
        }
        save_human_review(review_payload, ROOT / "outputs/evaluation/human_reviews.json", ROOT / "outputs/evaluation/human_reviews.csv")
        st.success("Human review saved.")


def _render_quality_analysis() -> None:
    st.title("Quality analysis")
    quality_report = _read_json(ROOT / "outputs/quality_evaluation_and_validation/quality_report.json", {})
    failure_analysis = _read_json(ROOT / "outputs/quality_evaluation_and_validation/failure_analysis.json", {})
    ranked = _load_list(ROOT / "outputs/promo_ranking_and_selection/ranked_promos.json")
    human_reviews = load_human_reviews(ROOT / "outputs/evaluation/human_reviews.json")

    if not quality_report and not failure_analysis and not ranked:
        st.info("No quality data is available yet. Run the pipeline to generate evaluation reports.")
        return

    if quality_report:
        st.subheader("Technical validation summary")
        st.json(quality_report.get("technical_validation_summary", {}))

    if failure_analysis:
        st.subheader("Failure categories")
        st.json(failure_analysis.get("failure_frequency_by_category", {}))

    if human_reviews:
        metrics, errors, rows = compute_system_evaluation(
            ranked,
            quality_report,
            _load_list(ROOT / "outputs/episodes/parsed_episodes.json"),
            human_reviews,
            module6_metadata=_read_json(ROOT / "outputs/multi_segment_promo_generation/run_metadata.json", {}),
        )
        st.subheader("Human evaluation metrics")
        st.json(metrics)
        if errors:
            st.subheader("Error analysis")
            st.json(errors)
    else:
        st.info("Human annotations are not yet available, so agreement and precision metrics remain unavailable.")


def _render_export_page() -> None:
    st.title("Export results")
    files = [
        "outputs/promo_ranking_and_selection/ranked_promos.json",
        "outputs/promo_ranking_and_selection/ranked_promos.csv",
        "outputs/quality_evaluation_and_validation/quality_report.json",
        "outputs/quality_evaluation_and_validation/quality_report.csv",
        "outputs/quality_evaluation_and_validation/failure_analysis.json",
        "outputs/candidates/multi_segment_candidates.json",
        "outputs/evaluation/human_reviews.json",
        "outputs/evaluation/human_reviews.csv",
        "outputs/evaluation/evaluation_metrics.json",
        "outputs/evaluation/evaluation_report.csv",
        "outputs/evaluation/error_analysis.json",
    ]
    for relative in files:
        path = ROOT / relative
        if path.exists():
            with open(path, "rb") as handle:
                st.download_button(
                    label=f"Download {relative}",
                    data=handle.read(),
                    file_name=relative,
                    mime="application/json" if relative.endswith(".json") else "text/csv",
                )


def _run_files(run_root: Path) -> dict[str, Path]:
    return {
        "ranked_json": run_root / "promo_ranking_and_selection" / "ranked_promos.json",
        "ranked_csv": run_root / "promo_ranking_and_selection" / "ranked_promos.csv",
        "quality_json": run_root / "quality_evaluation_and_validation" / "quality_report.json",
        "quality_csv": run_root / "quality_evaluation_and_validation" / "quality_report.csv",
        "failures": run_root / "quality_evaluation_and_validation" / "failure_analysis.json",
        "human_metrics": run_root / "evaluation" / "evaluation_metrics.json",
        "human_report": run_root / "evaluation" / "evaluation_report.csv",
        "human_errors": run_root / "evaluation" / "error_analysis.json",
        "human_summary": run_root / "evaluation" / "evaluation_summary.md",
        "reviews": run_root / "evaluation" / "human_reviews.json",
    }


def _display_value(value: Any, suffix: str = "") -> str:
    return "Not available" if value is None or value == "" else f"{value}{suffix}"


def _render_upload_screen() -> None:
    st.title("PROMO AI")
    st.caption("Intelligent promo discovery from your screenplay files.")
    st.subheader("Upload screenplays")
    st.write("Add one or more TXT, PDF, or DOCX files. Remove any file from the list before processing.")
    st.caption("Pipeline order: " + " → ".join(str(module) for module in PIPELINE_MODULE_ORDER))

    uploaded_files = st.file_uploader(
        "Screenplay files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="pipeline_screenplay_uploads",
        help="Accepted formats: PDF, TXT, DOCX.",
    )
    if uploaded_files:
        st.markdown("#### Selected files")
        for uploaded in uploaded_files:
            size = len(uploaded.getvalue())
            try:
                _validate_upload(uploaded)
                status = "Ready"
            except ValueError as exc:
                status = f"Error: {exc}"
            st.write(f"**{uploaded.name}** · {size:,} bytes · {status}")

    if st.button("Start processing", type="primary", disabled=not uploaded_files, key="run_uploaded_pipeline"):
        try:
            run_id, input_directory = _store_uploaded_files(uploaded_files)
        except (OSError, ValueError) as exc:
            st.error(f"Upload validation failed: {exc}", title="Unable to start processing")
            return

        st.session_state["promo_ai_run_id"] = run_id
        st.session_state["promo_ai_run_root"] = str(RUNS_ROOT / run_id)
        st.session_state["promo_ai_input_directory"] = str(input_directory)
        with st.status("Processing screenplays", expanded=True) as progress:
            def update_step(label: str, state: str) -> None:
                if state == "running":
                    progress.update(label=label, state="running")
                elif state == "complete":
                    progress.update(label=f"{label} complete", state="running")
                else:
                    st.session_state["promo_ai_failed_module"] = int(label.split()[1])
                    progress.update(label=f"{label} failed", state="error")

            code, log, run_output = _execute_uploaded_pipeline(
                input_directory, run_id, get_settings().llm.provider, on_step=update_step
            )
            progress.update(
                label="Processing complete" if code == 0 else "Processing stopped at the failed module",
                state="complete" if code == 0 else "error",
            )

        st.session_state["promo_ai_pipeline_error"] = "" if code == 0 else log
        if code == 0:
            st.session_state.pop("promo_ai_failed_module", None)
        st.session_state["promo_ai_run_root"] = str(run_output)
        st.session_state["promo_ai_navigate_results"] = True
        if code == 0:
            st.success("Processing completed in order: " + " → ".join(map(str, PIPELINE_MODULE_ORDER)))
        else:
            st.error("Processing stopped after a module failed. Available outputs are shown in Final results.")
            with st.expander("Processing log", expanded=True):
                st.code(log or "No console output returned.")
        st.rerun()


def _render_promo_recommendations(ranked: list[dict[str, Any]], quality_by_id: dict[str, dict[str, Any]]) -> None:
    selected = [row for row in ranked if row.get("selection_status") == "selected"]
    selected.sort(key=lambda row: float(row.get("raw_score") or 0), reverse=True)
    st.subheader("Final promo recommendations")
    if not selected:
        st.info("No selected promo recommendations are available in this run.")
        return

    for index, promo in enumerate(selected, start=1):
        promo_id = str(promo.get("promo_id", "Unknown promo"))
        quality = quality_by_id.get(promo_id, {})
        score = promo.get("raw_score", "N/A")
        with st.expander(f"#{index} · {promo_id} · score {score}", expanded=index == 1):
            columns = st.columns(4)
            segments = promo.get("segments") or []
            episode_ids = promo.get("episode_ids") or list(dict.fromkeys(
                segment.get("episode_id") for segment in segments if segment.get("episode_id") is not None
            ))
            columns[0].metric("Episode", ", ".join(map(str, episode_ids)) or "N/A")
            columns[1].metric("Type", "Multi-segment" if promo.get("promo_type") == "multi_segment" or len(segments) > 1 else "Continuous")
            columns[2].metric("Duration", _display_value(promo.get("total_duration"), " sec"))
            columns[3].metric("Quality", quality.get("quality_decision", "Pending"))
            st.write(f"**Posting priority:** {_display_value(promo.get('posting_priority') or promo.get('rank'))}")
            st.write(f"**Category:** {_display_value(promo.get('promotional_category') or promo.get('quality_category'))}")
            if segments:
                st.markdown("**Segments**")
                for segment_index, segment in enumerate(segments, start=1):
                    st.write(
                        f"Segment {segment.get('segment_order') or segment_index}: Episode {segment.get('episode_id', 'N/A')} · "
                        f"{segment.get('start_time', 'N/A')} – {segment.get('end_time', 'N/A')}"
                    )
            else:
                st.write("Segment timestamps: Not available in the ranking output.")
            st.write(f"**Hook:** {(promo.get('hook') or {}).get('description', 'Not available')}")
            st.write(f"**Cliffhanger:** {(promo.get('cliffhanger') or {}).get('description', 'Not available')}")
            st.write(f"**Why it works:** {promo.get('overall_summary') or promo.get('combined_summary') or 'No explanation was provided.'}")
            if promo.get("ranking_explanation"):
                st.caption(f"Ranking explanation: {promo['ranking_explanation']}")


def _render_quality_results(quality: dict[str, Any], failures: dict[str, Any]) -> None:
    st.subheader("Module 5 · Automated quality evaluation")
    if not quality:
        st.warning("Module 5 quality evaluation is unavailable for this run.")
        return
    counts = quality.get("quality_decision_counts") or {}
    summary_columns = st.columns(4)
    summary_columns[0].metric("Evaluated", quality.get("candidates_evaluated", 0))
    summary_columns[1].metric("PASS", counts.get("PASS", 0))
    summary_columns[2].metric("PASS_WITH_WARNINGS", counts.get("PASS_WITH_WARNINGS", 0))
    summary_columns[3].metric("FAIL", counts.get("FAIL", 0))
    quality_rows = quality.get("candidate_results") or []
    technical_by_id = {
        str(item.get("promo_id")): item
        for item in quality.get("technical_validation_results", [])
        if isinstance(item, dict)
    }
    with st.expander("Detailed checks, issues, and explanations"):
        if not quality_rows:
            st.info("Module 5 did not produce per-promo quality records.")
        for row in quality_rows:
            promo_id = str(row.get("promo_id", "Unknown promo"))
            technical = technical_by_id.get(promo_id, {})
            with st.container(border=True):
                st.markdown(f"**{promo_id} — {row.get('quality_decision', 'Unknown status')}**")
                technical_status = technical.get("status", row.get("technical_status", "Unavailable"))
                st.write(
                    "Technical checks (timestamps, 20–60 second duration, episode/scene references): "
                    f"{technical_status}"
                )
                narrative = row.get("narrative_evaluation") or {}
                criterion_labels = {
                    "opening_hook": "Hook quality", "ending_cliffhanger": "Cliffhanger quality",
                    "standalone_understandability": "Standalone understandability",
                    "narrative_coherence": "Narrative coherence", "emotional_engagement": "Emotional impact",
                }
                if narrative:
                    for key, label in criterion_labels.items():
                        check = narrative.get(key) or {}
                        score = _display_value(check.get("score"))
                        st.write(f"{label}: {score}/10 — {check.get('explanation', '')}")
                else:
                    st.write(row.get("narrative_evaluation_message") or "Narrative checks were not available in the Module 5 output.")
                findings = [
                    *technical.get("errors", row.get("technical_errors", [])),
                    *technical.get("warnings", row.get("technical_warnings", [])),
                    *row.get("failure_labels", []),
                ]
                if findings:
                    st.markdown("**Issues and warnings**")
                    for finding in findings:
                        if isinstance(finding, dict):
                            st.write(
                                f"{str(finding.get('severity', 'info')).upper()} · "
                                f"{finding.get('category', 'Finding')}: "
                                f"{finding.get('explanation') or finding.get('message', '')}"
                            )
                st.caption(row.get("explanation", ""))
    if failures:
        with st.expander("Module 5 failure analysis"):
            st.json(failures)


def _render_human_results(metrics: dict[str, Any], error_analysis: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    st.subheader("Module 8 · Human evaluation")
    if not metrics:
        st.warning("Module 8 human evaluation report is unavailable for this run.")
        return
    reviewed = int(metrics.get("reviewed_promo_count") or 0)
    if reviewed == 0 and not reviews:
        st.info("Human evaluation is not available yet. Review data will appear here once reviewers submit their assessments.")
        if metrics.get("human_review_dataset_size"):
            st.caption("A review set was prepared, but no reviewer submissions were recorded.")
        return
    st.write(f"Reviewed promos: **{reviewed}** · Review submissions: **{metrics.get('review_submission_count', len(reviews))}** · Reviewers: **{metrics.get('reviewer_count', 'N/A')}**")
    distribution = metrics.get("approval_counts") or {}
    cols = st.columns(3)
    cols[0].metric("Strong", distribution.get("strong", 0))
    cols[1].metric("Average", distribution.get("average", 0))
    cols[2].metric("Poor", distribution.get("poor", 0))
    averages = metrics.get("average_human_ratings") or {}
    st.markdown("**Average reviewer ratings**")
    rating_names = {
        "hook_rating": "Hook", "cliffhanger_rating": "Cliffhanger",
        "standalone_rating": "Standalone understandability", "coherence_rating": "Coherence",
        "emotional_impact_rating": "Emotional impact", "promotional_potential_rating": "Promotional potential",
    }
    rating_cols = st.columns(3)
    for index, (key, label) in enumerate(rating_names.items()):
        rating_cols[index % 3].metric(label, _display_value(averages.get(key)))
    agreement = metrics.get("human_ai_agreement") or {}
    precision = metrics.get("precision_at_k") or {}
    precision_text = ", ".join(
        f"P@{key} {_display_value((value or {}).get('value'))}" for key, value in precision.items()
    ) or "Not available"
    inter = metrics.get("inter_reviewer_agreement") or {}
    st.markdown("**Ranking and reviewer agreement**")
    st.write(f"Precision@K: {precision_text}")
    st.write(f"Spearman rank agreement: {_display_value(agreement.get('spearman_rank_correlation'))}")
    st.write(f"Kendall rank agreement: {_display_value(agreement.get('kendall_tau_b'))}")
    st.write(f"Inter-reviewer agreement ({inter.get('method') or 'method unavailable'}): {_display_value(inter.get('value'))}")
    disagreements = (error_analysis or {}).get("human_ai_disagreements") or []
    st.write(f"AI–human ranking disagreements: {len(disagreements)}")
    if disagreements:
        with st.expander("Disagreements and reviewer feedback"):
            st.json(disagreements)
    if reviews:
        with st.expander("Reviewer comments"):
            for review in reviews:
                comment = review.get("comments") or "No comment provided."
                st.write(f"**{review.get('promo_id')} · {review.get('reviewer_id')}**: {comment}")


def _render_exports(files: dict[str, Path]) -> None:
    st.subheader("Download reports")
    downloads = [
        ("Final ranked recommendations (JSON)", "ranked_json", "application/json"),
        ("Final ranked recommendations (CSV)", "ranked_csv", "text/csv"),
        ("Module 5 quality report", "quality_json", "application/json"),
        ("Module 5 quality report (CSV)", "quality_csv", "text/csv"),
        ("Module 5 failure analysis", "failures", "application/json"),
        ("Module 8 human evaluation metrics", "human_metrics", "application/json"),
        ("Module 8 human evaluation report", "human_report", "text/csv"),
        ("Module 8 error analysis", "human_errors", "application/json"),
        ("Module 8 evaluation summary", "human_summary", "text/markdown"),
    ]
    available = [(label, files[key], mime) for label, key, mime in downloads if files[key].is_file()]
    if not available:
        st.info("No downloadable reports are available for this run yet.")
        return
    for label, path, mime in available:
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=f"download_{path.name}")


def _render_results_screen() -> None:
    run_root_value = st.session_state.get("promo_ai_run_root")
    if not run_root_value:
        st.info("Upload screenplays and start processing to see final recommendations here.")
        return
    run_root = Path(run_root_value)
    st.title("Final results")
    st.caption(f"Run: {st.session_state.get('promo_ai_run_id', run_root.name)}")
    pipeline_error = st.session_state.get("promo_ai_pipeline_error")
    if pipeline_error:
        st.error("The pipeline did not complete. Results below include outputs available before the failed module.")
        with st.expander("Pipeline error details"):
            st.code(pipeline_error)
        failed_module = int(st.session_state.get("promo_ai_failed_module", 1))
        input_directory_value = st.session_state.get("promo_ai_input_directory")
        input_directory = Path(input_directory_value) if input_directory_value else None
        if input_directory is not None and input_directory.is_dir():
            st.caption(
                f"Retry from Module {failed_module}. Completed earlier stages will be left alone; "
                "Module 3 resumes from its saved candidate checkpoint."
            )
            if st.button("Retry failed stage", type="primary", key="retry_failed_stage"):
                with st.status(f"Resuming at Module {failed_module}", expanded=True) as progress:
                    def update_step(label: str, state: str) -> None:
                        if state == "running":
                            progress.update(label=label, state="running")
                        elif state == "complete":
                            progress.update(label=f"{label} complete", state="running")
                        else:
                            st.session_state["promo_ai_failed_module"] = int(label.split()[1])
                            progress.update(label=f"{label} failed", state="error")

                    code, log, run_output = _execute_uploaded_pipeline(
                        input_directory,
                        str(st.session_state.get("promo_ai_run_id", run_root.name)),
                        get_settings().llm.provider,
                        on_step=update_step,
                        start_at_module=failed_module,
                    )
                    progress.update(
                        label="Processing complete" if code == 0 else "Processing stopped at the failed module",
                        state="complete" if code == 0 else "error",
                    )
                st.session_state["promo_ai_pipeline_error"] = "" if code == 0 else log
                st.session_state["promo_ai_run_root"] = str(run_output)
                if code == 0:
                    st.session_state.pop("promo_ai_failed_module", None)
                st.rerun()
    files = _run_files(run_root)
    ranked = _load_list(files["ranked_json"])
    quality = _read_json(files["quality_json"], {})
    failures = _read_json(files["failures"], {})
    metrics = _read_json(files["human_metrics"], {})
    error_analysis = _read_json(files["human_errors"], {})
    reviews = load_human_reviews(files["reviews"])
    quality_by_id = {
        str(item.get("promo_id")): item
        for item in quality.get("candidate_results", [])
        if isinstance(item, dict)
    } if isinstance(quality, dict) else {}
    _render_promo_recommendations(ranked, quality_by_id)
    _render_quality_results(quality if isinstance(quality, dict) else {}, failures if isinstance(failures, dict) else {})
    _render_human_results(metrics if isinstance(metrics, dict) else {}, error_analysis if isinstance(error_analysis, dict) else {}, reviews)
    _render_exports(files)


def _main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run: python -m pip install -r requirements.txt")

    st.set_page_config(page_title="PROMO AI - Intelligent Promo Discovery", page_icon=":material/movie:", layout="wide")
    if st.session_state.pop("promo_ai_navigate_results", False):
        st.session_state["promo_ai_view"] = "Final results"
    st.sidebar.title("PROMO AI")
    st.sidebar.caption("Intelligent promo discovery")
    current_run = st.session_state.get("promo_ai_run_id")
    st.sidebar.caption(f"Status: {'Results ready' if current_run else 'Ready for upload'}")
    view = st.sidebar.radio(
        "Section",
        ["Upload screen", "Final results"],
        key="promo_ai_view",
        label_visibility="collapsed",
    )
    if view == "Upload screen":
        _render_upload_screen()
    else:
        _render_results_screen()


if __name__ == "__main__":
    os.chdir(ROOT)
    _main()

"""Tests for Streamlit screenplay uploads and isolated pipeline orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.cmd import app


class UploadedFile:
    def __init__(self, name: str, content: bytes, mime_type: str = "") -> None:
        self.name = name
        self._content = content
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._content


def test_uploads_are_saved_under_data_uploads_with_safe_names(tmp_path):
    root = tmp_path / "data" / "uploads"
    run_id, destination = app._store_uploaded_files(
        [UploadedFile("..\\episode-001.txt", b"episode text")],
        upload_root=root,
    )

    saved = list(destination.iterdir())
    assert destination.parent == root
    assert destination.name == run_id
    assert len(saved) == 1
    assert saved[0].name == "001_episode-001.txt"
    assert saved[0].read_bytes() == b"episode text"


def test_upload_rejects_unsupported_and_empty_files_before_creating_run_folder(tmp_path):
    root = tmp_path / "uploads"
    with pytest.raises(ValueError, match="Unsupported upload"):
        app._store_uploaded_files([UploadedFile("script.exe", b"no")], upload_root=root)
    with pytest.raises(ValueError, match="empty"):
        app._store_uploaded_files([UploadedFile("episode.txt", b"")], upload_root=root)
    assert not root.exists()


def test_upload_rejects_incompatible_mime_and_corrupted_supported_formats(tmp_path):
    root = tmp_path / "uploads"
    with pytest.raises(ValueError, match="MIME type"):
        app._store_uploaded_files(
            [UploadedFile("episode.txt", b"screenplay text", "application/pdf")], upload_root=root
        )
    with pytest.raises(ValueError, match="invalid or corrupted"):
        app._store_uploaded_files(
            [UploadedFile("episode.pdf", b"not a PDF", "application/pdf")], upload_root=root
        )
    with pytest.raises(ValueError, match="invalid or corrupted"):
        app._store_uploaded_files(
            [UploadedFile("episode.docx", b"not a DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")],
            upload_root=root,
        )
    assert not root.exists()


def test_uploaded_pipeline_isolated_commands_run_modules_in_dependency_order(tmp_path):
    run_id = "upload-123"
    source_dir = tmp_path / "data" / "uploads" / run_id
    output_root, commands = app._build_uploaded_pipeline_commands(source_dir, run_id, "mock")

    labels = [label for label, _ in commands]
    assert [label.split(" — ", 1)[0] for label in labels] == [
        "Module 1", "Module 2", "Module 3", "Module 6", "Module 4", "Module 5", "Module 8"
    ]
    assert str(source_dir) in commands[0][1]
    assert str(output_root / "episodes" / "parsed_episodes.json") in commands[0][1]
    assert all(str(output_root) in " ".join(command) for _, command in commands)
    assert str(output_root / "multi_segment_promo_generation" / "evaluation_cache.json") in commands[3][1]
    assert str(output_root / "promo_ranking_and_selection" / "ranked_promos.json") in commands[4][1]


def test_results_exports_resolve_inside_the_selected_run(tmp_path):
    run_root = tmp_path / "outputs" / "runs" / "run-123"
    files = app._run_files(run_root)

    assert files["ranked_json"] == run_root / "promo_ranking_and_selection" / "ranked_promos.json"
    assert files["quality_json"] == run_root / "quality_evaluation_and_validation" / "quality_report.json"
    assert files["human_metrics"] == run_root / "evaluation" / "evaluation_metrics.json"


def test_uploaded_pipeline_stops_after_first_failed_stage(tmp_path):
    calls: list[list[str]] = []
    updates: list[tuple[str, str]] = []

    def fake_runner(command: list[str]) -> tuple[int, str]:
        calls.append(command)
        return (1, "ingestion failed") if len(calls) == 1 else (0, "ok")

    code, output, _ = app._execute_uploaded_pipeline(
        tmp_path / "input",
        "upload-fail",
        "mock",
        command_runner=fake_runner,
        on_step=lambda label, state: updates.append((label, state)),
    )

    assert code == 1
    assert len(calls) == 1
    assert "ingestion failed" in output
    assert updates[-1][1] == "error"


def test_uploaded_pipeline_can_resume_at_failed_module_without_rerunning_prior_stages(tmp_path):
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> tuple[int, str]:
        calls.append(command)
        return 0, "ok"

    code, _, _ = app._execute_uploaded_pipeline(
        tmp_path / "uploads" / "resume-run",
        "resume-run",
        "mock",
        command_runner=fake_runner,
        start_at_module=3,
    )

    assert code == 0
    assert [Path(command[1]).name for command in calls] == [
        "run_promo_generation.py",
        "run_module6_multi_segment.py",
        "run_module4_rank.py",
        "run_module5_quality.py",
        "run_module8_evaluation.py",
    ]


def test_existing_data_pipeline_runs_modules_3_6_4_5_in_order(monkeypatch):
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> tuple[int, str]:
        calls.append(command)
        return 0, "ok"

    monkeypatch.setattr(app, "_run_script", fake_runner)

    code, output = app._run_pipeline_command("Rank/quality/multi-segment (existing data)", "mock")

    assert code == 0
    scripts = [Path(command[1]).name for command in calls]
    assert scripts == [
        "run_promo_generation.py",
        "run_module6_multi_segment.py",
        "run_module4_rank.py",
        "run_module5_quality.py",
    ]
    assert "outputs/candidates/promo_candidates.json" in calls[1]
    assert "outputs/candidates/multi_segment_candidates.json" in calls[2]
    assert "outputs/promo_ranking_and_selection/ranked_promos.json" in calls[3]
    assert "ok" in output


def test_existing_data_pipeline_stops_after_failed_stage(monkeypatch):
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> tuple[int, str]:
        calls.append(command)
        return (1, "Module 6 failed") if len(calls) == 2 else (0, "ok")

    monkeypatch.setattr(app, "_run_script", fake_runner)

    code, output = app._run_pipeline_command("Rank/quality/multi-segment (existing data)", "mock")

    assert code == 1
    assert len(calls) == 2
    assert "Module 6 failed" in output


def test_script_runner_preserves_unicode_child_output():
    expected = "non‑breaking hyphen: ‑"
    code, output = app._run_script(
        [sys.executable, "-c", f"print({expected!r})"]
    )

    assert code == 0
    assert output.strip() == expected

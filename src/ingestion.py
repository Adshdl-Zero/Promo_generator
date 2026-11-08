"""Module 1: data ingestion and screenplay parsing logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
TIME_PATTERN = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?"
SCENE_PATTERN = re.compile(r"^\s*SCENE\s+(\d+)\b.*$", re.IGNORECASE | re.MULTILINE)
EPISODE_PATTERN = re.compile(r"^\s*EPISODE NUMBER\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
TIMING_PATTERN = re.compile(
    rf"^\s*Start:\s*\(?({TIME_PATTERN})\)?\s*\|\s*End:\s*\(?({TIME_PATTERN})\)?\s*$",
    re.IGNORECASE,
)
START_PATTERN = re.compile(rf"^\s*Start:\s*\(?({TIME_PATTERN})\)?\s*$", re.IGNORECASE)
END_PATTERN = re.compile(rf"^\s*End:\s*\(?({TIME_PATTERN})\)?\s*$", re.IGNORECASE)
EPISODE_FILE_PATTERN = re.compile(r"episode[-_ ]?(\d+)", re.IGNORECASE)
NON_EVENT_LABELS = {
    "SCENE HEADING", "SCENE START", "SCENE END", "ACTIONS AND DIALOGUE", "DIALOGUE",
    "SCRIPT START", "SCRIPT END", "EPISODE NUMBER", "EPISODE LOGLINE", "CONTINUITY NOTES",
}


def time_to_seconds(value: str | None) -> float | None:
    """Convert SS, MM:SS, or HH:MM:SS timestamps to seconds."""
    if value is None:
        return None
    parts = value.strip().strip("()").split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None


def read_txt(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path}")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError("PDF input requires PyMuPDF. Install it with: pip install pymupdf") from exc
        with fitz.open(path) as document:
            return "\n".join(page.get_text("text") for page in document)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX input requires python-docx. Install it with: pip install python-docx") from exc
        document = Document(path)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            paragraphs.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        return "\n".join(paragraphs)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _episode_number(text: str, source: Path, fallback: int) -> int:
    match = EPISODE_PATTERN.search(text)
    if match:
        number = re.search(r"\d+", match.group(1))
        if number:
            return int(number.group())
    match = EPISODE_FILE_PATTERN.search(source.stem)
    return int(match.group(1)) if match else fallback


def _split_episode_documents(text: str) -> list[str]:
    matches = list(EPISODE_PATTERN.finditer(text))
    if len(matches) <= 1:
        return [text]
    return [text[matches[index].start(): matches[index + 1].start() if index + 1 < len(matches) else len(text)]
            for index in range(len(matches))]


def _clean_label(line: str) -> str | None:
    candidate = line.strip()
    if not candidate.endswith(":") and not re.search(r":\*+\s*$", candidate):
        return None
    candidate = re.sub(r"^\*+|\*+$", "", candidate).strip()
    candidate = candidate.rstrip("*").strip()
    if not candidate.endswith(":"):
        return None
    label = candidate[:-1].strip()
    return label or None


def _is_event_label(label: str) -> bool:
    return label.upper() not in NON_EVENT_LABELS | {"CAMERA"}


def _parse_event(lines: list[str], index: int, label: str) -> tuple[dict[str, Any], int]:
    cursor = index + 1
    start_time = end_time = None
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor < len(lines):
        timing = TIMING_PATTERN.match(lines[cursor])
        if timing:
            start_time = time_to_seconds(timing.group(1))
            end_time = time_to_seconds(timing.group(2))
            cursor += 1
        else:
            start_match = START_PATTERN.match(lines[cursor])
            if start_match:
                start_time = time_to_seconds(start_match.group(1))
                cursor += 1
                if cursor < len(lines) and (end_match := END_PATTERN.match(lines[cursor])):
                    end_time = time_to_seconds(end_match.group(1))
                    cursor += 1
    body: list[str] = []
    while cursor < len(lines):
        next_label = _clean_label(lines[cursor])
        if next_label is not None:
            break
        body.append(lines[cursor])
        cursor += 1
    text = "\n".join(line.strip() for line in body).strip()
    if text:
        text = re.sub(r"\n{3,}", "\n\n", text)
    return {"start_time": start_time, "end_time": end_time, "text": text}, cursor


def _parse_scene(scene_text: str, scene_id: int) -> dict[str, Any]:
    lines = scene_text.splitlines()
    heading = None
    scene_start = scene_end = None
    dialogue: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    characters: list[str] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("scene heading:"):
            heading = stripped.split(":", 1)[1].strip() or None
        elif stripped.lower().startswith("scene start:"):
            scene_start = time_to_seconds(stripped.split(":", 1)[1].strip())
        elif stripped.lower().startswith("scene end:"):
            scene_end = time_to_seconds(stripped.split(":", 1)[1].strip())

    index = 0
    while index < len(lines):
        label = _clean_label(lines[index])
        if label is None:
            index += 1
            continue
        if label.upper() in NON_EVENT_LABELS:
            index += 1
            continue
        event, next_index = _parse_event(lines, index, label)
        if event["text"]:
            if label.upper() == "ACTION":
                actions.append(event)
            elif _is_event_label(label):
                # Character labels may include delivery notes in parentheses.
                character = re.sub(r"\s*\([^)]*\)", "", label).strip()
                character = re.sub(r"\s+\(.*$", "", character).strip()
                if character and character not in characters:
                    characters.append(character)
                dialogue.append({"character": character or label, **event})
        index = max(next_index, index + 1)

    if scene_start is None:
        starts = [event["start_time"] for event in dialogue + actions if event["start_time"] is not None]
        scene_start = min(starts) if starts else None
    if scene_end is None:
        ends = [event["end_time"] for event in dialogue + actions if event["end_time"] is not None]
        scene_end = max(ends) if ends else None

    # Keep useful content from non-screenplay text instead of silently dropping it.
    if not dialogue and not actions:
        body = [line.strip() for line in lines if line.strip() and not re.match(
            r"^\s*(SCENE\s+\d+|Scene heading:|Scene start:|Scene end:|Actions and dialogue:)",
            line, re.IGNORECASE,
        )]
        if body:
            actions.append({"start_time": None, "end_time": None, "text": "\n".join(body)})

    return {
        "scene_id": scene_id,
        "heading": heading,
        "start_time": scene_start,
        "end_time": scene_end,
        "characters": characters,
        "dialogue": dialogue,
        "actions": actions,
    }


def parse_episode(text: str, source: Path, fallback_id: int) -> dict[str, Any]:
    episode_id = _episode_number(text, source, fallback_id)
    logline_match = re.search(r"^\s*EPISODE LOGLINE\s*:\s*(.*?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    continuity_match = re.search(r"^\s*CONTINUITY NOTES\s*:\s*(.*?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    scene_matches = list(SCENE_PATTERN.finditer(text))
    scenes: list[dict[str, Any]] = []

    if scene_matches:
        for position, match in enumerate(scene_matches):
            end = scene_matches[position + 1].start() if position + 1 < len(scene_matches) else len(text)
            scenes.append(_parse_scene(text[match.start():end], int(match.group(1))))
    else:
        scenes.append(_parse_scene(text, 1))

    return {
        "episode_id": episode_id,
        "source_file": source.name,
        "logline": logline_match.group(1).strip() if logline_match else None,
        "continuity_notes": continuity_match.group(1).strip() if continuity_match else None,
        "scenes": scenes,
    }


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input type: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def ingest(input_path: Path) -> list[dict[str, Any]]:
    files = discover_input_files(input_path)
    if not files:
        raise FileNotFoundError(f"No TXT, PDF, or DOCX files found under {input_path}")
    episodes: list[dict[str, Any]] = []
    next_id = 1
    for path in files:
        extracted = extract_text(path)
        for episode_text in _split_episode_documents(extracted):
            episodes.append(parse_episode(episode_text, path, next_id))
            next_id += 1
    return episodes



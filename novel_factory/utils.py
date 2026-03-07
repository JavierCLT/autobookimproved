"""Utility helpers shared across the repository."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from novel_factory.schemas import ChapterPlan, Outline, SceneCard

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def slugify(value: str) -> str:
    """Converts a label into a filesystem-safe slug."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "project"


def ensure_directory(path: Path) -> Path:
    """Creates a directory if needed and returns it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_utc() -> str:
    """Returns an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Reads a text file."""

    return path.read_text(encoding=encoding)


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Writes a text file, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def json_dumps(data: object) -> str:
    """Serializes JSON with stable formatting."""

    return json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)


def count_words(text: str) -> int:
    """Counts word-like tokens."""

    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Splits text into rough sentence units."""

    stripped = text.strip()
    if not stripped:
        return []
    return [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(stripped) if chunk.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Splits text into paragraphs."""

    stripped = text.strip()
    if not stripped:
        return []
    return [chunk.strip() for chunk in PARAGRAPH_SPLIT_RE.split(stripped) if chunk.strip()]


def first_token(text: str) -> str:
    """Returns the first alphanumeric token in a string."""

    match = WORD_RE.search(text)
    return match.group(0).lower() if match else ""


def plain_text_from_markdown(markdown_text: str) -> str:
    """Converts simple markdown headings into plain text output."""

    lines = markdown_text.splitlines()
    converted = []
    for line in lines:
        if line.startswith("#"):
            converted.append(line.lstrip("#").strip().upper())
        else:
            converted.append(line)
    return "\n".join(converted).strip() + "\n"


def format_scene_number(scene_number: int) -> str:
    """Formats scene numbering for filenames."""

    return f"{scene_number:02d}"


def format_chapter_number(chapter_number: int) -> str:
    """Formats chapter numbering for filenames and headings."""

    return f"{chapter_number:02d}"


def serialise_model(model: object) -> str:
    """Serializes a Pydantic model or dict to pretty JSON."""

    if hasattr(model, "model_dump"):
        data = model.model_dump()
    else:
        data = model
    return json_dumps(data)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncates long context blocks without destroying utility."""

    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def get_chapter_plan(outline: Outline, chapter_number: int) -> ChapterPlan:
    """Returns the chapter plan for a chapter number."""

    for chapter in outline.chapters:
        if chapter.chapter_number == chapter_number:
            return chapter
    raise KeyError(f"Chapter {chapter_number} was not found in the outline.")


def get_scene_card(scene_cards: Sequence[SceneCard], scene_number: int) -> SceneCard:
    """Returns a scene card by its global scene number."""

    for scene_card in scene_cards:
        if scene_card.scene_number == scene_number:
            return scene_card
    raise KeyError(f"Scene {scene_number} was not found in the scene cards.")


def chapter_scene_numbers(scene_cards: Iterable[SceneCard], chapter_number: int) -> list[int]:
    """Returns ordered scene numbers belonging to a chapter."""

    return sorted(
        [scene.scene_number for scene in scene_cards if scene.chapter_number == chapter_number]
    )

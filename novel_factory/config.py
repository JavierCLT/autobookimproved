"""Environment-backed application configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from rich.logging import RichHandler


class ReasoningProfiles(BaseModel):
    """Reasoning effort profiles used across the pipeline."""

    planning: str = "medium"
    drafting: str = "medium"
    rewriting: str = "high"
    qa: str = "medium"
    global_qa: str = "high"
    repair: str = "high"
    voice_calibration: str = "high"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    model: str = "gpt-5.4"
    drafting_model: str = ""
    qa_model: str = ""
    run_root: Path = Path("runs")
    synopsis_encoding: str = "utf-8"
    default_audience: str = "Adult"
    default_rating_ceiling: str = "R"
    default_market_position: str = "adult thriller"
    target_words: int = 40_000
    target_chapters: int = 14
    target_scenes: int = 28
    recent_scene_summaries: int = 3
    max_scene_rewrites: int = 2
    anchor_best_of_n_enabled: bool = True
    anchor_best_of_n_candidates: int = 3
    retry_attempts: int = 4
    retry_base_delay_seconds: float = 1.5
    request_timeout_seconds: float = 240.0
    planning_temperature: float = 0.2
    drafting_temperature: float = 0.85
    rewriting_temperature: float = 0.7
    qa_temperature: float = 0.1
    global_qa_temperature: float = 0.1
    max_recent_scene_summary_chars: int = 1800
    max_synopsis_context_chars: int = 18_000
    max_scene_context_chars: int = 10_000
    reasoning: ReasoningProfiles = Field(default_factory=ReasoningProfiles)

    def get_drafting_model(self) -> str:
        """Returns the prose-generation model, falling back to the primary model."""

        return self.drafting_model or self.model

    def get_qa_model(self) -> str:
        """Returns the judge-analysis model, falling back to the primary model."""

        return self.qa_model or self.model


def load_config(require_api_key: bool = True) -> AppConfig:
    """Loads configuration from the environment."""

    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env.")

    return AppConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4",
        drafting_model=os.getenv("OPENAI_DRAFTING_MODEL", "").strip(),
        qa_model=os.getenv("OPENAI_QA_MODEL", "").strip(),
        run_root=Path(os.getenv("NOVEL_FACTORY_RUN_ROOT", "runs")),
        default_audience=os.getenv("NOVEL_FACTORY_DEFAULT_AUDIENCE", "Adult").strip() or "Adult",
        default_rating_ceiling=os.getenv("NOVEL_FACTORY_DEFAULT_RATING_CEILING", "R").strip() or "R",
        default_market_position=os.getenv("NOVEL_FACTORY_DEFAULT_MARKET_POSITION", "adult thriller").strip()
        or "adult thriller",
        anchor_best_of_n_enabled=(
            os.getenv("NOVEL_FACTORY_ANCHOR_BEST_OF_N_ENABLED", "true").strip().lower() == "true"
        ),
        anchor_best_of_n_candidates=max(
            1,
            int(os.getenv("NOVEL_FACTORY_ANCHOR_BEST_OF_N", "3")),
        ),
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Configures rich logging for the CLI."""

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

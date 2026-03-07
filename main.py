"""Typer CLI entrypoint for Novel Factory."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from novel_factory.config import configure_logging, load_config
from novel_factory.pipeline import NovelPipeline

app = typer.Typer(add_completion=False, help="Generate thriller manuscripts from long synopses.")
console = Console()


def build_pipeline() -> NovelPipeline:
    """Creates a configured pipeline instance."""

    configure_logging()
    return NovelPipeline(load_config(require_api_key=True))


@app.command("bootstrap")
def bootstrap_command(
    project: str = typer.Option(..., help="Project slug or label."),
    synopsis_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Creates StorySpec, Outline, SceneCards, and initial continuity artifacts."""

    pipeline = build_pipeline()
    storage = pipeline.bootstrap(project=project, synopsis_file=synopsis_file)
    console.print(f"Bootstrapped project at {storage.root}")


@app.command("draft-scene")
def draft_scene_command(
    project: str = typer.Option(..., help="Project slug or label."),
    scene_index: int = typer.Option(..., min=1, help="1-based global scene index."),
    force: bool = typer.Option(False, help="Rewrite the scene even if it is already approved."),
) -> None:
    """Drafts one scene and runs deterministic plus model QA."""

    pipeline = build_pipeline()
    report = pipeline.draft_scene(project=project, scene_index=scene_index, force=force)
    console.print(f"Scene {scene_index:02d} pass_fail={report.pass_fail}")


@app.command("run-project")
def run_project_command(
    project: str = typer.Option(..., help="Project slug or label."),
    synopsis_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Runs planning, drafting, assembly, global QA, and repairs."""

    pipeline = build_pipeline()
    report = pipeline.run_project(project=project, synopsis_file=synopsis_file)
    console.print(f"Project complete. Global QA pass_fail={report.pass_fail}")


@app.command("global-qa")
def global_qa_command(
    project: str = typer.Option(..., help="Project slug or label."),
) -> None:
    """Runs manuscript-level QA on existing outputs."""

    pipeline = build_pipeline()
    report = pipeline.global_qa(project=project)
    console.print(f"Global QA pass_fail={report.pass_fail}")


@app.command("repair-project")
def repair_project_command(
    project: str = typer.Option(..., help="Project slug or label."),
) -> None:
    """Applies targeted repairs from the last failed global QA pass."""

    pipeline = build_pipeline()
    report = pipeline.repair_project(project=project)
    console.print(f"Repair pass complete. Global QA pass_fail={report.pass_fail}")


if __name__ == "__main__":
    app()

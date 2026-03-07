"""Checkpointed end-to-end pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from novel_factory.config import AppConfig
from novel_factory.generators import NovelGenerator
from novel_factory.judges import GlobalJudge, SceneJudge
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.schemas import (
    ContinuityState,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    SceneCard,
    SceneCardCollection,
    SceneQaReport,
    StorySpec,
)
from novel_factory.storage import RunStorage
from novel_factory.utils import (
    chapter_scene_numbers,
    get_scene_card,
    plain_text_from_markdown,
    truncate_text,
)
from novel_factory.validators import SceneValidator

logger = logging.getLogger(__name__)


class SceneApprovalError(RuntimeError):
    """Raised when a scene cannot pass QA after the rewrite budget is exhausted."""


@dataclass
class ProjectArtifacts:
    """Loaded project state required by pipeline phases."""

    synopsis: str
    story_spec: StorySpec
    outline: Outline
    scene_cards: list[SceneCard]
    continuity_state: ContinuityState


class NovelPipeline:
    """Runs the planning, drafting, QA, repair, and assembly pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = OpenAIResponsesClient(config)
        self.generators = NovelGenerator(self.llm, config)
        self.validator = SceneValidator()
        self.scene_judge = SceneJudge(self.llm, config)
        self.global_judge = GlobalJudge(self.llm, config)

    def bootstrap(self, *, project: str, synopsis_file: Path) -> RunStorage:
        """Creates or resumes the planning artifacts for a project."""

        storage = RunStorage(self.config, project)
        synopsis_text = synopsis_file.read_text(encoding=self.config.synopsis_encoding).strip()
        if not synopsis_text:
            raise ValueError(f"Synopsis file is empty: {synopsis_file}")

        storage.save_text(storage.synopsis_path, synopsis_text + "\n")
        storage.append_log(
            "bootstrap_started",
            {"project": storage.project_slug, "synopsis_file": str(synopsis_file)},
        )

        if storage.story_spec_path.exists():
            story_spec = storage.load_model(storage.story_spec_path, StorySpec)
            logger.info("Loaded existing story_spec.json")
        else:
            story_spec = self.generators.generate_story_spec(synopsis_text)
            storage.save_model(storage.story_spec_path, story_spec)
            storage.append_log("story_spec_generated", {"title": story_spec.title_working})

        if storage.outline_path.exists():
            outline = storage.load_model(storage.outline_path, Outline)
            logger.info("Loaded existing outline.json")
        else:
            outline = self.generators.generate_outline(synopsis_text, story_spec)
            storage.save_model(storage.outline_path, outline)
            storage.append_log("outline_generated", {"chapters": len(outline.chapters)})

        if storage.scene_cards_path.exists():
            scene_cards = storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards
            logger.info("Loaded existing scene_cards.json")
        else:
            scene_cards = self.generators.generate_scene_cards(synopsis_text, story_spec, outline)
            storage.save_model(
                storage.scene_cards_path,
                SceneCardCollection(scene_cards=scene_cards),
            )
            storage.append_log("scene_cards_generated", {"scenes": len(scene_cards)})

        if storage.initial_continuity_path.exists():
            initial_continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        else:
            initial_continuity = self.generators.generate_initial_continuity(story_spec, outline)
            storage.save_model(storage.initial_continuity_path, initial_continuity)
            storage.append_log("initial_continuity_generated", {})

        if not storage.continuity_path.exists():
            storage.save_model(storage.continuity_path, initial_continuity)

        storage.append_log(
            "bootstrap_completed",
            {
                "story_spec_path": str(storage.story_spec_path),
                "outline_path": str(storage.outline_path),
                "scene_cards_path": str(storage.scene_cards_path),
            },
        )
        return storage

    def draft_scene(self, *, project: str, scene_index: int, force: bool = False) -> SceneQaReport:
        """Drafts a single scene and runs the full approval loop."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if storage.has_approved_scene(scene_index) and not force:
            logger.info("Scene %s is already approved; skipping.", scene_index)
            return storage.load_model(storage.scene_qa_path(scene_index), SceneQaReport)

        scene_card = get_scene_card(artifacts.scene_cards, scene_index)
        continuity_before = self._continuity_before_scene(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
            scene_number=scene_index,
        )

        scene_text, qa_report = self._run_scene_loop(
            storage=storage,
            story_spec=artifacts.story_spec,
            outline=artifacts.outline,
            scene_card=scene_card,
            continuity_before=continuity_before,
            phase_label="draft",
            attempt_generator=lambda rewrite_brief, current_draft: self.generators.draft_scene(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                scene_card=scene_card,
                continuity_state=continuity_before,
                recent_scene_summaries=self._recent_summaries(continuity_before),
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
            ),
        )

        self._approve_scene(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_card=scene_card,
            continuity_before=continuity_before,
            scene_text=scene_text,
            qa_report=qa_report,
        )

        # Keep continuity_state.json aligned with the full approved prefix after ad hoc rewrites.
        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )
        return qa_report

    def run_project(self, *, project: str, synopsis_file: Path) -> GlobalQaReport:
        """Runs the full project from planning through QA and repair."""

        storage = self.bootstrap(project=project, synopsis_file=synopsis_file)
        artifacts = self._load_project_artifacts(storage)
        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )

        approved_prefix = self._approved_prefix_length(storage, len(artifacts.scene_cards))
        for scene_number in range(approved_prefix + 1, len(artifacts.scene_cards) + 1):
            self.draft_scene(project=project, scene_index=scene_number)

        self.assemble_manuscript(project=project)
        report = self.global_qa(project=project)
        if report.pass_fail:
            return report
        repaired_report = self.repair_project(project=project)
        if not repaired_report.pass_fail:
            raise RuntimeError("Targeted repairs completed, but global QA still fails.")
        return repaired_report

    def assemble_manuscript(self, *, project: str) -> Path:
        """Assembles approved scenes into chapter files and final manuscript outputs."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        for scene_card in artifacts.scene_cards:
            if not storage.has_approved_scene(scene_card.scene_number):
                raise RuntimeError(
                    f"Cannot assemble manuscript before scene {scene_card.scene_number} is approved."
                )

        chapter_texts: list[str] = []
        for chapter in artifacts.outline.chapters:
            scene_numbers = chapter_scene_numbers(artifacts.scene_cards, chapter.chapter_number)
            scene_texts = [storage.load_text(storage.scene_path(scene_number)) for scene_number in scene_numbers]
            chapter_markdown = (
                f"# Chapter {chapter.chapter_number:02d}: {chapter.title}\n\n"
                + "\n\n".join(scene_texts).strip()
                + "\n"
            )
            storage.save_text(storage.chapter_path(chapter.chapter_number), chapter_markdown)
            chapter_texts.append(chapter_markdown.strip())

        final_markdown = f"# {artifacts.story_spec.title_working}\n\n" + "\n\n".join(chapter_texts) + "\n"
        storage.save_text(storage.final_markdown_path, final_markdown)
        storage.save_text(storage.final_text_path, plain_text_from_markdown(final_markdown))
        storage.append_log("manuscript_assembled", {"path": str(storage.final_markdown_path)})
        return storage.final_markdown_path

    def global_qa(self, *, project: str) -> GlobalQaReport:
        """Runs manuscript-level QA on the assembled manuscript."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if not storage.final_markdown_path.exists():
            self.assemble_manuscript(project=project)

        manuscript_text = storage.load_text(storage.final_markdown_path)
        report = self.global_judge.judge(
            story_spec=artifacts.story_spec,
            outline=artifacts.outline,
            manuscript_text=manuscript_text,
        )
        storage.save_model(storage.global_qa_path, report)
        storage.append_log("global_qa_completed", {"pass_fail": report.pass_fail})
        return report

    def repair_project(self, *, project: str) -> GlobalQaReport:
        """Applies targeted scene repairs from a failed global QA report."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if storage.global_qa_path.exists():
            global_report = storage.load_model(storage.global_qa_path, GlobalQaReport)
        else:
            global_report = self.global_qa(project=project)

        if global_report.pass_fail:
            return global_report
        if not global_report.repair_targets:
            raise RuntimeError("Global QA failed without repair targets.")

        seen_scene_numbers: set[int] = set()
        for target in sorted(global_report.repair_targets, key=lambda item: item.scene_number):
            if target.scene_number in seen_scene_numbers:
                continue
            seen_scene_numbers.add(target.scene_number)
            scene_card = get_scene_card(artifacts.scene_cards, target.scene_number)
            continuity_before = self._continuity_before_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_cards=artifacts.scene_cards,
                scene_number=target.scene_number,
            )
            current_scene = storage.load_text(storage.scene_path(target.scene_number))
            repaired_scene, repaired_qa = self._run_scene_loop(
                storage=storage,
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                scene_card=scene_card,
                continuity_before=continuity_before,
                phase_label="repair",
                attempt_generator=lambda rewrite_brief, current_draft, scene_card=scene_card, continuity_before=continuity_before, current_scene=current_scene: self.generators.repair_scene(
                    story_spec=artifacts.story_spec,
                    outline=artifacts.outline,
                    scene_card=scene_card,
                    continuity_state=continuity_before,
                    current_scene=current_draft or current_scene,
                    global_qa_report=global_report,
                    rewrite_brief=rewrite_brief or target.rewrite_brief,
                ),
                initial_rewrite_brief=target.rewrite_brief,
                initial_current_draft=current_scene,
            )
            self._approve_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_card=scene_card,
                continuity_before=continuity_before,
                scene_text=repaired_scene,
                qa_report=repaired_qa,
            )
            storage.append_log(
                "scene_repaired",
                {"scene_number": target.scene_number, "reason": target.reason},
            )

        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )
        self.assemble_manuscript(project=project)
        return self.global_qa(project=project)

    def _load_project_artifacts(self, storage: RunStorage) -> ProjectArtifacts:
        """Loads the persisted project artifacts."""

        if not storage.synopsis_path.exists():
            raise RuntimeError("Project has not been bootstrapped yet.")

        return ProjectArtifacts(
            synopsis=storage.load_text(storage.synopsis_path).strip(),
            story_spec=storage.load_model(storage.story_spec_path, StorySpec),
            outline=storage.load_model(storage.outline_path, Outline),
            scene_cards=storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards,
            continuity_state=storage.load_model(storage.continuity_path, ContinuityState),
        )

    def _recent_summaries(self, continuity_state: ContinuityState) -> list[str]:
        """Returns the recent scene summaries trimmed for drafting context."""

        summaries = continuity_state.recent_scene_summaries[-self.config.recent_scene_summaries :]
        return [truncate_text(summary, self.config.max_recent_scene_summary_chars) for summary in summaries]

    def _run_scene_loop(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        outline: Outline,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        phase_label: str,
        attempt_generator: Callable[[str | None, str | None], str],
        initial_rewrite_brief: str | None = None,
        initial_current_draft: str | None = None,
    ) -> tuple[str, SceneQaReport]:
        """Runs drafting, validation, judging, and rewrites until approval or failure."""

        rewrite_brief = initial_rewrite_brief
        current_draft = initial_current_draft
        latest_report: SceneQaReport | None = None
        latest_validation: DeterministicValidationReport | None = None

        for attempt_number in range(0, self.config.max_scene_rewrites + 1):
            scene_text = attempt_generator(rewrite_brief, current_draft).strip()
            storage.save_text(
                storage.rewrite_path(
                    scene_card.scene_number,
                    attempt_number,
                    phase_label=phase_label,
                ),
                scene_text + "\n",
            )

            validation_report = self.validator.validate(
                scene_text=scene_text,
                scene_card=scene_card,
                story_spec=story_spec,
                continuity_state=continuity_before,
            )
            qa_report = self.scene_judge.judge(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity_before,
                validation_report=validation_report,
                scene_text=scene_text,
            )
            merged_report = self._merge_validation_into_qa(qa_report, validation_report)
            storage.save_model(storage.scene_qa_path(scene_card.scene_number), merged_report)
            storage.append_log(
                "scene_attempt_completed",
                {
                    "scene_number": scene_card.scene_number,
                    "phase": phase_label,
                    "attempt_number": attempt_number,
                    "pass_fail": merged_report.pass_fail,
                },
            )

            if merged_report.pass_fail:
                return scene_text, merged_report

            latest_report = merged_report
            latest_validation = validation_report
            current_draft = scene_text
            rewrite_brief = self._compose_rewrite_brief(merged_report, validation_report)

        assert latest_report is not None
        assert latest_validation is not None
        storage.append_log(
            "scene_failed",
            {
                "scene_number": scene_card.scene_number,
                "phase": phase_label,
                "hard_fail_reasons": latest_report.hard_fail_reasons,
            },
        )
        raise SceneApprovalError(
            f"Scene {scene_card.scene_number} failed QA after {self.config.max_scene_rewrites + 1} attempts."
        )

    def _merge_validation_into_qa(
        self,
        qa_report: SceneQaReport,
        validation_report: DeterministicValidationReport,
    ) -> SceneQaReport:
        """Applies deterministic failures and score thresholds to the model verdict."""

        hard_fail_reasons = list(qa_report.hard_fail_reasons)
        hard_fail_reasons.extend(
            finding.message for finding in validation_report.findings if finding.severity == "error"
        )

        threshold_failures = []
        for field_name in [
            "continuity_score",
            "engagement_score",
            "voice_score",
            "pacing_score",
            "specificity_score",
            "prose_freshness_score",
            "emotional_movement_score",
        ]:
            if getattr(qa_report, field_name) < 3:
                threshold_failures.append(f"{field_name} fell below the acceptance threshold.")
        if qa_report.ai_smell_score >= 4:
            threshold_failures.append("AI-smell risk is above the acceptance threshold.")

        hard_fail_reasons.extend(threshold_failures)
        pass_fail = qa_report.pass_fail and validation_report.pass_fail and not threshold_failures
        return qa_report.model_copy(
            update={
                "pass_fail": pass_fail,
                "hard_fail_reasons": list(dict.fromkeys(hard_fail_reasons)),
                "deterministic_pass": validation_report.pass_fail,
                "deterministic_findings": validation_report.findings,
            }
        )

    def _compose_rewrite_brief(
        self,
        qa_report: SceneQaReport,
        validation_report: DeterministicValidationReport,
    ) -> str:
        """Combines model QA feedback and deterministic findings into one rewrite brief."""

        error_messages = [
            finding.message for finding in validation_report.findings if finding.severity == "error"
        ]
        warning_messages = [
            finding.message for finding in validation_report.findings if finding.severity == "warning"
        ]
        parts = [qa_report.rewrite_brief.strip()]
        if error_messages:
            parts.append("Fix deterministic hard failures: " + " ".join(error_messages))
        if warning_messages:
            parts.append("Reduce these validator risks: " + " ".join(warning_messages[:3]))
        if qa_report.ai_smell_score >= 4:
            parts.append(
                "Vary sentence openings and cadence, replace generic emotional shorthand with scene-specific detail, and cut explanatory dialogue."
            )
        return "\n".join(part for part in parts if part).strip()

    def _approve_scene(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        scene_text: str,
        qa_report: SceneQaReport,
    ) -> ContinuityState:
        """Persists an approved scene and advances continuity."""

        updated_continuity = self.generators.update_continuity(
            story_spec=story_spec,
            scene_card=scene_card,
            continuity_state=continuity_before,
            scene_text=scene_text,
        )
        storage.save_text(storage.scene_path(scene_card.scene_number), scene_text + "\n")
        storage.save_model(storage.scene_qa_path(scene_card.scene_number), qa_report)
        storage.save_model(storage.continuity_path, updated_continuity)
        storage.append_log(
            "scene_approved",
            {
                "scene_number": scene_card.scene_number,
                "chapter_number": scene_card.chapter_number,
                "word_count_hint": scene_card.target_words,
            },
        )
        return updated_continuity

    def _approved_prefix_length(self, storage: RunStorage, total_scenes: int) -> int:
        """Returns the length of the contiguous approved scene prefix."""

        prefix_length = 0
        for scene_number in range(1, total_scenes + 1):
            if storage.has_approved_scene(scene_number):
                prefix_length += 1
            else:
                break

        for scene_number in range(prefix_length + 1, total_scenes + 1):
            if storage.has_approved_scene(scene_number):
                raise RuntimeError(
                    "Approved scenes are not contiguous. Repair the run folder before resuming."
                )
        return prefix_length

    def _continuity_before_scene(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_cards: list[SceneCard],
        scene_number: int,
    ) -> ContinuityState:
        """Replays continuity up to the scene immediately before the target scene."""

        if not storage.initial_continuity_path.exists():
            raise RuntimeError("Initial continuity snapshot is missing.")

        continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        for prior_scene_number in range(1, scene_number):
            if not storage.has_approved_scene(prior_scene_number):
                raise RuntimeError(
                    f"Scene {prior_scene_number} must be approved before drafting scene {scene_number}."
                )
            scene_card = get_scene_card(scene_cards, prior_scene_number)
            scene_text = storage.load_text(storage.scene_path(prior_scene_number))
            continuity = self.generators.update_continuity(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity,
                scene_text=scene_text,
            )
        return continuity

    def _rebuild_continuity_from_approved_scenes(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_cards: list[SceneCard],
    ) -> ContinuityState:
        """Recomputes the continuity state from the approved contiguous prefix."""

        continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        prefix_length = self._approved_prefix_length(storage, len(scene_cards))
        for scene_number in range(1, prefix_length + 1):
            scene_card = get_scene_card(scene_cards, scene_number)
            scene_text = storage.load_text(storage.scene_path(scene_number))
            continuity = self.generators.update_continuity(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity,
                scene_text=scene_text,
            )
        storage.save_model(storage.continuity_path, continuity)
        return continuity

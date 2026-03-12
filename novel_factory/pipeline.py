"""Checkpointed end-to-end pipeline orchestration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from novel_factory.config import AppConfig
from novel_factory.generators import NovelGenerator
from novel_factory.intake import parse_book_intake
from novel_factory.judges import ColdReaderJudge, GlobalJudge, PacingAnalyzer, SceneJudge
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.schemas import (
    ArcQaReport,
    BookIntake,
    ChapterQaReport,
    ColdReaderReport,
    ContinuityState,
    DeterministicValidationReport,
    EditorialBlueprint,
    GlobalQaReport,
    Outline,
    PacingAnalysis,
    PlantPayoffMap,
    RepairTarget,
    SceneCard,
    SceneCardCollection,
    SceneQaReport,
    StorySpec,
    SubplotWeaveMap,
    VoiceDNA,
)
from novel_factory.storage import RunStorage
from novel_factory.utils import (
    chapter_scene_numbers,
    count_words,
    get_scene_card,
    plain_text_from_markdown,
    truncate_text,
)
from novel_factory.validators import PlanValidator, SceneValidator

logger = logging.getLogger(__name__)


class SceneApprovalError(RuntimeError):
    """Raised when a scene cannot pass QA after the rewrite budget is exhausted."""


@dataclass
class SceneAttemptEvaluation:
    """One evaluated scene draft candidate inside an attempt slot."""

    scene_text: str
    validation_report: DeterministicValidationReport
    qa_report: SceneQaReport
    candidate_number: int = 1


@dataclass
class ProjectArtifacts:
    """Loaded project state required by pipeline phases."""

    synopsis: str
    book_intake: BookIntake | None
    story_spec: StorySpec
    editorial_blueprint: EditorialBlueprint
    voice_dna: VoiceDNA | None
    plant_payoff_map: PlantPayoffMap | None
    subplot_weave_map: SubplotWeaveMap | None
    outline: Outline
    scene_cards: list[SceneCard]
    continuity_state: ContinuityState


class NovelPipeline:
    """Runs the planning, drafting, QA, repair, and assembly pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = OpenAIResponsesClient(config)
        self.generators = NovelGenerator(self.llm, config)
        self.plan_validator = PlanValidator()
        self.validator = SceneValidator()
        self.scene_judge = SceneJudge(self.llm, config)
        self.global_judge = GlobalJudge(self.llm, config)
        self.cold_reader = ColdReaderJudge(self.llm, config)
        self.pacing_analyzer = PacingAnalyzer(self.llm, config)

    def bootstrap(
        self,
        *,
        project: str,
        synopsis_file: Path | None = None,
        intake_file: Path | None = None,
    ) -> RunStorage:
        """Creates or resumes the planning artifacts for a project."""

        storage = RunStorage(self.config, project)
        if synopsis_file is None and intake_file is None:
            raise ValueError("Provide either synopsis_file or intake_file.")

        book_intake: BookIntake | None = None
        if intake_file is not None:
            intake_text = intake_file.read_text(encoding=self.config.synopsis_encoding).strip()
            if not intake_text:
                raise ValueError(f"Intake file is empty: {intake_file}")
            book_intake = parse_book_intake(intake_text)
            storage.save_text(storage.intake_markdown_path, intake_text + "\n")
            storage.save_model(storage.intake_json_path, book_intake)

        if synopsis_file is not None:
            synopsis_text = synopsis_file.read_text(encoding=self.config.synopsis_encoding).strip()
            if not synopsis_text:
                raise ValueError(f"Synopsis file is empty: {synopsis_file}")
        else:
            assert book_intake is not None
            synopsis_text = book_intake.fields.get("synopsis", "").strip()
            if not synopsis_text:
                raise ValueError("Intake file does not contain a synopsis field.")

        storage.save_text(storage.synopsis_path, synopsis_text + "\n")
        storage.append_log(
            "bootstrap_started",
            {
                "project": storage.project_slug,
                "synopsis_file": str(synopsis_file) if synopsis_file else None,
                "intake_file": str(intake_file) if intake_file else None,
            },
        )

        voice_dna: VoiceDNA | None = None
        if storage.voice_dna_path.exists():
            voice_dna = storage.load_model(storage.voice_dna_path, VoiceDNA)
            logger.info("Loaded existing voice_dna.json")
        else:
            voice_dna = self.generators.calibrate_voice(book_intake, synopsis=synopsis_text)
            if voice_dna is not None:
                storage.save_model(storage.voice_dna_path, voice_dna)
                storage.append_log("voice_dna_generated", {})

        if storage.story_spec_path.exists():
            story_spec = storage.load_model(storage.story_spec_path, StorySpec)
            logger.info("Loaded existing story_spec.json")
        else:
            story_spec = self.generators.generate_story_spec(
                synopsis_text,
                book_intake=book_intake,
                voice_dna=voice_dna,
            )
            storage.save_model(storage.story_spec_path, story_spec)
            storage.append_log("story_spec_generated", {"title": story_spec.title_working})

        if storage.editorial_blueprint_path.exists():
            editorial_blueprint = storage.load_model(storage.editorial_blueprint_path, EditorialBlueprint)
            logger.info("Loaded existing editorial_blueprint.json")
        else:
            editorial_blueprint = self.generators.generate_editorial_blueprint(
                synopsis_text,
                story_spec,
                book_intake=book_intake,
            )
            storage.save_model(storage.editorial_blueprint_path, editorial_blueprint)
            storage.append_log(
                "editorial_blueprint_generated",
                {"commercial_hook": editorial_blueprint.commercial_hook},
            )

        if storage.outline_path.exists():
            outline = storage.load_model(storage.outline_path, Outline)
            logger.info("Loaded existing outline.json")
        else:
            outline = self.generators.generate_outline(
                synopsis_text,
                story_spec,
                editorial_blueprint,
                book_intake=book_intake,
            )
            storage.save_model(storage.outline_path, outline)
            storage.append_log("outline_generated", {"chapters": len(outline.chapters)})

        if storage.plant_payoff_map_path.exists():
            plant_payoff_map = storage.load_model(storage.plant_payoff_map_path, PlantPayoffMap)
            logger.info("Loaded existing plant_payoff_map.json")
        else:
            plant_payoff_map = self.generators.generate_plant_payoff_map(
                story_spec,
                editorial_blueprint,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.plant_payoff_map_path, plant_payoff_map)
            storage.append_log("plant_payoff_map_generated", {"entries": len(plant_payoff_map.entries)})

        if storage.subplot_weave_path.exists():
            subplot_weave_map = storage.load_model(storage.subplot_weave_path, SubplotWeaveMap)
            logger.info("Loaded existing subplot_weave_map.json")
        else:
            subplot_weave_map = self.generators.generate_subplot_weave(
                story_spec,
                editorial_blueprint,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.subplot_weave_path, subplot_weave_map)
            storage.append_log(
                "subplot_weave_generated",
                {"subplots": len(subplot_weave_map.subplots)},
            )

        if storage.scene_cards_path.exists():
            scene_cards = storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards
            logger.info("Loaded existing scene_cards.json")
        else:
            scene_cards = self.generators.generate_scene_cards(
                synopsis_text,
                story_spec,
                outline,
                editorial_blueprint,
                plant_payoff_map=plant_payoff_map,
                subplot_weave_map=subplot_weave_map,
                book_intake=book_intake,
            )
            storage.save_model(
                storage.scene_cards_path,
                SceneCardCollection(scene_cards=scene_cards),
            )
            storage.append_log("scene_cards_generated", {"scenes": len(scene_cards)})

        if storage.initial_continuity_path.exists():
            initial_continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        else:
            initial_continuity = self.generators.generate_initial_continuity(
                story_spec,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.initial_continuity_path, initial_continuity)
            storage.append_log("initial_continuity_generated", {})

        plan_report = self.plan_validator.validate(
            story_spec=story_spec,
            editorial_blueprint=editorial_blueprint,
            plant_payoff_map=plant_payoff_map,
            subplot_weave_map=subplot_weave_map,
            outline=outline,
            scene_cards=scene_cards,
            initial_continuity=initial_continuity,
        )
        storage.save_model(storage.plan_qa_path, plan_report)
        storage.append_log("plan_validation_completed", {"pass_fail": plan_report.pass_fail})
        if not plan_report.pass_fail:
            raise RuntimeError(
                "Planning artifacts failed deterministic plan QA. Adjust the bootstrap prompts or start from a fresh project slug."
            )

        if not storage.continuity_path.exists():
            storage.save_model(storage.continuity_path, initial_continuity)

        storage.append_log(
            "bootstrap_completed",
            {
                "story_spec_path": str(storage.story_spec_path),
                "editorial_blueprint_path": str(storage.editorial_blueprint_path),
                "voice_dna_path": str(storage.voice_dna_path) if voice_dna is not None else None,
                "outline_path": str(storage.outline_path),
                "plant_payoff_map_path": str(storage.plant_payoff_map_path),
                "subplot_weave_map_path": str(storage.subplot_weave_path),
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
            editorial_blueprint=artifacts.editorial_blueprint,
            outline=artifacts.outline,
            book_intake=artifacts.book_intake,
            scene_card=scene_card,
            continuity_before=continuity_before,
            phase_label="draft",
            attempt_generator=lambda rewrite_brief, current_draft: self.generators.draft_scene(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                editorial_blueprint=artifacts.editorial_blueprint,
                scene_card=scene_card,
                continuity_state=continuity_before,
                recent_scene_summaries=self._recent_summaries(continuity_before),
                voice_dna=artifacts.voice_dna,
                book_intake=artifacts.book_intake,
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

    def run_project(
        self,
        *,
        project: str,
        synopsis_file: Path | None = None,
        intake_file: Path | None = None,
    ) -> GlobalQaReport:
        """Runs the full project from planning through QA and repair."""

        storage = self.bootstrap(project=project, synopsis_file=synopsis_file, intake_file=intake_file)
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
        self._run_editorial_qa(project=project)
        report = self.global_qa(project=project)
        reader_targets = self._collect_reader_repair_targets(storage=storage, artifacts=artifacts)
        if report.pass_fail and not reader_targets:
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
            editorial_blueprint=artifacts.editorial_blueprint,
            manuscript_text=manuscript_text,
            book_intake=artifacts.book_intake,
        )
        cold_reader_report = self.cold_reader.judge(
            story_spec=artifacts.story_spec,
            manuscript_text=manuscript_text,
        )
        pacing_analysis = self.pacing_analyzer.analyze(
            story_spec=artifacts.story_spec,
            manuscript_text=manuscript_text,
            scene_count=len(artifacts.scene_cards),
        )
        storage.save_model(storage.global_qa_path, report)
        storage.save_model(storage.cold_reader_path, cold_reader_report)
        storage.save_model(storage.pacing_analysis_path, pacing_analysis)
        storage.append_log(
            "global_qa_completed",
            {
                "pass_fail": report.pass_fail,
                "cold_reader_score": cold_reader_report.overall_score,
                "weakest_scenes": cold_reader_report.weakest_scenes,
                "pacing_recommendations": len(pacing_analysis.recommendations),
            },
        )
        return report

    def editorial_qa(self, *, project: str) -> None:
        """Runs chapter and arc QA, assembling the manuscript first when needed."""

        storage = RunStorage(self.config, project)
        if not storage.final_markdown_path.exists():
            self.assemble_manuscript(project=project)
        self._run_editorial_qa(project=project)

    def _run_editorial_qa(self, *, project: str) -> None:
        """Runs chapter and arc QA before the manuscript-level judge."""

        storage = RunStorage(self.config, project)
        max_editorial_rounds = 3
        for round_number in range(0, max_editorial_rounds):
            artifacts = self._load_project_artifacts(storage)
            chapter_reports, arc_reports = self._judge_editorial_layers(
                storage=storage,
                artifacts=artifacts,
            )
            repair_targets = self._collect_editorial_repair_targets(chapter_reports, arc_reports)
            if not repair_targets:
                return
            if round_number >= max_editorial_rounds - 1:
                raise RuntimeError(
                    "Editorial QA still failed after targeted scene repairs. Inspect chapter and arc QA reports."
                )
            self._apply_editorial_repair_targets(
                storage=storage,
                artifacts=artifacts,
                repair_targets=repair_targets,
            )
            self._rebuild_continuity_from_approved_scenes(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_cards=artifacts.scene_cards,
            )
            self.assemble_manuscript(project=project)

    def repair_project(self, *, project: str) -> GlobalQaReport:
        """Applies targeted scene repairs from a failed global QA report."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if (
            storage.global_qa_path.exists()
            and storage.cold_reader_path.exists()
            and storage.pacing_analysis_path.exists()
        ):
            global_report = storage.load_model(storage.global_qa_path, GlobalQaReport)
        else:
            global_report = self.global_qa(project=project)

        repair_targets = self._merge_repair_targets(
            global_report.repair_targets,
            self._collect_reader_repair_targets(storage=storage, artifacts=artifacts),
        )
        if global_report.pass_fail and not repair_targets:
            return global_report
        if not repair_targets:
            raise RuntimeError("Global QA failed without repair targets.")
        repair_context_report = self._build_repair_context_report(
            global_report=global_report,
            repair_targets=repair_targets,
        )

        seen_scene_numbers: set[int] = set()
        for target in repair_targets:
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
                editorial_blueprint=artifacts.editorial_blueprint,
                outline=artifacts.outline,
                book_intake=artifacts.book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                phase_label="repair",
                attempt_generator=lambda rewrite_brief, current_draft, scene_card=scene_card, continuity_before=continuity_before, current_scene=current_scene: self.generators.repair_scene(
                    story_spec=artifacts.story_spec,
                    outline=artifacts.outline,
                    editorial_blueprint=artifacts.editorial_blueprint,
                    scene_card=scene_card,
                    continuity_state=continuity_before,
                    current_scene=current_draft or current_scene,
                    global_qa_report=repair_context_report,
                    rewrite_brief=rewrite_brief or target.rewrite_brief,
                    voice_dna=artifacts.voice_dna,
                    book_intake=artifacts.book_intake,
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
        self._run_editorial_qa(project=project)
        return self.global_qa(project=project)

    def _judge_editorial_layers(
        self,
        *,
        storage: RunStorage,
        artifacts: ProjectArtifacts,
    ) -> tuple[list[ChapterQaReport], list[ArcQaReport]]:
        """Runs chapter-level and targeted arc-level QA."""

        chapter_reports: list[ChapterQaReport] = []
        for chapter in artifacts.outline.chapters:
            chapter_text = storage.load_text(storage.chapter_path(chapter.chapter_number))
            report = self.global_judge.judge_chapter(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                editorial_blueprint=artifacts.editorial_blueprint,
                chapter_number=chapter.chapter_number,
                chapter_text=chapter_text,
                scene_cards=artifacts.scene_cards,
            )
            storage.save_model(storage.chapter_qa_path(chapter.chapter_number), report)
            chapter_reports.append(report)

        arc_reports: list[ArcQaReport] = []
        for arc_name, arc_focus, scene_numbers in self._editorial_arc_specs(
            artifacts.scene_cards,
            artifacts.editorial_blueprint,
        ):
            arc_text = "\n\n".join(
                storage.load_text(storage.scene_path(scene_number)).strip()
                for scene_number in scene_numbers
            ).strip()
            report = self.global_judge.judge_arc(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                editorial_blueprint=artifacts.editorial_blueprint,
                arc_name=arc_name,
                arc_focus=arc_focus,
                scene_numbers=scene_numbers,
                arc_text=arc_text,
            )
            storage.save_model(storage.arc_qa_path(arc_name), report)
            arc_reports.append(report)

        storage.append_log(
            "editorial_qa_completed",
            {
                "chapter_failures": sum(1 for report in chapter_reports if not report.pass_fail),
                "arc_failures": sum(1 for report in arc_reports if not report.pass_fail),
            },
        )
        return chapter_reports, arc_reports

    def _collect_editorial_repair_targets(
        self,
        chapter_reports: list[ChapterQaReport],
        arc_reports: list[ArcQaReport],
    ) -> list[RepairTarget]:
        """Collapses chapter and arc repair targets into a minimal scene list."""

        targets: list[RepairTarget] = []
        for report in [*chapter_reports, *arc_reports]:
            if report.pass_fail:
                continue
            if not report.repair_targets:
                raise RuntimeError("Editorial QA failed without scene-level repair targets.")
            targets.extend(report.repair_targets)
        return self._merge_repair_targets(targets)

    def _apply_editorial_repair_targets(
        self,
        *,
        storage: RunStorage,
        artifacts: ProjectArtifacts,
        repair_targets: list[RepairTarget],
    ) -> None:
        """Rewrites scene targets raised by chapter or arc QA."""

        for target in repair_targets:
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
                editorial_blueprint=artifacts.editorial_blueprint,
                outline=artifacts.outline,
                book_intake=artifacts.book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                phase_label="editorial_repair",
                attempt_generator=lambda rewrite_brief, current_draft, scene_card=scene_card, continuity_before=continuity_before, current_scene=current_scene, target=target: self.generators.repair_scene(
                    story_spec=artifacts.story_spec,
                    outline=artifacts.outline,
                    editorial_blueprint=artifacts.editorial_blueprint,
                    scene_card=scene_card,
                    continuity_state=continuity_before,
                    current_scene=current_draft or current_scene,
                    global_qa_report=GlobalQaReport(
                        pass_fail=False,
                        hook_strength_score=3,
                        midpoint_turn_score=3,
                        climax_payoff_score=3,
                        ending_payoff_score=3,
                        relationship_progression_score=3,
                        antagonist_pressure_score=3,
                        continuity_score=3,
                        emotional_aftershock_score=3,
                        boredom_risk_score=3,
                        voice_consistency_score=3,
                        ai_smell_score=3,
                        major_problems=[target.reason],
                        repair_targets=[target],
                    ),
                    rewrite_brief=rewrite_brief or target.rewrite_brief,
                    voice_dna=artifacts.voice_dna,
                    book_intake=artifacts.book_intake,
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
                "scene_editorially_repaired",
                {"scene_number": target.scene_number, "reason": target.reason},
            )

    def _load_project_artifacts(self, storage: RunStorage) -> ProjectArtifacts:
        """Loads the persisted project artifacts."""

        if not storage.synopsis_path.exists():
            raise RuntimeError("Project has not been bootstrapped yet.")

        book_intake = None
        if storage.intake_json_path.exists():
            book_intake = storage.load_model(storage.intake_json_path, BookIntake)
        elif storage.intake_markdown_path.exists():
            book_intake = parse_book_intake(storage.load_text(storage.intake_markdown_path))

        synopsis = storage.load_text(storage.synopsis_path).strip()
        story_spec = storage.load_model(storage.story_spec_path, StorySpec)
        voice_dna: VoiceDNA | None = None
        if storage.voice_dna_path.exists():
            voice_dna = storage.load_model(storage.voice_dna_path, VoiceDNA)
        else:
            voice_dna = self.generators.calibrate_voice(book_intake, synopsis=synopsis)
            if voice_dna is not None:
                storage.save_model(storage.voice_dna_path, voice_dna)
                storage.append_log("voice_dna_backfilled", {})

        if storage.editorial_blueprint_path.exists():
            editorial_blueprint = storage.load_model(storage.editorial_blueprint_path, EditorialBlueprint)
        else:
            editorial_blueprint = self.generators.generate_editorial_blueprint(
                synopsis,
                story_spec,
                book_intake=book_intake,
            )
            storage.save_model(storage.editorial_blueprint_path, editorial_blueprint)
            storage.append_log("editorial_blueprint_backfilled", {})

        if storage.outline_path.exists():
            outline = storage.load_model(storage.outline_path, Outline)
        else:
            outline = self.generators.generate_outline(
                synopsis,
                story_spec,
                editorial_blueprint,
                book_intake=book_intake,
            )
            storage.save_model(storage.outline_path, outline)
            storage.append_log("outline_backfilled", {"chapters": len(outline.chapters)})

        if storage.plant_payoff_map_path.exists():
            plant_payoff_map = storage.load_model(storage.plant_payoff_map_path, PlantPayoffMap)
        else:
            plant_payoff_map = self.generators.generate_plant_payoff_map(
                story_spec,
                editorial_blueprint,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.plant_payoff_map_path, plant_payoff_map)
            storage.append_log(
                "plant_payoff_map_backfilled",
                {"entries": len(plant_payoff_map.entries)},
            )

        if storage.subplot_weave_path.exists():
            subplot_weave_map = storage.load_model(storage.subplot_weave_path, SubplotWeaveMap)
        else:
            subplot_weave_map = self.generators.generate_subplot_weave(
                story_spec,
                editorial_blueprint,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.subplot_weave_path, subplot_weave_map)
            storage.append_log(
                "subplot_weave_backfilled",
                {"subplots": len(subplot_weave_map.subplots)},
            )

        if storage.scene_cards_path.exists():
            scene_cards = storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards
        else:
            scene_cards = self.generators.generate_scene_cards(
                synopsis,
                story_spec,
                outline,
                editorial_blueprint,
                plant_payoff_map=plant_payoff_map,
                subplot_weave_map=subplot_weave_map,
                book_intake=book_intake,
            )
            storage.save_model(storage.scene_cards_path, SceneCardCollection(scene_cards=scene_cards))
            storage.append_log("scene_cards_backfilled", {"scenes": len(scene_cards)})

        if storage.initial_continuity_path.exists():
            initial_continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        else:
            initial_continuity = self.generators.generate_initial_continuity(
                story_spec,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.initial_continuity_path, initial_continuity)
            storage.append_log("initial_continuity_backfilled", {})

        if storage.continuity_path.exists():
            continuity_state = storage.load_model(storage.continuity_path, ContinuityState)
        else:
            continuity_state = initial_continuity
            storage.save_model(storage.continuity_path, continuity_state)
            storage.append_log("continuity_state_backfilled", {})

        return ProjectArtifacts(
            synopsis=synopsis,
            book_intake=book_intake,
            story_spec=story_spec,
            editorial_blueprint=editorial_blueprint,
            voice_dna=voice_dna,
            plant_payoff_map=plant_payoff_map,
            subplot_weave_map=subplot_weave_map,
            outline=outline,
            scene_cards=scene_cards,
            continuity_state=continuity_state,
        )

    def _recent_summaries(self, continuity_state: ContinuityState) -> list[str]:
        """Returns the recent scene summaries trimmed for drafting context."""

        summaries = continuity_state.recent_scene_summaries[-self.config.recent_scene_summaries :]
        return [truncate_text(summary, self.config.max_recent_scene_summary_chars) for summary in summaries]

    def _editorial_arc_specs(
        self,
        scene_cards: list[SceneCard],
        editorial_blueprint: EditorialBlueprint,
    ) -> list[tuple[str, str, list[int]]]:
        """Returns targeted manuscript slices for intermediate arc QA."""

        total_scenes = len(scene_cards)
        opening_numbers = list(range(1, min(total_scenes, 4) + 1))
        midpoint_center = max(1, total_scenes // 2)
        midpoint_numbers = list(
            range(max(1, midpoint_center - 1), min(total_scenes, midpoint_center + 2) + 1)
        )
        counterforce_numbers = [
            scene.scene_number
            for scene in scene_cards
            if self._scene_has_counterforce(scene, editorial_blueprint)
        ][:6]
        relationship_numbers = [
            scene.scene_number
            for scene in scene_cards
            if self._scene_has_relationship_pressure(scene, editorial_blueprint)
        ][:6]
        ending_numbers = list(range(max(1, total_scenes - 3), total_scenes + 1))

        relationship_label = editorial_blueprint.relationship_focus_name.strip() or "the core relationship"
        counterforce_label = editorial_blueprint.counterforce_name.strip() or "the counterforce"
        specs: list[tuple[str, str, list[int]]] = [
            (
                "opening",
                "Does the opening create immediate compulsion, visible human cost, and a concrete shadow of pursuit or consequence?",
                opening_numbers,
            ),
            (
                "midpoint",
                "Does the middle of the book deliver an irreversible reframe, reveal, or escalation rather than competent procedural drift?",
                midpoint_numbers,
            ),
            (
                "ending",
                "Does the ending convert containment, loss, and escape into emotional aftershock rather than elegant distance?",
                ending_numbers,
            ),
        ]
        if len(counterforce_numbers) >= 2:
            specs.append(
                (
                    "counterforce",
                    f"Does {counterforce_label} feel like a hunter rather than a remote abstraction, and does suspicion or exposure narrow the field?",
                    counterforce_numbers,
                )
            )
        if len(relationship_numbers) >= 2:
            specs.append(
                (
                    "relationship",
                    f"Does pressure on {relationship_label} materialize through refusals, absences, concealment, tenderness, and cost rather than summary or theme?",
                    relationship_numbers,
                )
            )
        return specs

    def _scene_has_counterforce(
        self,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
    ) -> bool:
        """Returns True when a scene visibly advances pursuit or institutional shadow."""

        text = " ".join(
            [scene_card.counterforce_trace, scene_card.suspicion_delta, scene_card.pressure_source]
        ).lower()
        if any(keyword in text for keyword in self._counterforce_keywords(editorial_blueprint)):
            return True
        return "review" in text and any(
            keyword in text for keyword in self._counterforce_keywords(editorial_blueprint)
        )

    def _scene_has_relationship_pressure(
        self,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
    ) -> bool:
        """Returns True when a scene materially advances the marriage/relationship arc."""

        text = " ".join(
            [scene_card.relationship_delta, scene_card.secret_pressure, scene_card.cost_paid]
        ).lower()
        relationship_name = editorial_blueprint.relationship_focus_name.lower().strip()
        return (
            bool(relationship_name and relationship_name in scene_card.pov_character.lower())
            or any(keyword in text for keyword in self._relationship_keywords(editorial_blueprint))
        )

    def _run_scene_loop(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        outline: Outline,
        book_intake: BookIntake | None,
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
            evaluation = self._select_scene_attempt(
                storage=storage,
                story_spec=story_spec,
                editorial_blueprint=editorial_blueprint,
                book_intake=book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                attempt_generator=attempt_generator,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
                phase_label=phase_label,
                attempt_number=attempt_number,
            )
            scene_text = evaluation.scene_text
            storage.save_text(
                storage.rewrite_path(
                    scene_card.scene_number,
                    attempt_number,
                    phase_label=phase_label,
                ),
                scene_text + "\n",
            )

            validation_report = evaluation.validation_report
            merged_report = evaluation.qa_report
            if merged_report.pass_fail or not storage.has_approved_scene(scene_card.scene_number):
                storage.save_model(storage.scene_qa_path(scene_card.scene_number), merged_report)
            storage.append_log(
                "scene_attempt_completed",
                {
                    "scene_number": scene_card.scene_number,
                    "phase": phase_label,
                    "attempt_number": attempt_number,
                    "candidate_number": evaluation.candidate_number,
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

    def _select_scene_attempt(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        book_intake: BookIntake | None,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        attempt_generator: Callable[[str | None, str | None], str],
        rewrite_brief: str | None,
        current_draft: str | None,
        phase_label: str,
        attempt_number: int,
    ) -> SceneAttemptEvaluation:
        """Evaluates one draft attempt, optionally using anchor-scene best-of-N selection."""

        if not self._should_use_anchor_best_of_n(
            scene_card=scene_card,
            editorial_blueprint=editorial_blueprint,
            total_scenes=story_spec.expected_scenes,
            phase_label=phase_label,
            attempt_number=attempt_number,
            rewrite_brief=rewrite_brief,
            current_draft=current_draft,
        ):
            scene_text = self._generate_scene_text(
                attempt_generator=attempt_generator,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
                scene_card=scene_card,
            )
            return self._evaluate_scene_attempt(
                story_spec=story_spec,
                editorial_blueprint=editorial_blueprint,
                book_intake=book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                scene_text=scene_text,
                candidate_number=1,
            )

        evaluations: list[SceneAttemptEvaluation] = []
        candidate_count = max(2, self.config.anchor_best_of_n_candidates)
        for candidate_number in range(1, candidate_count + 1):
            scene_text = self._generate_scene_text(
                attempt_generator=attempt_generator,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
                scene_card=scene_card,
            )
            storage.save_text(
                storage.candidate_path(
                    scene_card.scene_number,
                    attempt_number,
                    candidate_number,
                    phase_label=phase_label,
                ),
                scene_text + "\n",
            )
            evaluations.append(
                self._evaluate_scene_attempt(
                    story_spec=story_spec,
                    editorial_blueprint=editorial_blueprint,
                    book_intake=book_intake,
                    scene_card=scene_card,
                    continuity_before=continuity_before,
                    scene_text=scene_text,
                    candidate_number=candidate_number,
                )
            )

        selected = self._select_best_scene_candidate(evaluations, scene_card=scene_card)
        storage.append_log(
            "anchor_best_of_n_completed",
            {
                "scene_number": scene_card.scene_number,
                "attempt_number": attempt_number,
                "selected_candidate": selected.candidate_number,
                "candidate_count": candidate_count,
                "candidates": [
                    {
                        "candidate_number": evaluation.candidate_number,
                        "pass_fail": evaluation.qa_report.pass_fail,
                        "quality_score": self._scene_quality_total(evaluation.qa_report),
                        "ai_smell_score": evaluation.qa_report.ai_smell_score,
                        "hard_fail_reasons": len(evaluation.qa_report.hard_fail_reasons),
                    }
                    for evaluation in evaluations
                ],
            },
        )
        return selected

    def _generate_scene_text(
        self,
        *,
        attempt_generator: Callable[[str | None, str | None], str],
        rewrite_brief: str | None,
        current_draft: str | None,
        scene_card: SceneCard,
    ) -> str:
        """Retries obviously truncated drafts before spending a rewrite attempt."""

        latest_scene_text = ""
        for _ in range(0, 3):
            latest_scene_text = attempt_generator(rewrite_brief, current_draft).strip()
            if not self._scene_text_looks_truncated(latest_scene_text, scene_card):
                return latest_scene_text
            logger.warning(
                "Scene %s draft looked truncated; retrying generation within the same attempt.",
                scene_card.scene_number,
            )
        return latest_scene_text

    def _evaluate_scene_attempt(
        self,
        *,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        book_intake: BookIntake | None,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        scene_text: str,
        candidate_number: int,
    ) -> SceneAttemptEvaluation:
        """Runs deterministic and model QA for one candidate scene draft."""

        validation_report = self.validator.validate(
            scene_text=scene_text,
            scene_card=scene_card,
            story_spec=story_spec,
            continuity_state=continuity_before,
        )
        qa_report = self.scene_judge.judge(
            story_spec=story_spec,
            editorial_blueprint=editorial_blueprint,
            scene_card=scene_card,
            continuity_state=continuity_before,
            validation_report=validation_report,
            scene_text=scene_text,
            book_intake=book_intake,
        )
        merged_report = self._merge_validation_into_qa(
            qa_report,
            validation_report,
            editorial_blueprint=editorial_blueprint,
            scene_card=scene_card,
            total_scenes=story_spec.expected_scenes,
        )
        return SceneAttemptEvaluation(
            scene_text=scene_text,
            validation_report=validation_report,
            qa_report=merged_report,
            candidate_number=candidate_number,
        )

    def _should_use_anchor_best_of_n(
        self,
        *,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
        total_scenes: int,
        phase_label: str,
        attempt_number: int,
        rewrite_brief: str | None,
        current_draft: str | None,
    ) -> bool:
        """Returns True when the pipeline should evaluate several candidates before choosing."""

        if not self.config.anchor_best_of_n_enabled or self.config.anchor_best_of_n_candidates < 2:
            return False
        if phase_label != "draft" or attempt_number != 0:
            return False
        if rewrite_brief is not None or current_draft is not None:
            return False
        return self._is_anchor_scene(
            scene_card=scene_card,
            editorial_blueprint=editorial_blueprint,
            total_scenes=total_scenes,
        )

    def _select_best_scene_candidate(
        self,
        evaluations: list[SceneAttemptEvaluation],
        *,
        scene_card: SceneCard,
    ) -> SceneAttemptEvaluation:
        """Chooses the strongest candidate from an anchor-scene best-of-N batch."""

        if not evaluations:
            raise RuntimeError("Anchor best-of-N completed without any candidate evaluations.")
        return max(
            evaluations,
            key=lambda evaluation: (
                1 if evaluation.qa_report.pass_fail else 0,
                1 if evaluation.qa_report.deterministic_pass else 0,
                -self._validation_error_count(evaluation.validation_report),
                -self._validation_warning_count(evaluation.validation_report),
                -len(evaluation.qa_report.hard_fail_reasons),
                self._scene_quality_total(evaluation.qa_report),
                -evaluation.qa_report.ai_smell_score,
                -abs(evaluation.validation_report.word_count - scene_card.target_words),
            ),
        )

    def _scene_quality_total(self, qa_report: SceneQaReport) -> int:
        """Aggregates positive scene-QA scores for candidate ranking."""

        score_fields = (
            "continuity_score",
            "engagement_score",
            "voice_score",
            "pacing_score",
            "specificity_score",
            "prose_freshness_score",
            "emotional_movement_score",
            "subtext_score",
            "concealment_score",
            "leverage_shift_score",
            "relationship_cost_score",
            "commercial_hook_score",
        )
        return sum(getattr(qa_report, field_name) for field_name in score_fields)

    def _validation_error_count(self, validation_report: DeterministicValidationReport) -> int:
        """Returns the number of deterministic hard failures in a candidate draft."""

        return sum(1 for finding in validation_report.findings if finding.severity == "error")

    def _validation_warning_count(self, validation_report: DeterministicValidationReport) -> int:
        """Returns the number of deterministic warnings in a candidate draft."""

        return sum(1 for finding in validation_report.findings if finding.severity == "warning")

    def _scene_text_looks_truncated(self, scene_text: str, scene_card: SceneCard) -> bool:
        """Returns True when a scene draft is too short or ends like an accidental cutoff."""

        if not scene_text:
            return True
        minimum_retry_floor = max(500, int(scene_card.target_words * 0.55))
        if count_words(scene_text) < minimum_retry_floor:
            return True
        trimmed = scene_text.rstrip()
        if not trimmed:
            return True
        if trimmed[-1] in ".!?\"'”’":
            return False
        return True

    def _merge_validation_into_qa(
        self,
        qa_report: SceneQaReport,
        validation_report: DeterministicValidationReport,
        *,
        editorial_blueprint: EditorialBlueprint,
        scene_card: SceneCard,
        total_scenes: int,
    ) -> SceneQaReport:
        """Applies deterministic failures and score thresholds to the model verdict."""

        hard_fail_reasons = list(qa_report.hard_fail_reasons)
        hard_fail_reasons.extend(
            finding.message for finding in validation_report.findings if finding.severity == "error"
        )

        threshold_failures = []
        thresholds = {
            "continuity_score": 4,
            "engagement_score": 4,
            "voice_score": 3,
            "pacing_score": 4,
            "specificity_score": 4,
            "prose_freshness_score": 4,
            "emotional_movement_score": 4,
            "subtext_score": 4,
            "concealment_score": 4,
            "leverage_shift_score": 4,
            "relationship_cost_score": 3,
            "commercial_hook_score": 4,
        }
        if self._is_anchor_scene(
            scene_card=scene_card,
            editorial_blueprint=editorial_blueprint,
            total_scenes=total_scenes,
        ):
            thresholds["voice_score"] = 4
        if self._scene_needs_relationship_cost(scene_card, editorial_blueprint):
            thresholds["relationship_cost_score"] = 4
        if self._scene_allows_quieter_opening_drive(scene_card, editorial_blueprint):
            thresholds["pacing_score"] = 3
            thresholds["commercial_hook_score"] = 3
        for field_name, minimum_score in thresholds.items():
            if getattr(qa_report, field_name) < minimum_score:
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
        score_guidance = {
            "engagement_score": "Increase immediate pressure and make the scene harder to stop reading within the first page.",
            "pacing_score": "Cut explanatory drag, tighten turn timing, and ensure each beat changes the pressure.",
            "prose_freshness_score": "Replace generic thriller phrasing with scene-specific language and sharper physical detail.",
            "emotional_movement_score": "Make the emotional turn visible through behavior, choice, and consequence rather than summary.",
            "subtext_score": "Reduce direct explanation and let omission, evasion, and behavior carry the scene's hidden argument.",
            "concealment_score": "Put an active lie, omission, or cover story under pressure on-page.",
            "leverage_shift_score": "Make the power balance clearly change by the end of the scene.",
            "relationship_cost_score": "Ensure the scene exacts a visible interpersonal cost rather than only a thematic one.",
            "commercial_hook_score": "Sharpen the opening disturbance and the closing choice so the scene creates stronger forward pull.",
        }
        for field_name, guidance in score_guidance.items():
            if getattr(qa_report, field_name) < 4:
                parts.append(guidance)
        return "\n".join(part for part in parts if part).strip()

    def _is_anchor_scene(
        self,
        *,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
        total_scenes: int,
    ) -> bool:
        """Returns True when a scene should clear a higher QA bar."""

        lower_type = scene_card.scene_type.lower()
        anchor_types = {
            "opening",
            "confrontation",
            "domestic fracture",
            "pursuit",
            "breakup",
            "midpoint",
            "climax",
            "fallout",
            "ending",
        }
        anchor_numbers = {1, max(1, total_scenes // 2), min(total_scenes, (total_scenes // 2) + 1), max(1, total_scenes - 1), total_scenes}
        relationship_name = editorial_blueprint.relationship_focus_name.lower().strip()
        return (
            lower_type in anchor_types
            or scene_card.scene_number in anchor_numbers
            or bool(relationship_name and relationship_name in scene_card.pov_character.lower())
        )

    def _scene_needs_relationship_cost(
        self,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
    ) -> bool:
        """Returns True when the scene should clear a higher interpersonal-cost bar."""

        text = " ".join([scene_card.relationship_delta, scene_card.secret_pressure, scene_card.cost_paid]).lower()
        required_entities_text = " ".join(scene_card.required_entities).lower()
        location_text = scene_card.location.lower()
        relationship_keywords = self._relationship_keywords(editorial_blueprint)
        relationship_name = editorial_blueprint.relationship_focus_name.lower().strip()
        if "off-page" in text:
            return False
        if scene_card.scene_type.lower() == "domestic fracture":
            return True
        if relationship_name and relationship_name in scene_card.pov_character.lower():
            return True
        if any(keyword in required_entities_text for keyword in relationship_keywords):
            return True
        if any(keyword in location_text for keyword in ("apartment", "home", "kitchen", "bedroom")):
            return True
        return any(keyword in text for keyword in relationship_keywords)

    def _scene_allows_quieter_opening_drive(
        self,
        scene_card: SceneCard,
        editorial_blueprint: EditorialBlueprint,
    ) -> bool:
        """Returns True when a scene can trade some hook velocity for intimacy or set-up."""

        lower_type = scene_card.scene_type.lower()
        quiet_types = {
            "domestic fracture",
            "cover story",
            "fallout",
        }
        return lower_type in quiet_types and not self._scene_has_counterforce(
            scene_card,
            editorial_blueprint,
        )

    def _collect_reader_repair_targets(
        self,
        *,
        storage: RunStorage,
        artifacts: ProjectArtifacts,
    ) -> list[RepairTarget]:
        """Builds repair targets from cold-reader and pacing-analysis reports."""

        targets: list[RepairTarget] = []
        total_scenes = len(artifacts.scene_cards)

        if storage.cold_reader_path.exists():
            cold_reader_report = storage.load_model(storage.cold_reader_path, ColdReaderReport)
            targets.extend(self._cold_reader_repair_targets(cold_reader_report, total_scenes=total_scenes))
        if storage.pacing_analysis_path.exists():
            pacing_analysis = storage.load_model(storage.pacing_analysis_path, PacingAnalysis)
            targets.extend(self._pacing_repair_targets(pacing_analysis, total_scenes=total_scenes))
        return self._merge_repair_targets(targets)

    def _cold_reader_repair_targets(
        self,
        cold_reader_report: ColdReaderReport,
        *,
        total_scenes: int,
    ) -> list[RepairTarget]:
        """Converts cold-reader observations into scene repair targets."""

        targets: list[RepairTarget] = []
        for scene_number in cold_reader_report.weakest_scenes:
            if 1 <= scene_number <= total_scenes:
                targets.append(
                    RepairTarget(
                        scene_number=scene_number,
                        reason="Cold-reader pass marked this as a weakest scene.",
                        rewrite_brief=(
                            "Make this scene less predictable, sharpen causal clarity, and ensure the "
                            "emotional or narrative turn feels indispensable instead of optional."
                        ),
                    )
                )

        note_groups = [
            (
                cold_reader_report.confusion_points,
                "Cold-reader confusion point",
                "Clarify who knows what, where everyone is, and why the key turn matters without adding explanatory drag.",
            ),
            (
                cold_reader_report.engagement_drops,
                "Cold-reader engagement drop",
                "Tighten the opening pressure, cut drift, and land on a sharper choice or destabilizing turn.",
            ),
            (
                cold_reader_report.character_tracking_issues,
                "Cold-reader character-tracking issue",
                "Anchor who is present, who is acting, and how the interpersonal geometry changes on-page.",
            ),
            (
                cold_reader_report.predictable_moments,
                "Cold-reader predictability note",
                "Replace the expected beat with a more surprising but inevitable turn, and make the cost feel specific.",
            ),
        ]
        for notes, label, rewrite_brief in note_groups:
            for note in notes:
                for scene_number in self._extract_scene_numbers(note, total_scenes=total_scenes):
                    targets.append(
                        RepairTarget(
                            scene_number=scene_number,
                            reason=f"{label}: {truncate_text(note, 180)}",
                            rewrite_brief=rewrite_brief,
                        )
                    )
        return targets

    def _pacing_repair_targets(
        self,
        pacing_analysis: PacingAnalysis,
        *,
        total_scenes: int,
    ) -> list[RepairTarget]:
        """Converts pacing-analysis observations into scene repair targets."""

        targets: list[RepairTarget] = []
        for note in [*pacing_analysis.tension_sags, *pacing_analysis.fatigue_zones]:
            for scene_number in self._extract_scene_numbers(note, total_scenes=total_scenes):
                targets.append(
                    RepairTarget(
                        scene_number=scene_number,
                        reason=f"Pacing analysis flagged a sag: {truncate_text(note, 180)}",
                        rewrite_brief=(
                            "Reduce explanatory drag, make pressure or emotional movement visibly change within "
                            "the scene, and strengthen the forward pull into the next scene."
                        ),
                    )
                )

        low_metric_scenes = sorted(
            [
                scene_data
                for scene_data in pacing_analysis.scene_data
                if 1 <= scene_data.scene_number <= total_scenes
                and (
                    (
                        scene_data.tension_level
                        + scene_data.stakes_level
                        + scene_data.action_density
                        + scene_data.emotional_intensity
                    )
                    / 4
                )
                <= 3.0
            ],
            key=lambda scene_data: (
                (
                    scene_data.tension_level
                    + scene_data.stakes_level
                    + scene_data.action_density
                    + scene_data.emotional_intensity
                )
                / 4,
                scene_data.tension_level,
                scene_data.action_density,
            ),
        )[:3]
        for scene_data in low_metric_scenes:
            targets.append(
                RepairTarget(
                    scene_number=scene_data.scene_number,
                    reason="Pacing curve shows this scene as a low-energy pocket.",
                    rewrite_brief=self._build_pacing_metric_rewrite_brief(scene_data),
                )
            )
        return targets

    def _build_pacing_metric_rewrite_brief(self, scene_data: object) -> str:
        """Builds a focused rewrite brief from low pacing metrics."""

        parts: list[str] = []
        tension_level = getattr(scene_data, "tension_level", 0)
        stakes_level = getattr(scene_data, "stakes_level", 0)
        action_density = getattr(scene_data, "action_density", 0)
        emotional_intensity = getattr(scene_data, "emotional_intensity", 0)
        if tension_level <= 3:
            parts.append("Raise immediate pressure or consequence.")
        if stakes_level <= 3:
            parts.append("Make the stakes concrete, personal, and harder to dismiss.")
        if action_density <= 3:
            parts.append("Replace static explanation with consequential movement, confrontation, or decision.")
        if emotional_intensity <= 3:
            parts.append("Make the emotional turn legible through behavior, cost, and changed leverage.")
        if not parts:
            parts.append("Tighten pacing and make the turn in this scene more consequential.")
        return " ".join(parts)

    def _extract_scene_numbers(self, note: str, *, total_scenes: int) -> list[int]:
        """Extracts scene numbers from free-form QA notes when they are explicitly referenced."""

        lowered = (note or "").lower()
        if "scene" not in lowered:
            return []

        scene_numbers: list[int] = []
        for match in re.finditer(r"scenes?\s+#?(\d+)\s*(?:-|to|through)\s*(\d+)", lowered):
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                start, end = end, start
            if end - start <= 4:
                scene_numbers.extend(range(start, end + 1))
            else:
                scene_numbers.extend([start, end])

        for match in re.finditer(r"scene\s+#?(\d+)", lowered):
            scene_numbers.append(int(match.group(1)))

        if not scene_numbers:
            scene_numbers.extend(int(match.group(0)) for match in re.finditer(r"\b\d+\b", lowered))

        unique_numbers: list[int] = []
        for scene_number in scene_numbers:
            if 1 <= scene_number <= total_scenes and scene_number not in unique_numbers:
                unique_numbers.append(scene_number)
        return unique_numbers

    def _merge_repair_targets(self, *repair_target_groups: list[RepairTarget]) -> list[RepairTarget]:
        """Merges multiple repair-target lists into one deduplicated scene list."""

        merged: dict[int, RepairTarget] = {}
        for group in repair_target_groups:
            for target in group:
                existing = merged.get(target.scene_number)
                if existing is None:
                    merged[target.scene_number] = RepairTarget(
                        scene_number=target.scene_number,
                        reason=target.reason.strip(),
                        rewrite_brief=target.rewrite_brief.strip(),
                    )
                    continue
                merged[target.scene_number] = RepairTarget(
                    scene_number=target.scene_number,
                    reason="; ".join(
                        dict.fromkeys(
                            [
                                existing.reason.strip(),
                                target.reason.strip(),
                            ]
                        )
                    ).strip(),
                    rewrite_brief="\n".join(
                        dict.fromkeys(
                            [
                                existing.rewrite_brief.strip(),
                                target.rewrite_brief.strip(),
                            ]
                        )
                    ).strip(),
                )
        return [merged[number] for number in sorted(merged)]

    def _build_repair_context_report(
        self,
        *,
        global_report: GlobalQaReport,
        repair_targets: list[RepairTarget],
    ) -> GlobalQaReport:
        """Builds the report object passed into targeted scene repairs."""

        return global_report.model_copy(
            update={
                "pass_fail": False,
                "major_problems": list(
                    dict.fromkeys(
                        list(global_report.major_problems)
                        + [target.reason for target in repair_targets]
                    )
                )[:12],
                "repair_targets": repair_targets,
            }
        )

    def _counterforce_keywords(self, editorial_blueprint: EditorialBlueprint) -> set[str]:
        """Returns story-specific and generic keywords for pursuit pressure."""

        return self._keyword_tokens(
            editorial_blueprint.counterforce_name,
            editorial_blueprint.counterforce_role,
            "review",
            "suspicion",
            "audit",
            "memo",
            "trace",
            "footprint",
            "inquiry",
            "request",
            "pursuit",
            "exposure",
            "hunter",
            "threat",
        )

    def _relationship_keywords(self, editorial_blueprint: EditorialBlueprint) -> set[str]:
        """Returns story-specific and generic keywords for relationship cost."""

        return self._keyword_tokens(
            editorial_blueprint.relationship_focus_name,
            editorial_blueprint.relationship_focus_role,
            "relationship",
            "marriage",
            "trust",
            "distance",
            "withdraw",
            "silence",
            "leave",
            "refuse",
            "cold",
            "home",
            "love",
            "betray",
            "intimacy",
            "family",
        )

    def _keyword_tokens(self, *values: str) -> set[str]:
        """Normalizes keyword tokens from names, roles, and generic hints."""

        tokens: set[str] = set()
        for value in values:
            for token in re.findall(r"[a-z0-9']+", (value or "").lower()):
                if len(token) >= 4:
                    tokens.add(token)
        return tokens

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

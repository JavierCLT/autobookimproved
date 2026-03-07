"""Planning, drafting, rewriting, and continuity generation."""

from __future__ import annotations

import re

from novel_factory.config import AppConfig
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    initial_continuity_user_prompt,
    planning_system_prompt,
    repair_scene_user_prompt,
    scene_cards_user_prompt,
    scene_draft_system_prompt,
    scene_draft_user_prompt,
    story_spec_user_prompt,
    outline_user_prompt,
)
from novel_factory.schemas import (
    ContinuityState,
    Outline,
    SceneCard,
    SceneCardCollection,
    StorySpec,
)
from novel_factory.utils import get_chapter_plan


class NovelGenerator:
    """Generates planning artifacts and prose with the Responses API."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def generate_story_spec(self, synopsis: str) -> StorySpec:
        """Creates the locked story specification from the synopsis."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=self.config.default_audience,
                rating_ceiling=self.config.default_rating_ceiling,
                market_position=self.config.default_market_position,
            ),
            user_prompt=story_spec_user_prompt(
                synopsis=synopsis,
                target_words=self.config.target_words,
                chapters=self.config.target_chapters,
                scenes=self.config.target_scenes,
                audience=self.config.default_audience,
                rating_ceiling=self.config.default_rating_ceiling,
                market_position=self.config.default_market_position,
            ),
            schema=StorySpec,
            task_name="story_spec",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=8_000,
            verbosity="low",
        )

    def generate_outline(self, synopsis: str, story_spec: StorySpec) -> Outline:
        """Creates the macro outline from the locked story spec."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=outline_user_prompt(synopsis=synopsis, story_spec=story_spec),
            schema=Outline,
            task_name="outline",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=16_000,
            verbosity="low",
        )

    def generate_scene_cards(
        self,
        synopsis: str,
        story_spec: StorySpec,
        outline: Outline,
    ) -> list[SceneCard]:
        """Creates and normalizes all scene cards."""

        collection = self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=scene_cards_user_prompt(
                synopsis=synopsis,
                story_spec=story_spec,
                outline=outline,
            ),
            schema=SceneCardCollection,
            task_name="scene_cards",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=24_000,
            verbosity="low",
        )
        return self._normalize_scene_cards(collection.scene_cards, story_spec, outline)

    def generate_initial_continuity(
        self,
        story_spec: StorySpec,
        outline: Outline,
    ) -> ContinuityState:
        """Creates the initial continuity state."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=initial_continuity_user_prompt(story_spec=story_spec, outline=outline),
            schema=ContinuityState,
            task_name="initial_continuity",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=3_000,
            verbosity="low",
        )

    def draft_scene(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        recent_scene_summaries: list[str],
        rewrite_brief: str | None = None,
        current_draft: str | None = None,
    ) -> str:
        """Drafts or rewrites a scene in prose."""

        is_rewrite = rewrite_brief is not None or current_draft is not None
        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=scene_draft_user_prompt(
                story_spec=story_spec,
                outline=outline,
                scene_card=scene_card,
                continuity_state=continuity_state,
                recent_scene_summaries=recent_scene_summaries,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
            ),
            task_name=f"{'rewrite' if is_rewrite else 'draft'}_scene_{scene_card.scene_number:02d}",
            reasoning_effort=(
                self.config.reasoning.rewriting if is_rewrite else self.config.reasoning.drafting
            ),
            temperature=(
                self.config.rewriting_temperature
                if is_rewrite
                else self.config.drafting_temperature
            ),
            max_output_tokens=7_000,
        )

    def update_continuity(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        scene_text: str,
    ) -> ContinuityState:
        """Updates continuity deterministically from the locked scene contract."""

        facts_to_add = list(scene_card.continuity_outputs)
        open_threads_to_add = [scene_card.revelation_or_shift, scene_card.pressure_source]
        relationship_updates = [f"{scene_card.pov_character}: {scene_card.emotional_turn}"]
        costs_to_add = self._select_matching_lines(
            scene_card.continuity_outputs + [scene_card.conflict, scene_card.pressure_source],
            keywords=("cost", "risk", "threat", "exposure", "loss", "injury", "suspicion"),
        )
        evidence_to_add = self._extract_evidence_like_items(
            scene_card.required_entities + scene_card.continuity_outputs
        )
        promises_to_add = self._select_matching_lines(
            scene_card.continuity_outputs + [scene_card.ending_mode],
            keywords=("will", "must", "needs to", "promise", "plan", "agrees to"),
        )
        recent_scene_summary = self._build_scene_summary(scene_card)

        return ContinuityState(
            current_day=scene_card.time_marker.strip() or continuity_state.current_day,
            current_location=scene_card.location.strip() or continuity_state.current_location,
            known_facts=self._merge_list(continuity_state.known_facts, facts_to_add, limit=40),
            open_threads=self._merge_list(continuity_state.open_threads, open_threads_to_add, limit=20),
            relationship_state=self._merge_list(
                continuity_state.relationship_state,
                relationship_updates,
                limit=20,
            ),
            injuries_or_costs=self._merge_list(
                continuity_state.injuries_or_costs,
                costs_to_add,
                limit=20,
            ),
            evidence_or_objects=self._merge_list(
                continuity_state.evidence_or_objects,
                evidence_to_add,
                limit=25,
            ),
            unresolved_promises=self._merge_list(
                continuity_state.unresolved_promises,
                promises_to_add,
                limit=20,
            ),
            disallowed_entities=self._merge_list(
                continuity_state.disallowed_entities,
                [],
                limit=25,
            ),
            recent_scene_summaries=(continuity_state.recent_scene_summaries + [recent_scene_summary])[-3:],
            last_approved_scene_number=scene_card.scene_number,
        )

    def repair_scene(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        current_scene: str,
        global_qa_report,
        rewrite_brief: str,
    ) -> str:
        """Runs a targeted repair on an already approved scene."""

        return self.llm.text(
            system_prompt=scene_draft_system_prompt(),
            user_prompt=repair_scene_user_prompt(
                story_spec=story_spec,
                outline=outline,
                scene_card=scene_card,
                continuity_state=continuity_state,
                current_scene=current_scene,
                global_qa_report=global_qa_report,
                rewrite_brief=rewrite_brief,
            ),
            task_name=f"repair_scene_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.repair,
            temperature=self.config.rewriting_temperature,
            max_output_tokens=7_000,
        )

    def _normalize_scene_cards(
        self,
        scene_cards: list[SceneCard],
        story_spec: StorySpec,
        outline: Outline,
    ) -> list[SceneCard]:
        """Stabilizes numbering, titles, and target lengths for scene cards."""

        if len(scene_cards) != story_spec.expected_scenes:
            raise ValueError(
                f"Expected {story_spec.expected_scenes} scene cards, received {len(scene_cards)}."
            )

        ordered = sorted(scene_cards, key=lambda card: (card.chapter_number, card.scene_number))
        normalized: list[SceneCard] = []
        for index, scene_card in enumerate(ordered, start=1):
            chapter_plan = get_chapter_plan(outline, scene_card.chapter_number)
            target_words = min(2_200, max(900, scene_card.target_words))
            normalized.append(
                scene_card.model_copy(
                    update={
                        "scene_number": index,
                        "chapter_title": chapter_plan.title,
                        "target_words": target_words,
                    }
                )
            )
        return normalized

    def _merge_list(self, current: list[str], additions: list[str], *, limit: int) -> list[str]:
        """Appends unique normalized strings and caps the list."""

        merged = list(current)
        for item in additions:
            cleaned = item.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged[-limit:]

    def _select_matching_lines(self, lines: list[str], *, keywords: tuple[str, ...]) -> list[str]:
        """Returns lines containing operational keywords."""

        selected = []
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in keywords):
                selected.append(line)
        return selected

    def _extract_evidence_like_items(self, values: list[str]) -> list[str]:
        """Pulls object-like entities into continuity."""

        keywords = (
            "file",
            "ledger",
            "account",
            "loan",
            "email",
            "badge",
            "key",
            "phone",
            "report",
            "document",
            "api",
            "patch",
            "code",
            "provision",
            "model",
            "statement",
        )
        selected = []
        for value in values:
            lower_value = value.lower()
            if any(keyword in lower_value for keyword in keywords):
                selected.append(value)
        return selected

    def _build_scene_summary(self, scene_card: SceneCard) -> str:
        """Builds a short operational summary from the scene card."""

        summary = (
            f"{scene_card.pov_character} in {scene_card.location} faces {scene_card.conflict}; "
            f"{scene_card.revelation_or_shift.lower()}."
        )
        return re.sub(r"\s+", " ", summary).strip()[:220]

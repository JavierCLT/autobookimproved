"""Planning, drafting, rewriting, and continuity generation."""

from __future__ import annotations

import re

from novel_factory.config import AppConfig
from novel_factory.intake import (
    build_drafting_guidance,
    build_planning_guidance,
    get_reference_passages,
    resolve_planning_defaults,
)
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    editorial_blueprint_user_prompt,
    initial_continuity_user_prompt,
    outline_user_prompt,
    pacing_analysis_system_prompt,
    pacing_analysis_user_prompt,
    planning_system_prompt,
    plant_payoff_system_prompt,
    plant_payoff_user_prompt,
    repair_scene_user_prompt,
    scene_cards_user_prompt,
    scene_draft_system_prompt,
    scene_draft_user_prompt,
    story_spec_user_prompt,
    subplot_weave_system_prompt,
    subplot_weave_user_prompt,
    voice_calibration_system_prompt,
    voice_calibration_user_prompt,
)
from novel_factory.schemas import (
    BookIntake,
    ContinuityState,
    EditorialBlueprint,
    Outline,
    PacingAnalysis,
    PlantPayoffMap,
    SceneCard,
    SceneCardCollection,
    StorySpec,
    SubplotWeaveMap,
    VoiceDNA,
)
from novel_factory.utils import get_chapter_plan, truncate_text


class NovelGenerator:
    """Generates planning artifacts and prose with the Responses API."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def calibrate_voice(self, book_intake: BookIntake | None, *, synopsis: str) -> VoiceDNA | None:
        """Extracts an optional voice fingerprint from reference passages in the intake."""

        reference_passages = get_reference_passages(book_intake).strip()
        if not reference_passages:
            return None

        planning_defaults = resolve_planning_defaults(
            intake=book_intake,
            default_audience=self.config.default_audience,
            default_rating_ceiling=self.config.default_rating_ceiling,
            default_market_position=self.config.default_market_position,
            default_target_words=self.config.target_words,
            default_expected_chapters=self.config.target_chapters,
            default_expected_scenes=self.config.target_scenes,
        )
        return self.llm.structured(
            system_prompt=voice_calibration_system_prompt(),
            user_prompt=voice_calibration_user_prompt(
                reference_passages=reference_passages,
                genre=(book_intake.fields.get("genre", "").strip() if book_intake else "")
                or planning_defaults.market_position
                or "thriller",
                audience=planning_defaults.audience,
            ),
            schema=VoiceDNA,
            task_name="voice_calibration",
            reasoning_effort=self.config.reasoning.voice_calibration,
            temperature=self.config.qa_temperature,
            max_output_tokens=3_500,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )

    def generate_story_spec(
        self,
        synopsis: str,
        book_intake: BookIntake | None = None,
        voice_dna: VoiceDNA | None = None,
    ) -> StorySpec:
        """Creates the locked story specification from the synopsis."""

        planning_defaults = resolve_planning_defaults(
            intake=book_intake,
            default_audience=self.config.default_audience,
            default_rating_ceiling=self.config.default_rating_ceiling,
            default_market_position=self.config.default_market_position,
            default_target_words=self.config.target_words,
            default_expected_chapters=self.config.target_chapters,
            default_expected_scenes=self.config.target_scenes,
        )
        intake_guidance = build_planning_guidance(book_intake)
        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=planning_defaults.audience,
                rating_ceiling=planning_defaults.rating_ceiling,
                market_position=planning_defaults.market_position,
            ),
            user_prompt=story_spec_user_prompt(
                synopsis=synopsis,
                target_words=planning_defaults.target_words,
                chapters=planning_defaults.expected_chapters,
                scenes=planning_defaults.expected_scenes,
                audience=planning_defaults.audience,
                rating_ceiling=planning_defaults.rating_ceiling,
                market_position=planning_defaults.market_position,
                intake_guidance=intake_guidance,
                voice_dna_summary=self._build_voice_dna_summary(voice_dna),
            ),
            schema=StorySpec,
            task_name="story_spec",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=8_000,
            verbosity="low",
        )

    def generate_outline(
        self,
        synopsis: str,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        book_intake: BookIntake | None = None,
    ) -> Outline:
        """Creates the macro outline from the locked story spec."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=outline_user_prompt(
                synopsis=synopsis,
                story_spec=story_spec,
                editorial_blueprint=editorial_blueprint,
                intake_guidance=build_planning_guidance(book_intake),
            ),
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
        editorial_blueprint: EditorialBlueprint,
        plant_payoff_map: PlantPayoffMap | None = None,
        subplot_weave_map: SubplotWeaveMap | None = None,
        book_intake: BookIntake | None = None,
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
                editorial_blueprint=editorial_blueprint,
                plant_payoff_map=plant_payoff_map,
                subplot_weave_map=subplot_weave_map,
                intake_guidance=build_planning_guidance(book_intake),
            ),
            schema=SceneCardCollection,
            task_name="scene_cards",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=24_000,
            verbosity="low",
        )
        return self._normalize_scene_cards(collection.scene_cards, story_spec, outline)

    def generate_plant_payoff_map(
        self,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        outline: Outline,
        book_intake: BookIntake | None = None,
    ) -> PlantPayoffMap:
        """Creates the plant/payoff registry used during scene planning and QA."""

        return self.llm.structured(
            system_prompt=plant_payoff_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=plant_payoff_user_prompt(
                story_spec=story_spec,
                editorial_blueprint=editorial_blueprint,
                outline=outline,
                intake_guidance=build_planning_guidance(book_intake),
            ),
            schema=PlantPayoffMap,
            task_name="plant_payoff_map",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=6_000,
            verbosity="low",
        )

    def generate_subplot_weave(
        self,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
        outline: Outline,
        book_intake: BookIntake | None = None,
    ) -> SubplotWeaveMap:
        """Creates the subplot weave registry used during scene planning and QA."""

        return self.llm.structured(
            system_prompt=subplot_weave_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=subplot_weave_user_prompt(
                story_spec=story_spec,
                editorial_blueprint=editorial_blueprint,
                outline=outline,
                intake_guidance=build_planning_guidance(book_intake),
            ),
            schema=SubplotWeaveMap,
            task_name="subplot_weave_map",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=6_000,
            verbosity="low",
        )

    def generate_editorial_blueprint(
        self,
        synopsis: str,
        story_spec: StorySpec,
        book_intake: BookIntake | None = None,
    ) -> EditorialBlueprint:
        """Creates the pre-draft editorial scaffold used to sharpen the pipeline."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=editorial_blueprint_user_prompt(
                synopsis=synopsis,
                story_spec=story_spec,
                intake_guidance=build_planning_guidance(book_intake),
            ),
            schema=EditorialBlueprint,
            task_name="editorial_blueprint",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=10_000,
            verbosity="low",
        )

    def generate_initial_continuity(
        self,
        story_spec: StorySpec,
        outline: Outline,
        book_intake: BookIntake | None = None,
    ) -> ContinuityState:
        """Creates the initial continuity state."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=initial_continuity_user_prompt(
                story_spec=story_spec,
                outline=outline,
                intake_guidance=build_planning_guidance(book_intake),
            ),
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
        editorial_blueprint: EditorialBlueprint,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        recent_scene_summaries: list[str],
        voice_dna: VoiceDNA | None = None,
        book_intake: BookIntake | None = None,
        rewrite_brief: str | None = None,
        current_draft: str | None = None,
    ) -> str:
        """Drafts or rewrites a scene in prose."""

        is_rewrite = rewrite_brief is not None or current_draft is not None
        story_brief = self._build_story_brief(story_spec, editorial_blueprint)
        chapter_brief = self._build_chapter_brief(outline, scene_card)
        continuity_brief = self._build_continuity_brief(continuity_state)
        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=scene_draft_user_prompt(
                intake_guidance=build_drafting_guidance(book_intake),
                story_brief=story_brief,
                chapter_brief=chapter_brief,
                editorial_blueprint=editorial_blueprint,
                scene_card=scene_card,
                continuity_brief=continuity_brief,
                recent_scene_summaries=recent_scene_summaries,
                voice_dna_summary=self._build_voice_dna_summary(voice_dna),
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
            model_override=self.config.get_drafting_model(),
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

        open_threads_to_add = self._merge_candidate_updates(
            [
                scene_card.counterforce_trace,
                scene_card.revelation_or_shift,
                scene_card.pressure_source,
                scene_card.secret_pressure,
            ]
        )

        relationship_updates = self._merge_candidate_updates(
            [
                scene_card.relationship_delta,
                f"{scene_card.pov_character}: {scene_card.emotional_turn}",
            ]
        )

        suspicion_updates = self._merge_candidate_updates(
            [scene_card.suspicion_delta]
        )

        leverage_updates = self._merge_candidate_updates(
            [scene_card.power_shift]
        )

        moral_lines_crossed_to_add = self._detect_moral_line_crossings(
            story_spec=story_spec,
            scene_card=scene_card,
            scene_text=scene_text,
        )

        costs_to_add = self._merge_candidate_updates(
            [scene_card.cost_paid]
            + self._select_matching_lines(
                scene_card.continuity_outputs
                + [scene_card.conflict, scene_card.pressure_source, scene_card.secret_pressure],
                keywords=(
                    "cost",
                    "risk",
                    "threat",
                    "exposure",
                    "loss",
                    "injury",
                    "damage",
                    "suspicion",
                    "compromise",
                    "sacrifice",
                ),
            )
        )

        evidence_to_add = self._extract_evidence_like_items(
            scene_card.required_entities
            + scene_card.continuity_outputs
            + [scene_card.sensory_anchor, scene_card.counterforce_trace]
        )

        promises_to_add = self._select_matching_lines(
            scene_card.continuity_outputs
            + [scene_card.ending_mode, scene_card.scene_desire, scene_card.scene_fear],
            keywords=("will", "must", "needs to", "promise", "plan", "agrees to", "vows to"),
        )

        recent_scene_summary = self._build_scene_summary(scene_card)
        protagonist_name = self._primary_character_name(story_spec, fallback=scene_card.pov_character)
        knowledge_updates = self._character_knowledge_updates(
            protagonist_name=protagonist_name,
            scene_card=scene_card,
        )
        emotional_updates = self._emotional_state_updates(
            protagonist_name=protagonist_name,
            scene_card=scene_card,
        )
        character_locations = dict(continuity_state.character_locations)
        character_locations[scene_card.pov_character] = scene_card.location.strip() or continuity_state.current_location
        active_promises = self._merge_list(
            continuity_state.active_promises or continuity_state.unresolved_promises,
            promises_to_add,
            limit=25,
        )

        return ContinuityState(
            current_day=scene_card.time_marker.strip() or continuity_state.current_day,
            current_location=scene_card.location.strip() or continuity_state.current_location,
            character_locations=character_locations,
            known_facts=self._merge_list(continuity_state.known_facts, facts_to_add, limit=50),
            open_threads=self._merge_list(continuity_state.open_threads, open_threads_to_add, limit=25),
            relationship_state=self._merge_list(
                continuity_state.relationship_state,
                relationship_updates,
                limit=25,
            ),
            suspicion_state=self._merge_list(
                continuity_state.suspicion_state,
                suspicion_updates,
                limit=25,
            ),
            leverage_state=self._merge_list(
                continuity_state.leverage_state,
                leverage_updates,
                limit=25,
            ),
            moral_lines_crossed=self._merge_list(
                continuity_state.moral_lines_crossed,
                moral_lines_crossed_to_add,
                limit=20,
            ),
            injuries_or_costs=self._merge_list(
                continuity_state.injuries_or_costs,
                costs_to_add,
                limit=25,
            ),
            evidence_or_objects=self._merge_list(
                continuity_state.evidence_or_objects,
                evidence_to_add,
                limit=30,
            ),
            unresolved_promises=self._merge_list(
                continuity_state.unresolved_promises,
                promises_to_add,
                limit=25,
            ),
            active_promises=active_promises,
            character_knowledge=self._merge_mapping_lists(
                continuity_state.character_knowledge,
                knowledge_updates,
                limit=20,
            ),
            emotional_states={
                **continuity_state.emotional_states,
                **emotional_updates,
            },
            disallowed_entities=self._merge_list(
                continuity_state.disallowed_entities,
                [],
                limit=25,
            ),
            recent_scene_summaries=(
                continuity_state.recent_scene_summaries + [recent_scene_summary]
            )[-self.config.recent_scene_summaries :],
            last_approved_scene_number=scene_card.scene_number,
        )

    def repair_scene(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        editorial_blueprint: EditorialBlueprint,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        current_scene: str,
        global_qa_report,
        rewrite_brief: str,
        voice_dna: VoiceDNA | None = None,
        book_intake: BookIntake | None = None,
    ) -> str:
        """Runs a targeted repair on an already approved scene."""

        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=repair_scene_user_prompt(
                story_spec=story_spec,
                outline=outline,
                editorial_blueprint=editorial_blueprint,
                scene_card=scene_card,
                continuity_state=continuity_state,
                current_scene=current_scene,
                global_qa_report=global_qa_report,
                rewrite_brief=rewrite_brief,
                voice_dna_summary=self._build_voice_dna_summary(voice_dna),
                intake_guidance=build_drafting_guidance(book_intake),
            ),
            task_name=f"repair_scene_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.repair,
            temperature=self.config.rewriting_temperature,
            max_output_tokens=7_000,
            model_override=self.config.get_drafting_model(),
        )

    def analyze_pacing(self, manuscript_text: str, *, scene_count: int, story_spec: StorySpec) -> PacingAnalysis:
        """Runs a scene-by-scene pacing analysis on the assembled manuscript."""

        return self.llm.structured(
            system_prompt=pacing_analysis_system_prompt(story_spec),
            user_prompt=pacing_analysis_user_prompt(manuscript_text, scene_count),
            schema=PacingAnalysis,
            task_name="pacing_analysis",
            reasoning_effort=self.config.reasoning.global_qa,
            temperature=self.config.global_qa_temperature,
            max_output_tokens=5_000,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )

    def _merge_candidate_updates(self, values: list[str]) -> list[str]:
        """Normalizes candidate status strings and drops empty placeholders."""

        cleaned: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", value or "").strip()
            if not normalized:
                continue
            if normalized.lower() in {"none", "n/a", "no change", "unchanged"}:
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    def _detect_moral_line_crossings(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        scene_text: str,
    ) -> list[str]:
        """Infers irreversible ethical thresholds crossed in the scene."""

        candidates = [
            scene_card.cost_paid,
            scene_card.secret_pressure,
            scene_card.revelation_or_shift,
            scene_card.power_shift,
            scene_card.suspicion_delta,
            scene_card.relationship_delta,
        ]

        trigger_patterns = (
            r"\blie\b",
            r"\blies\b",
            r"\blied\b",
            r"\bforges?\b",
            r"\bsteals?\b",
            r"\btheft\b",
            r"\bdeceiv(?:e|es|ed|ing)\b",
            r"\bdeception\b",
            r"\bbetray(?:s|ed|al)?\b",
            r"\bsabotag(?:e|es|ed|ing)\b",
            r"\btamper(?:s|ed|ing)?\b",
            r"\bhide\s+evidence\b",
            r"\bconceal(?:s|ed|ing)?\b",
            r"\bcover-?up\b",
            r"\bcrosses?\s+a\s+line\b",
            r"\bbreaks?\s+protocol\b",
            r"\bviolates?\s+policy\b",
            r"\bdestroys?\s+trust\b",
        )

        def _matches_any_trigger(text: str) -> bool:
            lowered = (text or "").lower()
            return any(re.search(pattern, lowered) for pattern in trigger_patterns)

        def _contains_word(text: str, word: str) -> bool:
            return bool(re.search(rf"\b{re.escape(word.lower())}\b", text.lower()))

        hits = []
        for candidate in candidates:
            if _matches_any_trigger(candidate):
                hits.append(candidate.strip())

        text_lower = scene_text.lower()
        protagonist_name = self._primary_character_name(story_spec, fallback=scene_card.pov_character)
        text_triggers = [
            (
                f"{protagonist_name} lies on-page",
                tuple(filter(None, (protagonist_name.lower(),))),
                (r"\blie\b", r"\blies\b", r"\blied\b"),
            ),
            (
                "Evidence is concealed on-page",
                ("evidence",),
                (r"\bconceal(?:s|ed|ing)?\b", r"\bhide\s+evidence\b"),
            ),
            (
                "A critical system is tampered with on-page",
                ("system",),
                (r"\btamper(?:s|ed|ing)?\b",),
            ),
            (
                "Professional protocol is violated on-page",
                ("policy", "protocol", "compliance"),
                (r"\bviolates?\b", r"\bbreaks?\s+protocol\b"),
            ),
        ]
        for label, required_terms, trigger_regexes in text_triggers:
            if all(_contains_word(text_lower, term) for term in required_terms) and any(
                re.search(pattern, text_lower) for pattern in trigger_regexes
            ):
                hits.append(label)

        deduped: list[str] = []
        for hit in hits:
            normalized = re.sub(r"\s+", " ", hit).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

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
            cleaned = re.sub(r"\s+", " ", item or "").strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged[-limit:]

    def _merge_mapping_lists(
        self,
        current: dict[str, list[str]],
        additions: dict[str, list[str]],
        *,
        limit: int,
    ) -> dict[str, list[str]]:
        """Merges dictionary-backed list state while preserving uniqueness."""

        merged = {key: list(value) for key, value in current.items()}
        for key, values in additions.items():
            bucket = merged.setdefault(key, [])
            for value in values:
                cleaned = re.sub(r"\s+", " ", value or "").strip()
                if cleaned and cleaned not in bucket:
                    bucket.append(cleaned)
            if len(bucket) > limit:
                merged[key] = bucket[-limit:]
        return merged

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
            "notebook",
            "spreadsheet",
            "reserve",
            "reconciliation",
            "exception",
            "ticket",
            "memo",
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
            f"{scene_card.pov_character} in {scene_card.location} pursues {scene_card.scene_desire}; "
            f"{scene_card.power_shift.lower()} after {scene_card.closing_choice.lower()}."
        )
        return re.sub(r"\s+", " ", summary).strip()[:220]

    def _build_voice_dna_summary(self, voice_dna: VoiceDNA | None) -> str | None:
        """Builds a compact textual summary of the calibrated voice."""

        if voice_dna is None:
            return None
        techniques = ", ".join(voice_dna.characteristic_techniques[:5]) or "None listed"
        avoid = ", ".join(voice_dna.avoid_patterns[:5]) or "None listed"
        return (
            f"Register: {voice_dna.vocabulary_register}\n"
            f"Rhythm: {voice_dna.rhythm_signature}\n"
            f"Techniques: {techniques}\n"
            f"Avoid: {avoid}\n"
            f"Sample paragraph: {voice_dna.sample_paragraph}"
        )

    def _build_story_brief(
        self,
        story_spec: StorySpec,
        editorial_blueprint: EditorialBlueprint,
    ) -> str:
        """Builds a compact drafting brief from the locked story contract."""

        cast_lines = [
            f"- {character.name}: {character.role}; needs {character.private_need}; fears {character.fear}."
            for character in story_spec.cast[:6]
        ]
        style_lines = list(dict.fromkeys(story_spec.style_guide.prose_traits[:4] + story_spec.style_guide.banned_tells[:3]))
        parts = [
            f"Title: {story_spec.title_working}",
            f"Promise: {story_spec.one_sentence_promise}",
            f"Commercial hook: {editorial_blueprint.commercial_hook}",
            f"Premise core: {story_spec.premise_core}",
            f"Emotional engine: {story_spec.emotional_engine}",
            f"Adversarial engine: {story_spec.adversarial_engine}",
            f"Moral fault line: {story_spec.moral_fault_line}",
            f"Relationship focus: {editorial_blueprint.relationship_focus_name} ({editorial_blueprint.relationship_focus_role})",
            f"Counterforce: {editorial_blueprint.counterforce_name} ({editorial_blueprint.counterforce_role})",
            "Key cast:\n" + ("\n".join(cast_lines) if cast_lines else "- None specified"),
            "Style guardrails:\n" + ("\n".join(f"- {line}" for line in style_lines) if style_lines else "- Keep pressure specific and readable."),
            "Voice anchors:\n" + (
                "\n".join(f"- {line}" for line in editorial_blueprint.voice_anchors[:6])
                if editorial_blueprint.voice_anchors
                else "- Keep pressure specific and readable."
            ),
            "Motif threads:\n" + (
                "\n".join(f"- {line}" for line in editorial_blueprint.motif_threads[:5])
                if editorial_blueprint.motif_threads
                else "- No motifs locked."
            ),
        ]
        return "\n".join(parts)

    def _build_chapter_brief(self, outline: Outline, scene_card: SceneCard) -> str:
        """Builds a compact drafting brief for the current chapter."""

        chapter_plan = get_chapter_plan(outline, scene_card.chapter_number)
        beat_lines = [
            f"- Beat {beat.slot_number}: {beat.beat_summary} ({beat.purpose})"
            for beat in chapter_plan.scenes[:4]
        ]
        parts = [
            f"Chapter purpose: {chapter_plan.purpose}",
            f"Chapter target words: {chapter_plan.target_words}",
            "Chapter beats:\n" + ("\n".join(beat_lines) if beat_lines else "- No chapter beats provided."),
        ]
        return "\n".join(parts)

    def _build_continuity_brief(self, continuity_state: ContinuityState) -> str:
        """Builds a compact continuity brief for prose drafting."""

        def block(title: str, lines: list[str], *, limit: int) -> str:
            trimmed = [truncate_text(line, 180) for line in lines[-limit:] if line.strip()]
            if not trimmed:
                return f"{title}:\n- None"
            return f"{title}:\n" + "\n".join(f"- {line}" for line in trimmed)

        parts = [
            f"Current day: {continuity_state.current_day}",
            f"Current location: {continuity_state.current_location}",
            block(
                "Character locations",
                [f"{name}: {location}" for name, location in continuity_state.character_locations.items()],
                limit=6,
            ),
            block("Known facts", continuity_state.known_facts, limit=6),
            block("Open threads", continuity_state.open_threads, limit=6),
            block("Relationship state", continuity_state.relationship_state, limit=5),
            block("Suspicion state", continuity_state.suspicion_state, limit=5),
            block("Leverage state", continuity_state.leverage_state, limit=5),
            block("Moral lines crossed", continuity_state.moral_lines_crossed, limit=4),
            block("Costs already in play", continuity_state.injuries_or_costs, limit=4),
            block("Evidence or objects", continuity_state.evidence_or_objects, limit=5),
            block("Unresolved promises", continuity_state.unresolved_promises, limit=4),
            block("Active promises", continuity_state.active_promises, limit=4),
            block(
                "Character knowledge",
                [
                    f"{name}: {', '.join(facts[-3:])}"
                    for name, facts in continuity_state.character_knowledge.items()
                    if facts
                ],
                limit=5,
            ),
            block(
                "Emotional states",
                [f"{name}: {state}" for name, state in continuity_state.emotional_states.items()],
                limit=5,
            ),
            block("Disallowed entities", continuity_state.disallowed_entities, limit=4),
        ]
        return "\n".join(parts)

    def _primary_character_name(self, story_spec: StorySpec, *, fallback: str) -> str:
        """Returns the likeliest protagonist name for generic continuity labels."""

        for character in story_spec.cast:
            role = character.role.lower()
            if any(keyword in role for keyword in ("protagonist", "lead", "main")):
                return character.name.strip() or fallback
        if story_spec.cast and story_spec.cast[0].name.strip():
            return story_spec.cast[0].name.strip()
        return fallback.strip()

    def _character_knowledge_updates(
        self,
        *,
        protagonist_name: str,
        scene_card: SceneCard,
    ) -> dict[str, list[str]]:
        """Builds deterministic character-knowledge updates from the approved scene card."""

        knowledge_points = self._merge_candidate_updates(
            [scene_card.revelation_or_shift] + list(scene_card.continuity_outputs)
        )
        if not knowledge_points:
            return {}
        return {protagonist_name: knowledge_points[:5]}

    def _emotional_state_updates(
        self,
        *,
        protagonist_name: str,
        scene_card: SceneCard,
    ) -> dict[str, str]:
        """Builds deterministic emotional-state updates from the approved scene card."""

        emotional_turn = re.sub(r"\s+", " ", scene_card.emotional_turn or "").strip()
        if not emotional_turn:
            return {}
        return {protagonist_name: emotional_turn}

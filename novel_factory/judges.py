"""Model-based QA judges for scenes and full manuscripts."""

from __future__ import annotations

from novel_factory.config import AppConfig
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    global_qa_system_prompt,
    global_qa_user_prompt,
    scene_qa_system_prompt,
    scene_qa_user_prompt,
)
from novel_factory.schemas import (
    ContinuityState,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    SceneCard,
    SceneQaReport,
    StorySpec,
)


class SceneJudge:
    """Runs the scene-level model judge."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        validation_report: DeterministicValidationReport,
        scene_text: str,
    ) -> SceneQaReport:
        """Scores and judges a drafted scene."""

        return self.llm.structured(
            system_prompt=scene_qa_system_prompt(story_spec),
            user_prompt=scene_qa_user_prompt(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity_state,
                validation_report=validation_report,
                scene_text=scene_text,
            ),
            schema=SceneQaReport,
            task_name=f"scene_qa_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
        )


class GlobalJudge:
    """Runs the global manuscript-level judge."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        manuscript_text: str,
    ) -> GlobalQaReport:
        """Scores and judges the assembled manuscript."""

        return self.llm.structured(
            system_prompt=global_qa_system_prompt(story_spec),
            user_prompt=global_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                manuscript_text=manuscript_text,
            ),
            schema=GlobalQaReport,
            task_name="global_qa",
            reasoning_effort=self.config.reasoning.global_qa,
            temperature=self.config.global_qa_temperature,
            max_output_tokens=3_500,
            verbosity="low",
        )

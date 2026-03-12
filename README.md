# Novel Factory

Novel Factory is a production-minded Python CLI that turns a long synopsis into a checkpointed thriller manuscript pipeline. It plans first, locks a story contract and outline, generates an editorial blueprint plus plant/payoff and subplot-weave maps, drafts scenes one at a time with continuity context, rejects weak scenes through deterministic and model-based QA, rewrites failed scenes, assembles chapter files, and runs manuscript-level QA plus targeted repair.

The repository is optimized for output quality rather than speed or token efficiency. It uses the official OpenAI Python SDK, the Responses API, and structured outputs for planning and QA artifacts.

The pipeline can now take either a plain synopsis file or a richer markdown intake file based on [BOOK_INTAKE_TEMPLATE.md](./BOOK_INTAKE_TEMPLATE.md). When an intake file is provided, its explicit market, relationship, style, structural, and non-negotiable constraints are threaded into planning and drafting. Optional `reference_passages` in the intake can also calibrate a reusable voice fingerprint that informs both drafting and repairs.

## What It Produces

Each project is stored under `runs/<project_slug>/` and includes:

- `book_intake.md` and `book_intake.json` when an intake file is used
- `input_synopsis.md`
- `story_spec.json`
- `editorial_blueprint.json`
- `voice/voice_dna.json` when reference passages are provided
- `outline.json`
- `plant_payoff_map.json`
- `subplot_weave_map.json`
- `scene_cards.json`
- `initial_continuity_state.json`
- `continuity_state.json`
- `scenes/scene_01.md`, `scene_02.md`, ...
- `qa/scene_01_qa.json`, `scene_02_qa.json`, ...
- `rewrites/scene_01_draft_attempt_00.md`, ...
- `candidates/scene_01_draft_attempt_00_candidate_01.md`, ... for anchor-scene best-of-N selection
- `chapters/Chapter_01.md`, `Chapter_02.md`, ...
- `final_manuscript.md`
- `final_manuscript.txt`
- `global_qa_report.json`
- `cold_reader_report.json`
- `pacing_analysis.json`
- `run_log.jsonl`

## Pipeline

1. `bootstrap`
   Reads either a synopsis or a structured intake file and generates `StorySpec`, an `EditorialBlueprint`, optional `VoiceDNA`, `Outline`, `PlantPayoffMap`, `SubplotWeaveMap`, `SceneCards`, and initial continuity.
2. `draft-scene`
   Builds scene context, drafts a scene, runs deterministic validators, runs scene QA, and rewrites up to two times if needed. Anchor scenes can use selective best-of-N candidate drafting before a single candidate advances into the rewrite loop.
3. `assemble-manuscript`
   Assembles approved scenes into chapter files and final manuscript outputs.
4. `editorial-qa`
   Runs chapter-level and arc-level editorial QA before manuscript-level judging.
5. `run-project`
   Runs the full pipeline from planning through manuscript assembly, editorial QA, global QA, and targeted repairs.
6. `global-qa`
   Evaluates the assembled manuscript for hook strength, payoff, continuity, boredom risk, voice consistency, and AI-smell risk, then adds cold-reader and pacing-analysis reports.
7. `repair-project`
   Rewrites only the scenes flagged by the latest global QA, cold-reader, or pacing-analysis pass, then reassembles and re-runs QA.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set your API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.4
OPENAI_DRAFTING_MODEL=
OPENAI_QA_MODEL=
NOVEL_FACTORY_RUN_ROOT=runs
NOVEL_FACTORY_DEFAULT_AUDIENCE=Adult
NOVEL_FACTORY_DEFAULT_RATING_CEILING=R
NOVEL_FACTORY_DEFAULT_MARKET_POSITION=adult thriller
NOVEL_FACTORY_ANCHOR_BEST_OF_N_ENABLED=true
NOVEL_FACTORY_ANCHOR_BEST_OF_N=3
```

`OPENAI_MODEL` defaults to `gpt-5.4`. You can optionally split prose generation and QA onto different models with `OPENAI_DRAFTING_MODEL` and `OPENAI_QA_MODEL`. `NOVEL_FACTORY_ANCHOR_BEST_OF_N_*` controls selective multi-candidate drafting for anchor scenes only. The audience and rating defaults can be changed through the `NOVEL_FACTORY_DEFAULT_*` variables when you want to switch between adult and YA projects.

## Usage

### Bootstrap a project from a synopsis

```bash
python main.py bootstrap --project my_book --synopsis-file synopsis.md
```

### Bootstrap a project from an intake template

```bash
python main.py bootstrap --project my_book --intake-file my_book_intake.md
```

### Draft one scene

```bash
python main.py draft-scene --project my_book --scene-index 5
```

Force a rewrite of an already approved scene:

```bash
python main.py draft-scene --project my_book --scene-index 5 --force
```

### Run the full project from a synopsis

```bash
python main.py run-project --project my_book --synopsis-file synopsis.md
```

### Assemble a manuscript from approved scenes

```bash
python main.py assemble-manuscript --project my_book
```

### Run chapter and arc editorial QA

```bash
python main.py editorial-qa --project my_book
```

### Run the full project from an intake template

```bash
python main.py run-project --project my_book --intake-file my_book_intake.md
```

### Run manuscript-level QA

```bash
python main.py global-qa --project my_book
```

### Apply targeted repairs from the last failed global QA

```bash
python main.py repair-project --project my_book
```

## Checkpointing And Resume Behavior

- Planning artifacts are saved once and reused on later runs.
- If an intake file is used, both the raw markdown and parsed JSON are saved with the run.
- Approved scenes are saved only after they pass deterministic validation and model QA.
- Failed attempts are preserved in `rewrites/`.
- Anchor-scene candidate drafts are preserved in `candidates/` before the selected candidate enters the normal rewrite loop.
- `continuity_state.json` is updated after each approved scene.
- `run_log.jsonl` records phase transitions and scene attempts.
- `run-project` resumes from the longest contiguous prefix of approved scenes.
- The editorial blueprint locks chapter missions, motif threads, escalation ladders, and ending payoffs before drafting starts.
- Plant/payoff and subplot-weave artifacts give the planner explicit foreshadowing and subplot threading before scene drafting begins.
- Cold-reader and pacing reports can trigger targeted scene repairs even when the main global QA pass is otherwise clean.
- Continuity can be rebuilt from `initial_continuity_state.json` plus approved scenes, which keeps the run recoverable after interruption or repair.

If a scene fails repeatedly, the run stops without saving it into `scenes/`. Fix the issue, adjust prompts or inputs if needed, and rerun `python main.py draft-scene ...` or `python main.py run-project ...`.

## Repository Layout

```text
.
|- main.py
|- BOOK_INTAKE_TEMPLATE.md
|- novel_factory/
|  |- config.py
|  |- generators.py
|  |- intake.py
|  |- judges.py
|  |- llm.py
|  |- pipeline.py
|  |- prompts.py
|  |- schemas.py
|  |- storage.py
|  |- utils.py
|  `- validators.py
|- requirements.txt
|- .env.example
`- runs/
```

## Notes

- The code uses `client.responses.create(...)` for prose generation.
- The code uses `client.responses.parse(...)` for structured artifacts and QA models.
- The pipeline reads `OPENAI_API_KEY` from environment variables through `python-dotenv`.
- Structured planning artifacts use Pydantic models to keep the run restartable and inspectable.
- Intake-template runs still produce `input_synopsis.md`; the synopsis is either read directly from `--synopsis-file` or extracted from the `synopsis:` field inside the intake file.

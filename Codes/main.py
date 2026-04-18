import json

from Codes.Orchestrators.AdventureContext import AdventureContext
from Codes.Databases.AdventureLogger import AdventureLogger
from Codes.Databases.FaissVectorDB import FAISSVectorDB

from Codes.Orchestrators.AdventureEngine import AdventureEngine
from Codes.Interfaces.ConsoleInterface import ConsoleInterface

from Codes.Orchestrators.GenerationUnit import GenerationUnit
from Codes.LLMHandlers.SimpleOllamaLLM import SimpleOllamaLLM

from Codes.PromptConstructors.NarratorPromptConstructor import NarratorPromptConstructor
from Codes.PromptConstructors.PlannerPromptConstructor import PlannerPromptConstructor
from Codes.Orchestrators.MemoryManager import MemoryManager


# =========================================================
# CONFIG
# =========================================================

ADVENTURE_NAME = "vanilla_fantasy"
PROMPT_CORE_JSON_PATH: str = "../SettingRawDataJSON/vanilla_fantasy/PromptCore.json"
ENGINE_CONFIG_PATH: str = "../Configs/AdventureEngineConfig.json"
USE_ENGINE_CONFIG_JSON: bool = True

NARRATOR_MODEL_NAME: str = "cogito:8b-v1-preview-llama-q4_K_M"
PLANNER_MODEL_NAME: str = "cogito:8b-v1-preview-llama-q4_K_M"
SUMMARIZER_MODEL_NAME: str = "cogito:8b-v1-preview-llama-q4_K_M"

# NARRATOR_MODEL_NAME: str = "qwen3:14b"
# PLANNER_MODEL_NAME: str = "qwen3:14b"
# SUMMARIZER_MODEL_NAME: str = "qwen3:14b"

MEMORY_INTERVAL: int = 4
GLOBAL_SUMMARY_INTERVAL: int = 12
NARRATOR_VISIBLE_TURNS: int = 4


# =========================================================
# BUILD ENGINE
# =========================================================

def build_engine() -> AdventureEngine:

    # =========================
    # LOAD CONFIG
    # =========================
    data: dict = load_json(PROMPT_CORE_JSON_PATH)

    narrator_role, narrator_instructions = pick_role_and_instructions(
        data,
        use_planner=False
    )

    planner_role, planner_instructions = pick_role_and_instructions(
        data,
        use_planner=True
    )

    narrator_system_prompt: str = f"{narrator_role}\n\n{narrator_instructions}"
    planner_system_prompt: str = f"{planner_role}\n\n{planner_instructions}"

    # =========================
    # DATABASES
    # =========================
    sql_db: AdventureLogger = AdventureLogger(adventure_name="vanilla_fantasy")
    vector_db: FAISSVectorDB = FAISSVectorDB(adventure_name="vanilla_fantasy")

    context: AdventureContext = AdventureContext(sql_db, vector_db)

    # =========================
    # NARRATOR
    # =========================
    narrator_prompt: NarratorPromptConstructor = NarratorPromptConstructor(
        system_prompt=narrator_system_prompt
    )

    narrator_llm: SimpleOllamaLLM = SimpleOllamaLLM(
        model=NARRATOR_MODEL_NAME,
        prompt_constructor=narrator_prompt
    )

    narrator_unit: GenerationUnit = GenerationUnit(
        narrator_llm,
        name="Narrator"
    )

    # =========================
    # PLANNER
    # =========================
    planner_prompt: PlannerPromptConstructor = PlannerPromptConstructor(
        system_prompt=planner_system_prompt
    )

    planner_llm: SimpleOllamaLLM = SimpleOllamaLLM(
        model=PLANNER_MODEL_NAME,
        prompt_constructor=planner_prompt
    )

    planner_unit: GenerationUnit = GenerationUnit(
        planner_llm,
        name="Planner"
    )

    # =========================
    # MEMORY
    # =========================
    memory_manager: MemoryManager = MemoryManager(
        model=SUMMARIZER_MODEL_NAME
    )

    # =========================
    # ENGINE
    # =========================
    engine: AdventureEngine = AdventureEngine(
        context=context,
        narrator_unit=narrator_unit,
        planner_unit=planner_unit,
        memory_manager=memory_manager,
        memory_interval=MEMORY_INTERVAL,
        global_summary_interval=GLOBAL_SUMMARY_INTERVAL,
        narrator_visible_turns=NARRATOR_VISIBLE_TURNS,
        load_config_from_json=USE_ENGINE_CONFIG_JSON,
        json_config_path=ENGINE_CONFIG_PATH,
        core_prompt=data
    )

    return engine

# =========================================================
# HELPERS
# =========================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_role_and_instructions(data: dict, use_planner: bool) -> tuple[str, str]:
    if use_planner:
        role = data.get("planner_role", "").strip()
        instructions = data.get("planner_instructions", "").strip()
    else:
        role = data.get("narrator_role", "").strip()
        instructions = data.get("narrator_instructions", "").strip()
    return role, instructions

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("=== INITIALIZING ADVENTURE ENGINE ===")

    engine: AdventureEngine = build_engine()

    print("=== ENGINE READY ===")

    cli: ConsoleInterface = ConsoleInterface(engine)
    cli.run()


if __name__ == "__main__":
    main()
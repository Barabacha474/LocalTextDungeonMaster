import json
import time
from typing import Optional, Tuple, Iterator

from Codes.Orchestrators.AdventureContext import AdventureContext
from Codes.Orchestrators.GenerationUnit import GenerationUnit
from Codes.Orchestrators.MemoryManager import MemoryManager


class AdventureEngine:
    """
    Core game engine.

    Responsible for:
    - processing player input
    - managing turns
    - running planner / narrator (STREAMING PIPELINE)
    - logging
    - memory lifecycle
    """

    def __init__(
        self,
        context: AdventureContext,
        narrator_unit: GenerationUnit,
        planner_unit: Optional[GenerationUnit],
        memory_manager: Optional[MemoryManager],
        memory_interval: int = 4,
        global_summary_interval: int = 12,
        narrator_visible_turns: int = 4,
        load_config_from_json: bool = False,  # NEW
        json_config_path: str = "Configs/AdventureEngineConfig.json",  # NEW
        core_prompt: Optional[dict] = None,  # NEW
    ):
        self.context: AdventureContext = context
        self.narrator: GenerationUnit = narrator_unit
        self.planner: Optional[GenerationUnit] = planner_unit
        self.memory_manager: Optional[MemoryManager] = memory_manager

        self.memory_interval: int = memory_interval
        self.global_summary_interval: int = global_summary_interval
        self.narrator_visible_turns: int = narrator_visible_turns

        # Planner toggle
        self.planner_enabled: bool = False

        # Engine-level settings
        self.show_planner: bool = True
        self.show_system_messages: bool = True

        self.core_prompt = core_prompt or {}
        # Continue message (NEW)
        self.continue_message = self.core_prompt.get("continue_message", "continue")

        # =========================================================
        # CONFIG STORAGE (UPDATED)
        # =========================================================

        self.narrator_settings = {
            "temperature": 0.6,
            "num_predict": 800,
            "auto_ctx": True,
            "ctx_margin": 1.1,
            "max_ctx": 8192,
            "debug_info": False,
            "debug_prompt": False
        }

        self.planner_settings = {
            "temperature": 0.5,
            "num_predict": 400,
            "auto_ctx": True,
            "ctx_margin": 1.1,
            "max_ctx": 4096,
            "debug_info": False,
            "debug_prompt": False
        }

        self.memory_settings = {
            "temperature": 0.2,
            "num_predict": 800,
            "num_turns": 6,
            "auto_ctx": True,
            "ctx_margin": 1.1,
            "max_ctx": 8192,
            "debug_info": False,
            "debug_prompt": False
        }

        self.rag_narrator_settings = {
            "n_history": 5,
            "k_per_cascade": 5,
            "number_of_cascades": 2,
            "threshold": 0.3,
            "chunk_size": None
        }
        self.rag_planner_settings = {
            "n_history": 5,
            "k_per_cascade": 5,
            "number_of_cascades": 2,
            "threshold": 0.3,
            "chunk_size": None
        }

        # =========================================================
        # CONFIG AUTO LOAD (NEW)
        # =========================================================

        if load_config_from_json:
            try:
                self.load_config_from_file(json_config_path)
                print(f"[CONFIG LOADED] {json_config_path}")
            except Exception as e:
                print(f"[CONFIG LOAD ERROR] {json_config_path}")
                print(f"Error: {e}")

        self._config_path = json_config_path

        # =========================================================
        # HELP SYSTEM (NEW)
        # =========================================================

        self.help_text = """
        Available commands:

        undo turn
            Delete entire last turn (Player + Planner + Narrator)

        undo narrator
            Delete only Narrator output of last turn

        regenerate plan
            Regenerate planner output for current turn

        regenerate
            Regenerate narrator output using existing plan

        continue
            Continue generation (planner if needed + narrator)

        planner on / planner off
            Enable or disable planner

        config save
            Save current configuration

        config load
            Reload configuration from file
        """

    # =========================================================
    # TURN MANAGEMENT
    # =========================================================

    def _get_current_turn_id(self) -> int:
        return self.context.get_turn_count() + 1

    # =========================================================
    # MAIN ENTRY (STREAM ONLY)
    # =========================================================

    def process_player_input_stream(self, user_input: str) -> Tuple[Iterator[str], float]:
        """
        Main streaming entry point.
        ALL flows (normal + commands) go through stream.
        """

        # 1. Commands
        command_result = self._handle_command_stream(user_input)
        if command_result is not None:
            return command_result

        # 2. Normal turn
        turn_id = self._get_current_turn_id()

        self.context.set_player_input(user_input)
        self.context.log_turn(turn_id, "Player", user_input)

        # 3. Run full pipeline
        return self._run_generation_pipeline(
            turn_id=turn_id,
            run_planner=self.planner_enabled,
            run_narrator=True
        )

    # =========================================================
    # ADVENTURE INITIALIZATION
    # =========================================================

    def initialize_adventure_stream(self):
        """
        Separate initialization call.
        Returns intro stream if needed.
        """

        if self.context.get_turn_count() > 0:
            return None

        plot_start = self.core_prompt.get("plot_start", "")
        opening_text = self.core_prompt.get("player_opening_text", "")

        def stream():
            parts = []

            if plot_start:
                parts.append(plot_start)

            if opening_text:
                yield self._event("system", "[ADVENTURE START]")
                yield self._event("system", opening_text)
                parts.append(opening_text)

            result = ""
            for part in parts:
                result += part + "\n"

            if result != "":
                self.context.log_turn(0, "Narrator", result)

            yield self._event("system", "[You may now act]")

        return stream(), 0.0

    # =========================================================
    # CORE PIPELINE
    # =========================================================

    def _run_generation_pipeline(
        self,
        turn_id: int,
        run_planner: bool,
        run_narrator: bool
    ) -> Tuple[Iterator[str], float]:
        """
        Universal streaming pipeline:
        planner → log → narrator → log

        This is the ONLY place where generation happens.
        """

        start_time = time.time()

        def stream():
            # =========================
            # PLANNER
            # =========================
            has_plan = self._has_planner_in_turn(turn_id)

            if run_planner and self.planner_enabled and self.planner and not has_plan:
                if self.show_system_messages:
                    yield self._event("system", "\n[PLANNER START]\n")

                planner_stream = self.planner.generate_stream(
                    context=self.context,
                    prompt_kwargs={
                        "core_prompt": self.core_prompt,
                        "rag_config": self.rag_planner_settings,
                        "use_few_shot_examples": self.planner_settings.get("use_few_shot_examples", False)
                    },
                    generation_kwargs=self.planner_settings
                )

                plan_text = ""

                for token in planner_stream:
                    plan_text += token

                    if self.show_planner:
                        yield self._event("llm", token)

                self.context.log_turn(turn_id, "Planner", plan_text)

                if self.show_system_messages:
                    yield self._event("system", "\n[PLANNER END]\n")

            # =========================
            # NARRATOR
            # =========================
            if run_narrator:
                if self.show_system_messages:
                    yield self._event("system", "\n[NARRATOR START]\n")

                narrator_stream = self.narrator.generate_stream(
                    context=self.context,
                    prompt_kwargs={
                        "core_prompt": self.core_prompt,
                        "rag_config": self.rag_narrator_settings,
                        "use_few_shot_examples": self.narrator_settings.get("use_few_shot_examples", False)
                    },
                    generation_kwargs=self.narrator_settings
                )

                full_text = ""

                for token in narrator_stream:
                    full_text += token
                    yield self._event("llm", token)

                self.context.log_turn(turn_id, "Narrator", full_text)

                if self.show_system_messages:
                    yield self._event("system", "\n[NARRATOR END]\n")

                # memory lifecycle
                for event in self._handle_memory(turn_id):
                    yield event

        return stream(), start_time

    # =========================================================
    # COMMANDS (STREAM-BASED)
    # =========================================================

    def _handle_command_stream(self, user_input: str):
        cmd = user_input.strip().lower()

        # -----------------------------------------------------
        # UNDO TURN (FULL DELETE)
        # -----------------------------------------------------
        if cmd == "undo turn":
            last = self.context.get_latest_turn()

            if not last:
                return self._single_stream("[Nothing to undo]")

            turn_id = last["turn_id"]

            success = self.context.delete_turn(turn_id)

            if success:
                self.context.delete_memories_after_turn(turn_id - 1)
                return self._single_stream(f"[Turn {turn_id} fully removed]")

            return self._single_stream("[Undo failed]")

        # -----------------------------------------------------
        # UNDO NARRATOR (PARTIAL DELETE)
        # -----------------------------------------------------
        if cmd == "undo narrator":
            last = self.context.get_latest_turn()

            if not last:
                return self._single_stream("[Nothing to undo]")

            turn_id = last["turn_id"]

            deleted = self.context.delete_turn_roles(turn_id, ["Narrator"])

            if deleted > 0:
                return self._single_stream(f"[Narrator removed from turn {turn_id}]")

            return self._single_stream("[No narrator to remove]")

        # -----------------------------------------------------
        # REGENERATE PLAN
        # -----------------------------------------------------
        if cmd == "regenerate plan":
            last = self.context.get_latest_turn()
            if not last:
                return self._single_stream("[Nothing to regenerate]")

            turn_id = last["turn_id"]

            self.context.delete_turn_roles(turn_id, ["Narrator", "Planner"])

            return self._run_generation_pipeline(
                turn_id=turn_id,
                run_planner=True,
                run_narrator=False
            )

        # -----------------------------------------------------
        # REGENERATE (NARRATOR ONLY)
        # -----------------------------------------------------
        if cmd == "regenerate":
            last = self.context.get_latest_turn()
            if not last:
                return self._single_stream("[Nothing to regenerate]")

            turn_id = last["turn_id"]

            self.context.delete_turn_roles(turn_id, ["Narrator"])

            return self._run_generation_pipeline(
                turn_id=turn_id,
                run_planner=False,
                run_narrator=True
            )

        # -----------------------------------------------------
        # CONTINUE
        # -----------------------------------------------------
        if cmd == "continue":
            turn_id = self._get_current_turn_id()

            continue_text = self.continue_message

            self.context.set_player_input(continue_text)
            self.context.log_turn(turn_id, "Player", continue_text)

            has_plan = self._has_planner_in_turn(turn_id)

            return self._run_generation_pipeline(
                turn_id=turn_id,
                run_planner=(self.planner_enabled and not has_plan),
                run_narrator=True
            )

        # -----------------------------------------------------
        # PLANNER TOGGLE
        # -----------------------------------------------------
        if cmd == "planner on":
            self.planner_enabled = True
            return self._single_stream("[Planner enabled]")

        if cmd == "planner off":
            self.planner_enabled = False
            return self._single_stream("[Planner disabled]")

        return None

    # =========================================================
    # HELPERS
    # =========================================================

    def _single_stream(self, text: str):
        """
        Wrap simple responses into stream format
        (for CLI compatibility)
        """

        def gen():
            yield self._event("system", text)

        return gen(), 0.0

    def _has_planner_in_turn(self, turn_id: int) -> bool:
        """
        Check if planner already exists in given turn
        """
        turns = self.context.get_turns_range(turn_id, turn_id)
        return any(t["role"] == "Planner" for t in turns)

    def get_help_text(self) -> str:
        return self.help_text

    def _event(self, type_: str, content: str):
        return {
            "type": type_,
            "content": content
        }

    # =========================================================
    # MEMORY SYSTEM
    # =========================================================

    def _handle_memory(self, turn_id: int):
        def events():
            if not self.memory_manager:
                return iter([])

            # -------------------------
            # SHORT MEMORY
            # -------------------------
            if turn_id % self.memory_interval == 0 and turn_id > self.memory_interval:

                if self.show_system_messages:
                    yield self._event("system", "[MEMORY START]")

                print("[START GENERATING MEMORY]")

                summary = self.memory_manager.generate_memory(
                    context=self.context,
                    **self.memory_settings
                )

                print(f"[MEMORY GENERATED] - {summary}")

                if self.show_system_messages:
                    yield self._event("system", "[MEMORY SAVED]")

                start_turn = max(1, turn_id - (self.memory_interval + self.narrator_visible_turns))
                end_turn = turn_id - self.narrator_visible_turns

                if end_turn >= start_turn:
                    self.context.save_memory(
                        text=summary,
                        turn_start=start_turn,
                        turn_end=end_turn
                    )

            # -------------------------
            # GLOBAL SUMMARY
            # -------------------------
            if turn_id % self.global_summary_interval == 0:

                if self.show_system_messages:
                    yield self._event("system", "[GLOBAL MEMORY START]")

                global_summary = self.memory_manager.generate_global_summary(
                    context=self.context,
                    **self.memory_settings
                )

                if self.show_system_messages:
                    yield self._event("system", "[GLOBAL MEMORY SAVED]")

                self.context.vector_db.insert_single(
                    text=global_summary,
                    metadata={
                        "type": "global_summary",
                        "turn": turn_id
                    }
                )
                self.context.vector_db.save()

        return events()

    # =========================================================
    # CONFIG HANDLING
    # =========================================================

    def load_config(self, config: dict):
        """
        Load settings from config dict
        """

        # engine
        engine_cfg = config.get("engine", {})
        self.show_planner = engine_cfg.get("show_planner", True)
        self.show_system_messages = engine_cfg.get("show_system_messages", True)
        self.planner_enabled = engine_cfg.get("use_planner", False)

        # narrator
        self.narrator_settings.update(config.get("narrator", {}))

        # planner
        self.planner_settings.update(config.get("planner", {}))

        # memory
        self.memory_settings.update(config.get("memory", {}))

        # rag
        self.rag_narrator_settings.update(config.get("rag_narrator", {}))
        self.rag_planner_settings.update(config.get("rag_planner", {}))

    def export_config(self) -> dict:
        """
        Export current settings to dict
        """
        return {
            "engine": {
                "show_planner": self.show_planner,
                "show_system_messages": self.show_system_messages,
                "use_planner": self.planner_enabled
            },
            "narrator": self.narrator_settings,
            "planner": self.planner_settings,
            "memory": self.memory_settings,
            "rag_narrator": self.rag_narrator_settings,
            "rag_planner": self.rag_planner_settings
        }

    def save_config(self, filepath: str = None):
        if not filepath:
            filepath = self._config_path
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_config(), f, indent=4, ensure_ascii=False)

    def load_config_from_file(self, filepath: str = None):
        if not filepath:
            filepath = self._config_path

        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.load_config(config)
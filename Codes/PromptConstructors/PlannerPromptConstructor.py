from typing import Optional, Dict, Any, List
from Codes.AbstractClasses.AbstractPromptConstructor import AbstractPromptConstructor


class PlannerPromptConstructor(AbstractPromptConstructor):
    """
    Advanced planner prompt constructor.

    Planner sees:
    - instructions
    - setting
    - global summary
    - RAG (context-aware)
    - full recent history (ALL roles)
    - ALL planner outputs in history window
    - current player input
    """

    def __init__(
        self,
        planner_instructions: str,
        history_turns: int = 10,
        include_setting: bool = True,
        include_summary: bool = True,
        include_relevant_info: bool = True
    ):
        self.planner_instructions = planner_instructions
        self.history_turns = history_turns

        self.include_setting = include_setting
        self.include_summary = include_summary
        self.include_relevant_info = include_relevant_info

    # =========================================================
    # MAIN
    # =========================================================

    def construct_prompt(
        self,
        context,
        **kwargs
    ) -> str:

        core_prompt = kwargs.get("core_prompt")
        rag_config = kwargs.get("rag_config")

        parts: List[str] = []

        # =========================================================
        # ROLE + INSTRUCTIONS
        # =========================================================
        parts.append("ROLE:\nAdventure Planner")
        parts.append(f"INSTRUCTIONS:\n{self.planner_instructions}")

        # =========================================================
        # SETTING
        # =========================================================
        if self.include_setting and core_prompt:
            setting = core_prompt.get("setting", "")
            if setting:
                parts.append(f"SETTING:\n{setting}")

        # =========================================================
        # GLOBAL SUMMARY
        # =========================================================
        if self.include_summary:
            summary = context.get_latest_global_summary()
            if summary:
                parts.append(f"GLOBAL STORY SUMMARY:\n{summary}")

        # =========================================================
        # RAG (CONTEXT-AWARE)
        # =========================================================
        if self.include_relevant_info:
            rag_block = self._build_rag_block(context, rag_config)
            if rag_block:
                parts.append(rag_block)

        # =========================================================
        # FULL HISTORY (ALL ROLES)
        # =========================================================
        history = context.get_last_turns(self.history_turns)

        if history:
            history_block = "\n".join(
                f"{t['role']}: {t['content']}"
                for t in history
            )
            parts.append(f"RECENT HISTORY:\n{history_block}")

        # =========================================================
        # PLANNER MEMORY (ВСЕ ПЛАНЫ В ОКНЕ)
        # =========================================================
        planner_history = context.get_last_turns(
            self.history_turns,
            roles=["Planner"]
        )

        if planner_history:
            planner_block = "\n".join(
                f"{t['content']}"
                for t in planner_history
            )
            parts.append(f"PLANNER MEMORY:\n{planner_block}")

        # =========================================================
        # PLAYER INPUT
        # =========================================================
        player_input = context.get_player_input()
        if player_input:
            parts.append(f"CURRENT PLAYER ACTION:\n{player_input}")

        # =========================================================
        # OUTPUT FORMAT
        # =========================================================
        parts.append(
            "PLANNER OUTPUT FORMAT:\n"
            "- scene intent\n"
            "- hidden events\n"
            "- pacing\n"
            "- tension\n"
            "- twists\n"
            "- constraints\n"
        )

        parts.append("Planner:")

        return "\n\n".join(parts)

    # =========================================================
    # INTERNAL
    # =========================================================

    def _build_rag_block(self, context, rag_config: Optional[Dict[str, Any]]):

        player_input = context.get_player_input()
        if not player_input:
            return None

        rag_config = rag_config or {}

        n_history = rag_config.get("n_history", 6)
        k = rag_config.get("k_per_cascade", 5)
        cascades = rag_config.get("number_of_cascades", 2)
        threshold = rag_config.get("threshold", 0.3)
        chunk_size = rag_config.get("chunk_size")

        # 👉 используем helper из AdventureContext
        query = context.build_search_query(player_input, n_history)

        relevant = context.get_relevant_info(
            query=query,
            k_per_cascade=k,
            number_of_cascades=cascades,
            threshold=threshold,
            chunk_size=chunk_size
        )

        if not relevant:
            return None

        info_text = "\n".join(m["text"] for m in relevant)

        return f"RELEVANT INFORMATION:\n{info_text}"
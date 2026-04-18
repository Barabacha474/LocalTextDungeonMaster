from typing import Optional, Dict, Any, List

from Codes.AbstractClasses.AbstractPromptConstructor import AbstractPromptConstructor
from Codes.Orchestrators.AdventureContext import AdventureContext


class NarratorPromptConstructor(AbstractPromptConstructor):
    """
    Full narrator prompt constructor.

    Includes:
    - system instructions
    - setting (core prompt)
    - global summary
    - RAG (context-aware)
    - recent story (Player + Narrator)
    - last planner output (previous turn only)
    - player input
    """

    def __init__(
        self,
        system_prompt: str,
        history_turns: int = 6,
        include_setting: bool = True,
        include_global_summary: bool = True,
        include_relevant_info: bool = True
    ):
        self.system_prompt = system_prompt

        self.history_turns = history_turns
        self.include_setting = include_setting
        self.include_global_summary = include_global_summary
        self.include_relevant_info = include_relevant_info

    # =========================================================
    # MAIN
    # =========================================================

    def construct_prompt(
        self,
        context: AdventureContext,
        **kwargs
    ) -> str:

        core_prompt: Optional[Dict[str, Any]] = kwargs.get("core_prompt")
        rag_config: Optional[Dict[str, Any]] = kwargs.get("rag_config")
        use_few_shot = kwargs.get("use_few_shot_examples", False)

        separator = "=" * 30

        parts: List[str] = []

        # =========================================================
        # SYSTEM / INSTRUCTIONS
        # =========================================================
        parts.append(self.system_prompt)
        parts.append(separator)

        # =========================================================
        # FEW SHOT EXAMPLE (OPTIONAL)
        # =========================================================

        if use_few_shot and core_prompt:
            example = core_prompt.get("narrator_few_shot_example")
            if example:
                parts.append(example)
                parts.append(separator)

        # =========================================================
        # SETTING
        # =========================================================
        if self.include_setting and core_prompt:
            setting = core_prompt.get("setting", "")
            if setting:
                parts.append(f"SETTING:\n{setting}")
                parts.append(separator)

        # =========================================================
        # GLOBAL SUMMARY
        # =========================================================
        if self.include_global_summary:
            summary = context.get_latest_global_summary()
            if summary:
                parts.append(f"GLOBAL STORY SUMMARY:\n{summary}")
                parts.append(separator)

        # =========================================================
        # RAG (CONTEXT-AWARE)
        # =========================================================
        if self.include_relevant_info:
            rag_block = self._build_rag_block(context, rag_config)
            if rag_block:
                parts.append(rag_block)
                parts.append(separator)

        # =========================================================
        # HISTORY (Player + Narrator only)
        # =========================================================
        history = context.get_last_turns(
            n=self.history_turns,
            roles=["Player", "Narrator"]
        )

        if history:
            history_text = "\n".join(
                f"{t['role']}: {t['content']}"
                for t in history
            )
            parts.append(f"RECENT STORY:\n{history_text}")
            parts.append(separator)

        # =========================================================
        # LAST PLANNER OUTPUT (previous turn only)
        # =========================================================
        turn_count = context.get_turn_count()

        if turn_count > 0:
            planner_turn = context.get_turns_range(
                start=turn_count,
                end=turn_count,
                roles=["Planner"]
            )

            if planner_turn:
                last_plan = planner_turn[-1]["content"]
                parts.append(f"[PLANNER PLAN]\n{last_plan}")
                parts.append(separator)

        # =========================================================
        # PLAYER INPUT
        # =========================================================
        player_input = context.get_player_input()

        if player_input:
            parts.append(f"Player: {player_input}")
        else:
            parts.append("Player:")
        parts.append(separator)

        # =========================================================
        # OUTPUT PREFIX
        # =========================================================
        parts.append("Narrator:")

        return "\n\n".join(parts)

    # =========================================================
    # INTERNAL: RAG
    # =========================================================

    def _build_rag_block(
        self,
        context: AdventureContext,
        rag_config: Optional[Dict[str, Any]]
    ) -> Optional[str]:

        player_input = context.get_player_input()
        if not player_input:
            return None

        rag_config = rag_config or {}

        # -------------------------
        # CONFIG
        # -------------------------
        n_history = rag_config.get("n_history", 5)
        k = rag_config.get("k_per_cascade", 5)
        cascades = rag_config.get("number_of_cascades", 1)
        threshold = rag_config.get("threshold", 0.3)
        chunk_size = rag_config.get("chunk_size")

        # -------------------------
        # QUERY (history-aware)
        # -------------------------
        query = context.build_search_query(
            user_input=player_input,
            n_history=n_history
        )

        # -------------------------
        # SEARCH
        # -------------------------
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
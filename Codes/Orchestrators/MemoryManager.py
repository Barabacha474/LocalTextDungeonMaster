from typing import Optional
from GenerationUnit import GenerationUnit
from LastNTurnsPromptConstructor import LastNTurnsPromptConstructor
from SimpleOllamaLLM import SimpleOllamaLLM
from AdventureContext import AdventureContext


class MemoryManager:
    """
    Handles summarization logic:
    - short-term memory (last N turns)
    - global adventure summary

    Does NOT save anything to DB.
    Only generates and returns text.

    Now fully config-driven (generation params passed via arguments).
    """

    def __init__(
        self,
        model: str,
        max_context_length: int = 8192
    ):
        self.model = model
        self.max_context_length = max_context_length

        # =========================================================
        # PROMPT CONSTRUCTORS
        # =========================================================

        self.memory_prompt_constructor = LastNTurnsPromptConstructor(
            num_turns=6,
            roles=["Player", "Narrator"]
        )

        self.global_prompt_constructor = LastNTurnsPromptConstructor(
            num_turns=-1,
            roles=["Player", "Narrator"]
        )

        # =========================================================
        # LLMs
        # =========================================================

        self.memory_llm = SimpleOllamaLLM(
            model=self.model,
            prompt_constructor=self.memory_prompt_constructor
        )

        self.global_llm = SimpleOllamaLLM(
            model=self.model,
            prompt_constructor=self.global_prompt_constructor
        )

        # =========================================================
        # GENERATION UNITS
        # =========================================================

        self.memory_unit = GenerationUnit(
            self.memory_llm,
            name="MemorySummarizer"
        )

        self.global_unit = GenerationUnit(
            self.global_llm,
            name="GlobalSummarizer"
        )

        # =========================================================
        # INSTRUCTIONS
        # =========================================================

        self.memory_instruction = self._build_memory_instruction()
        self.global_instruction = self._build_global_instruction()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def generate_memory(
        self,
        context: AdventureContext,
        num_turns: int = 6,
        offset: int = 0,
        temperature: float = 0.2,
        seed: int = 42,
        num_predict: int = 800,
        num_ctx: Optional[int] = None,
        # 🔥 NEW
        auto_ctx: bool = False,
        ctx_margin: float = 1.1,
        max_ctx: Optional[int] = None,
        debug_info: bool = False,
        debug_prompt: bool = False
    ) -> str:
        """
        Generate short-term memory summary
        """

        ctx = num_ctx or self.max_context_length

        self.memory_prompt_constructor.set_special_command(
            self.memory_instruction
        )

        return self.memory_unit.generate(
            context=context,
            prompt_kwargs={
                "num_turns": num_turns,
                "offset": offset
            },
            generation_kwargs={
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": ctx,

                "auto_ctx": auto_ctx,
                "ctx_margin": ctx_margin,
                "max_ctx": max_ctx,

                "debug_info": debug_info,
                "debug_prompt": debug_prompt
            }
        )

    def generate_global_summary(
        self,
        context: AdventureContext,
        temperature: float = 0.2,
        seed: int = 42,
        num_predict: int = 1500,
        num_ctx: Optional[int] = None,
        # 🔥 NEW
        auto_ctx: bool = False,
        ctx_margin: float = 1.1,
        max_ctx: Optional[int] = None,
        debug_info: bool = False,
        debug_prompt: bool = False
    ) -> str:
        """
        Generate full adventure summary
        """

        ctx = num_ctx or self.max_context_length

        self.global_prompt_constructor.set_special_command(
            self.global_instruction
        )

        return self.global_unit.generate(
            context=context,
            generation_kwargs={
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": ctx,

                # 🔥 NEW
                "auto_ctx": auto_ctx,
                "ctx_margin": ctx_margin,
                "max_ctx": max_ctx,

                "debug_info": debug_info,
                "debug_prompt": debug_prompt
            }
        )

    # =========================================================
    # INSTRUCTIONS
    # =========================================================

    def _build_memory_instruction(self) -> str:
        return """You are an expert summarizer for fantasy adventure logs.

Create a detailed memory of recent events from the following adventure log.

IMPORTANT:
- Do NOT invent new facts
- Do NOT omit important details
- Maintain chronological order

Focus on:
1. Key events
2. Characters and their actions
3. Locations
4. Important discoveries
5. Player decisions
6. Dialogue

Rules:
- Write in narrative form (NOT bullet points)
- Refer to the player as "player"
- Preserve important context for future storytelling

Output should be rich but concise.
"""

    def _build_global_instruction(self) -> str:
        return """You are an expert narrative analyst for long-running fantasy adventures.

Create a comprehensive summary of the entire adventure so far.

IMPORTANT:
- Do NOT invent new facts
- Preserve all major story arcs
- Maintain chronological coherence

Focus on:
1. Main plot progression
2. Major events and turning points
3. Important characters and relationships
4. Key locations and discoveries
5. Conflicts and their outcomes
6. Player’s long-term impact on the world

Also include:
- Current state of the story
- Unresolved threads
- Ongoing tensions

Style:
- Clear, structured narrative
- Not too verbose, but information-dense
- No bullet points

This summary will be used as long-term memory for a Dungeon Master.
"""
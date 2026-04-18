from typing import Optional, Iterator, Dict
from Codes.AbstractClasses.AbstractOllamaLLM import AbstractOllamaLLM
from Codes.Orchestrators.AdventureContext import AdventureContext


class GenerationUnit:
    """
    Universal wrapper for LLM + PromptConstructor pair.

    Handles:
    - prompt building
    - generation
    - streaming
    - dynamic context estimation (NEW)
    - debug access (last prompt/result)
    """

    def __init__(self, llm: AbstractOllamaLLM, name: str):
        self.llm = llm
        self.name = name

        self._last_prompt: Optional[str] = None
        self._last_result: Optional[str] = None

    # =========================================================
    # PROMPT
    # =========================================================

    def build_prompt(self, context: AdventureContext = None, prompt_kwargs: Optional[Dict] = None) -> str:
        try:
            if prompt_kwargs is None:
                prompt_kwargs = {}

            prompt = self.llm.get_prompt(
                prompt=None,
                prompt_kwargs={"context": context, **prompt_kwargs}
            )

            self._last_prompt = prompt
            return prompt

        except Exception as e:
            print("\n[PROMPT BUILD ERROR]")
            print(f"Unit: {self.name}")
            print(f"Error: {e}")
            raise

    # =========================================================
    # CONTEXT ESTIMATION (NEW)
    # =========================================================

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation:
        1 token ≈ 4 characters (safe heuristic)
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _apply_dynamic_context(
        self,
        prompt: str,
        generation_kwargs: Dict
    ) -> Dict:
        """
        Dynamically estimate required context size and inject num_ctx.

        Logic:
        num_ctx ≈ (prompt_tokens + expected_output_tokens) * margin

        Does NOT override num_ctx if explicitly provided.
        """

        if "num_ctx" in generation_kwargs:
            return generation_kwargs

        auto_ctx = generation_kwargs.get("auto_ctx", False)

        if not auto_ctx:
            return generation_kwargs

        # -------------------------
        # ESTIMATE PROMPT TOKENS
        # -------------------------
        prompt_tokens = self._estimate_tokens(prompt)

        # -------------------------
        # OUTPUT RESERVE
        # -------------------------
        output_tokens = generation_kwargs.get("num_predict", 512)

        # -------------------------
        # SAFETY MARGIN
        # -------------------------
        margin = generation_kwargs.get("ctx_margin", 1.1)

        # -------------------------
        # FINAL CONTEXT SIZE
        # -------------------------
        estimated_ctx = int((prompt_tokens + output_tokens) * margin)

        # -------------------------
        # MAX LIMIT (optional)
        # -------------------------
        max_ctx = generation_kwargs.get("max_ctx")

        if max_ctx is not None:
            estimated_ctx = min(estimated_ctx, max_ctx)

        # -------------------------
        # DEBUG (optional)
        # -------------------------
        if generation_kwargs.get("debug_ctx", False):
            print(f"\n[CTX DEBUG] Unit: {self.name}")
            print(f"Prompt tokens (est): {prompt_tokens}")
            print(f"Output reserve: {output_tokens}")
            print(f"Final num_ctx: {estimated_ctx}")

        generation_kwargs = dict(generation_kwargs)
        generation_kwargs["num_ctx"] = estimated_ctx

        return generation_kwargs

    # =========================================================
    # GENERATION
    # =========================================================

    def generate(
        self,
        context: AdventureContext = None,
        prompt_kwargs: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None
    ) -> str:
        try:
            if generation_kwargs is None:
                generation_kwargs = {}

            prompt = self.build_prompt(
                context=context,
                prompt_kwargs=prompt_kwargs
            )

            # dynamic context
            generation_kwargs = self._apply_dynamic_context(
                prompt,
                generation_kwargs
            )

            # debug output
            self._debug_print(prompt, generation_kwargs)

            result = self.llm.generate(
                prompt=prompt,
                **generation_kwargs
            )

            self._last_result = result
            return result

        except Exception as e:
            print("\n[GENERATION ERROR]")
            print(f"Unit: {self.name}")
            print(f"Error: {e}")

            if self._last_prompt:
                print("\n--- LAST PROMPT ---")
                print(self._last_prompt[:1000])

            raise

    # =========================================================
    # STREAM
    # =========================================================

    def generate_stream(
        self,
        context: AdventureContext = None,
        prompt_kwargs: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None
    ) -> Iterator[str]:
        try:
            if generation_kwargs is None:
                generation_kwargs = {}

            prompt = self.build_prompt(
                context=context,
                prompt_kwargs=prompt_kwargs
            )

            # dynamic context
            generation_kwargs = self._apply_dynamic_context(
                prompt,
                generation_kwargs
            )

            # debug output
            self._debug_print(prompt, generation_kwargs)

            return self.llm.generate_stream(
                prompt=prompt,
                **generation_kwargs
            )

        except Exception as e:
            print("\n[GENERATION ERROR]")
            print(f"Unit: {self.name}")
            print(f"Error: {e}")

            if self._last_prompt:
                print("\n--- LAST PROMPT ---")
                print(self._last_prompt[:1000])

            raise

    # =========================================================
    # DEBUG
    # =========================================================

    def get_last_prompt(self) -> Optional[str]:
        return self._last_prompt

    def get_last_result(self) -> Optional[str]:
        return self._last_result

    def _debug_print(
            self,
            prompt: str,
            generation_kwargs: Dict
    ):
        """
        Debug output for prompt and model info.

        Controlled by:
        - debug_info
        - debug_prompt
        """

        debug_info = generation_kwargs.get("debug_info", False)
        debug_prompt = generation_kwargs.get("debug_prompt", False)

        if not debug_info and not debug_prompt:
            return

        model_name = getattr(self.llm, "model", "UNKNOWN_MODEL")
        prompt_tokens = self._estimate_tokens(prompt)
        num_ctx = generation_kwargs.get("num_ctx", "N/A")

        if debug_info:
            print("\n[GEN DEBUG INFO]")
            print(f"Unit: {self.name}")
            print(f"Model: {model_name}")
            print(f"Estimated prompt tokens: {prompt_tokens}")
            print(f"num_ctx: {num_ctx}")

        if debug_prompt:
            print("\n[GEN DEBUG PROMPT - FULL]")
            print(prompt)
            print("\n[END OF PROMPT]\n")
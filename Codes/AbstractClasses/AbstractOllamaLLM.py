from abc import ABC, abstractmethod
from typing import Optional, Iterator
import ollama
from Codes.AbstractClasses.AbstractPromptConstructor import AbstractPromptConstructor


class AbstractOllamaLLM(ABC):
    """Abstract base class for Ollama LLM generation with prompt generation support."""

    def __init__(
            self,
            model: str,
            default_prompt: Optional[str] = None,
            prompt_constructor: Optional[AbstractPromptConstructor] = None
    ):
        """
        Initialize the Ollama LLM.

        Args:
            model: Ollama model name (e.g., "llama2", "mistral")
            default_prompt: Default prompt to use if no prompt generator provided
            prompt_constructor: Optional PromptGenerator instance for dynamic prompt creation
        """
        self.model = model
        self.default_prompt = default_prompt
        self.prompt_constructor = prompt_constructor

        if not default_prompt and not prompt_constructor:
            raise ValueError("Either default_prompt or prompt_constructor must be provided")

    @abstractmethod
    def generate(
            self,
            prompt: Optional[str] = None,
            prompt_kwargs: Optional[dict] = None,
            **generation_kwargs
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Optional explicit prompt string. If None, uses prompt_constructor or default_prompt
            prompt_kwargs: Keyword arguments to pass to prompt_constructor if using one
            **generation_kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            str: Generated response

        Raises:
            ValueError: If no prompt can be generated
            RuntimeError: If Ollama API request fails
        """
        pass

    @abstractmethod
    def generate_stream(
            self,
            prompt: Optional[str] = None,
            prompt_kwargs: Optional[dict] = None,
            **generation_kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response from the model.

        Args:
            prompt: Optional explicit prompt string. If None, uses prompt_constructor or default_prompt
            prompt_kwargs: Keyword arguments to pass to prompt_constructor if using one
            **generation_kwargs: Additional generation parameters (temperature, top_p, etc.)

        Yields:
            str: Stream of response tokens

        Raises:
            ValueError: If no prompt can be generated
            RuntimeError: If Ollama API request fails
        """
        pass

    def get_prompt(self, prompt=None, prompt_kwargs=None) -> str:
        if prompt:
            return prompt

        if not self.prompt_constructor:
            raise ValueError("No prompt or prompt_constructor provided.")

        if not prompt_kwargs:
            prompt_kwargs = {}

        return self.prompt_constructor.construct_prompt(**prompt_kwargs)

    def _make_generate_request(
            self,
            prompt: str,
            stream: bool = False,
            keep_alive: str | int = "5m",
            **generation_kwargs
    ):
        """
        Sends request to Ollama with proper argument mapping.
        Filters out unsupported custom arguments.
        """

        # =========================================================
        # SPLIT ARGS
        # =========================================================

        options = {}

        allowed_options = {
            "temperature",
            "seed",
            "num_predict",
            "num_ctx",
            "top_k",
            "top_p",
            "repeat_penalty",
            "stop"
        }

        allowed_direct = {
            "format",
            "raw",
            "system",
            "template"
        }

        internal_keys = {
            "auto_ctx",
            "ctx_margin",
            "max_ctx",
            "debug_ctx",
            "debug_info",
            "debug_prompt"
        }

        clean_kwargs = {}

        for key in list(generation_kwargs.keys()):
            value = generation_kwargs[key]

            if key in allowed_options:
                options[key] = value

            elif key in allowed_direct:
                clean_kwargs[key] = value

            elif key in internal_keys:
                continue

            else:
                # print(f"[WARN] Unknown generation arg: {key}")
                continue

        # =========================================================
        # REQUEST
        # =========================================================

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=stream,
                keep_alive=keep_alive,
                options=options,
                **clean_kwargs
            )

            return response

        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def _parse_stream_response(self, stream_generator: Iterator[dict]) -> Iterator[str]:
        """
        Parse streaming response from ollama library.

        Args:
            stream_generator: Iterator of response dictionaries from ollama.generate(stream=True)

        Yields:
            str: Token from the response
        """
        for chunk in stream_generator:
            if chunk.get("done", False):
                break
            if "response" in chunk:
                yield chunk["response"]

    def set_new_prompt_constructor(self, prompt_constructor: AbstractPromptConstructor):
        """
        Sets new prompt constructor to existing LLM generator

        Args:
            prompt_constructor: new prompt constructor
        """
        self.prompt_constructor = prompt_constructor

    def set_new_model_name(self, model: str):
        """
        Sets new model to utilise new LLM

        Args:
            model: new model name
        """
        self.model = model

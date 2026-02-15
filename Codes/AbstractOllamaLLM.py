from abc import ABC, abstractmethod
from typing import Optional, Iterator, Any
import ollama
from AbstractPromptConstructor import AbstractPromptConstructor


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

    def _get_prompt(
            self,
            prompt: Optional[str],
            prompt_kwargs: Optional[dict] = None
    ) -> str:
        """
        Get the prompt to use for generation.

        Args:
            prompt: Optional explicit prompt string
            prompt_kwargs: Keyword arguments for prompt generator

        Returns:
            str: The prompt to use

        Raises:
            ValueError: If no prompt can be determined
        """
        if prompt is not None:
            return prompt

        if self.prompt_constructor and prompt_kwargs is not None:
            return self.prompt_constructor.generate_prompt(**prompt_kwargs)

        if self.default_prompt is not None:
            return self.default_prompt

        if self.prompt_constructor:
            # Try with empty kwargs if prompt_constructor can handle it
            return self.prompt_constructor.generate_prompt()

        raise ValueError(
            "No prompt provided and no default_prompt or valid prompt_constructor available"
        )

    def _make_generate_request(
            self,
            prompt: str,
            stream: bool = False,
            **generation_kwargs
    ) -> Any:
        """
        Make a generate request using the ollama library.
        This is a helper method that concrete implementations can use.

        Args:
            prompt: The prompt to send
            stream: Whether to stream the response
            **generation_kwargs: Additional generation parameters

        Returns:
            For non-streaming: dict response from ollama
            For streaming: generator yielding response chunks

        Raises:
            RuntimeError: If Ollama request fails
        """
        try:
            if stream:
                # For streaming, return the generator directly
                return ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=stream,
                    **generation_kwargs
                )
            else:
                # For non-streaming, return the complete response
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=stream,
                    **generation_kwargs
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

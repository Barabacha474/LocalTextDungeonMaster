from abc import ABC, abstractmethod
class AbstractPromptConstructor(ABC):
    """Abstract base class for generating prompts from various input types."""

    @abstractmethod
    def construct_prompt(self, **kwargs) -> str:
        """
        Generate a prompt string based on input arguments.

        Args:
            **kwargs: Arbitrary keyword arguments that vary by implementation

        Returns:
            str: The constructed prompt string

        Note:
            Concrete implementations should handle very different argument types.
            Examples:
            - For text generation: {"text": "Hello", "context": "..."}
            - For image description: {"image_path": "path/to/image.jpg"}
            - For code generation: {"language": "python", "task": "sort list"}
        """
        pass
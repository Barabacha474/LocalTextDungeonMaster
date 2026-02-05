from typing import Iterator
from AbstractOllamaLLM import AbstractOllamaLLM
from LastNTurnsPromptConstructor import LastNTurnsPromptConstructor
import time


class SummarizerOllamaLLM(AbstractOllamaLLM):
    """
    Ollama LLM specifically for summarization tasks.
    Uses LastNTurnsPromptConstructor to generate prompts for summarization.
    """

    def __init__(
            self,
            model: str,
            prompt_constructor: LastNTurnsPromptConstructor
    ):
        """
        Initialize the Summarizer Ollama LLM.

        Args:
            model: Ollama model name for summarization
            prompt_constructor: LastNTurnsPromptConstructor instance for generating prompts
        """
        # Initialize parent class with the prompt constructor
        super().__init__(
            model=model,
            prompt_constructor=prompt_constructor
        )

        self.prompt_constructor = prompt_constructor
        self.prompt_constructor.set_special_command(SummarizerOllamaLLM._get_default_summarization_prompt())

    def generate(
            self,
            num_turns: int = None,
            desired_response_size: int = 100,
            temperature: float = 0.2,
            seed: int = 42
    ) -> str:
        """
        Generate a summary of recent turns.

        Args:
            num_turns: Number of turns to summarize (overrides prompt constructor default)
            desired_response_size: Target word count for summary
            temperature: Generation temperature
            seed: Random seed for consistent generation

        Returns:
            str: Generated summary
        """
        # Generate prompt using the prompt constructor with optional num_turns override
        prompt = self.prompt_constructor.construct_prompt(
            player_input="[Generate summary]",
            num_turns=num_turns
        )

        # Add summarization-specific options
        options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": desired_response_size * 2  # Rough estimate
        }

        # Make the request
        response = self._make_generate_request(
            prompt=prompt,
            stream=False,
            options=options
        )

        # Extract and clean the response
        summary = response.get("response", "").strip()
        summary = self._clean_summary(summary)

        return summary

    def generate_stream(
            self,
            num_turns: int = None,
            desired_response_size: int = 100,
            temperature: float = 0.2,
            seed: int = 42
    ) -> Iterator[str]:
        """
        Generate a streaming summary of recent turns.

        Args:
            num_turns: Number of turns to summarize (overrides prompt constructor default)
            desired_response_size: Target word count for summary
            temperature: Generation temperature
            seed: Random seed for consistent generation

        Yields:
            str: Stream of summary tokens
        """
        # Generate prompt using the prompt constructor with optional num_turns override
        prompt = self.prompt_constructor.construct_prompt(
            player_input="[Generate summary]",
            num_turns=num_turns
        )

        # Add summarization-specific options
        options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": desired_response_size * 2  # Rough estimate
        }

        # Make streaming request
        stream = self._make_generate_request(
            prompt=prompt,
            stream=True,
            options=options
        )

        # Parse and yield stream
        for token in self._parse_stream_response(stream):
            yield token

    def _clean_summary(self, text: str) -> str:
        """
        Clean up common formatting artifacts from the summary.
        Same as original Summarizer._clean_summary method.

        Args:
            text: Raw summary text

        Returns:
            str: Cleaned summary
        """
        # Remove surrounding quotes
        text = text.strip('"\'`')

        # Remove markdown code blocks
        if text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        elif text.startswith('```'):
            text = text[3:].strip()
        elif text.endswith('```'):
            text = text[:-3].strip()

        # Remove redundant labels like "Summary:" at start
        if text.lower().startswith('summary:'):
            text = text[8:].strip()

        # Remove extra whitespace and normalize
        text = ' '.join(text.split())

        return text

    @staticmethod
    def _get_default_summarization_prompt() -> str:
        """
        Get the default summarization prompt (from original Summarizer).

        Returns:
            str: Default summarization prompt
        """
        return """You are an expert summarizer for fantasy adventure logs. 
    Create a summary of about 100 words from the following adventure log (Number of words is important!). Create as memory of 
    adventure flow for Dungeon Master. DM's response marked with 'DM:', player actions and speech marked with 
    'Player:', player speech is in "", player commands for DM is in ## - ignore them, player actions in plain text. There is only one player. 

    Focus on:
    1. Key events and plot developments
    2. Important characters and their actions
    3. Locations visited and discoveries
    4. Relationships and conflicts
    5. Major decisions made
    6. Dialogs

    DO NOT GENERATE NEW FACTS OR INFORMATION, ONLY SUM UP MENTIONED INFORMATION!
    Mention player as 'player', not 'you'.

    Write in a concise, narrative style. Maintain chronological order.

    Adventure log:"""


# Example usage
if __name__ == "__main__":
    from AdventureLogger import AdventureLogger

    # Create AdventureLogger
    logger = AdventureLogger()

    prompt_constructor = LastNTurnsPromptConstructor(
        adventure_logger=logger,
        num_turns=4
    )

    # Create Summarizer LLM
    summarizer = SummarizerOllamaLLM(
        model="mistral:7b-instruct-v0.3-q4_0",
        prompt_constructor=prompt_constructor
    )

    # prompt print
    print(prompt_constructor.construct_prompt())

    # Start timer
    start_time = time.time()

    # Stream summary
    print("\n\nStreaming summary:")
    print("=" * 50)
    for token in summarizer.generate_stream(
            desired_response_size=150,
            temperature=0.3,
            seed=42
    ):
        print(token, end="", flush=True)
    print("\n" + "=" * 50)

    # End timer
    end_time = time.time()
    generation_time = end_time - start_time

    # Print statistics
    print(f"Generation time: {generation_time:.2f} seconds")

    # Clean up
    logger.close()
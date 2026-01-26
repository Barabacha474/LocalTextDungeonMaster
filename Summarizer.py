import ollama
from typing import Optional


class Summarizer:
    """
    Summarizer for adventure logs using Ollama models.
    Simple interface for text summarization.
    """

    def __init__(self,
                 # default_model: str = "llama3.2:1b"
                 # default_model: str = "gemma2:2b"
                 default_model: str = "phi3:mini"
                 # default_model: str = "deepseek-r1:7b"
                 ):
        """
        Initialize the summarizer with default model.

        Args:
            default_model (str): Default model name for summarization
        """
        self.default_model = default_model

        # Test connection to Ollama
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama. Is it running? Error: {e}")

    def summarize(
            self,
            input_text: str,
            summary_model_name: Optional[str] = None,
            desired_response_size: int = 100
    ) -> str:
        """
        Generate a summary of the input text.

        Args:
            input_text (str): Text to summarize
            summary_model_name (str, optional): Model to use for summarization.
                If None, uses default_model.
            desired_response_size (int): Target approximate word count for summary.
                Default: 100 words

        Returns:
            str: Generated summary
        """
        model = summary_model_name or self.default_model

        # Validate input
        if desired_response_size < 10:
            raise ValueError("desired_response_size must be at least 10 words")

        # Construct prompt for summarization
        prompt = f"""You are an expert summarizer for fantasy adventure logs. 
Create a summary of about {desired_response_size} words from the following adventure log.

Focus on:
1. Key events and plot developments
2. Important characters and their actions
3. Locations visited and discoveries
4. Relationships and conflicts
5. Major decisions made
6. Dialogs

Write in a concise, narrative style. Maintain chronological order.

Adventure log:
{input_text}

Summary:"""

        try:
            # Calculate max tokens based on word count (rough estimate)
            max_tokens = desired_response_size * 2

            # Call Ollama
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.2,  # Low temperature for consistent summaries
                    "num_predict": max_tokens,
                    "stop": ["\n\n", "##", "---"]  # Stop sequences to prevent run-on
                }
            )

            summary = response['response'].strip()

            # Clean up common formatting artifacts
            summary = self._clean_summary(summary)

            return summary

        except Exception as e:
            raise RuntimeError(f"Failed to generate summary with model {model}: {e}")

    def _clean_summary(self, text: str) -> str:
        """Clean up common formatting artifacts from the summary."""
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

    def list_available_models(self) -> list:
        """List available Ollama models."""
        try:
            models_response = ollama.list()
            models = models_response.get('models', [])
            return [model['name'] for model in models]
        except Exception:
            return []
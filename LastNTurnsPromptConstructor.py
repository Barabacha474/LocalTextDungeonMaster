from typing import Optional, Any, Dict, List
from AbstractPromptConstructor import AbstractPromptConstructor
from AdventureLogger import AdventureLogger


class LastNTurnsPromptConstructor(AbstractPromptConstructor):
    """
    Prompt constructor that summarizes last N turns from AdventureLogger
    and combines with special command and player input.
    """

    def __init__(
            self,
            adventure_logger: AdventureLogger,
            num_turns: int = 4,
            special_command: Optional[str] = None
    ):
        """
        Initialize the LastNTurnsPromptConstructor.

        Args:
            adventure_logger: Instance of AdventureLogger object
            num_turns: Number of recent turns to include in summary (default: 4)
            special_command: Default command to prepend to the prompt
        """
        self.special_command = special_command
        self.adventure_logger = adventure_logger
        self.num_turns = num_turns

        if num_turns <= 0:
            raise ValueError("num_turns must be positive")

    def construct_prompt(self, player_input: Optional[str] = None, num_turns: Optional[int] = None) -> str:
        """
        Generate a prompt combining special command, last N turns, and player input.

        Expected kwargs:
            player_input (str): The current player's input/action
            special_command (str, optional): Special instruction/command.
                If not provided, uses default_special_command
            num_turns_override (int, optional): Override for number of turns to include

        Returns:
            str: Formatted prompt string
        """

        # Get last N turns from AdventureLogger object
        last_turns = self._get_last_n_turns(num_turns or self.num_turns)

        # Format the prompt
        prompt = self._format_prompt(
            last_turns=last_turns,
            player_input=player_input
        )

        return prompt

    def _get_last_n_turns(self, num_turns: int) -> List[Dict[str, Any]]:
        """
        Get the last N turns from AdventureLogger object.

        Args:
            num_turns: Number of turns to retrieve

        Returns:
            List of turn dictionaries from AdventureLogger
        """
        try:
            # Call the AdventureLogger object's method
            return self.adventure_logger.get_last_n_turns(num_turns)
        except AttributeError as e:
            raise AttributeError(
                f"AdventureLogger object missing required method: {e}. "
                f"Expected object with 'get_last_n_turns' method."
            )
        except Exception as e:
            # Log the error and return empty list
            print(f"Error getting last {num_turns} turns from AdventureLogger: {e}")
            return []

    def _format_prompt(
            self,
            last_turns: List[Dict[str, Any]],
            player_input: Optional[str]
    ) -> str:
        """
        Format the prompt with special command, last turns, and player input.

        Args:
            last_turns: List of last N turns from AdventureLogger
            player_input: Current player input

        Returns:
            Formatted prompt string
        """
        # Start with special command
        prompt_parts = []
        if self.special_command:
            prompt_parts.append(self.special_command + "\n")

        # Add last turns section if we have any
        if last_turns:
            for turn in last_turns:
                role = turn.get('role', 'Unknown')
                content = turn.get('content', '')
                # Format as "Role: Content"
                prompt_parts.append(f"{role}: {content}")

        # Add player input
        if player_input:
            prompt_parts.append(f"\n\nPlayer: {player_input}")

        return "\n".join(prompt_parts)

    def get_adventure_logger(self):
        """
        Get the AdventureLogger object instance.

        Returns:
            AdventureLogger object
        """
        return self.adventure_logger

    def set_special_command(self, special_command: str):
        self.special_command = special_command


if __name__ == "__main__":
    sql_db = AdventureLogger()
    constructor = LastNTurnsPromptConstructor(adventure_logger=sql_db,special_command="SPECIAL COMMAND")
    print(constructor.construct_prompt("player input"))
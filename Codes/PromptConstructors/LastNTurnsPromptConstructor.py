from typing import Optional, Any, Dict, List
from Codes.AbstractClasses.AbstractPromptConstructor import AbstractPromptConstructor
from Codes.Orchestrators.AdventureContext import AdventureContext

class LastNTurnsPromptConstructor(AbstractPromptConstructor):
    """
    Prompt constructor that uses AdventureContext
    to fetch last N turns and build a prompt.
    """

    def __init__(
        self,
        num_turns: int = 4,
        special_command: Optional[str] = None,
        roles: Optional[List[str]] = None
    ):
        """
        Args:
            num_turns:
                > 0 → last N turns
                < 0 → ALL turns
            special_command: optional system instruction
            roles: filter roles (e.g. ["Player", "Narrator", "PlannerDjn "])
        """
        self.special_command = special_command
        self.num_turns = num_turns
        self.roles = roles

        if num_turns == 0:
            raise ValueError("num_turns cannot be 0")

    # =========================================================
    # MAIN
    # =========================================================

    def construct_prompt(
        self,
        context: AdventureContext,
        player_input: Optional[str] = None,
        num_turns: Optional[int] = None,
        offset: int = 0
    ) -> str:
        turns = self._get_last_n_turns(
            context=context,
            num_turns=num_turns or self.num_turns,
            offset=offset
        )

        return self._format_prompt(
            last_turns=turns,
            player_input=player_input or context.get_player_input()
        )

    # =========================================================
    # INTERNAL
    # =========================================================

    def _get_last_n_turns(
            self,
            context: AdventureContext,
            num_turns: int,
            offset: int = 0
    ) -> List[Dict[str, Any]]:
        try:
            turn_count = context.get_turn_count()

            if turn_count == 0:
                return []

            # =========================================================
            # ALL TURNS MODE (num_turns < 0)
            # =========================================================
            if num_turns < 0:
                end = turn_count - offset
                start = 1

                if end < 1:
                    return []

                return context.get_turns_range(
                    start=start,
                    end=end,
                    roles=self.roles
                )

            # =========================================================
            # WINDOW MODE (N with offset M)
            # =========================================================
            end = turn_count - offset
            start = end - num_turns + 1

            if end < 1:
                return []

            if start < 1:
                start = 1

            return context.get_turns_range(
                start=start,
                end=end,
                roles=self.roles
            )

        except Exception as e:
            print(f"Error getting turns with offset: {e}")
            return []

    def _format_prompt(
        self,
        last_turns: List[Dict[str, Any]],
        player_input: Optional[str]
    ) -> str:
        parts = []

        # Special command
        if self.special_command:
            parts.append(self.special_command + "\n")

        # Turns
        for turn in last_turns:
            role = turn.get("role", "Unknown")
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")

        # Player input
        if player_input:
            parts.append(f"\n\nPlayer: {player_input}")

        return "\n".join(parts)

    # =========================================================
    # SETTERS
    # =========================================================

    def set_special_command(self, special_command: str):
        self.special_command = special_command

    def set_roles(self, roles: List[str]):
        self.roles = roles
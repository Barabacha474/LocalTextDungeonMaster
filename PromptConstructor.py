import json
import os
from SettingDataLoaders.FaissVectorDB import FAISSVectorDB
from AdventureLogger import AdventureLogger
from typing import Optional, List, Dict, Any


class PromptConstructor:
    def __init__(self, adventure_logger, vector_db, config_path="./SettingRawDataJSON/vanilla_fantasy/PromptCore.json"):
        """
        Initialize PromptConstructor with database instances and configuration.

        Args:
            adventure_logger (AdventureLogger): SQL logger instance
            vector_db (FAISSVectorDB): Vector database instance
            config_path (str): Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = None
        self.adventure_logger = adventure_logger
        self.vector_db = vector_db
        self.load_config()
        self.has_logged_opening_text = False

    def load_config(self, config_path=None):
        """
        Load configuration from JSON file.

        Args:
            config_path (str, optional): Path to JSON configuration file.
                                        If None, uses the path from initialization.
        """
        path = config_path or self.config_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")

    def _combine_with_context(self, user_input: str, last_n_turns: int = 5) -> str:
        """
        Combine user input with recent conversation history for better context.

        Args:
            user_input (str): Current user input/query
            last_n_turns (int): Number of recent turns to include

        Returns:
            str: Combined query string with context
        """
        if not user_input:
            return ""

        # Get recent conversation from SQL database
        recent_conversation = ""
        if last_n_turns > 0 and self.adventure_logger:
            try:
                turns = self.adventure_logger.get_last_n_turns(last_n_turns)
                if turns:
                    # Format conversation context
                    context_parts = []
                    for turn in turns[-last_n_turns:]:  # Ensure we only get requested number
                        role = turn.get('role', 'Unknown')
                        content = turn.get('content', '')
                        context_parts.append(f"{role}: {content}")

                    # Join with newlines
                    recent_conversation = "\n".join(context_parts)
            except Exception as e:
                print(f"Error retrieving conversation history: {e}")

        # Combine with user input
        if recent_conversation:
            return f"Recent conversation:\n{recent_conversation}\nCurrent query: {user_input}"
        else:
            return user_input

    def _search_relevant_information(self, user_input: str, last_n_turns: int = 5,
                                     k: int = 3, threshold: float = 0.1,
                                     chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant information in vector database with context from recent conversation.

        Args:
            user_input (str): User's input/query
            last_n_turns (int): Number of recent turns to include for context
            k (int): Number of relevant items to retrieve
            threshold (float): Minimum similarity threshold
            chunk_size (Optional[int]): Chunk size for vector search

        Returns:
            list: Relevant information items with metadata
        """
        try:
            # Combine user input with recent context
            search_query = self._combine_with_context(user_input, last_n_turns)

            # Perform the vector search
            if search_query:
                results = self.vector_db.search(
                    query=search_query,
                    k=k,
                    threshold=threshold,
                    chunk_size=chunk_size
                )
                return results
            else:
                return []

        except Exception as e:
            print(f"Error searching vector DB: {e}")
            return []

    def _get_formatted_conversation(self, n: int = 5) -> str:
        """
        Get formatted conversation string for inclusion in prompt.

        Args:
            n (int): Number of turns to format

        Returns:
            str: Formatted conversation string
        """
        try:
            turns = self.adventure_logger.get_last_n_turns(n)
            if not turns:
                return ""

            formatted = []
            for turn in turns:
                role = "Player" if turn.get('role') == 'Player' else "Dungeon Master"
                content = turn.get('content', '')
                formatted.append(f"\n{role}: {content}")

            return "".join(formatted)
        except Exception as e:
            print(f"Error formatting conversation: {e}")
            return ""

    def get_prompt(self, user_input: str = "", is_initial_prompt: bool = True,
                   n_last_turns: int = 5, vector_search_k: int = 3,
                   vector_threshold: float = 0.1, chunk_size: Optional[int] = None) -> str:
        """
        Construct a SINGLE prompt string for use with ollama.generate.
        Now includes relevant information and conversation history.

        Args:
            user_input (str): The player's current input/action
            is_initial_prompt (bool): Whether this is the first prompt of the session
            n_last_turns (int): Number of previous turns to include in prompt and vector search
            vector_search_k (int): Number of relevant vector DB items to retrieve
            vector_threshold (float): Minimum similarity threshold for vector search
            chunk_size (Optional[int]): Chunk size for vector search

        Returns:
            str: A single prompt string formatted for ollama.generate
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        # Start with role and instructions
        prompt_parts = []
        prompt_parts.append(f"{self.config.get('role', '')}\n\n{self.config.get('instructions', '')}")

        # Always include setting
        setting_text = self.config.get('setting', '')
        if setting_text:
            prompt_parts.append(f"WORLD SETTING:\n{setting_text}")

        # Include plot_start only on initial prompt
        if is_initial_prompt:
            plot_start_text = self.config.get('plot_start', '')
            if plot_start_text:
                prompt_parts.append(f"PLOT START:\n{plot_start_text}")

            player_opening_text = self.config.get('player_opening_text', '')
            if player_opening_text:
                prompt_parts.append(f"PLAYER OPENING TEXT:\n{player_opening_text}")

                # Log the player opening text to SQL database if not already logged
                if not self.has_logged_opening_text and self.adventure_logger:
                    try:
                        self.adventure_logger.write('System', player_opening_text)
                        self.has_logged_opening_text = True
                        print(f"Logged player opening text to SQL database: {player_opening_text[:50]}...")
                    except Exception as e:
                        print(f"Error logging player opening text to SQL database: {e}")

        # Add relevant information from vector DB (with context-aware search)
        if user_input and self.vector_db:
            relevant_info_results = self._search_relevant_information(
                user_input=user_input,
                last_n_turns=n_last_turns,  # Use same number of turns for context
                k=vector_search_k,
                threshold=vector_threshold,
                chunk_size=chunk_size
            )

            if relevant_info_results:
                # Format results for prompt
                relevant_info = []
                for i, result in enumerate(relevant_info_results, 1):
                    info_text = result.get('text', '')
                    metadata = result.get('metadata', {})

                    # Add metadata context
                    if metadata:
                        context_parts = []
                        if 'type' in metadata:
                            context_parts.append(f"Type: {metadata['type']}")
                        if 'location' in metadata:
                            context_parts.append(f"Location: {metadata['location']}")
                        if 'character' in metadata:
                            context_parts.append(f"Character: {metadata['character']}")

                        if context_parts:
                            info_text = f"[{', '.join(context_parts)}]\n{info_text}"

                    relevant_info.append(f"{i}. {info_text}")

                if relevant_info:
                    prompt_parts.append(f"RELEVANT INFORMATION:\n" + "\n".join(relevant_info))

        # Add conversation history from SQL DB (formatted for the prompt)
        if n_last_turns > 0 and self.adventure_logger:
            formatted_conversation = self._get_formatted_conversation(n_last_turns)
            if formatted_conversation and not is_initial_prompt:
                prompt_parts.append(f"RECENT CONVERSATION:{formatted_conversation}")

        # Add the current user input and prompt for DM response
        if user_input:
            prompt_parts.append(f"PLAYER INPUT: {user_input}")

        # Add continue message if no user input (for continuing the story)
        if not user_input and self.config.get('continue_message'):
            prompt_parts.append(f"CONTINUE MESSAGE: {self.config.get('continue_message')}")

        prompt_parts.append("Dungeon Master response:")

        # Join all parts
        prompt = "\n\n".join(prompt_parts)

        return prompt.strip()

    def get_setting_info(self, include_plot_start: bool = False) -> Dict[str, str]:
        """
        Get all information except for plot_start (or include it if specified).

        Args:
            include_plot_start (bool): Whether to include plot_start information

        Returns:
            dict: Dictionary with role, instructions, setting, and optionally plot_start
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        result = {
            'role': self.config.get('role', ''),
            'instructions': self.config.get('instructions', ''),
            'setting': self.config.get('setting', '')
        }

        if include_plot_start:
            result['plot_start'] = self.config.get('plot_start', '')

        return result

    def get_system_prompt_only(self) -> str:
        """
        Get just the system prompt portion (role + instructions).

        Returns:
            str: The system prompt
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        return f"{self.config.get('role', '')}\n\n{self.config.get('instructions', '')}".strip()

    def get_setting_summary(self) -> str:
        """
        Get a summary of the setting for display purposes.

        Returns:
            str: Formatted setting summary (first 500 chars)
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        setting_text = self.config.get('setting', '')
        # Extract just the world name and first few lines
        lines = setting_text.split('\n')
        summary = lines[0] if lines else "No setting available"

        # Add a bit more context if available
        for i, line in enumerate(lines):
            if i > 0 and line.strip():
                summary += f"\n{line}"

        return summary.strip()

    def get_player_opening_text(self) -> str:
        """
        Get the player opening text from the JSON configuration.

        Returns:
            str: The player opening text, or empty string if not found
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        return self.config.get('player_opening_text', '')

    def get_continue_message(self) -> str:
        """
        Get the text for LLM to continue story from where it ends

        Returns:
            str: continue message
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        return self.config.get('continue_message', '')

    def update_vector_db(self, text: str, metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Helper method to add information to the vector database.

        Args:
            text (str): Text to add to vector DB
            metadata (dict, optional): Metadata for the text

        Returns:
            int: Document ID if successful, None otherwise
        """
        try:
            if self.vector_db:
                return self.vector_db.insert_single(text, metadata)
        except Exception as e:
            print(f"Error updating vector DB: {e}")
            return None

    def log_turn(self, role: str, content: str) -> Optional[int]:
        """
        Helper method to log a turn to the adventure logger.

        Args:
            role (str): 'Player' or 'System' (use 'System' for DM responses)
            content (str): Content of the turn

        Returns:
            int: Turn ID if successful, None otherwise
        """
        try:
            if self.adventure_logger:
                # Map 'System' to 'System' in AdventureLogger terms
                logger_role = 'System' if role == 'System' else 'Player'
                return self.adventure_logger.write(logger_role, content)
        except Exception as e:
            print(f"Error logging turn: {e}")
            return None


if __name__ == "__main__":
    # Test the simplified PromptConstructor
    vector_db = FAISSVectorDB("vanilla_fantasy", "D:/Work/PROJECTS/DEEPSEEK_Local/pythonProject1/adventure_memories")
    sql_db = AdventureLogger()
    prompt_constructor = PromptConstructor(sql_db, vector_db)

    print("=== Testing Combined Search ===")
    prompt = prompt_constructor.get_prompt(
        "Do I see elves?",
        is_initial_prompt=False,
        n_last_turns=3
    )
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
import json
import os
from SettingDataLoaders.FaissVectorDB import FAISSVectorDB
from AdventureLogger import AdventureLogger

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

    def _search_relevant_information(self, user_input, k=3, threshold=0.5):
        """
        Search for relevant information in vector database.

        Args:
            user_input (str): User's input/query
            k (int): Number of relevant items to retrieve
            threshold (float): Minimum similarity threshold

        Returns:
            list: Relevant information items
        """
        try:
            results = self.vector_db.search(
                query=user_input,
                k=k,
                threshold=threshold
            )

            # Format results for prompt
            relevant_info = []
            for i, result in enumerate(results, 1):
                info_text = result['text']
                # Add context if metadata available
                metadata = result.get('metadata', {})
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

            return relevant_info

        except Exception as e:
            print(f"Error searching vector DB: {e}")
            return []

    def _get_last_n_turns(self, n=5):
        """
        Get last N turns from SQL database.

        Args:
            n (int): Number of turns to retrieve

        Returns:
            list: Last N turns formatted as conversation
        """
        try:
            turns = self.adventure_logger.get_last_n_turns(n)

            # Format conversation from turns
            conversation = []
            for turn in turns:
                role = "Player" if turn['role'] == 'Player' else "Dungeon Master"
                conversation.append(f"{role}: {turn['content']}")

            return conversation

        except Exception as e:
            print(f"Error retrieving last turns: {e}")
            return []

    def get_prompt(self, user_input="", is_initial_prompt=True, n_last_turns=5, vector_search_k=3,
                   vector_threshold=0.5):
        """
        Construct a SINGLE prompt string for use with ollama.generate.
        Now includes relevant information and conversation history.

        Args:
            user_input (str): The player's current input/action
            is_initial_prompt (bool): Whether this is the first prompt of the session
            n_last_turns (int): Number of previous turns to include
            vector_search_k (int): Number of relevant vector DB items to retrieve
            vector_threshold (float): Minimum similarity threshold for vector search

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
                prompt_parts.append(f"ADVENTURE START:\n{plot_start_text}")

        # Add relevant information from vector DB
        if user_input and self.vector_db:
            relevant_info = self._search_relevant_information(
                user_input=user_input,
                k=vector_search_k,
                threshold=vector_threshold
            )
            if relevant_info:
                prompt_parts.append(f"RELEVANT INFORMATION:\n" + "\n".join(relevant_info))

        # Add conversation history from SQL DB
        if n_last_turns > 0 and self.adventure_logger:
            last_turns = self._get_last_n_turns(n_last_turns)
            if last_turns:
                # Skip the current user input if it's already logged
                # We'll add it explicitly later
                current_user_turn = f"Player: {user_input}"
                if current_user_turn in last_turns:
                    last_turns.remove(current_user_turn)

                if last_turns:
                    prompt_parts.append(f"RECENT CONVERSATION:\n" + "\n".join(last_turns))

        # Add the current user input and prompt for DM response
        prompt_parts.append(f"PLAYER INPUT: {user_input}")
        prompt_parts.append("Dungeon Master response:")

        # Join all parts
        prompt = "\n\n".join(prompt_parts)

        return prompt.strip()

    def get_setting_info(self, include_plot_start=False):
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

    def get_system_prompt_only(self):
        """
        Get just the system prompt portion (role + instructions).

        Returns:
            str: The system prompt
        """
        if not self.config:
            raise ValueError("No configuration loaded.")

        return f"{self.config.get('role', '')}\n\n{self.config.get('instructions', '')}".strip()

    def get_setting_summary(self):
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

    def update_vector_db(self, text, metadata=None):
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

    def log_turn(self, role, content):
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
    vector_db = FAISSVectorDB("vanilla_fantasy", "D:/Work/PROJECTS/DEEPSEEK_Local/pythonProject1/adventure_memories")
    sql_db = AdventureLogger()
    prompt_constructor = PromptConstructor(sql_db, vector_db)
    print(prompt_constructor.get_prompt("Do I see elfs?"))
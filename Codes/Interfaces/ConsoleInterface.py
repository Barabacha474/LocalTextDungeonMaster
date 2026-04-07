import time
import traceback
from typing import Optional, List, Dict

from Codes.Orchestrators.AdventureEngine import AdventureEngine
from Codes.Orchestrators.AdventureContext import AdventureContext


class ConsoleInterface:
    """
    CLI interface for interacting with AdventureEngine.

    Responsibilities:
    - read user input
    - pass it to engine
    - display output

    No game logic here.
    """

    def __init__(self, engine: AdventureEngine, history_preview: int = 4):
        self.engine: AdventureEngine = engine
        self.context: AdventureContext = engine.context

        self.running: bool = True
        self.history_preview: int = history_preview

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self) -> None:
        self._print_header()
        self._print_recent_history()

        # =========================
        # ADVENTURE INIT (NEW)
        # =========================
        init = self.engine.initialize_adventure_stream()
        if init is not None:
            stream, start_time = init
            self._print_stream(stream, start_time)

        while self.running:
            try:
                user_input: str = input("\n> ")

                if self._handle_exit(user_input):
                    continue

                print("CALLING STREAM METHOD")
                stream, start_time = self.engine.process_player_input_stream(user_input)

                self._print_stream(stream, start_time)

            except KeyboardInterrupt:
                print("\n[Interrupted]")
                break

            except Exception as e:
                print("\n[ERROR OCCURRED]")
                print(f"Message: {e}")
                print("\n--- TRACEBACK ---")
                traceback.print_exc()

        self._shutdown()

    # =========================================================
    # STARTUP
    # =========================================================

    def _print_header(self) -> None:
        print("=" * 60)
        print("ADVENTURE ENGINE CLI")
        print("=" * 60)
        print("Type 'help' for commands.")

    def _print_recent_history(self) -> None:
        print("\n=== RECENT HISTORY ===")

        turns: List[Dict] = self.context.get_last_turns(
            self.history_preview
        )

        if not turns:
            print("[No previous turns]")
            return

        for turn in turns:
            role: str = turn.get("role", "Unknown")
            content: str = turn.get("content", "")
            turn_id: int = turn.get("turn_id", -1)

            print(f"[Turn {turn_id}] {role}: {content}")

    # =========================================================
    # HELPERS
    # =========================================================

    def _handle_exit(self, user_input: str) -> bool:
        cmd: str = user_input.strip().lower()

        if cmd in ["exit", "quit", "q"]:
            self.running = False
            return True

        if cmd == "help":
            self._print_help()
            return True

        return False

    def _print_response(self, response: Optional[str]) -> None:
        if response:
            print("\n" + response)
        else:
            print("\n[No response]")

    def _print_help(self) -> None:
        print(self.engine.get_help_text())

    def _print_stream(self, stream, start_time: float) -> None:
        first_token_time: Optional[float] = None
        full_text: str = ""

        print()

        for event in stream:
            if event["type"] == "llm":
                if first_token_time is None:
                    first_token_time = time.time()
                    delta = first_token_time - start_time
                    print(f"\n[first token: {delta:.2f}s]\n")

                full_text += event["content"]

                print(event["content"], end="", flush=True)

            elif event["type"] == "system":
                print(f"\n{event['content']}")

        print()

        if start_time > 0:
            total_time = time.time() - start_time
            print(f"\n[done: {len(full_text)} chars in {total_time:.2f}s]")

    def _shutdown(self) -> None:
        print("\nShutting down...")
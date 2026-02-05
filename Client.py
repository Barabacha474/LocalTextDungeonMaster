import ollama
import time
import os
import random
from pathlib import Path
from PromptConstructor import PromptConstructor
from SettingDataLoaders.FaissVectorDB import FAISSVectorDB
from AdventureLogger import AdventureLogger
from SummarizerOllamaLLM import SummarizerOllamaLLM
from LastNTurnsPromptConstructor import LastNTurnsPromptConstructor


def main():
    # Adventure name
    adventure_name = "vanilla_fantasy"

    print(f"=== Initializing Adventure: '{adventure_name}' ===")

    # Initialize vector database
    print("Initializing vector database...")
    vector_db = FAISSVectorDB(
        adventure_name=adventure_name,
        storage_path="./adventure_memories"
    )

    # Initialize SQL database logger
    print("Initializing adventure logger...")
    sql_db = AdventureLogger(
        adventure_name=adventure_name,
        storage_path="./adventure_logs"
    )

    # Initialize Summarizer
    print("Initializing summarizer...")
    # Create LastNTurnsPromptConstructor for summarizer
    summarizer_prompt_constructor = LastNTurnsPromptConstructor(
        adventure_logger=sql_db,
        num_turns=4,
        special_command=SummarizerOllamaLLM._get_default_summarization_prompt()
    )

    # Initialize Summarizer OllamaLLM
    summarizer = SummarizerOllamaLLM(
        model='mistral:7b-instruct-v0.3-q4_0',
        prompt_constructor=summarizer_prompt_constructor
    )

    # Check if there are existing turns (not initial prompt)
    existing_turns = sql_db.get_turn_count()
    is_initial_prompt = existing_turns == 0

    print(f"Existing turns in database: {existing_turns}")
    print(f"Is initial prompt: {is_initial_prompt}")

    # Initialize the PromptConstructor with databases
    constructor = PromptConstructor(
        adventure_logger=sql_db,
        vector_db=vector_db,
        config_path="SettingRawDataJSON/vanilla_fantasy/PromptCore.json"
    )

    # Special continue message
    CONTINUE_MESSAGE = constructor.get_continue_message()

    # Optional: Warm up Ollama once at startup
    print("\n=== Warming up Ollama ===")
    try:
        warmup_response = ollama.generate(
            # model='deepseek-r1:7b',
            model='llama3.1:8b-instruct-q4_K_M',
            prompt='.',
            options={'temperature': 0.1, 'num_predict': 1}
        )
        print("Model warmed up.")
    except Exception as e:
        print(f"Warmup note: {e}")
        print("Continuing anyway...")

    print("\n" + "=" * 60)
    print("ADVENTURE STARTED")
    if is_initial_prompt:
        print("OPENING TEXT")
        print("\n" + "*" * 60)
        print(constructor.get_player_opening_text())
        print("*" * 60 + "\n")
    else:
        print("RECENT ACTIONS")
        print("\n" + "*" * 60)
        last_turns = sql_db.get_last_n_turns(10)
        if last_turns:
            # Format conversation for summarization
            conversation_text = ""
            for turn in last_turns:
                speaker = "Player" if turn['role'] == 'Player' else "DM"
                conversation_text += f"{speaker}: {turn['content']}\n\n"
            print(conversation_text)
        print("*" * 60 + "\n")

    # Enhanced command help
    print("AVAILABLE COMMANDS:")
    print("  'quit' - Exit the adventure")
    print("  'clear' - Clear all conversation history")
    print("  'stats' - Show database statistics")
    print("  'regenerate' - Regenerate last DM response with random seed")
    print("  'undo' - Remove last DM response (and player input if applicable)")
    print("  'continue' - continue story from where it ends")
    print("=" * 60 + "\n")

    # Main interaction loop
    while True:
        turn_counter = sql_db.get_turn_count()

        # Get player input
        print(f"\n[Turn {turn_counter + 1}]")
        player_input = input("Player: ").strip()

        # Check for commands
        if player_input.lower() == 'quit':
            print("Ending adventure...")
            break

        elif player_input.lower() == 'clear':
            sql_db.clear_all_turns()
            print("Conversation cleared!")
            continue

        elif player_input.lower() == 'stats':
            print(f"\n=== DATABASE STATISTICS ===")
            print(f"SQL Database turns: {sql_db.get_turn_count()}")
            print(f"Vector Database documents: {vector_db.get_document_count()}")

            # Show last turn info
            last_turn = sql_db.get_latest_turn()
            if last_turn:
                print(f"Latest turn ID: {last_turn['id']}")
                print(f"Latest turn role: {last_turn['role']}")
                print(f"Latest turn time: {last_turn['timestamp']}")
            continue

        elif player_input.lower() in ['regenarate', 'regenerate']:

            # Get the last two turns

            last_turns = sql_db.get_last_n_turns(2)

            if len(last_turns) == 2:

                # Determine the roles of the last two turns

                last_turn_role = last_turns[1]['role']

                second_last_turn_role = last_turns[0]['role']

                print(f"\n=== REGENERATING LAST DM RESPONSE ===")

                if last_turn_role == 'DM' and second_last_turn_role == 'Player':

                    # Case: [Player, DM] - Normal regeneration

                    last_player_turn = last_turns[1]  # Player turn

                    last_dm_turn = last_turns[0]  # DM turn

                    print(f"Original player input: {last_player_turn['content'][:100]}...")

                    print(f"Original DM response: {last_dm_turn['content'][:100]}...")

                    # Delete the last DM response

                    sql_db.delete(last_dm_turn['id'])

                    print(f"Deleted DM turn ID: {last_dm_turn['id']}")

                    # Use the same player input for regeneration

                    player_input = last_player_turn['content']

                    print(f"Using player input: {player_input[:100]}...")

                    # Skip logging player input (already exists)

                    skip_player_logging = True


                elif last_turn_role == 'DM' and second_last_turn_role == 'DM':

                    # Case: [DM, DM] - DM-DM pair

                    last_dm_turn = last_turns[1]  # Most recent DM

                    second_last_dm_turn = last_turns[0]  # Previous DM

                    print(f"Found DM-DM pair. Last DM response: {last_dm_turn['content'][:100]}...")

                    print(f"Previous DM response: {second_last_dm_turn['content'][:100]}...")

                    # Delete the last DM response

                    sql_db.delete(last_dm_turn['id'])

                    print(f"Deleted DM turn ID: {last_dm_turn['id']}")

                    # Use continue message for regeneration

                    player_input = CONTINUE_MESSAGE

                    print(f"Using continue message: {player_input[:100]}...")

                    # Don't log this as a player input

                    should_log_input = False

                    skip_player_logging = True

                # Generate a random seed for regeneration

                random_seed = random.randint(1, 10000)

                print(f"Using random seed: {random_seed}")


            else:

                print("Cannot regenerate: Need at least one DM response in history.")

                continue

        elif player_input.lower() == 'undo':

            # Get the last two turns

            last_turns = sql_db.get_last_n_turns(2)

            if len(last_turns) == 2:

                # Determine the roles of the last two turns

                last_turn = last_turns[1]

                second_last_turn = last_turns[0]

                last_turn_role = last_turn['role']

                second_last_turn_role = second_last_turn['role']

                print(f"\n=== UNDO ACTION ===")

                if last_turn_role == 'DM' and second_last_turn_role == 'Player':

                    # Case: [Player, DM] - Delete both player and DM

                    sql_db.delete(last_turn['id'])

                    print(f"Deleted DM turn ID {last_turn['id']}")

                    sql_db.delete(second_last_turn['id'])

                    print(f"Deleted Player turn ID {second_last_turn['id']}")

                    # Show the deleted player input

                    print(f"\nDeleted player input (copy if needed):")

                    print(f"  {second_last_turn['content']}")

                    print("Last player-DM pair removed.")


                elif last_turn_role == 'DM' and second_last_turn_role == 'DM':

                    # Case: [DM, DM] - Delete only the last DM

                    sql_db.delete(last_turn['id'])

                    print(f"Deleted DM turn ID {last_turn['id']} (DM-DM pair)")

                    print("Only last DM response removed.")


                else:

                    # Other cases (Player-Player or mixed)

                    sql_db.delete(last_turn['id'])

                    print(f"Deleted turn ID {last_turn['id']} (role: {last_turn['role']})")


            elif len(last_turns) == 1:

                # Only one turn exists

                last_turn = last_turns[0]

                sql_db.delete(last_turn['id'])

                print(f"\n=== UNDO ACTION ===")

                print(f"Deleted turn ID {last_turn['id']} (role: {last_turn['role']})")


            else:

                print("Nothing to undo.")

            continue

        elif player_input.lower() == 'continue':
            # Use special prompt to continue the story
            player_input = CONTINUE_MESSAGE
            print(f"\n=== CONTINUING STORY NATURALLY ===")
            print(f"Using special prompt: {player_input}...")

            # Don't log this as a player input
            skip_player_logging = True

        elif not player_input:
            print("Please enter some text.")
            continue
        else:
            skip_player_logging = False

        # Get the prompt
        prompt = constructor.get_prompt(
            user_input=player_input,
            is_initial_prompt=is_initial_prompt,
            n_last_turns=5,
            vector_search_k=10,
            vector_threshold=0.1,
            chunk_size=100
        )

        # Log player input to SQL database (unless skipping for regeneration)
        if not skip_player_logging:
            player_turn_id = sql_db.write("Player", player_input)
            print(f"Logged player input (Turn ID: {player_turn_id})")

        # Print full prompt for debugging (optional - can be toggled)
        show_full_prompt = True  # Set to False to hide full prompts
        if show_full_prompt:
            print("\n" + "=" * 60)
            print("FULL PROMPT (sent to model):")
            print("=" * 60)
            print(prompt)
            print("=" * 60 + "\n")
        else:
            print(f"\nPrompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        # Stream response from Ollama
        print("DM: ", end='', flush=True)

        stream_start = time.time()
        first_token_time = None
        full_response = ""

        try:
            # Determine seed (use random seed for regeneration, fixed seed otherwise)
            seed = random_seed if 'random_seed' in locals() else 42

            stream = ollama.generate(
                model='llama3.1:8b-instruct-q4_K_M',
                prompt=prompt,
                options={
                    'temperature': 0.6,
                    'seed': seed,
                    'num_predict': 1000
                },
                stream=True
            )

            for chunk in stream:
                content = chunk.get('response', '')

                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        first_token_delay = first_token_time - stream_start
                        print(f"[{first_token_delay:.1f}s] ", end='', flush=True)

                    print(content, end='', flush=True)
                    full_response += content

            print()  # Final newline

            # Print timing info
            if first_token_time:
                total_time = time.time() - stream_start
                print(f"[Response: {len(full_response)} chars, {total_time:.1f}s total]")

        except Exception as e:
            print(f"\nError generating response: {e}")
            full_response = f"[Error: {e}]"

        # Log DM response to SQL database
        dm_turn_id = sql_db.write("DM", full_response)
        print(f"Logged DM response (Turn ID: {dm_turn_id})")

        # Clean up regeneration variables
        if 'random_seed' in locals():
            del random_seed
            del skip_player_logging

        # Every 4 turns, summarize and add to vector DB
        if turn_counter % 4 == 0:
            print("\n" + "=" * 60)
            print("SUMMARIZING LAST 4 TURNS")
            print("=" * 60)

            try:
                # Generate summary using the SummarizerOllamaLLM
                print("Generating summary...")
                summary = summarizer.generate(
                    num_turns=4,  # Summarize last 4 turns
                    desired_response_size=150,  # Changed from 300 to 150 words for consistency
                    temperature=0.3,
                    seed=42
                )

                print(f"\nSUMMARY:\n {summary}\n")

                # Get the turn IDs for metadata (need last 4 turns for range)
                last_turns = sql_db.get_last_n_turns(4)

                if last_turns:
                    # Add summary to vector DB
                    memory_text = f"Summary of turns {last_turns[0]['id']}-{last_turns[-1]['id']}:\n{summary}"

                    vector_db.insert_single(
                        text=memory_text,
                        metadata={
                            'type': 'conversation_summary',
                            'turn_range': f"{last_turns[0]['id']}-{last_turns[-1]['id']}",
                            'turn_count': len(last_turns),
                            'summary_turn': turn_counter,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )
                    print("Summary added to vector database as memory.")
                else:
                    print("Not enough turns to summarize yet.")

            except Exception as e:
                print(f"Error during summarization: {e}")
                import traceback
                traceback.print_exc()

            print("=" * 60 + "\n")

        # Update is_initial_prompt flag after first turn
        if is_initial_prompt:
            is_initial_prompt = False

    # Cleanup on exit
    print("\n" + "=" * 60)
    print("ADVENTURE ENDED - FINAL STATISTICS")
    print("=" * 60)

    print(f"Total turns: {turn_counter}")
    print(f"SQL Database turns: {sql_db.get_turn_count()}")
    print(f"Vector Database documents: {vector_db.get_document_count()}")

    # Show last 5 turns
    last_turns = sql_db.get_last_n_turns(5)
    print(f"\nLast {len(last_turns)} turns:")
    for turn in last_turns:
        print(f"  [{turn['id']}] {turn['role']}: {turn['content'][:60]}...")

    # Close databases
    print("\nClosing databases...")
    vector_db.close()
    sql_db.close()
    print("Adventure saved successfully!")


if __name__ == "__main__":
    main()

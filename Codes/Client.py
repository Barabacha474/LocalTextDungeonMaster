import ollama
import time
import random
from PromptConstructor import PromptConstructor
from FaissVectorDB import FAISSVectorDB
from AdventureLogger import AdventureLogger
from SummarizerOllamaLLM import SummarizerOllamaLLM
from LastNTurnsPromptConstructor import LastNTurnsPromptConstructor


def main():
    # Adventure name
    adventure_name = "vanilla_fantasy"

    #narrator_model_name = 'Tohur/natsumura-storytelling-rp-llama-3.1:latest'
    #summarizer_model_name = 'Tohur/natsumura-storytelling-rp-llama-3.1:latest'
    narrator_model_name = 'cogito:8b-v1-preview-llama-q4_K_M'
    summarizer_model_name = 'cogito:8b-v1-preview-llama-q4_K_M'
    context_size = 1024 * 7

    print(f"=== Initializing Adventure: '{adventure_name}' ===")

    # Initialize vector database
    print("Initializing vector database...")
    vector_db = FAISSVectorDB(
        adventure_name=adventure_name,
        storage_path="../adventure_memories"
    )

    # Initialize SQL database logger (updated to support seeds and turn_id)
    print("Initializing adventure logger...")
    sql_db = AdventureLogger(
        adventure_name=adventure_name,
        storage_path="../adventure_logs"
    )

    # Initialize Summarizer
    print("Initializing summarizer...")
    summarizer_prompt_constructor = LastNTurnsPromptConstructor(
        adventure_logger=sql_db,
        num_turns=4,
        special_command=SummarizerOllamaLLM._get_default_summarization_prompt()
    )

    summarizer = SummarizerOllamaLLM(model=summarizer_model_name,
                                     prompt_constructor=summarizer_prompt_constructor)

    # Check if there are existing turns (not initial prompt)
    existing_turns = sql_db.get_turn_count()
    is_initial_prompt = existing_turns == 0

    print(f"Existing turns in database: {existing_turns}")
    print(f"Is initial prompt: {is_initial_prompt}")

    # Initialize the PromptConstructor with databases
    constructor = PromptConstructor(
        adventure_logger=sql_db,
        vector_db=vector_db,
        config_path="../SettingRawDataJSON/vanilla_fantasy/PromptCore.json"
    )

    # Special continue message
    CONTINUE_MESSAGE = constructor.get_continue_message()

    # Optional: Warm up Ollama once at startup
    print("\n=== Warming up Ollama ===")
    try:
        # warmup_response = ollama.generate(
        #     model=narrator_model_name,
        #     prompt='.',
        #     options={'temperature': 0.1, 'num_predict': 1, 'num_ctx': context_size}
        # )
        warmup_response = ollama.chat(
            model=narrator_model_name,
            messages=[
                {'role': 'system', 'content': 'Enable deep thinking subroutine.'},
                {'role': 'user', 'content': '.'}
            ],
            options={'temperature': 0.1, 'num_predict': 1, 'num_ctx': context_size}
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
            # Format conversation for summarization (roles now include Narrator)
            conversation_text = ""
            for turn in last_turns:
                # Use the role directly (should be 'Player' or 'Narrator')
                speaker = turn['role']
                conversation_text += f"{speaker}: {turn['content']}\n\n"
            print(conversation_text)
        print("*" * 60 + "\n")

    # Enhanced command help
    print("AVAILABLE COMMANDS:")
    print("  'quit' - Exit the adventure")
    print("  'clear' - Clear all conversation history")
    print("  'stats' - Show database statistics")
    print("  'regenerate' - Regenerate last Narrator response with random seed")
    print("  'undo' - Remove last Narrator response (and player input if applicable)")
    print("  'continue' - continue story from where it ends")
    print("=" * 60 + "\n")

    # Main interaction loop
    while True:
        turn_counter = sql_db.get_turn_count() + 1

        # Get player input
        print(f"\n[Turn {turn_counter}]")
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

            # Show last turn info (using turn_id)
            last_turn = sql_db.get_latest_turn()
            if last_turn:
                print(f"Latest turn ID: {last_turn['turn_id']}")
                print(f"Latest turn role: {last_turn['role']}")
                print(f"Latest turn time: {last_turn['timestamp']}")
                print(f"Seed used: {last_turn['seed']}")
            continue

        elif player_input.lower() in ['regenarate', 'regenerate']:
            # Get all entries for the last logical turn
            last_entries = sql_db.get_last_n_turns(1)
            if not last_entries:
                print("No turns to regenerate.")
                continue

            turn_id = last_entries[0]['turn_id']
            player_entry = None
            narrator_sql_ids = []

            # Separate player and narrator entries
            for entry in last_entries:
                if entry['role'] == 'Player':
                    player_entry = entry
                elif entry['role'] == 'Narrator':
                    narrator_sql_ids.append(entry['sql_id'])

            if not player_entry:
                print("Cannot regenerate: last turn has no player input (maybe a 'continue' turn).")
                continue

            if not narrator_sql_ids:
                print("No narrator response to regenerate.")
                continue

            # Delete all narrator entries for this turn (by sql_id)
            for sql_id in narrator_sql_ids:
                sql_db.delete_by_sql_id(sql_id)

            print(f"\n=== REGENERATING LAST NARRATOR RESPONSE ===")
            print(f"Original player input: {player_entry['content'][:100]}...")
            print(f"Deleted {len(narrator_sql_ids)} narrator entry(s).")

            # Use the saved player input and skip re-logging it
            player_input = player_entry['content']
            skip_player_logging = True
            random_seed = random.randint(1, 10000)
            print(f"Using random seed: {random_seed}")

        elif player_input.lower() == 'undo':

            # Get all entries for the last logical turn

            last_entries = sql_db.get_last_n_turns(1)

            if not last_entries:
                print("Nothing to undo.")

                continue

            turn_id = last_entries[0]['turn_id']

            player_content = None

            # Look for a player entry to display

            for entry in last_entries:

                if entry['role'] == 'Player':
                    player_content = entry['content']

                    break

            # Delete the entire turn (both player and narrator)

            if sql_db.delete(turn_id):

                print(f"\n=== UNDO ACTION ===")

                if player_content:
                    print(f"Last player input (copy if you need it):\n  {player_content}")

                print(f"Removed turn {turn_id}.")

            else:

                print("Failed to undo.")

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
            turn_id=turn_counter,
            user_input=player_input,
            is_initial_prompt=is_initial_prompt,
            n_last_turns=5,
            vector_search_k_per_cascade=5,
            number_of_cascades=1,
            vector_threshold=0.4,
            chunk_size=100
        )

        # Log player input to SQL database (unless skipping for regeneration)
        if not skip_player_logging:
            player_turn_id = sql_db.write(turn_id=turn_counter, role="Player", content=player_input)  # seed not needed for player
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
        print("Narrator: ", end='', flush=True)

        stream_start = time.time()
        first_token_time = None
        full_response = ""

        try:
            # Determine seed (use random seed for regeneration, fixed seed otherwise)
            seed = random_seed if 'random_seed' in locals() else 42

            stream = ollama.generate(
                model=narrator_model_name,
                prompt=prompt,
                options={
                    'temperature': 0.6,
                    'seed': seed,
                    'num_predict': 1000,
                    'num_ctx': context_size
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

        # Log Narrator response to SQL database, including the seed used
        narrator_turn_id = sql_db.write(turn_id=turn_counter, role="Narrator", content=full_response, model_name=narrator_model_name, seed=seed)
        print(f"Logged Narrator response (Turn ID: {narrator_turn_id}) with seed {seed}")

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
                    desired_response_size_in_sentences=20,
                    temperature=0.3,
                    seed=42,
                )

                print(f"\nSUMMARY:\n {summary}\n")

                # Get the last 4 turns for metadata (using logical turn_ids)
                last_turns = sql_db.get_last_n_turns(4)

                if last_turns:
                    # Add summary to vector DB – use turn_ids for range
                    memory_text = f"Summary of turns {last_turns[0]['turn_id']}-{last_turns[-1]['turn_id']}:\n{summary}"

                    vector_db.insert_single(
                        text=memory_text,
                        metadata={
                            'type': 'Memory',
                            'turn_range': f"{last_turns[0]['turn_id']}-{last_turns[-1]['turn_id']}",
                            'turn_count': len(last_turns),
                            'summary_turn': turn_counter,
                            'model_name': summarizer_model_name,
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

    # Show last 5 turns (using turn_id)
    last_turns = sql_db.get_last_n_turns(5)
    print(f"\nLast {len(last_turns)} turns:")
    for turn in last_turns:
        print(f"  [Turn {turn['turn_id']}] {turn['role']}: {turn['content'][:60]}... (seed: {turn['seed']})")

    # Close databases
    print("\nClosing databases...")
    vector_db.close()
    sql_db.close()
    print("Adventure saved successfully!")


if __name__ == "__main__":
    main()
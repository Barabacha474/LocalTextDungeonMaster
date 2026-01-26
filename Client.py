import ollama
import time
import os
from pathlib import Path
from PromptConstructor import PromptConstructor
from SettingDataLoaders.FaissVectorDB import FAISSVectorDB
from AdventureLogger import AdventureLogger
from Summarizer import Summarizer


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
    summarizer = Summarizer()

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

    # Optional: Warm up Ollama once at startup
    print("\n=== Warming up Ollama ===")
    try:
        warmup_response = ollama.generate(
            model='deepseek-r1:7b',
            prompt='.',
            options={'temperature': 0.1, 'num_predict': 1}
        )
        print("Model warmed up.")
    except Exception as e:
        print(f"Warmup note: {e}")
        print("Continuing anyway...")

    print("\n" + "=" * 60)
    print("ADVENTURE STARTED")
    print("Type 'quit' to exit, 'clear' to clear conversation")
    print("=" * 60 + "\n")

    # Main interaction loop
    turn_counter = 0
    while True:
        turn_counter += 1

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
            print(f"SQL DB turns: {sql_db.get_turn_count()}")
            print(f"Vector DB documents: {vector_db.get_document_count()}")
            continue
        elif not player_input:
            print("Please enter some text.")
            continue

        # Log player input to SQL database
        player_turn_id = sql_db.write("Player", player_input)
        print(f"Logged player input (Turn ID: {player_turn_id})")

        # Get the prompt
        prompt = constructor.get_prompt(
            user_input=player_input,
            is_initial_prompt=is_initial_prompt and turn_counter == 1,
            n_last_turns=5,
            vector_search_k=3,
            vector_threshold=0.5
        )

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
            stream = ollama.generate(
                model='deepseek-r1:7b',
                prompt=prompt,
                options={'temperature': 0.6},
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
        dm_turn_id = sql_db.write("System", full_response)
        print(f"Logged DM response (Turn ID: {dm_turn_id})")

        # Every 4 turns, summarize and add to vector DB
        if turn_counter % 4 == 0:
            print("\n" + "=" * 60)
            print("SUMMARIZING LAST 4 TURNS")
            print("=" * 60)

            try:
                # Get last 4 turns (which includes 2 player inputs and 2 DM responses)
                # Actually we want last 8 entries (4 pairs) for better context
                last_turns = sql_db.get_last_n_turns(8)

                if last_turns:
                    # Format conversation for summarization
                    conversation_text = ""
                    for turn in last_turns:
                        speaker = "Player" if turn['role'] == 'Player' else "DM"
                        conversation_text += f"{speaker}: {turn['content']}\n\n"

                    # Generate summary
                    print("Generating summary...")
                    summary = summarizer.summarize(
                        input_text=conversation_text,
                        desired_response_size=150
                    )

                    print(f"Summary: {summary[:200]}..." if len(summary) > 200 else f"Summary: {summary}")

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
        if is_initial_prompt and turn_counter == 1:
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
import json
import time
from pathlib import Path

import ollama


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_role_and_instructions(data: dict, use_planner: bool) -> tuple[str, str]:
    if use_planner:
        role = data.get("planner_role", "").strip()
        instructions = data.get("planner_instructions", "").strip()
    else:
        role = data.get("narrator_role", "").strip()
        instructions = data.get("narrator_instructions", "").strip()
    return role, instructions


def collect_turns(
    data: dict,
    start_turn: int,
    num_turns: int,
    turn_key_prefix: str = "turn_",
) -> list[str]:
    turns = []
    for i in range(start_turn, start_turn + num_turns):
        key = f"{turn_key_prefix}{i}"
        value = data.get(key, "")
        if value:
            turns.append(value.strip())
    return turns


def build_prompt(
    data: dict,
    use_planner: bool = True,
    include_role: bool = True,
    include_instructions: bool = True,
    include_setting: bool = True,
    include_story_summary: bool = True,
    include_continue_message: bool = False,
    include_ending_message: bool = True,
    start_turn: int = 1,
    num_turns: int = 4,
    section_separator: str = "\n\n" + "=" * 80 + "\n\n",
    header_role: str = "ROLE",
    header_instructions: str = "INSTRUCTIONS",
    header_setting: str = "SETTING",
    header_story_summary: str = "STORY SUMMARY",
    header_turns: str = "RECENT TURNS",
    header_continue: str = "CONTINUE MESSAGE",
    header_ending_message: str = "ENDING MESSAGE"
) -> str:
    role, instructions = pick_role_and_instructions(data, use_planner=use_planner)

    parts = []

    if include_role and role:
        parts.append(f"{header_role}:\n{role}")

    if include_instructions and instructions:
        parts.append(f"{header_instructions}:\n{instructions}")

    setting = data.get("setting", "").strip()
    if include_setting and setting:
        parts.append(f"{header_setting}:\n{setting}")

    story_summary = data.get("story_summary", "").strip()
    if include_story_summary and story_summary:
        parts.append(f"{header_story_summary}:\n{story_summary}")

    turns = collect_turns(
        data=data,
        start_turn=start_turn,
        num_turns=num_turns,
    )
    if turns:
        turns_block = "\n\n".join(turns)
        parts.append(f"{header_turns}:\n{turns_block}")

    continue_message = data.get("continue_message", "").strip()
    if include_continue_message and continue_message:
        parts.append(f"{header_continue}:\n{continue_message}")

    ending_message = data.get("planner_ending_message", "").strip() if use_planner else data.get("narrator_ending_message", "").strip()
    if include_ending_message and ending_message:
        parts.append(f"{header_ending_message}:\n{ending_message}")

    return section_separator.join(parts).strip()


def warmup_model(model_name: str, context_size: int) -> None:
    try:
        ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Warmup."},
                {"role": "user", "content": "."},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 1,
                "num_ctx": context_size,
            },
        )
        print("Warmup complete.\n")
    except Exception as e:
        print(f"Warmup skipped: {e}\n")


def stream_generation(
    model_name: str,
    prompt: str,
    temperature: float,
    seed: int,
    num_predict: int,
    context_size: int,
    keep_alive: str | int = "5m",
    print_first_token_delay: bool = True,
) -> str:
    full_response = ""
    stream_start = time.time()
    first_token_time = None

    stream = ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=True,
        keep_alive=keep_alive,
        options={
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": context_size,
        },
    )

    for chunk in stream:
        content = chunk.get("response", "")
        if not content:
            continue

        if first_token_time is None:
            first_token_time = time.time()
            if print_first_token_delay:
                delay = first_token_time - stream_start
                print(f"[first token: {delay:.2f}s]\n")

        print(content, end="", flush=True)
        full_response += content

    print()
    total_time = time.time() - stream_start
    print(f"\n[done: {len(full_response)} chars in {total_time:.2f}s]")
    return full_response


def main() -> None:
    # =========================
    # FILE / MODEL CONFIG
    # =========================
    json_path = "/TEST_PROMPT/Greyheaven_TEST_PROMPT.json"
    # model_name = "mistral-small:22b-instruct-2409-q3_K_S"
    # model_name = "cogito:8b-v1-preview-llama-q4_K_M"
    model_name = "qwen3:14b"
    # model_name = "Tohur/natsumura-storytelling-rp-llama-3.1:latest"
    context_size = 1024 * 8
    keep_alive = "10m"

    # =========================
    # PROMPT CONSTRUCTION CONFIG
    # =========================
    use_planner = True
    include_role = True
    include_instructions = True
    include_setting = True
    include_story_summary = True
    include_continue_message = False
    include_ending_message = True

    start_turn = 1
    num_turns = 4

    section_separator = "\n\n" + "=" * 80 + "\n\n"
    header_role = "ROLE"
    header_instructions = "INSTRUCTIONS"
    header_setting = "SETTING"
    header_story_summary = "STORY SUMMARY"
    header_turns = "RECENT TURNS"
    header_continue = "CONTINUE MESSAGE"
    header_ending_message = "ENDING MESSAGE"

    # =========================
    # GENERATION CONFIG
    # =========================
    temperature = 0.6
    seed = 42
    num_predict = 1200

    # =========================
    # DEBUG / DISPLAY CONFIG
    # =========================
    do_warmup = False
    show_full_prompt = True
    show_prompt_stats = True
    save_prompt_to_file = False
    prompt_output_path = "constructed_prompt.txt"

    # =========================
    # RUN
    # =========================
    print("=== Simple JSON Prompt Tester ===")
    print(f"JSON file: {Path(json_path).resolve()}")
    print(f"Model: {model_name}")
    print(f"Mode: {'Planner' if use_planner else 'Narrator'}")
    print(f"Turns: {start_turn}..{start_turn + num_turns - 1}")
    print()

    data = load_json(json_path)

    prompt = build_prompt(
        data=data,
        use_planner=use_planner,
        include_role=include_role,
        include_instructions=include_instructions,
        include_setting=include_setting,
        include_story_summary=include_story_summary,
        include_continue_message=include_continue_message,
        include_ending_message=include_ending_message,
        start_turn=start_turn,
        num_turns=num_turns,
        section_separator=section_separator,
        header_role=header_role,
        header_instructions=header_instructions,
        header_setting=header_setting,
        header_story_summary=header_story_summary,
        header_turns=header_turns,
        header_continue=header_continue,
        header_ending_message=header_ending_message
    )

    if show_prompt_stats:
        print("=== Prompt Stats ===")
        print(f"Characters: {len(prompt)}")
        print(f"Approx tokens: {len(prompt) // 4}")
        print()

    if save_prompt_to_file:
        with open(prompt_output_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Prompt saved to: {Path(prompt_output_path).resolve()}\n")

    if show_full_prompt:
        print("=" * 80)
        print("FULL PROMPT")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print()

    if do_warmup:
        warmup_model(model_name=model_name, context_size=context_size)

    print("=== Generation ===\n")
    _response = stream_generation(
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
        context_size=context_size,
        keep_alive=keep_alive,
        print_first_token_delay=True,
    )


if __name__ == "__main__":
    main()
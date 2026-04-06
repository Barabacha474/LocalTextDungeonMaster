import time

from Codes.Databases.AdventureLogger import AdventureLogger
from Codes.Databases.FaissVectorDB import FAISSVectorDB
from Codes.Orchestrators.AdventureContext import AdventureContext


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_turns(turns):
    for t in turns:
        print(f"[Turn {t['turn_id']}] {t['role']}: {t['content']}")


def print_memories(memories):
    for m in memories:
        meta = m.get("metadata", {})
        print(f"[ID {m['id']}] {meta.get('label')} | {m['text'][:60]}...")


def main():
    adventure_name = "vanilla_fantasy"

    print_header("INIT")

    sql_db = AdventureLogger(adventure_name=adventure_name, storage_path="../../adventure_logs")
    vector_db = FAISSVectorDB(adventure_name=adventure_name, storage_path="../../adventure_memories")

    context = AdventureContext(sql_db, vector_db)

    # Clean start
    context.clear_all_turns()
    vector_db.clear()
    vector_db.save()

    # =========================================================
    # TEST 1: LOGGING
    # =========================================================

    print_header("TEST 1: LOGGING")

    context.log_turn(1, "Player", "I enter the cave")
    context.log_turn(1, "Narrator", "The cave is dark and cold")

    context.log_turn(2, "Player", "I light a torch")
    context.log_turn(2, "Narrator", "The walls reveal ancient carvings")

    turns = context.get_last_turns(5)
    print_turns(turns)

    # =========================================================
    # TEST 2: ROLE FILTER
    # =========================================================

    print_header("TEST 2: ROLE FILTER (Narrator only)")

    narrator_turns = context.get_last_turns(5, roles=["Narrator"])
    print_turns(narrator_turns)

    # =========================================================
    # TEST 3: MEMORY SAVE
    # =========================================================

    print_header("TEST 3: MEMORY SAVE")

    memory_text = "Player explored a cave and discovered ancient carvings."
    context.save_memory(memory_text, turn_start=1, turn_end=2)

    memories = context.get_all_memory()
    print_memories(memories)

    # =========================================================
    # TEST 4: MEMORY SEARCH
    # =========================================================

    print_header("TEST 4: MEMORY SEARCH")

    results = context.get_relevant_info("cave carvings", k_per_cascade=3)
    print_memories(results)

    # =========================================================
    # TEST 5: UNDO
    # =========================================================

    print_header("TEST 5: UNDO LAST TURN")

    context.undo_last_turn()

    turns = context.get_last_turns(5)
    print_turns(turns)

    memories = context.get_all_memory()
    print_memories(memories)

    # =========================================================
    # TEST 5.5: PARTIAL DELETE (ROLE-BASED)
    # =========================================================

    print_header("TEST 5.5: DELETE ONLY NARRATOR FROM LAST TURN")

    # Добавим planner + narrator в один ход
    context.log_turn(3, "Player", "I inspect the carvings closely")
    context.log_turn(3, "Planner", "Introduce hidden mechanism in wall")
    context.log_turn(3, "Narrator", "You notice a strange indentation between stones")

    print("Before deletion:")
    print_turns(context.get_last_turns(5))

    # Удаляем только Narrator
    context.delete_turn_roles(3, ["Narrator"])

    print("\nAfter deleting Narrator only:")
    print_turns(context.get_last_turns(5))

    # =========================================================
    # TEST 6: ADD MORE + MEMORY CLEANUP
    # =========================================================

    print_header("TEST 6: MEMORY CLEANUP AFTER ROLLBACK")

    context.log_turn(2, "Player", "I leave the cave")
    context.log_turn(2, "Narrator", "You step back into daylight")

    context.save_memory(
        "Player exited the cave safely.",
        turn_start=2,
        turn_end=2
    )

    print("Before rollback:")
    print_memories(context.get_all_memory())

    context.delete_turn(2)
    context.delete_memories_after_turn(1)

    print("After rollback:")
    print_memories(context.get_all_memory())

    # =========================================================
    # FINAL
    # =========================================================

    print_header("FINAL STATE")

    print("Turns:")
    print_turns(context.get_last_turns(10))

    print("\nMemories:")
    print_memories(context.get_all_memory())

    # Cleanup
    sql_db.close()
    vector_db.close()

    print("\nDONE.")


if __name__ == "__main__":
    main()
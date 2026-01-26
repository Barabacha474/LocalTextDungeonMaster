import time
from typing import Dict, List, Tuple
import ollama
from Summarizer import Summarizer  # Assuming the class is in summarizer.py


class SummarizerTester:
    """Test harness for the Summarizer class with synthetic adventure logs."""

    def __init__(self):
        self.summarizer = Summarizer()
        self.test_logs = self._create_test_logs()
        self.results = []  # Store test results with timing

    def _create_test_logs(self) -> Dict[str, Dict[str, str]]:
        """Create synthetic adventure logs focusing on different aspects."""

        base_logs = {
            "location_exploration": {
                "purpose": "Focus on location descriptions and exploration",
                "word_count": 154,
                "content": """The party entered the Whispering Woods, a dense forest where trees seemed to lean in as if listening. Elara the ranger pointed out glowing mushrooms that pulsed with a soft blue light. "These are Mooncap fungi," she explained. "They only grow in places touched by fey magic." Deeper in, they found a clearing with a stone circle. The stones were carved with ancient elven runes that Kaelen the wizard recognized as wards against dark spirits. In the center of the circle, a small spring bubbled with crystal-clear water that shimmered with silver flecks. Thorn the druid detected a faint magical aura. They decided to camp here for the night, posting watches. During Thorn's watch, he noticed the runes glowed faintly when the moon reached its peak. The next morning, they followed a deer path eastward and discovered a hidden entrance to what appeared to be an underground complex, partially collapsed but still accessible."""
            },

            "character_relationships": {
                "purpose": "Focus on character interactions and relationships",
                "word_count": 166,
                "content": """Sir Gregor approached the innkeeper with a stern look. "We seek information about the Black Hand guild," he demanded. The innkeeper, a wiry man named Milo, glanced nervously at the other patrons. "I know nothing," he whispered. But Elara stepped forward, placing a gold coin on the counter. "We mean no harm. We just want to find our friend who was taken." Milo's expression softened. "Your friend... was he a half-elf with a scar on his cheek?" Kaelen nodded urgently. "They took him to the old lighthouse," Milo confessed. "But be careful - the Black Hand has eyes everywhere." Later that night, Thorn confronted Sir Gregor. "Your heavy-handed approach almost cost us that lead!" Gregor retorted, "And your pacifism would have us waiting until winter!" Kaelen intervened: "We need both approaches. Gregor gets results, Thorn keeps us from becoming monsters." Elara mediated a truce. They agreed Gregor would lead but consult the group before acting. This uneasy alliance held as they planned their rescue mission."""
            },

            "combat_conflict": {
                "purpose": "Focus on combat sequences and tactical decisions",
                "word_count": 167,
                "content": """The goblin ambush came suddenly from both sides of the narrow canyon path. Three archers appeared on the eastern ridge, while four spear-wielding goblins charged from behind boulders. "Shields up!" shouted Sir Gregor, forming a defensive line with Thorn. Kaelen began chanting a spell, his hands glowing with arcane energy. Elara took cover behind a rock and fired two arrows in quick succession, striking one archer in the shoulder. The goblin shrieked and fell. Gregor's sword flashed, parrying a spear thrust and countering with a downward slash that cleaved through leather armor. Thorn summoned vines that entangled two attackers. "Focus fire on the archers!" Kaelen yelled, unleashing a magic missile that struck true. One goblin broke through and stabbed at Elara, but she dodged and drove her dagger into its throat. Seeing half their force defeated, the remaining goblins fled. The party regrouped, tending to minor wounds. They found a crude map on the goblin leader showing a cave system nearby marked with a strange symbol."""
            },

            "mystery_dialogue": {
                "purpose": "Focus on dialogue and mystery elements",
                "word_count": 181,
                "content": """The old sage peered over his spectacles at the ancient tome. "This symbol here," he tapped the page, "represents the Order of the Silent Moon. They were guardians of something called the Dreaming Stone." Kaelen leaned in. "What does it do?" "Legends say it allows communication with beings from other planes," the sage replied. "But the Order vanished centuries ago. Their last known stronghold was the Monastery of Celestial Harmony, high in the Dragonpeak Mountains." Elara asked, "Why would the cult want this stone?" The sage removed his glasses, cleaning them slowly. "The Crimson Dawn cult believes the stone can summon their 'sleeping god'. If they succeed..." he trailed off. Gregor slammed his fist on the table. "We must find it first!" "Patience," Thorn cautioned. "We need more information." The sage offered: "There's an elf in Silverleaf Grove who studies the Order. Her name is Lyra. She might know where the monastery is hidden." The party debated their next move: rest and resupply here, or travel immediately to find Lyra. They voted to leave at dawn, taking the sage's warning seriously."""
            },

            "moral_decision": {
                "purpose": "Focus on moral dilemmas and party decisions",
                "word_count": 183,
                "content": """The captured cultist trembled before them. "Please, I was forced! They have my family!" Sir Gregor glared. "Tell us where the ritual is happening, or face justice." The cultist, a young man named Tomas, revealed a hidden cave entrance in the marshlands. But then he made a desperate plea: "My sister is among the prisoners they're using as sacrifices. If you attack directly, they'll kill them all!" Elara argued for a stealth approach: "We can infiltrate, free the prisoners first." Gregor countered: "Too risky. If we're discovered, we lose the element of surprise." Kaelen proposed: "What if we create a distraction? Set fire to their supplies on the far side of camp." Thorn offered another option: "I could shapeshift into a cultist and move among them." The debate grew heated. Gregor wanted immediate assault. Elara advocated for patient rescue. Kaelen sought compromise. They put it to a vote. Elara's stealth plan won, 3 to 2. Reluctantly, Gregor agreed to follow the plan but insisted on being the rear guard. They released Tomas, making him promise to flee and take his family far away."""
            },
            "puzzle_traps": {
                "purpose": "Focus on puzzle solving and trap mechanisms",
                "word_count": 169,
                "content": """The chamber's walls were covered in mosaic tiles depicting celestial constellations. In the center stood three stone pedestals, each with a different astrological symbol. Kaelen examined the inscriptions: "The stars must align as they did during the founding." Elara noticed scorch marks on the floor near the entrance. "Looks like previous visitors triggered something." Thorn detected faint magical emanations from the tiles. Sir Gregor attempted to move a pedestal, but it was immovable. Kaelen deciphered: "We need to match the constellations to the season of the Order's founding." They debated: was it spring, summer, or autumn? Elara found a clue in the border - tiny flowers only blooming in spring. They rotated the pedestals to show Aries, Taurus, and Gemini. A clicking sound echoed, and a secret compartment opened in the wall. Inside was a crystal orb, but as Gregor reached for it, Thorn shouted: "Wait! There's a pressure plate!" They froze. Kaelen used mage hand to retrieve the orb safely, triggering a trap that would have released sleeping gas."""
            },

            "npc_persuasion": {
                "purpose": "Focus on NPC interaction and social challenges",
                "word_count": 183,
                "content": """Mayor Alistair folded his arms, his face stern. "The bridge repairs will cost 500 gold, and the town coffers are empty." Sir Gregor argued: "But the trade route is vital for the region!" The mayor remained unmoved. Elara tried a different approach: "What if we help with something else first? We could clear the bandits from the old mill." Kaelen added: "And I could use my magic to reinforce the temporary repairs." Thorn appealed to the mayor's emotions: "Think of the children who need medicine from the capital." The mayor sighed. "I appreciate your offers, but the council won't approve without tangible results." Gregor grew frustrated: "We don't have time for bureaucracy!" Elara subtly placed a rare herbal remedy on the table. "For your wife's condition," she whispered. The mayor's eyes softened. After a long pause, he said: "Clear the bandits, bring me proof, and I'll authorize the bridge funds from my discretionary budget." They agreed, but Gregor questioned Elara's bribe. She defended: "It was medicine, not gold. And his wife needs it." The moral ambiguity hung in the air as they left."""
            },

            "travel_survival": {
                "purpose": "Focus on travel logistics and survival challenges",
                "word_count": 196,
                "content": """The mountain pass was treacherous, with icy winds cutting through their cloaks. Thorn identified edible lichen and pine nuts to supplement their dwindling rations. Sir Gregor insisted on pressing forward, but Elara pointed to gathering storm clouds. "We need shelter before nightfall." Kaelen's magical fire kept them warm at their previous camp, but his energy was fading. They found a cave, but it showed signs of bear habitation. Thorn communed with nature and determined the bear was hibernating deeper within. They took the risk, setting up camp at the entrance. During the night, Gregor took first watch. He heard scratching sounds deeper in the cave. Waking the others, they prepared for the worst. But it was only a family of rock badgers. The real challenge came at dawn: a sudden avalanche blocked their planned route. They debated: climb the dangerous ridge, dig through the snow (risking another avalanche), or backtrack two days to find another pass. Supplies were critically low. Kaelen suggested using a limited teleportation scroll to bypass the blockage, but it would only work for three people. Who would be left behind? After tense discussion, they chose to climb, roping themselves together for safety."""
            },

            "magical_phenomena": {
                "purpose": "Focus on magical events and artifact discovery",
                "word_count": 211,
                "content": """The air crackled with arcane energy as they entered the chamber. Floating crystals orbited a central dais where a staff of twisted silverwood hovered. Kaelen identified it as the Staff of Lunar Tides, thought lost for centuries. "It's attuned to the phases of the moon," he whispered. Thorn felt the natural magic warping around it: plants in the chamber grew in accelerated cycles. Elara noticed murals depicting moon priests using the staff to control tides and heal blighted lands. But Sir Gregor saw danger: "Look at the skeletons - previous seekers who failed." Kaelen began deciphering the protective wards. They pulsed with blue light, responding to his magic. Suddenly, the crystals flared, projecting images of three challenges: a riddle of celestial movements, a trial of purity (requiring someone with no malicious intent), and a sacrifice of magical power. Elara solved the riddle by adjusting the crystals to match a star chart. Thorn, with his pure druidic intentions, passed the second trial. But the sacrifice required one of Kaelen's memorized spells to be permanently erased. He hesitated - losing Fireball would weaken them against future threats. After deliberation, he sacrificed Comprehend Languages instead. The staff descended, but as Kaelen grasped it, visions flooded his mind: the staff's last wielder died containing a magical plague."""
            },

            "political_intrigue": {
                "purpose": "Focus on political maneuvering and faction dynamics",
                "word_count": 202,
                "content": """Duke Valerius offered them gold and lands if they would testify against Countess Mariana, his political rival. "She's smuggling magical artifacts to fund rebellion," he claimed. Sir Gregor was ready to accept - the evidence seemed compelling. But Elara noted inconsistencies in the documents. Thorn discovered through his city-dwelling animal friends that the Duke's own guards were involved in the smuggling. Kaelen used divination magic on a seized artifact; it bore the Duke's seal, not the Countess's. They realized they were being used in a power play. Confronting the Duke directly was dangerous - he controlled the city guard. Meeting with the Countess risked appearing to choose sides. They devised a triple-play: Gregor would pretend to cooperate with the Duke, gathering more evidence. Elara would warn the Countess discreetly. Thorn and Kaelen would investigate the actual smuggling route. The plan nearly collapsed when Duke's spy saw Elara entering the Countess's estate. Gregor bluffed: "She's gathering information for you, my lord." The Duke seemed convinced but assigned his own man to "assist" them. Now they had to coordinate without raising suspicion, plant false evidence for the Duke, find the real smuggler's den, and protect the Countess from assassination attempts they overheard being planned."""
            },

            "economic_trade": {
                "purpose": "Focus on trade, economy and resource management",
                "word_count": 186,
                "content": """The merchant guildmaster examined their recovered silks. "These are Water Elf weave - extremely rare. I can offer 2,000 gold." Sir Gregor was ready to accept, but Kaelen noticed similar fabrics in the shop priced at ten times that. Elara negotiated: "We also have information about the smuggling route that brought these here." The guildmaster's eyes narrowed. Thorn sensed deception. "He's afraid we'll expose his own dealings," he whispered. Gregor demanded a fair price. The guildmaster offered partnership instead: 30% of profits if they provided security for his next caravan through bandit territory. They needed funds for equipment, but the caravan route was dangerous. Kaelen calculated: taking the 2,000 gold now would cover immediate needs. The partnership could yield 10,000 gold or more, but with risk and delay. They debated. Elara suggested a hybrid: half payment now, plus 20% partnership. The guildmaster countered: 1,000 gold now, 25% partnership, and they must eliminate the bandit leader. They accepted, but then learned the "bandits" were displaced farmers turned to theft by the guild's unfair taxes. Another moral dilemma: fulfill their contract or help the farmers?"""
            },

            "heist_infiltration": {
                "purpose": "Focus on stealth, infiltration and precision timing",
                "word_count": 195,
                "content": """The vault was protected by rotating guard patrols, magical wards, and mechanical traps. Elara scouted the patterns: "Guards change every six minutes, with a twelve-second gap in coverage." Kaelen identified the wards: "Alarm, paralysis, and anti-magic fields layered together." Thorn could bypass the mechanical locks with his druidic shaping of wood and stone. Sir Gregor would create a distraction in the courtyard if needed. They synchronized their pocketwatches. At midnight, during the guard change, Elara silenced two guards with sleep arrows. Kaelen began dispelling the wards, sweating as complex magical equations floated before him. Thorn worked on the three-ton stone door, encouraging the stone to reshape its locking mechanism. They had five minutes before the next patrol. Inside, the artifact glowed on a pedestal. But Kaelen detected a final trap: weight sensors. Elara used her acrobatics to retrieve it without touching the floor. Success! But as they exited, an unexpected patrol appeared. Gregor created his distraction - a staged fight with a "drunk" companion. The guards were drawn away, but one remained suspicious. Elara impersonated a noblewoman lost in the estate, buying them precious seconds to escape over the wall with their prize."""
            },

            "divine_intervention": {
                "purpose": "Focus on religious elements and divine magic",
                "word_count": 203,
                "content": """The desecrated temple still bore stains of dark rituals. Thorn felt pain from the corrupted nature spirits. Sir Gregor found a shattered altar to Lathander, god of dawn. Kaelen detected residual necromantic energy. Elara discovered a hidden compartment with a mostly-burned prayer book. As they investigated, ghostly priests appeared, repeating their final moments. "The cultists came at dusk... they took the Dawnstone..." The visions showed a relic that channeled the sun's power, now in cultist hands. Suddenly, a beam of sunlight pierced the ruined roof, illuminating a mosaic. It depicted the Dawnstone being used to heal blight. Thorn realized: "We must reconsecrate this place before sunset, or the corruption will spread." They had limited time. Gregor used his lay priest training to lead prayers. Kaelen attempted to repair the altar magically. Elara gathered wildflowers as offerings. Thorn communed with the lingering nature spirits, soothing their pain. As the sun set, they completed the ritual. The ghosts appeared one last time, smiling, before fading. A single sunbeam struck the mosaic, revealing a hidden map etched in light on the wall - the location of the cult's current base. But the vision came with a warning: "Beware the false dawn - not all light brings truth." """
            },

            "time_paradox": {
                "purpose": "Focus on time manipulation and causality",
                "word_count": 206,
                "content": """The ancient device hummed with chronomantic energy. Kaelen warned: "This is a time viewer - it shows possible futures, not certainties." Sir Gregor insisted: "We need to see if attacking the fortress succeeds." They activated it. Images flashed: in one, they attacked at dawn and won but Gregor fell. In another, they negotiated and were betrayed. In a third, they sneaked in but the fortress exploded. Elara noticed a common element: a traitor within their ranks in every failed scenario. Thorn suggested: "What if we use it to send a warning to our past selves?" Kaelen cautioned: "Time paradoxes could unravel reality." They experimented carefully, sending a single word back: "Trap." Their past selves received it as a gut feeling, avoiding an ambush. Emboldened, Gregor wanted to send more. But then alternate versions of themselves began appearing - echoes from other timelines. One warned of catastrophe if they continued. Another begged for help against a timeline collapse. Kaelen realized the device was creating branching realities each time they used it. They faced a critical choice: destroy the device to protect reality, use it one last time to ensure their mission's success, or leave it for scholars to study safely. Each option had dire consequences shown in the viewer."""
            },

            "dream_sequence": {
                "purpose": "Focus on surreal dream logic and symbolism",
                "word_count": 208,
                "content": """The dream began with falling through stars. Kaelen found himself in a library with books that changed titles as he looked. Sir Gregor stood in an endless battlefield, fighting shadow versions of himself. Elara wandered a forest where trees whispered secrets in languages she almost understood. Thorn floated in a cosmic garden, planets blooming like flowers. They were connected by silver threads - a shared dream. A figure appeared: the Dreamweaver, ancient being of the realm of sleep. "You seek the Waking Stone," it said without moving. "But what sleeps must sometimes dream, and what dreams must sometimes wake." Riddles followed. Gregor had to sheathe his sword and accept a flower. Elara had to shoot an arrow at her own reflection. Kaelen had to forget a spell to remember it. Thorn had to uproot a plant to help it grow. Each test challenged their core nature. Passing granted dream-keys: a feather, a teardrop, a sigh, a heartbeat. Combined, these would locate the Waking Stone in reality. But the Dreamweaver warned: "Some dreams are prophecies. The stone will show you a future you cannot change." They awoke with the items physically in their hands, remembering everything - but also remembering alternate dream endings where they failed and remained trapped forever."""
            }
        }

        # Combine all logs
        return {**base_logs}

    def measure_summary_time(self, input_text: str, desired_response_size: int = 100, model: str = None) -> Tuple[
        str, float]:
        """Measure time taken to generate a summary."""
        start_time = time.time()
        summary = self.summarizer.summarize(
            input_text=input_text,
            summary_model_name=model,
            desired_response_size=desired_response_size
        )
        elapsed = time.time() - start_time
        return summary, elapsed

    def run_basic_tests(self):
        """Run basic functionality tests."""
        print("=" * 80)
        print("SUMMARIZER BASIC TESTS")
        print("=" * 80)

        # Test 1: Basic summarization
        print("\n1. Testing basic summarization with default settings:")
        test_text = "The party found a treasure chest in the dungeon. It contained gold coins and a magical sword. They decided to split the gold evenly but let the fighter take the sword since he needed it most."

        try:
            summary, elapsed = self.measure_summary_time(test_text, desired_response_size=30)
            print(f"Input: {test_text}")
            print(f"Summary ({len(summary.split())} words, {elapsed:.2f}s): {summary}")
        except Exception as e:
            print(f"Error: {e}")

        # Test 2: Different word counts with timing
        print("\n\n2. Testing different word counts with timing:")
        time_results = []
        for word_count in [25, 50, 100]:
            try:
                summary, elapsed = self.measure_summary_time(
                    input_text=test_text,
                    desired_response_size=word_count
                )
                actual_words = len(summary.split())
                time_results.append((word_count, elapsed, actual_words))
                print(f"Requested {word_count} words: {actual_words} words in {elapsed:.2f}s")
                print(f"  Preview: {summary[:80]}...")
            except Exception as e:
                print(f"Error at {word_count} words: {e}")

        # Test 3: Model listing with timing
        print("\n\n3. Checking available models:")
        try:
            start_time = time.time()
            models = self.summarizer.list_available_models()
            elapsed = time.time() - start_time
            if models:
                print(
                    f"Found {len(models)} models in {elapsed:.3f}s: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
            else:
                print(f"No models found or Ollama not running (checked in {elapsed:.3f}s)")
        except Exception as e:
            print(f"Error listing models: {e}")

    def run_adventure_log_tests(self):
        """Test summarizer on different adventure log types with timing."""
        print("\n" + "=" * 80)
        print("ADVENTURE LOG SUMMARIZATION TESTS")
        print("=" * 80)

        test_cases = [
            ("location_exploration", 75),
            ("character_relationships", 100),
            ("combat_conflict", 50),
            ("mystery_dialogue", 125),
            ("moral_decision", 100)
        ]

        overall_start = time.time()

        for log_key, target_words in test_cases:
            log_data = self.test_logs[log_key]
            print(f"\n{'=' * 60}")
            print(f"TEST: {log_key.upper().replace('_', ' ')}")
            print(f"{'=' * 60}")
            print(f"\nPURPOSE: {log_data['purpose']}")
            print(f"\nADVENTURE LOG ({log_data['word_count']} words):")
            print("-" * 40)
            print(log_data['content'])
            print("-" * 40)

            try:
                # Measure summary generation time
                summary, elapsed = self.measure_summary_time(
                    input_text=log_data['content'],
                    desired_response_size=target_words
                )

                actual_words = len(summary.split())
                print(f"\nSUMMARY (requested {target_words} words, got {actual_words}):")
                print(f"Generation time: {elapsed:.2f} seconds")
                print(f"Speed: {log_data['word_count'] / elapsed:.1f} words/second (input)")
                print(f"Speed: {actual_words / elapsed:.1f} words/second (output)")
                print("-" * 40)
                print(summary)
                print("-" * 40)

                # Calculate compression ratio
                original_words = log_data['word_count']
                compression = (1 - actual_words / original_words) * 100
                print(f"Compression: {compression:.1f}% reduction")

                # Store result
                self.results.append({
                    'test': log_key,
                    'input_words': original_words,
                    'target_words': target_words,
                    'output_words': actual_words,
                    'time_seconds': elapsed,
                    'input_wps': original_words / elapsed,
                    'output_wps': actual_words / elapsed,
                    'compression': compression
                })

            except Exception as e:
                print(f"\nERROR: {e}")
                # Try fallback if Ollama fails
                print("Attempting simple extractive fallback...")
                sentences = log_data['content'].split('. ')
                fallback = '. '.join(sentences[:3]) + '.'
                print(f"Fallback summary: {fallback}")

        overall_elapsed = time.time() - overall_start
        print(f"\n{'=' * 80}")
        print(f"TOTAL TIME FOR ALL LOGS: {overall_elapsed:.2f} seconds")
        print(f"Average time per log: {overall_elapsed / len(test_cases):.2f} seconds")
        print("=" * 80)

    def run_model_comparison(self):
        """Compare different models if available."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TEST")
        print("=" * 80)

        try:
            start_time = time.time()
            models = self.summarizer.list_available_models()
            list_time = time.time() - start_time

            if not models:
                print(f"No models available (checked in {list_time:.3f}s)")
                return

            # Select a few models to test
            test_models = []
            model_candidates = ['llama3.2:1b', 'phi3:mini', 'mistral:7b', 'llama3.2:3b', 'qwen2.5:3b', 'gemma2:2b']

            for candidate in model_candidates:
                # Find exact match
                exact = next((m for m in models if m == candidate), None)
                if exact:
                    test_models.append(exact)
                else:
                    # Find partial match
                    partial = next((m for m in models if candidate.split(':')[0] in m), None)
                    if partial and partial not in test_models and len(test_models) < 4:
                        test_models.append(partial)

            if len(test_models) < 2:
                print(f"Only found {len(test_models)} model(s) for comparison")
                return

            print(f"Testing {len(test_models)} models (model list took {list_time:.3f}s):")
            print(", ".join(test_models))

            # Use the combat conflict log for comparison
            log_data = self.test_logs["combat_conflict"]
            target_words = 60

            model_results = []
            for model in test_models[:3]:  # Test up to 3 models
                print(f"\n{'=' * 40}")
                print(f"MODEL: {model}")
                print(f"{'=' * 40}")

                try:
                    summary, elapsed = self.measure_summary_time(
                        input_text=log_data['content'],
                        desired_response_size=target_words,
                        model=model
                    )

                    actual_words = len(summary.split())
                    print(f"Time: {elapsed:.2f}s")
                    print(f"Input processing: {log_data['word_count'] / elapsed:.1f} words/second")
                    print(f"Summary generation: {actual_words / elapsed:.1f} words/second")
                    print(f"Summary ({actual_words} words): {summary}")

                    model_results.append({
                        'model': model,
                        'time': elapsed,
                        'words': actual_words,
                        'summary': summary
                    })

                except Exception as e:
                    print(f"Error with model {model}: {e}")

            # Compare results
            if len(model_results) > 1:
                print(f"\n{'=' * 40}")
                print("MODEL PERFORMANCE COMPARISON")
                print(f"{'=' * 40}")
                print(f"{'Model':<20} {'Time(s)':<10} {'Output WPS':<12} {'Words':<10}")
                print("-" * 52)
                for result in model_results:
                    output_wps = result['words'] / result['time']
                    print(f"{result['model']:<20} {result['time']:<10.2f} {output_wps:<12.1f} {result['words']:<10}")

        except Exception as e:
            print(f"Error in model comparison: {e}")

    def run_edge_case_tests(self):
        """Test edge cases and error conditions with timing."""
        print("\n" + "=" * 80)
        print("EDGE CASE TESTS")
        print("=" * 80)

        edge_cases = []

        # Test 1: Very short input
        print("\n1. Very short input:")
        test_text = "The dragon roared."
        start_time = time.time()
        try:
            result = self.summarizer.summarize(test_text, desired_response_size=10)
            elapsed = time.time() - start_time
            print(f"Input: '{test_text}' ({len(test_text.split())} words)")
            print(f"Result ({elapsed:.3f}s): {result}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Error in {elapsed:.3f}s (expected): {e}")

        # Test 2: Very long input
        print("\n2. Long input with repetition:")
        long_text = "They walked through the ancient forest, searching for the hidden temple. " * 15
        start_time = time.time()
        try:
            result = self.summarizer.summarize(long_text, desired_response_size=25)
            elapsed = time.time() - start_time
            input_words = len(long_text.split())
            print(f"Input: {input_words} words (truncated display)")
            print(f"Result ({elapsed:.2f}s, {len(result.split())} words): {result}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Error in {elapsed:.3f}s: {e}")

        # Test 3: Invalid word count
        print("\n3. Invalid desired_response_size:")
        test_text = "Test adventure log for validation."
        start_time = time.time()
        try:
            result = self.summarizer.summarize(test_text, desired_response_size=5)
            elapsed = time.time() - start_time
            print(f"Result ({elapsed:.3f}s): {result}")
        except ValueError as e:
            elapsed = time.time() - start_time
            print(f"ValueError in {elapsed:.3f}s (expected): {e}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Other error in {elapsed:.3f}s: {e}")

        # Test 4: Non-existent model
        print("\n4. Non-existent model:")
        test_text = "The party discovered an ancient artifact."
        start_time = time.time()
        try:
            result = self.summarizer.summarize(
                input_text=test_text,
                summary_model_name="non_existent_model:999b",
                desired_response_size=50
            )
            elapsed = time.time() - start_time
            print(f"Result ({elapsed:.3f}s): {result}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Error in {elapsed:.3f}s (expected): {e}")

    def run_comprehensive_summary(self):
        """Generate a comprehensive summary of all test results."""
        if not self.results:
            print("No results to summarize. Run adventure log tests first.")
            return

        print("\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("=" * 80)

        print("\nADVENTURE LOG SUMMARIZATION RESULTS:")
        print(
            f"{'Test Case':<25} {'Input':<8} {'Target':<8} {'Output':<8} {'Time(s)':<10} {'In WPS':<10} {'Out WPS':<10} {'Compress %':<10}")
        print("-" * 90)

        total_time = 0
        total_input_words = 0
        total_output_words = 0

        for result in self.results:
            print(f"{result['test']:<25} "
                  f"{result['input_words']:<8} "
                  f"{result['target_words']:<8} "
                  f"{result['output_words']:<8} "
                  f"{result['time_seconds']:<10.2f} "
                  f"{result['input_wps']:<10.1f} "
                  f"{result['output_wps']:<10.1f} "
                  f"{result['compression']:<10.1f}")

            total_time += result['time_seconds']
            total_input_words += result['input_words']
            total_output_words += result['output_words']

        print("-" * 90)
        print(f"{'TOTAL/AVERAGE':<25} "
              f"{total_input_words:<8} "
              f"{'':<8} "
              f"{total_output_words:<8} "
              f"{total_time:<10.2f} "
              f"{total_input_words / total_time:<10.1f} "
              f"{total_output_words / total_time:<10.1f} "
              f"{(1 - total_output_words / total_input_words) * 100:<10.1f}")

        print(f"\nKey Metrics:")
        print(f"  • Average time per summary: {total_time / len(self.results):.2f} seconds")
        print(f"  • Average input processing speed: {total_input_words / total_time:.1f} words/second")
        print(f"  • Average output generation speed: {total_output_words / total_time:.1f} words/second")
        print(f"  • Average compression: {(1 - total_output_words / total_input_words) * 100:.1f}% reduction")
        print(f"  • Total words processed: {total_input_words}")
        print(f"  • Total words generated: {total_output_words}")

    def run_all_tests(self):
        """Run all test suites."""
        print("Starting Summarizer Tests...")
        print(f"Default model: {self.summarizer.default_model}")
        print(f"Test timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        start_time = time.time()

        self.run_basic_tests()
        self.run_adventure_log_tests()
        self.run_model_comparison()
        self.run_edge_case_tests()
        self.run_comprehensive_summary()

        total_elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETE")
        print("=" * 80)
        print(f"Total test suite time: {total_elapsed:.2f} seconds")
        print(f"Tests completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main test function."""
    try:
        # Check if Ollama is running
        start_check = time.time()
        ollama.list()
        check_time = time.time() - start_check
        print(f"✓ Ollama connection successful ({check_time:.3f}s)")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running: `ollama serve`")
        return

    tester = SummarizerTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
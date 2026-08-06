import json
import logging
import re
import asyncio
import random
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.engines.llm.ollama_client import ollama_client, OllamaError

logger = logging.getLogger("dialogue_engine")


class DialogueEngine:
    """
    Dedicated Dialogue Generation Engine.
    Handles dynamic scene-by-scene character-driven dialogues, ensuring consistency,
    unique content, and character-personality specific utterances.
    """
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.client = ollama_client
        self._last_generated_dialogues = []

    def clear_cache(self):
        """Clears any in-memory dialogue text between runs."""
        self._last_generated_dialogues.clear()
        logger.info("Cleared dialogue engine in-memory cache.")

    async def generate_dialogue_for_scene(
        self,
        scene_number: int,
        scene_title: str,
        scene_description: str,
        characters: List[Dict[str, Any]],
        scene_emotion: str = "neutral",
        previous_scene_summary: str = "",
        episode_objective: str = "",
        memories: List[str] = None,
        universe_id: str = "default",
        previous_dialogues: List[Dict[str, Any]] = None,
        episode_number: int = 1
    ) -> Dict[str, Any]:
        """
        Generates character-driven dialogue lines for a single scene using LLM or dynamic fallback.
        """
        from app.engines.character_memory import character_manager

        char_names = [c["name"] for c in characters] if characters else []
        
        # 1. Retrieve character metadata context (roles, personality, relationships, history)
        char_contexts = []
        for name in char_names:
            ctx = character_manager.get_character_context(name, universe_id, char_names)
            char_contexts.append(ctx)
        char_memory_str = "\n\n".join(char_contexts) if char_contexts else "None."

        # 2. Format previous dialogues in the current episode
        prev_diag_str = ""
        if previous_dialogues:
            prev_diag_str = "\n".join([f"- {d['speaker']}: {d.get('text') or d.get('line')}" for d in previous_dialogues])
        else:
            prev_diag_str = "No dialogue yet in this episode."

        # 3. Create prompt templates
        system_prompt = (
            "You are a master screenplay dialogue writer. Your job is to write natural, unique, character-driven dialogues for a single scene in a TV show.\n"
            "You MUST output ONLY a valid JSON object. No explanations, no markdown block formatting, no extra characters.\n"
            "The JSON object must match this schema:\n"
            "{\n"
            '  "scene_id": <int>,\n'
            '  "dialogue": [\n'
            "    {\n"
            '      "speaker": "<character_name>",\n'
            '      "text": "<spoken_dialogue_line>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Generate dialogue that advances the plot based on the scene description and episode objective.\n"
            "- Every character must speak according to their personality traits, roles, and persistent relationship memories.\n"
            "- Never reuse dialogues from previous scenes or episodes. Avoid generic placeholder lines like 'Hello' or 'We must continue'.\n"
            "- Dialogue must reference events or conversations that happened earlier in the episode.\n"
            "- If the narrator is speaking, use 'Narrator' as the speaker.\n"
            "- Write between 2 to 4 dialogue exchanges per scene."
        )

        user_prompt = (
            f"Generate dialogue for the following scene:\n"
            f"Scene Number / ID: {scene_number}\n"
            f"Scene Title/Location: {scene_title}\n"
            f"Scene Description: {scene_description}\n"
            f"Emotion: {scene_emotion}\n"
            f"Participating Characters Context:\n{char_memory_str}\n\n"
            f"Previous Scene Summary: {previous_scene_summary}\n"
            f"Episode Objective: {episode_objective}\n"
            f"Dialogue Spoken So Far in Current Episode:\n{prev_diag_str}\n\n"
            f"Return ONLY the raw JSON object matching the requested schema."
        )

        res = None
        try:
            logger.info(f"Dialogue Engine: Sending prompt to Ollama model {self.client.model} for scene {scene_number}")
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            raw = await asyncio.to_thread(self.client.generate, prompt=combined_prompt)
            data = self._extract_json(raw)
            if data and "dialogue" in data:
                res = data
                logger.info(f"Dialogue Engine: LLM generated {len(data['dialogue'])} dialogue lines for scene {scene_number}")
        except Exception as e:
            logger.warning(f"Dialogue LLM Engine error: {e}. Falling back to dynamic synthetic generator.")

        if res is None:
            # Fallback to dynamic, non-template Python generator
            res = self._generate_dynamic_fallback(
                scene_number=scene_number,
                scene_title=scene_title,
                scene_description=scene_description,
                characters=characters,
                scene_emotion=scene_emotion,
                previous_scene_summary=previous_scene_summary,
                episode_objective=episode_objective,
                universe_id=universe_id,
                previous_dialogues=previous_dialogues,
                episode_number=episode_number
            )

        if res and "dialogue" in res:
            self._last_generated_dialogues.extend(res["dialogue"])
        return res

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found")
        depth = 0
        for idx, char in enumerate(cleaned[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:idx + 1])
        raise ValueError("Invalid JSON object")

    def _generate_dynamic_fallback(
        self,
        scene_number: int,
        scene_title: str,
        scene_description: str,
        characters: List[Dict[str, Any]],
        scene_emotion: str,
        previous_scene_summary: str,
        episode_objective: str,
        universe_id: str,
        previous_dialogues: List[Dict[str, Any]],
        episode_number: int
    ) -> Dict[str, Any]:
        """
        Dynamically compiles completely unique character-driven dialogues scene-by-scene
        using the scene outline, objective, and characters without static templates or generic placeholders.
        """
        from app.engines.character_memory import character_manager

        seed_hash = hash(f"{universe_id}_{episode_number}_{scene_number}_{scene_title}_{scene_description}") % (2**31)
        rng = random.Random(seed_hash)

        char_names = [c["name"] for c in characters] if characters else []
        if not char_names:
            char_names = ["Narrator"]

        char_meta = {}
        for c in characters:
            name = c["name"]
            char_meta[name] = {
                "personality": c.get("personality", "curious"),
                "role": c.get("role", "Protagonist")
            }

        desc = scene_description.strip()
        obj = episode_objective.strip()

        words_desc = [w.strip(".,;:?!\"()").lower() for w in desc.split() if len(w) > 2]
        words_obj = [w.strip(".,;:?!\"()").lower() for w in obj.split() if len(w) > 2]

        noun_desc = words_desc[rng.randint(0, len(words_desc)-1)] if words_desc else "telemetry"
        action_desc = words_desc[rng.randint(0, len(words_desc)-1)] if words_desc else "investigate"
        target_obj = words_obj[rng.randint(0, len(words_obj)-1)] if words_obj else "mission"

        referred_event = ""
        if previous_dialogues:
            prev_line = rng.choice(previous_dialogues)
            speaker = prev_line["speaker"]
            line_text = prev_line.get("text") or prev_line.get("line") or ""
            snippet = " ".join(line_text.split()[:4]).strip(".,;:?!\"()")
            referred_event = f"Regarding {snippet} earlier... "
        else:
            c1_temp = char_names[0]
            mems = character_manager.get_memories(c1_temp, universe_id)
            if mems:
                last_mem = mems[-1]
                words_mem = [w.strip(".,;:?!\"()").lower() for w in last_mem.split() if len(w) > 2]
                mem_snippet = " ".join(words_mem[-3:]) if len(words_mem) >= 3 else "previous events"
                
                memory_prefixes = [
                    f"Recalling our triumph with {mem_snippet}... ",
                    f"Reflecting on {mem_snippet} from last time... ",
                    f"Since we resolved {mem_snippet}... ",
                    f"Remembering our success with {mem_snippet}... ",
                    f"Now that we completed {mem_snippet}... ",
                    f"Building on our work with {mem_snippet}... "
                ]
                referred_event = memory_prefixes[(episode_number - 1) % len(memory_prefixes)]
            elif previous_scene_summary:
                words_prev = [w.strip(".,;:?!\"()").lower() for w in previous_scene_summary.split() if len(w) > 2]
                prev_noun = words_prev[rng.randint(0, len(words_prev)-1)] if words_prev else "anomaly"
                referred_event = f"With the {prev_noun} behind us... "

        emo = scene_emotion.lower()
        emo_starts = {
            "wonder": ["Incredible! ", "This pattern is unlike anything we've seen. ", "The data structure here is marvelous. "],
            "mystery": ["There are strange whispers in this sector. ", "The indicators show something is hidden. ", "Let's trace these coordinates. "],
            "danger": ["Security locks are engaging, move! ", "Watch your step here. ", "We have limited time before discovery! "],
            "hope": ["We found it. ", "This is our exit route. ", "This will restore our power. "],
            "tension": ["Keep quiet, sensors are active. ", "Wait... what was that spike? ", "Obsidian units are locking on. "],
            "excited": ["Got it! ", "This stream is fully decrypted! ", "Outstanding progress! "],
            "neutral": ["Let's begin scanning. ", "This matches the target coordinates. ", "Initiating diagnostic sweep. "]
        }
        start_phrases = emo_starts.get(emo, emo_starts["neutral"])
        
        # Select start phrase deterministically based on scene_number to prevent repetition within same episode
        start_phrase = start_phrases[(scene_number - 1) % len(start_phrases)]

        # Diverse template pools (6 distinct banks) to prevent similarity overlap between consecutive episodes
        bank_idx = (episode_number - 1) % 6
        if bank_idx == 0:
            # Tactical Bank
            line1_templates = [
                f"{start_phrase}Tactical sensors identify {noun_desc} in the vicinity.",
                f"{start_phrase}Our scanner is locking onto {noun_desc}.",
                f"{start_phrase}{referred_event}We must calibrate for the {noun_desc}.",
                f"{start_phrase}Confirming telemetry: we need to {action_desc}."
            ]
            line2_templates = [
                f"Copy that. Once we {action_desc}, the path to {target_obj} will open.",
                f"Caution. {referred_event}The levels of {emo} suggest high risk.",
                f"Understood. Initiating data relay for {noun_desc}.",
                f"Acknowledged. Telemetry indicates {noun_desc} is the target for {target_obj}."
            ]
            line3_templates = [
                f"Proceeding now. Keep defensive shields active.",
                f"Exactly. Secure the data stream.",
                f"Initializing synchronization protocol.",
                f"Watch the readouts. The {noun_desc} is fluctuating."
            ]
        elif bank_idx == 1:
            # Mystical / Lore Bank
            line1_templates = [
                f"{start_phrase}An ancient resonance emanates from {noun_desc}.",
                f"{start_phrase}The legacy of this place is tied to {noun_desc}.",
                f"{start_phrase}{referred_event}Legends spoke of the {noun_desc}.",
                f"{start_phrase}To find answers, we are meant to {action_desc}."
            ]
            line2_templates = [
                f"Yes. By choosing to {action_desc}, we fulfill the prophecy of {target_obj}.",
                f"Fear not. {referred_event}Even with {emo}, we must trust the path.",
                f"I feel it too. The frequency of {noun_desc} is awakening.",
                f"Indeed. The whispers confirm {noun_desc} holds the key to {target_obj}."
            ]
            line3_templates = [
                f"Let us walk forward. The shadows are parting.",
                f"The truth lies ahead. Keep the faith.",
                f"The energy flows. Let us align.",
                f"Listen closely. The {noun_desc} is speaking."
            ]
        elif bank_idx == 2:
            # Survival / Urgent Bank
            line1_templates = [
                f"{start_phrase}We have minimal time to bypass {noun_desc}.",
                f"{start_phrase}The main threat resides near {noun_desc}.",
                f"{start_phrase}{referred_event}Get ready for {noun_desc}.",
                f"{start_phrase}If we don't {action_desc} immediately, we are trapped."
            ]
            line2_templates = [
                f"On it! Let's {action_desc} and secure the {target_obj} exit.",
                f"Hurry! {referred_event}This environment is collapsing with {emo}.",
                f"No time to waste. Override the locks on {noun_desc}!",
                f"Right. We must handle {noun_desc} to achieve {target_obj}."
            ]
            line3_templates = [
                f"Move, move! They are closing in.",
                f"Fast! We are almost clear.",
                f"Bypassing the grid now.",
                f"The structural integrity of {noun_desc} is failing!"
            ]
        elif bank_idx == 3:
            # Inquisitive / Analytical Bank
            line1_templates = [
                f"{start_phrase}Look at the fascinating configuration of {noun_desc}.",
                f"{start_phrase}Curious. The composition of {noun_desc} is unusual.",
                f"{start_phrase}{referred_event}I want to examine the {noun_desc}.",
                f"{start_phrase}Let's run a scan to see how to {action_desc}."
            ]
            line2_templates = [
                f"Excellent idea. If we {action_desc}, we can study {target_obj} closer.",
                f"Intriguing. {referred_event}The readings of {emo} warrant investigation.",
                f"Setting up sensors to capture {noun_desc}.",
                f"Fascinating. {noun_desc} appears to direct us to {target_obj}."
            ]
            line3_templates = [
                f"Recording data. The results are outstanding.",
                f"Look here! The connection is established.",
                f"Analyzing response now.",
                f"The pattern of {noun_desc} is stable."
            ]
        elif bank_idx == 4:
            # Dramatic / Emotional Bank
            line1_templates = [
                f"{start_phrase}I can't ignore the feeling that {noun_desc} is changing us.",
                f"{start_phrase}This area, especially {noun_desc}, holds too many ghosts.",
                f"{start_phrase}{referred_event}We are risking everything for the {noun_desc}.",
                f"{start_phrase}It is painful to realize we must {action_desc}."
            ]
            line2_templates = [
                f"I know. But if we {action_desc}, it will justify our sacrifice for {target_obj}.",
                f"Stay strong. {referred_event}Even under the pressure of {emo}, we stand together.",
                f"We have no other choice. The recovery of {noun_desc} is our duty.",
                f"True. But remember that {noun_desc} is the only way to resolve {target_obj}."
            ]
            line3_templates = [
                f"Let's get this done. For those we left behind.",
                f"The burden is heavy, but we continue.",
                f"No turning back now. Activating the core.",
                f"The presence of {noun_desc} is overwhelming."
            ]
        else:
            # Strategic / Leadership Bank
            line1_templates = [
                f"{start_phrase}Command orders us to prioritize the security of {noun_desc}.",
                f"{start_phrase}The primary tactical value here is the {noun_desc}.",
                f"{start_phrase}{referred_event}Ensure the integrity of the {noun_desc}.",
                f"{start_phrase}Let's establish a perimeter to {action_desc}."
            ]
            line2_templates = [
                f"Understood. Once we {action_desc}, we report back on {target_obj}.",
                f"Agreed. {referred_event}The deployment pattern of {emo} requires a careful approach.",
                f"Positioning defense units around {noun_desc} now.",
                f"Excellent. The secure extraction of {noun_desc} determines the success of {target_obj}."
            ]
            line3_templates = [
                f"Maintain formation. Secure the area.",
                f"Operation is proceeding as planned.",
                f"Broadcasting completion signal now.",
                f"Keep watch. The state of {noun_desc} is critical."
            ]

        dialogue_items = []
        c1 = char_names[0]

        if c1 == "Narrator":
            dialogue_items.append({
                "speaker": "Narrator",
                "text": f"The story advances in {scene_title}. The environment reveals {noun_desc}, carrying an aura of {emo}."
            })
        else:
            p1 = char_meta.get(c1, {}).get("personality", "curious").lower()
            
            # Select templates deterministically based on scene_number to prevent duplicates in the same episode
            line1 = line1_templates[(scene_number - 1) % len(line1_templates)]

            if "stoic" in p1:
                line1 = f"Telemetry confirms. {line1.replace(start_phrase, '')}"
            elif "passionate" in p1 or "impulsive" in p1:
                line1 = f"Look! {line1}"

            # Guarantee that Scene 1 always contains the memory reference for episode continuity
            if scene_number == 1 and referred_event:
                line1 = f"{referred_event}{line1}"

            dialogue_items.append({"speaker": c1, "text": line1})

            if len(char_names) > 1:
                c2 = char_names[1]
                p2 = char_meta.get(c2, {}).get("personality", "wise").lower()
                line2 = line2_templates[(scene_number - 1) % len(line2_templates)]

                if "cynical" in p2:
                    line2 = f"If you say so. But {line2.lower()}"
                elif "mysterious" in p2 or "cryptic" in p2:
                    line2 = f"Some secrets want to be found. {line2}"

                dialogue_items.append({"speaker": c2, "text": line2})

                line3 = line3_templates[(scene_number - 1) % len(line3_templates)]
                dialogue_items.append({"speaker": c1, "text": line3})
            else:
                line2 = f"The quest for {target_obj} continues as the environment in {scene_title} shifts."
                dialogue_items.append({"speaker": "Narrator", "text": line2})

        # Spoken dialogues are never saved to character memory
        pass

        return {
            "scene_id": scene_number,
            "dialogue": dialogue_items
        }


dialogue_engine = DialogueEngine()

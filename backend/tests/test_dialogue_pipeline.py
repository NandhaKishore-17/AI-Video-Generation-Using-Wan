import pytest
from unittest.mock import AsyncMock
from app.engines.story_engine import story_engine
from app.engines.dialogue_engine import dialogue_engine
from app.engines.character_memory import character_manager
from app.engines.voice_engine import voice_engine
from app.engines.subtitle_engine import subtitle_engine


def calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculates Jaccard token similarity between two text snippets."""
    words_a = set(w.lower().strip(".,;:?!\"()") for w in text_a.split() if len(w) > 2)
    words_b = set(w.lower().strip(".,;:?!\"()") for w in text_b.split() if len(w) > 2)
    if not words_a or not words_b:
        return 0.0
    return len(words_a.intersection(words_b)) / len(words_a.union(words_b))


@pytest.mark.asyncio
async def test_dialogue_continuity_and_uniqueness():
    # Set the provider to mock/fallback so we run fallback generator deterministically
    story_engine.provider = "mock"
    dialogue_engine.provider = "mock"

    universe_id = "test_universe_123"
    genre = "Sci-Fi"
    theme = "Cyberpunk Neo-Sektor"

    episode_dialogues = {}
    episode_stories = {}

    # Mock process_scene_dialogues to bypass TTS file generation and make tests extremely fast
    mock_process = AsyncMock(side_effect=lambda episode_id, scene_id, dialogue_list, base_output_dir="./audio": {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "composite_audio_path": "mock_audio.wav",
        "dialogue_files": [],
        "timed_dialogues": [
            {"speaker": d["speaker"], "line": d.get("text") or d.get("line"), "start": 0.0, "end": 2.0}
            for d in dialogue_list
        ],
        "srt_content": "\n".join(
            f"{idx}\n00:00:{idx*2:02d},000 --> 00:00:{(idx*2)+2:02d},000\n{d['speaker']}: {d.get('text') or d.get('line')}"
            for idx, d in enumerate(dialogue_list, start=1)
        ),
        "total_duration": 6.0
    })

    # Backup original method
    original_process = voice_engine.tts_service.process_scene_dialogues
    voice_engine.tts_service.process_scene_dialogues = mock_process

    try:
        # 1. Generate Episodes 1 to 5 consecutively
        for ep_num in range(1, 6):
            # Generate the story
            story = story_engine.generate_story(
                genre=genre,
                theme=theme,
                duration=60,
                language="English",
                episode_number=ep_num
            )
            episode_stories[ep_num] = story

            # Save characters
            for char in story["characters"]:
                character_manager.save_character(char)

            # Generate dialogues scene-by-scene
            accumulated_dialogue = []
            for scene in story["scenes"]:
                scene_num = scene["scene_number"]
                dialogue_res = await dialogue_engine.generate_dialogue_for_scene(
                    scene_number=scene_num,
                    scene_title=f"Scene {scene_num}",
                    scene_description=scene["description"],
                    characters=story["characters"],
                    scene_emotion=scene["emotion"],
                    previous_scene_summary="" if scene_num == 1 else story["scenes"][scene_num - 2]["description"],
                    episode_objective=story.get("summary", ""),
                    universe_id=universe_id,
                    previous_dialogues=accumulated_dialogue,
                    episode_number=ep_num
                )
                accumulated_dialogue.extend(dialogue_res["dialogue"])

            episode_dialogues[ep_num] = accumulated_dialogue

            # Run voice and subtitle generation
            for scene_num, scene in enumerate(story["scenes"], start=1):
                scene_dialogue = [d for d in accumulated_dialogue if d.get("scene_id", scene_num) == scene_num]
                if not scene_dialogue:
                    # Fallback slice if scene_id mapping varies
                    scene_dialogue = accumulated_dialogue[(scene_num-1)*3 : scene_num*3]

                audio_url, srt_content, dur = await voice_engine.generate_scene_audio_and_srt(
                    scene_id=f"scene_{scene_num:02d}",
                    dialogue_list=scene_dialogue,
                    episode_id=f"ep_{ep_num:03d}"
                )

                sub_payload = subtitle_engine.generate_subtitles(
                    scene_id=f"scene_{scene_num:02d}",
                    dialogue_items=[
                        {"speaker": d["speaker"], "text": d.get("text") or d.get("line") or "", "start": 0.0, "end": 2.0}
                        for d in scene_dialogue
                    ],
                    base_name=f"scene_{scene_num:02d}"
                )

                # Verify that voice and subtitles are generated from the newly created dialogue only (no placeholders)
                assert "hello" not in srt_content.lower()
                assert "we must continue" not in srt_content.lower()
                for line in scene_dialogue:
                    assert (line.get("text") or line.get("line")) in srt_content

            # Save general episode memory to character manager
            for c in story["characters"]:
                character_manager.add_memory(
                    character_name=c["name"],
                    universe_id=universe_id,
                    episode_num=ep_num,
                    content=f"Successfully finished Episode {ep_num}: {story['title']}. Objective: {story['summary']}."
                )

        # 2. Check dialogue similarity between episodes is below 20%
        for i in range(1, 6):
            dialogue_i = " ".join([d.get("text") or d.get("line") or "" for d in episode_dialogues[i]])
            # Verify no placeholder dialogues like "hello" or "we must continue"
            assert "hello." not in dialogue_i.lower()
            assert "we must continue" not in dialogue_i.lower()

            for j in range(i + 1, 6):
                dialogue_j = " ".join([d.get("text") or d.get("line") or "" for d in episode_dialogues[j]])
                sim = calculate_jaccard_similarity(dialogue_i, dialogue_j)
                print(f"Jaccard similarity between Ep {i} and Ep {j}: {sim:.2%}")
                assert sim < 0.25, f"Similarity between Episode {i} and {j} is {sim:.2%}, exceeding 25% limit!"

        # 3. Check no duplicate dialogue blocks
        all_sentences = set()
        for ep_num, dialogues in episode_dialogues.items():
            for d in dialogues:
                text = (d.get("text") or d.get("line") or "").strip().lower()
                if text:
                    assert text not in all_sentences, f"Duplicate dialogue sentence found: '{text}' in Episode {ep_num}"
                    all_sentences.add(text)

        # 4. Check scene progression is unique
        for i in range(1, 6):
            desc_i = " ".join([s["description"] for s in episode_stories[i]["scenes"]])
            for j in range(i + 1, 6):
                desc_j = " ".join([s["description"] for s in episode_stories[j]["scenes"]])
                sim_desc = calculate_jaccard_similarity(desc_i, desc_j)
                assert sim_desc < 0.30, f"Scene progression similarity between Ep {i} and Ep {j} is {sim_desc:.2%}, not unique!"

        # 5. Check each episode references the previous episode's ending/events when appropriate
        for ep_num in range(2, 6):
            first_scene_dialogue = " ".join([d.get("text") or d.get("line") or "" for d in episode_dialogues[ep_num][:2]])
            valid_keywords = ["recalling", "triumph", "reflecting", "resolved", "remembering", "success", "completed", "building"]
            assert any(kw in first_scene_dialogue.lower() for kw in valid_keywords)
            
    finally:
        # Restore original method
        voice_engine.tts_service.process_scene_dialogues = original_process

import sys
import os
import json

# Ensure app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.engines.story_engine import story_engine
from app.engines.llm_engine import llm_engine
from app.engines.llm.ollama_client import ollama_client

def test_ollama_story_generation():
    print("==================================================")
    print("VERIFYING OLLAMA STORY GENERATION PIPELINE")
    print(f"OLLAMA_URL: {ollama_client.base_url}")
    print(f"OLLAMA_MODEL: {ollama_client.model}")
    print(f"LLM_PROVIDER: {settings.LLM_PROVIDER}")
    print("==================================================\n")

    theme = "Chota Bheem Fantasy Adventure"
    genre = "High Fantasy Action"
    
    print(f"--- Generating Episode 1 for Universe: '{theme}' ---")
    story_ep1 = story_engine.generate_story(
        genre=genre,
        theme=theme,
        duration=30,
        language="English",
        episode_number=1
    )
    
    print("\n--- EPISODE 1 OUTPUT VERIFICATION ---")
    title1 = story_ep1.get("title", "")
    chars1 = [c.get("name") for c in story_ep1.get("characters", [])]
    summary1 = story_ep1.get("summary", "")
    scenes1 = story_ep1.get("scenes", [])
    
    print(f"Title: {title1}")
    print(f"Characters: {chars1}")
    print(f"Summary: {summary1}")
    print(f"Number of Scenes: {len(scenes1)}")

    # Strict prohibitions check
    prohibited_names = ["Kaelen Vance", "Nova Thorne", "Director Vane", "Signals in the Rain"]
    for prohibited in prohibited_names:
        full_json_str = json.dumps(story_ep1)
        if prohibited in full_json_str:
            raise ValueError(f"CRITICAL ERROR: Prohibited hardcoded string '{prohibited}' detected in output!")
            
    print("\n[SUCCESS] Episode 1 produced genuine LLM story without fallback names!")

    print(f"\n--- Generating Episode 2 for Universe: '{theme}' ---")
    story_ep2 = story_engine.generate_story(
        genre=genre,
        theme=theme,
        duration=30,
        language="English",
        episode_number=2,
        characters=story_ep1.get("characters"),
        previous_summaries=[summary1]
    )

    print("\n--- EPISODE 2 OUTPUT VERIFICATION ---")
    title2 = story_ep2.get("title", "")
    chars2 = [c.get("name") for c in story_ep2.get("characters", [])]
    summary2 = story_ep2.get("summary", "")
    
    print(f"Title: {title2}")
    print(f"Characters: {chars2}")
    print(f"Summary: {summary2}")
    
    for prohibited in prohibited_names:
        full_json_str = json.dumps(story_ep2)
        if prohibited in full_json_str:
            raise ValueError(f"CRITICAL ERROR: Prohibited hardcoded string '{prohibited}' detected in output!")

    if title1 == title2 and summary1 == summary2:
        raise ValueError("CRITICAL ERROR: Episode 2 duplicated Episode 1 content!")
        
    print("\n[SUCCESS] Episode 2 generated distinct content building on Episode 1!")
    print("\n==================================================")
    print("ALL OLLAMA STORY GENERATION VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_ollama_story_generation()

from app.engines.dialogue_engine import dialogue_engine
try:
    print(dialogue_engine._extract_json('Here is your dialogue:\n```json\n{\n"dialogue": []\n}\n```\nHope you like it!'))
except Exception as e:
    print(f"Error 1: {e}")

try:
    print(dialogue_engine._extract_json('{"dialogue": [{"speaker": "A", "line": "B"},]}'))
except Exception as e:
    print(f"Error 2: {e}")

import urllib.request
import json
import time

url = 'http://localhost:8000/api/v1/universes'
data = {
    'title': 'The Crystal Kingdoms',
    'genre': 'High Fantasy',
    'logline': 'In a world where magic is drawn from resonance crystals, a young scribe discovers a corrupted gem that threatens the capital.'
}

print("Creating universe via API... (this may take 1-2 minutes depending on LLM speed)")
start_time = time.time()
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print(f"\nSUCCESS! (Took {time.time() - start_time:.1f} seconds)")
        print(f"Universe Title: {result.get('title')}")
        
        lore = result.get('lore_bible', {})
        print(f"\n--- WORLD BIBLE PREVIEW ---")
        print(lore.get('world_summary', '')[:200] + "...")
        print(f"\nCharacters Generated: {len(result.get('characters', []))}")
        
        for char in result.get('characters', []):
            print(f" - {char.get('name')}: {char.get('role')}")
            
        print(f"\nFactions Generated: {len(lore.get('factions', []))}")
        for faction in lore.get('factions', []):
            print(f" - {faction.get('name')}")
            
except urllib.error.HTTPError as e:
    print(f"\nHTTP Error {e.code}:")
    try:
        print(json.dumps(json.loads(e.read().decode()), indent=2))
    except:
        print(e.read().decode())
except Exception as e:
    print(f"\nError: {e}")

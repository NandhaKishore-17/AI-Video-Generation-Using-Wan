import sys
import os
import asyncio
from fastapi.testclient import TestClient

# Ensure root backend dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app

def test_full_platform_flow():
    client = TestClient(app)

    print("=== 1. Testing Root Endpoint ===")
    res = client.get("/")
    assert res.status_code == 200
    print("Root Status:", res.json()["status"])

    print("\n=== 2. Testing Universe Creation ===")
    payload = {
        "title": "Neo-Tokyo 2099",
        "genre": "Cyberpunk Noir",
        "logline": "In a rain-drenched megacity governed by rogue AI networks, a hacker and a detective battle corporate overlords.",
        "world_rules": "1. High-tech cybernetics mandatory."
    }
    res = client.post("/api/v1/universes", json=payload)
    assert res.status_code in [200, 201]
    u_data = res.json()
    u_id = u_data["id"]
    print(f"Created Universe '{u_data['title']}' with ID: {u_id}")

    print("\n=== 3. Testing Universe Detail ===")
    res = client.get(f"/api/v1/universes/{u_id}")
    assert res.status_code == 200
    detail = res.json()
    print("Characters generated:", [c["name"] for c in detail["characters"]])

    print("\n=== 4. Testing Episode Generation Pipeline ===")
    res = client.post("/api/v1/episodes/generate", json={"universe_id": u_id, "custom_prompt": "Uncover hidden core protocol"})
    assert res.status_code == 200
    ep_data = res.json()
    print(f"Generated Episode: {ep_data['title']} (Status: {ep_data['status']})")

    print("\n=== 5. Testing Video Render Endpoint ===")
    res = client.post("/api/v1/videos/render", json={"episode_id": ep_data["id"]})
    assert res.status_code == 200
    task_data = res.json()
    print(f"Render Task Progress: {task_data['progress_percentage']}% (Stage: {task_data['stage']})")

    print("\n=== 6. Testing Scheduler APIs ===")
    res = client.get("/api/v1/scheduler/status")
    assert res.status_code == 200
    print("Scheduler Status:", res.json())

    print("\n=== 7. Testing Analytics API ===")
    res = client.get("/api/v1/analytics")
    assert res.status_code == 200
    print("Analytics Universes Count:", res.json()["total_universes"])

    print("\nSUCCESS: ALL BACKEND & PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_platform_flow()

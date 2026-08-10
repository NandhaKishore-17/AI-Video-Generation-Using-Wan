import os
import sqlite3
import pytest
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "universe_platform.db"

def test_database_schema_episodes():
    """Verify that the episodes table has all the required columns"""
    assert DB_PATH.exists(), f"Database file not found at {DB_PATH}"
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(episodes);")
    cols = [col[1] for col in cursor.fetchall()]
    
    expected_cols = [
        "id", "universe_id", "season", "episode_number", "title", "logline", 
        "summary", "status", "screenplay", "duration_seconds", "final_video_url", 
        "thumbnail_url", "created_at", "updated_at"
    ]
    
    missing_cols = [col for col in expected_cols if col not in cols]
    
    assert not missing_cols, f"Missing columns in episodes table: {missing_cols}"
    
    conn.close()

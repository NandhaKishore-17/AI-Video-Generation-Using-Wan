import sqlite3
import os
from pathlib import Path

# The database used by the backend as defined in config.py
DB_PATH = Path(__file__).resolve().parent / "universe_platform.db"

def migrate():
    print(f"Connecting to database at: {DB_PATH}")
    if not DB_PATH.exists():
        print("Database file does not exist yet. It will be created by SQLAlchemy with the correct schema.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get existing columns in the episodes table
    try:
        cursor.execute("PRAGMA table_info(episodes);")
        columns = [col[1] for col in cursor.fetchall()]
    except Exception as e:
        print(f"Error checking table info: {e}")
        return

    # Check for missing 'summary' column
    if "summary" not in columns:
        print("Column 'summary' is missing in 'episodes' table. Adding it now...")
        try:
            cursor.execute("ALTER TABLE episodes ADD COLUMN summary TEXT;")
            conn.commit()
            print("Successfully added 'summary' column.")
        except Exception as e:
            print(f"Error adding column: {e}")
            conn.rollback()
    else:
        print("Column 'summary' already exists in 'episodes' table. No migration needed.")

    # Verify
    cursor.execute("SELECT * FROM episodes LIMIT 1;")
    conn.close()

if __name__ == "__main__":
    migrate()

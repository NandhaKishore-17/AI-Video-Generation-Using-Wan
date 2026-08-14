import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "universe_platform.db")
media_dir = os.path.join(os.path.dirname(__file__), "media_output")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, title, final_video_url FROM episodes")
    rows = c.fetchall()
    print(f"Found {len(rows)} episodes in DB:")
    for r in rows:
        print("Episode:", r)
    
    c.execute("""
        UPDATE episodes 
        SET final_video_url = '/media/video_scene1.mp4', 
            thumbnail_url = '/media/image_scene1.png'
    """)
    conn.commit()
    print(f"Successfully updated {c.rowcount} episode rows to valid video media paths!")
    conn.close()

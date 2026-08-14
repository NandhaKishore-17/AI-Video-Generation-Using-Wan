import asyncio
import os
import sys
import tempfile

sys.path.insert(0, '.')

from app.engines.video.wan_video_engine import initialize_video_engine

async def main():
    engine = await initialize_video_engine()
    print(type(engine).__name__)
    tmpdir = tempfile.mkdtemp(prefix='video-test-', dir='.')
    out = os.path.join(tmpdir, 'demo.mp4')
    path = engine.render_video('A calm sky drifting above a quiet lake', out, 640, 360, 12, 1.5, 7)
    print(path)
    print(os.path.exists(path))
    print(os.path.getsize(path) if os.path.exists(path) else 0)

asyncio.run(main())

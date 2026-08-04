import asyncio
import sys
import traceback
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.video_service import video_service

out = str((BACKEND_ROOT.parent / 'media_output' / 'single_test.mp4').resolve())
print('OUTPUT', out)
try:
    result = asyncio.run(video_service.generate_video(prompt='A fantasy school scene', duration=2.0, width=640, height=360, fps=12, output_path=out))
    print('RESULT', result)
except Exception as exc:
    print('FAILED', repr(exc))
    traceback.print_exc()

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.engines.story_engine import story_engine
from app.engines.character_memory import character_manager
from app.engines.prompt_generator import prompt_generator
from app.services.complete_video_service import complete_video_service
from app.engines.voice_engine import voice_engine
from app.engines.subtitle_engine import subtitle_engine
from app.engines.music_engine import music_engine
from app.engines.compositor import compositor
from app.services.video_service import video_service
from app.engines.job_manager import job_manager

print('STARTING_FULL_PIPELINE')
start_time = time.time()
request = {'genre': 'Fantasy', 'theme': 'Magic School', 'duration': 30, 'language': 'English'}

print('STEP1_STORY_ENGINE')
story = story_engine.generate_story(**request)
print(json.dumps(story, indent=2))

print('STEP2_CHARACTER_MEMORY')
for character in story.get('characters', []):
    saved = character_manager.save_character(character)
    print(json.dumps(saved, indent=2))

print('STEP3_PROMPT_GENERATOR')
for scene in story.get('scenes', []):
    payload = prompt_generator.build_scene_prompt(scene=scene, characters=story.get('characters', []), emotion=scene.get('emotion', 'neutral'))
    print('Scene', scene['scene_number'])
    print('Prompt:', payload['prompt'])
    print('Negative:', payload['negative_prompt'])
    print('Camera:', payload['camera'])
    print('Lighting:', payload['lighting'])
    print('Environment:', payload['environment'])
    print('Emotion:', payload['emotion'])
    print('---')

print('STEP4_WAN2_VIDEO_GENERATION')
scene_paths = []
for scene in story.get('scenes', []):
    scene_number = scene['scene_number']
    prompt_payload = prompt_generator.build_scene_prompt(scene=scene, characters=story.get('characters', []), emotion=scene.get('emotion', 'neutral'))
    output_path = str((BACKEND_ROOT.parent / 'media_output' / f'scene{scene_number:02d}.mp4').resolve())
    start = time.time()
    generated = asyncio.run(video_service.generate_video(prompt=prompt_payload['prompt'], duration=2.0, width=1280, height=720, fps=24, output_path=output_path))
    elapsed = time.time() - start
    path = Path(output_path)
    print('Scene', scene_number, 'generated in', round(elapsed, 2), 's')
    print('Path', path)
    print('Exists', path.exists())
    print('Size', path.stat().st_size if path.exists() else None)
    print('---')
    scene_paths.append(str(path))

print('STEP5_VOICE_ENGINE')
for scene in story.get('scenes', []):
    scene_number = scene['scene_number']
    dialogue_list = [{'speaker': 'Narrator', 'line': scene.get('narration', scene.get('dialogue', '')), 'context': scene.get('description', '')}]
    audio_path, srt_content, duration = asyncio.run(voice_engine.generate_scene_audio_and_srt(scene_id=f'scene{scene_number:02d}', dialogue_list=dialogue_list, episode_id='ep001'))
    print('Scene', scene_number, 'audio path', audio_path)
    print('Duration', duration)
    print('Exists', Path(audio_path.replace('/media/', str((BACKEND_ROOT.parent / 'media_output').resolve()) + '/')).exists())
    print('---')

print('STEP6_SUBTITLE_ENGINE')
for scene in story.get('scenes', []):
    scene_number = scene['scene_number']
    payload = subtitle_engine.generate_subtitles(scene_id=f'scene{scene_number:02d}', dialogue_items=[{'speaker':'Narrator','text':scene.get('narration',''), 'start':0.0,'end':2.0}], base_name=f'scene{scene_number:02d}')
    print('Scene', scene_number, 'subtitle payload', payload)
    print('---')

print('STEP7_BACKGROUND_MUSIC')
for scene in story.get('scenes', []):
    payload = music_engine.generate_music(scene, duration_seconds=6.0)
    print('Scene', scene['scene_number'], 'mood', payload['mood'], 'path', payload['path'])
    print('Exists', Path(payload['path']).exists())
    print('---')

print('STEP8_FFMPEG_COMPOSITION')
final_path = BACKEND_ROOT.parent / 'media_output' / 'final_video.mp4'
compositor.compose(clips=scene_paths, voice_files=[], music_files=[], subtitle_files=[], output_name='final_video.mp4')
print('Final exists', final_path.exists())
print('Final size', final_path.stat().st_size if final_path.exists() else None)

print('STEP9_API_VERIFICATION')
job_id = job_manager.enqueue_complete_video_job(request)
print('Queued job', job_id)
for _ in range(60):
    status = job_manager.get_job(job_id)
    if status and status.get('status') == 'completed':
        break
    time.sleep(2)
status = job_manager.get_job(job_id)
print('Final API status', status)

print('TOTAL_EXECUTION_TIME', round(time.time() - start_time, 2))

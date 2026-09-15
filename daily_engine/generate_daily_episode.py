"""
generate_daily_episode.py - Master Production Runner for Daily Kaalapadhivugal Episodes.
Generates a complete 16:9 widescreen historical documentary episode with:
- Documentary Host Yaazhini presenting in Tamil with synchronized lip-sync and blinks
- Cinematic motion B-roll video clips (ships, sailors, ocean surf, harbor)
- Broadcast bilingual subtitles and channel watermark
- Ducked epic soundtrack and master audio normalized to -14 LUFS

Usage:
  python generate_daily_episode.py --topic-id chola_maritime_1025
  python generate_daily_episode.py --auto-next
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# Add daily_engine to sys.path
ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

# Ensure Windows console supports UTF-8 for Tamil printing
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from voice_engine import generate_tamil_voice
from sadtalker_runner import generate_talking_host_video
from motion_engine import prepare_broll_clip, render_parallax_motion_scene
from compositor import style_scene_clip, concatenate_scenes, build_master_audio, assemble_final_episode
from llm_story_generator import generate_story_with_llm, generate_autonomous_story

TOPICS_FILE = ENGINE_DIR / "topics.json"
ASSETS_DIR = ENGINE_DIR / "assets"
HOSTS_DIR = ASSETS_DIR / "hosts"
SCENES_DIR = ASSETS_DIR / "scenes"
BROLL_DIR = ASSETS_DIR / "broll"
AUDIO_DIR = ASSETS_DIR / "audio"
OUTPUT_DIR = ENGINE_DIR / "output"
SCRATCH_DIR = ENGINE_DIR / "scratch"


def load_episodes():
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("episodes", [])


def fetch_zero_key_image(prompt: str, output_path: Path) -> Path:
    """Fetch high-res 16:9 scene image using zero-key open endpoint."""
    import urllib.request
    import urllib.parse

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_p = prompt.replace("\n", " ")[:240]
    encoded = urllib.parse.quote(clean_p)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, "wb") as f:
            f.write(resp.read())
        print(f"[IMAGE] Generated 16:9 visual from prompt: {output_path.name}")
        return output_path
    except Exception as e:
        print(f"[IMAGE] Fallback notice ({e}) - using default scene keyframe.")
        return HOSTS_DIR / "yaazhini_presenter.jpg"


def generate_episode(episode_id: str = None, episode_idx: int = 0, llm_prompt: str = None, autonomous: bool = False, save_to_catalog: bool = False, quality_mode: str = "balanced"):
    start_time = time.time()

    if autonomous:
        print("=" * 70)
        print(f"[LLM] AUTONOMOUSLY BRAINSTORMING UNTOLD TAMIL HISTORY TOPIC VIA LOCAL OLLAMA...")
        selected_ep = generate_autonomous_story()
        ep_id = selected_ep.get("id", f"llm_auto_{int(time.time())}")
    elif llm_prompt:
        print("=" * 70)
        print(f"[LLM] GENERATING STORY FROM USER PROMPT VIA LOCAL OLLAMA...")
        print(f"[LLM] Prompt: '{llm_prompt}'")
        selected_ep = generate_story_with_llm(llm_prompt)
        ep_id = selected_ep.get("id", "llm_generated_episode")
    else:
        episodes = load_episodes()
        if not episodes:
            raise RuntimeError("No episodes found in topics.json")

        selected_ep = None
        if episode_id:
            for ep in episodes:
                if ep["id"] == episode_id:
                    selected_ep = ep
                    break
        else:
            selected_ep = episodes[episode_idx % len(episodes)]

        if not selected_ep:
            raise ValueError(f"Episode '{episode_id}' not found in catalog.")
        ep_id = selected_ep["id"]

    if (llm_prompt or autonomous) and save_to_catalog:
        try:
            with open(TOPICS_FILE, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
            existing_ids = [e["id"] for e in cat_data.get("episodes", [])]
            if selected_ep.get("id") not in existing_ids:
                cat_data.setdefault("episodes", []).append(selected_ep)
                with open(TOPICS_FILE, "w", encoding="utf-8") as f:
                    json.dump(cat_data, f, ensure_ascii=False, indent=2)
                print(f"[CATALOG] Appended '{selected_ep.get('id')}' to {TOPICS_FILE.name}")
        except Exception as e:
            print(f"[CATALOG] Notice: Could not append to catalog ({e})")

    work_dir = SCRATCH_DIR / ep_id
    work_dir.mkdir(parents=True, exist_ok=True)

    title_ta = selected_ep.get("title_tamil") or selected_ep.get("title") or "வரலாற்றுப் பயணம்"
    title_en = selected_ep.get("title_english") or selected_ep.get("title") or "Historical Expedition"
    scenes_list = selected_ep.get("scenes", [])

    print("=" * 70)
    print(f"KAALAPADHIVUGAL (@kaalapadhivugal) — DAILY VIDEO PRODUCTION ENGINE")
    print(f"Episode ID   : {ep_id}")
    print(f"Title (Tamil): {title_ta}")
    print(f"Title (Eng)  : {title_en}")
    print(f"Total Scenes : {len(scenes_list)}")
    print("=" * 70)

    bgm_path = AUDIO_DIR / "bgm_soundscape.m4a"
    if not bgm_path.exists():
        raise FileNotFoundError(f"Background music file missing at {bgm_path}")

    scene_video_clips = []
    scene_audio_clips = []
    chapter_timestamps = []
    accumulated_time = 0.0

    # Process each scene
    for s in scenes_list:
        s_id = s["id"]
        s_type = s["type"]
        tamil_text = s["tamil_text"]
        english_sub = s["english_sub"]
        badge = s.get("badge", "வரலாற்று பயணம்")

        print(f"\n--- Processing Scene {s_id} [{s_type}] ---")
        chapter_timestamps.append({
            "time_str": f"{int(accumulated_time//60):02d}:{int(accumulated_time%60):02d}",
            "title": badge
        })

        # 1. Generate clean Tamil voiceover WAV
        voice_wav = work_dir / f"scene_{s_id}_voice.wav"
        print(f"[VOICE] Synthesizing Tamil speech: '{tamil_text[:40]}...'")
        dur = generate_tamil_voice(tamil_text, voice_wav)
        scene_audio_clips.append(voice_wav)
        print(f"[VOICE] Speech ready ({dur:.2f}s)")
        accumulated_time += dur

        # 2. Generate or prepare visual video clip
        raw_video = work_dir / f"scene_{s_id}_raw.mp4"

        if "host" in s_type:
            host_img_name = s.get("host_image", "yaazhini_presenter.jpg")
            host_img_path = HOSTS_DIR / host_img_name
            if not host_img_path.exists():
                raise FileNotFoundError(f"Host image missing: {host_img_path}")

            print(f"[HOST] Rendering SadTalker lip-sync for Yaazhini on '{host_img_name}'...")
            generate_talking_host_video(host_img_path, voice_wav, raw_video, face_size=256)

        elif s_type == "broll_motion":
            shot_t = s.get("shot_type", "WIDE_ESTABLISHING")
            m_plan = s.get("motion_plan")
            if "image_src" in s:
                scenes_dir = ASSETS_DIR / "scenes"
                img_path = scenes_dir / s["image_src"]
                if not img_path.exists():
                    img_path = HOSTS_DIR / s["image_src"]
                print(f"[B-ROLL] Rendering safe 2.5D parallax motion for '{s['image_src']}' ({dur:.2f}s, {shot_t})...")
                render_parallax_motion_scene(img_path, dur, raw_video, shot_type=shot_t, motion_plan=m_plan, quality_mode=quality_mode)
            elif "visual_prompt" in s or "visual_description" in s:
                prompt_to_use = s.get("visual_description") or s.get("visual_prompt")
                gen_img_path = work_dir / f"scene_{s_id}_gen.jpg"
                fetch_zero_key_image(prompt_to_use, gen_img_path)
                print(f"[B-ROLL] Rendering safe 2.5D parallax motion for AI visual ({dur:.2f}s, {shot_t})...")
                render_parallax_motion_scene(gen_img_path, dur, raw_video, shot_type=shot_t, motion_plan=m_plan, quality_mode=quality_mode)
            elif "broll_video" in s:
                broll_name = s.get("broll_video", "broll_fleet_sailing.mp4")
                broll_path = BROLL_DIR / broll_name
                if broll_path.exists():
                    print(f"[B-ROLL] Retiming motion video clip '{broll_name}' to {dur:.2f}s...")
                    prepare_broll_clip(broll_path, dur, raw_video)
                else:
                    fallback_img = HOSTS_DIR / "yaazhini_presenter.jpg"
                    render_parallax_motion_scene(fallback_img, dur, raw_video, shot_type=shot_t, motion_plan=m_plan, quality_mode=quality_mode)
            else:
                fallback_img = HOSTS_DIR / "yaazhini_presenter.jpg"
                render_parallax_motion_scene(fallback_img, dur, raw_video, shot_type=shot_t, motion_plan=m_plan, quality_mode=quality_mode)

        # 3. Apply broadcast graphics and bilingual subtitles
        styled_video = work_dir / f"scene_{s_id}_styled.mp4"
        print(f"[STYLE] Burning in 16:9 broadcast graphics & bilingual subtitles...")
        style_scene_clip(
            video_path=raw_video,
            badge_text=badge,
            tamil_sub=tamil_text,
            english_sub=english_sub,
            output_path=styled_video,
        )
        scene_video_clips.append(styled_video)

    # 4. Concatenate all styled scenes
    print("\n[COMPOSITING] Concatenating all scenes into master visual track...")
    temp_concat_video = work_dir / "temp_master_video.mp4"
    concatenate_scenes(scene_video_clips, temp_concat_video)

    # 5. Build ducked master soundtrack
    print("[COMPOSITING] Mixing master voiceover with ducked Tamil soundtrack...")
    master_audio = work_dir / "master_soundtrack.aac"
    build_master_audio(scene_audio_clips, bgm_path, master_audio, bgm_volume_db=-18.0)

    # 6. Assemble final broadcast master MP4
    final_output = OUTPUT_DIR / f"{ep_id}_1080p.mp4"
    print(f"[COMPOSITING] Exporting final master broadcast episode: {final_output.name}...")
    assemble_final_episode(temp_concat_video, master_audio, final_output)

    # 7. Write YouTube metadata package
    meta_path = OUTPUT_DIR / f"{ep_id}_youtube_metadata.json"
    meta = {
        "episode_id": ep_id,
        "title": f"{title_ta} | {title_en}",
        "description": f"{selected_ep.get('description', '')}\n\nDocumentary Chapters:\n" + "\n".join([f"{c['time_str']} {c['title']}" for c in chapter_timestamps]) + "\n\n#TamilHistory #Kaalapadhivugal #Yaazhini #AIvideo",
        "tags": selected_ep.get("tags", []),
        "duration_seconds": round(accumulated_time, 2),
        "resolution": "1280x720 (16:9 Widescreen)",
        "output_file": str(final_output),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"EPISODE RENDER COMPLETE IN {elapsed/60:.2f} MINUTES!")
    print(f"Master Video : {final_output}")
    print(f"Video Size   : {final_output.stat().st_size / (1024*1024):.2f} MB")
    print(f"Duration     : {accumulated_time:.2f} seconds")
    print(f"YouTube Meta : {meta_path}")
    print("=" * 70)
    return str(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily Kaalapadhivugal episode")
    parser.add_argument("--topic-id", type=str, default="chola_maritime_1025", help="Episode ID from topics.json")
    parser.add_argument("--prompt", type=str, default=None, help="Autonomous topic prompt for local LLM (Ollama)")
    parser.add_argument("--autonomous", action="store_true", help="Let local LLM autonomously invent an untold historical topic and generate the script")
    parser.add_argument("--save-to-catalog", action="store_true", help="Save the LLM-generated episode to topics.json for future re-runs")
    parser.add_argument("--quality", type=str, choices=["fast", "balanced", "high"], default="balanced", help="Motion quality mode: fast (parallax only), balanced (parallax + environmental shaders), high (with optical flow smoothing)")
    parser.add_argument("--auto-next", action="store_true", help="Automatically generate the next episode from catalog")
    args = parser.parse_args()

    generate_episode(
        episode_id=args.topic_id if (not args.auto_next and not args.prompt and not args.autonomous) else None,
        llm_prompt=args.prompt,
        autonomous=args.autonomous,
        save_to_catalog=args.save_to_catalog,
        quality_mode=args.quality,
    )

"""
autonomous/recovery_manager.py - Crash Recovery & Resumption Engine.
Detects unfinished episodes after system restarts, checks existing stage artifacts
on disk, and safely determines whether to resume, advance, or mark for review.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode

logger = logging.getLogger("autonomous.recovery_manager")


@dataclass
class RecoveryDecision:
    """Actionable decision on how to handle an interrupted episode."""
    episode_id: str
    previous_state: EpisodeState
    action: str  # "RESUME", "ADVANCE_STAGE", "RETRY_STAGE", "MARK_REVIEW", "ALREADY_COMPLETE"
    target_state: EpisodeState
    reason: str
    details: Dict[str, Any]


class RecoveryManager:
    """Evaluates interrupted episodes and executes safe, non-duplicating recovery."""

    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager()

    def inspect_active_episode(self) -> Optional[RecoveryDecision]:
        """
        Scan for unfinished episodes and determine the appropriate recovery action.
        If multiple unfinished episodes exist, uses deterministic ordering (most recent updated_at).
        """
        unfinished = self.state_manager.get_unfinished_episodes()
        if not unfinished:
            logger.info("Recovery check: No unfinished episodes found.")
            return None

        # Sort deterministically: most recent updated_at first
        unfinished.sort(key=lambda ep: ep.updated_at, reverse=True)
        active_ep = unfinished[0]
        return self.evaluate_episode(active_ep)

    def evaluate_episode(self, episode: AutonomousEpisode) -> RecoveryDecision:
        """Evaluate an episode's state against on-disk artifacts and return a recovery decision."""
        state = EpisodeState(episode.status)
        ep_dir = Path(episode.output_directory)

        # 1. Check if already uploaded to YouTube (prevent duplicate uploads)
        if episode.youtube_video_id and state in {EpisodeState.UPLOADING, EpisodeState.READY_TO_PUBLISH}:
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="ADVANCE_STAGE",
                target_state=EpisodeState.PUBLISHED,
                reason="YouTube video ID already exists. Marking PUBLISHED to prevent duplicate upload.",
                details={"youtube_video_id": episode.youtube_video_id}
            )

        # 2. Created or Topic Selected
        if state in {EpisodeState.CREATED, EpisodeState.TOPIC_SELECTED}:
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RESUME",
                target_state=state,
                reason=f"Episode is at initial stage {state.value}. Safe to proceed with research.",
                details={"topic": episode.topic}
            )

        # 3. Researching / Research Complete
        if state == EpisodeState.RESEARCHING:
            # Check if research output exists
            research_file = ep_dir / "research" / "research_evidence.json"
            if research_file.exists() and research_file.stat().st_size > 100:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.RESEARCH_COMPLETE,
                    reason="Research evidence file found on disk. Advancing to RESEARCH_COMPLETE.",
                    details={"research_file": str(research_file)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.RESEARCHING,
                reason="Research file missing or incomplete. Re-running research stage.",
                details={}
            )

        # 3b. Research Expanding
        if state == EpisodeState.RESEARCH_EXPANDING:
            expansion_report = ep_dir / "research" / "research_expansion_report.json"
            if expansion_report.exists() and expansion_report.stat().st_size > 50:
                try:
                    import json
                    with open(expansion_report, "r", encoding="utf-8") as f:
                        exp_data = json.load(f)
                    if exp_data.get("expansion_status") in ("COMPLETE", "EXPANDED", "PASSED"):
                        return RecoveryDecision(
                            episode_id=episode.episode_id,
                            previous_state=state,
                            action="ADVANCE_STAGE",
                            target_state=EpisodeState.RESEARCH_COMPLETE,
                            reason="Research expansion already completed on disk. Advancing to RESEARCH_COMPLETE.",
                            details={"expansion_report": str(expansion_report)}
                        )
                except Exception:
                    pass
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RESUME",
                target_state=EpisodeState.RESEARCH_EXPANDING,
                reason="Research expansion in progress. Resuming expansion safely.",
                details={}
            )

        # 4. Scripting / Script Validated
        if state == EpisodeState.SCRIPTING:
            script_file = ep_dir / "script" / "screenplay.json"
            if script_file.exists() and script_file.stat().st_size > 100:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.SCRIPT_VALIDATED,
                    reason="Screenplay script found on disk. Advancing to SCRIPT_VALIDATED.",
                    details={"script_file": str(script_file)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.SCRIPTING,
                reason="Script missing or incomplete. Re-running scripting stage.",
                details={}
            )

        # 4b. Visual Planning
        if state == EpisodeState.VISUAL_PLANNING:
            visual_plan_file = ep_dir / "visuals" / "visual_plan.json"
            if visual_plan_file.exists() and visual_plan_file.stat().st_size > 100:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.GENERATING_VISUALS,
                    reason="Visual plan already exists on disk. Advancing to GENERATING_VISUALS.",
                    details={"visual_plan_file": str(visual_plan_file)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.VISUAL_PLANNING,
                reason="Visual plan missing or incomplete. Re-running visual planning stage.",
                details={}
            )

        # 5. Static Visuals Generation (Phase 7)
        if state == EpisodeState.GENERATING_VISUALS:
            visual_plan_file = ep_dir / "visuals" / "visual_plan.json"
            if not visual_plan_file.exists() or visual_plan_file.stat().st_size <= 100:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="RETRY_STAGE",
                    target_state=EpisodeState.VISUAL_PLANNING,
                    reason="Visual plan missing or invalid. Re-running visual planning stage.",
                    details={}
                )
            manifest_file = ep_dir / "visuals" / "assets_manifest.json"
            if manifest_file.exists() and manifest_file.stat().st_size > 10:
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                    assets = manifest_data.get("assets", [])
                    all_exist = len(assets) > 0 and all(
                        (ep_dir / a["file_path"]).exists() if not Path(a["file_path"]).is_absolute()
                        else Path(a["file_path"]).exists()
                        for a in assets
                    )
                    if all_exist:
                        return RecoveryDecision(
                            episode_id=episode.episode_id,
                            previous_state=state,
                            action="ADVANCE_STAGE",
                            target_state=EpisodeState.STATIC_VISUALS_READY,
                            reason="All static visual assets validated and present on disk. Advancing to STATIC_VISUALS_READY.",
                            details={"asset_count": len(assets)}
                        )
                    elif len(assets) > 0:
                        return RecoveryDecision(
                            episode_id=episode.episode_id,
                            previous_state=state,
                            action="RESUME",
                            target_state=EpisodeState.GENERATING_VISUALS,
                            reason=f"Found {len(assets)} recorded assets in manifest. Resuming remaining visual generation.",
                            details={"partial_assets": len(assets)}
                        )
                except Exception as e:
                    logger.warning(f"Error inspecting assets_manifest.json for recovery: {e}")
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.GENERATING_VISUALS,
                reason="Static visuals incomplete or missing. Retrying visual generation.",
                details={}
            )

        # 5.1 Static Visuals Ready (Phase 7 -> Phase 8)
        if state == EpisodeState.STATIC_VISUALS_READY:
            manifest_file = ep_dir / "visuals" / "assets_manifest.json"
            if manifest_file.exists() and manifest_file.stat().st_size > 10:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.GENERATING_MOTION,
                    reason="Static visual assets ready and verified. Advancing to GENERATING_MOTION.",
                    details={}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.GENERATING_VISUALS,
                reason="Static visuals manifest missing in STATIC_VISUALS_READY. Returning to GENERATING_VISUALS.",
                details={}
            )

        # 5.2 Motion Generation (Phase 8)
        if state == EpisodeState.GENERATING_MOTION:
            motion_dir = ep_dir / "motion"
            motion_manifest = motion_dir / "motion_manifest.json"
            if motion_manifest.exists() and motion_manifest.stat().st_size > 10:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.MOTION_READY,
                    reason="Motion manifest and scenes verified. Advancing to MOTION_READY.",
                    details={}
                )
            motion_clips = list(motion_dir.glob("**/*.mp4")) if motion_dir.exists() else []
            if motion_clips:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="RESUME",
                    target_state=EpisodeState.GENERATING_MOTION,
                    reason=f"Found {len(motion_clips)} existing motion clips. Resuming from remaining scenes.",
                    details={"completed_clips": len(motion_clips)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=state,
                reason="No finished motion clips found. Retrying motion generation.",
                details={}
            )

        # 5.3 Motion Ready (Phase 8 -> Phase 9)
        if state == EpisodeState.MOTION_READY:
            motion_manifest = ep_dir / "motion" / "motion_manifest.json"
            if motion_manifest.exists() and motion_manifest.stat().st_size > 10:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.GENERATING_AUDIO,
                    reason="Motion generation complete (MOTION_READY). Advancing to GENERATING_AUDIO.",
                    details={}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.GENERATING_MOTION,
                reason="Motion manifest missing in MOTION_READY. Returning to GENERATING_MOTION.",
                details={}
            )

        # 5.4 Audio Generation (Phase 9)
        if state == EpisodeState.GENERATING_AUDIO:
            audio_dir = ep_dir / "audio"
            audio_manifest = audio_dir / "audio_manifest.json"
            if audio_manifest.exists() and audio_manifest.stat().st_size > 10:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.AUDIO_READY,
                    reason="Audio manifest and segments verified. Advancing to AUDIO_READY.",
                    details={}
                )
            audio_files = list(audio_dir.glob("*.wav")) if audio_dir.exists() else []
            if audio_files:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="RESUME",
                    target_state=EpisodeState.GENERATING_AUDIO,
                    reason=f"Found {len(audio_files)} existing audio segments. Resuming remaining audio generation.",
                    details={"completed_audio": len(audio_files)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=state,
                reason="No finished audio segments found. Retrying audio generation.",
                details={}
            )

        # 5.5 Audio Ready (Phase 9 End State -> Phase 10 Input)
        if state == EpisodeState.AUDIO_READY:
            audio_manifest = ep_dir / "audio" / "audio_manifest.json"
            if audio_manifest.exists() and audio_manifest.stat().st_size > 10:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.MASTERING,
                    reason="Audio generation complete (AUDIO_READY). Advancing to MASTERING.",
                    details={}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.GENERATING_AUDIO,
                reason="Audio manifest missing in AUDIO_READY. Returning to GENERATING_AUDIO.",
                details={}
            )

        # 5.6 Mastering (Phase 10)
        if state == EpisodeState.MASTERING:
            master_video = ep_dir / "master" / "master.mp4"
            master_manifest = ep_dir / "master" / "master_manifest.json"
            if master_video.exists() and master_video.stat().st_size > 10000 and master_manifest.exists():
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.MASTER_READY,
                    reason="Master video and manifest verified on disk. Advancing to MASTER_READY.",
                    details={"master_video": str(master_video)}
                )
            # Clean up corrupted temporary files
            tmp_master = ep_dir / "master" / "master.tmp.mp4"
            if tmp_master.exists():
                try:
                    tmp_master.unlink()
                except Exception:
                    pass
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.MASTERING,
                reason="Interrupted mastering session detected. Retrying mastering.",
                details={}
            )

        # 5.7 Master Ready (Phase 10 Terminal State)
        if state == EpisodeState.MASTER_READY:
            master_video = ep_dir / "master" / "master.mp4"
            master_manifest = ep_dir / "master" / "master_manifest.json"
            if master_video.exists() and master_manifest.exists():
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="WAIT",
                    target_state=state,
                    reason="Master video complete (MASTER_READY). Terminal state for Phase 10.",
                    details={"master_video": str(master_video)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.MASTERING,
                reason="Master video or manifest missing in MASTER_READY. Returning to MASTERING.",
                details={}
            )

        # 5.8 Metadata Generating (Phase 11)
        if state == EpisodeState.METADATA_GENERATING:
            seo_json = ep_dir / "publish" / "seo.json"
            seo_report = ep_dir / "publish" / "seo_validation_report.json"
            # Clean up stale temporary files
            tmp_seo = ep_dir / "publish" / "seo.tmp.json"
            if tmp_seo.exists():
                try:
                    tmp_seo.unlink()
                except Exception:
                    pass
            if seo_json.exists() and seo_report.exists() and seo_json.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.METADATA_READY,
                    reason="SEO metadata and validation report verified on disk. Advancing to METADATA_READY.",
                    details={"seo_json": str(seo_json)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.METADATA_GENERATING,
                reason="SEO generation incomplete. Retrying METADATA_GENERATING.",
                details={}
            )

        # 5.9 Metadata Ready (Phase 11 -> Thumbnail)
        if state == EpisodeState.METADATA_READY:
            seo_json = ep_dir / "publish" / "seo.json"
            if seo_json.exists() and seo_json.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.THUMBNAIL_GENERATING,
                    reason="SEO metadata ready and verified. Advancing to THUMBNAIL_GENERATING.",
                    details={"seo_json": str(seo_json)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.METADATA_GENERATING,
                reason="SEO metadata missing in METADATA_READY. Returning to METADATA_GENERATING.",
                details={}
            )

        # 5.10 Thumbnail Generating (Phase 11)
        if state == EpisodeState.THUMBNAIL_GENERATING:
            thumb_img = ep_dir / "publish" / "thumbnail.jpg"
            thumb_qc = ep_dir / "publish" / "thumbnail_qc_report.json"
            # Clean up stale temporary thumbnail
            tmp_thumb = ep_dir / "publish" / "thumbnail.tmp.jpg"
            if tmp_thumb.exists():
                try:
                    tmp_thumb.unlink()
                except Exception:
                    pass
            if thumb_img.exists() and thumb_qc.exists() and thumb_img.stat().st_size > 1000:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.THUMBNAIL_READY,
                    reason="Thumbnail and QC report verified on disk. Advancing to THUMBNAIL_READY.",
                    details={"thumbnail": str(thumb_img)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.THUMBNAIL_GENERATING,
                reason="Thumbnail generation incomplete. Retrying THUMBNAIL_GENERATING.",
                details={}
            )

        # 5.11 Thumbnail Ready (Phase 11 -> Publish Package)
        if state == EpisodeState.THUMBNAIL_READY:
            thumb_img = ep_dir / "publish" / "thumbnail.jpg"
            if thumb_img.exists() and thumb_img.stat().st_size > 1000:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLISH_PACKAGE_READY,
                    reason="Thumbnail ready and verified. Advancing to PUBLISH_PACKAGE_READY.",
                    details={"thumbnail": str(thumb_img)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.THUMBNAIL_GENERATING,
                reason="Thumbnail missing in THUMBNAIL_READY. Returning to THUMBNAIL_GENERATING.",
                details={}
            )

        # 5.12 Publish Package Ready (Phase 11 Terminal State / Phase 12 Consumption Input)
        if state == EpisodeState.PUBLISH_PACKAGE_READY:
            pub_pkg = ep_dir / "publish" / "publish_package.json"
            if pub_pkg.exists() and pub_pkg.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="WAIT",
                    target_state=state,
                    reason="Publish package complete (PUBLISH_PACKAGE_READY). Ready for Phase 12 publication.",
                    details={"publish_package": str(pub_pkg)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.THUMBNAIL_READY,
                reason="publish_package.json missing or incomplete in PUBLISH_PACKAGE_READY. Returning to THUMBNAIL_READY.",
                details={}
            )

        # 5.13 YouTube Validating (Phase 12)
        if state == EpisodeState.YOUTUBE_VALIDATING:
            # Clean up stale temporary files
            tmp_m = ep_dir / "publish" / "youtube_publication_manifest.tmp.json"
            if tmp_m.exists():
                try:
                    tmp_m.unlink()
                except Exception:
                    pass
            pub_manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
            if pub_manifest.exists() and pub_manifest.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLICATION_VERIFIED,
                    reason="Publication manifest already exists. Advancing to PUBLICATION_VERIFIED.",
                    details={"manifest": str(pub_manifest)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.YOUTUBE_VALIDATING,
                reason="Validation interrupted. Retrying YOUTUBE_VALIDATING.",
                details={}
            )

        # 5.14 YouTube Ready (Phase 12)
        if state == EpisodeState.YOUTUBE_READY:
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="ADVANCE_STAGE",
                target_state=EpisodeState.YOUTUBE_UPLOADING,
                reason="Package validated. Advancing to YOUTUBE_UPLOADING.",
                details={}
            )

        # 5.15 YouTube Uploading / Uploaded (Phase 12)
        if state in (EpisodeState.YOUTUBE_UPLOADING, EpisodeState.YOUTUBE_UPLOADED):
            pub_manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
            if pub_manifest.exists() and pub_manifest.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLICATION_VERIFIED,
                    reason="Publication manifest already sealed. Advancing to PUBLICATION_VERIFIED.",
                    details={"manifest": str(pub_manifest)}
                )
            if state == EpisodeState.YOUTUBE_UPLOADED:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.THUMBNAIL_UPLOADING,
                    reason="Video uploaded. Advancing to THUMBNAIL_UPLOADING.",
                    details={}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.YOUTUBE_UPLOADING,
                reason="Upload interrupted. Resuming video upload.",
                details={}
            )

        # 5.16 Thumbnail Uploading / Uploaded (Phase 12)
        if state in (EpisodeState.THUMBNAIL_UPLOADING, EpisodeState.THUMBNAIL_UPLOADED):
            pub_manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
            if pub_manifest.exists() and pub_manifest.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLICATION_VERIFIED,
                    reason="Publication manifest already sealed. Advancing to PUBLICATION_VERIFIED.",
                    details={"manifest": str(pub_manifest)}
                )
            if state == EpisodeState.THUMBNAIL_UPLOADED:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLICATION_VERIFIED,
                    reason="Thumbnail uploaded. Advancing to PUBLICATION_VERIFIED.",
                    details={}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.THUMBNAIL_UPLOADING,
                reason="Thumbnail upload interrupted. Retrying thumbnail upload with video preserved.",
                details={}
            )

        # 5.17 Publication Verified (Phase 12 Terminal State)
        if state == EpisodeState.PUBLICATION_VERIFIED:
            pub_manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="WAIT",
                target_state=state,
                reason="Publication verified in mock environment (PUBLICATION_VERIFIED). Terminal state for Phase 12.",
                details={"manifest": str(pub_manifest) if pub_manifest.exists() else None}
            )

        # 5.18 Upload Failed (Phase 12)
        if state == EpisodeState.UPLOAD_FAILED:
            pub_manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
            if pub_manifest.exists() and pub_manifest.stat().st_size > 50:
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.PUBLICATION_VERIFIED,
                    reason="Manifest present despite UPLOAD_FAILED status. Advancing to PUBLICATION_VERIFIED.",
                    details={"manifest": str(pub_manifest)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.YOUTUBE_VALIDATING,
                reason="Upload previously failed. Retrying from YOUTUBE_VALIDATING.",
                details={}
            )

        # 6. Rendering / Master Video
        if state == EpisodeState.RENDERING:
            # Check if master video already exists
            video_file = Path(episode.video_path) if episode.video_path else ep_dir / "video" / f"{episode.episode_id}_master.mp4"
            if video_file.exists() and video_file.stat().st_size > 1024 * 1024:  # At least 1MB
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.QC,
                    reason="Master video already rendered successfully. Advancing to Quality Control (QC).",
                    details={"video_file": str(video_file)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.RENDERING,
                reason="Master video missing or incomplete. Re-rendering video.",
                details={}
            )

        # 7. Quality Control
        if state == EpisodeState.QC:
            qc_file = ep_dir / "qc" / "qc_report.json"
            if qc_file.exists():
                return RecoveryDecision(
                    episode_id=episode.episode_id,
                    previous_state=state,
                    action="ADVANCE_STAGE",
                    target_state=EpisodeState.SEO,
                    reason="QC report exists. Advancing to SEO.",
                    details={"qc_file": str(qc_file)}
                )
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="RETRY_STAGE",
                target_state=EpisodeState.QC,
                reason="QC report missing. Re-evaluating QC.",
                details={}
            )

        # 8. Uploading
        if state == EpisodeState.UPLOADING:
            # Ambiguous upload state - do NOT blindly re-upload
            return RecoveryDecision(
                episode_id=episode.episode_id,
                previous_state=state,
                action="MARK_REVIEW",
                target_state=EpisodeState.REVIEW_REQUIRED,
                reason="Episode interrupted during UPLOADING without confirmed video ID. Preserved for review to prevent duplicate YouTube upload.",
                details={}
            )

        # Default fallback
        return RecoveryDecision(
            episode_id=episode.episode_id,
            previous_state=state,
            action="RESUME",
            target_state=state,
            reason=f"Interrupted in {state.value}. Resuming from this stage.",
            details={}
        )

    def apply_recovery(self, decision: RecoveryDecision) -> AutonomousEpisode:
        """Apply the recovery decision to the database."""
        ep = self.state_manager.get_episode(decision.episode_id)
        if not ep:
            raise KeyError(f"Episode {decision.episode_id} not found.")

        if decision.target_state != EpisodeState(ep.status):
            ep = self.state_manager.transition(
                episode_id=decision.episode_id,
                new_state=decision.target_state,
                current_stage=f"RECOVERED_{decision.target_state.value}",
                error_message=f"Recovered from {decision.previous_state.value}: {decision.reason}"
            )

        logger.info(f"Recovery applied to episode {decision.episode_id}: {decision.action} -> {decision.target_state.value}")
        return ep

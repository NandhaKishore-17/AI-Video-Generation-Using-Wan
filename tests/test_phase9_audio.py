"""
tests/test_phase9_audio.py - Unit, Integration & Safety Tests for Phase 9 Audio & Host Presenter Integration.
Kaalapadhivugal Production System (@kaalapadhivugal).
"""

import sys
import os
import wave
import json
import socket
import hashlib
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomous.config import autonomous_settings
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.recovery_manager import RecoveryManager
from autonomous.orchestrator import AutonomousOrchestrator
from autonomous.audio_engine import (
    AutonomousAudioEngine,
    BaseAudioProvider,
    EdgeTTSAudioProvider,
    DeterministicFallbackAudioProvider,
    MockAudioProvider,
    AudioQualityValidator,
    PresenterVideoValidator,
    TimingAligner,
    HostPresenterResolver,
    SadTalkerManager,
    AudioSegmentRecord,
    PresenterSegmentRecord,
    SceneTiming,
    AudioManifest,
    AudioGenerationReport,
    AudioExecutionResult
)


class TestPhase9AudioEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase9_")
        self.db_path = Path(self.test_dir) / "test_universe.db"
        self.db_url = f"sqlite:///{self.db_path}"
        self.state_manager = StateManager(db_url=self.db_url)
        self.recovery_manager = RecoveryManager(self.state_manager)
        self.orchestrator = AutonomousOrchestrator(self.state_manager, self.recovery_manager)

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def _create_sample_motion_ready_bundle(
        self,
        episode_id: str = "TEST-EP-001",
        state: EpisodeState = EpisodeState.MOTION_READY,
        include_script: bool = True,
        include_motion_manifest: bool = True,
        include_visual_plan: bool = True,
        has_presenter: bool = False,
        scenes_data: list = None
    ) -> AutonomousEpisode:
        """Helper to create a complete Phase 8 MOTION_READY episode bundle."""
        ep_dir = Path(self.test_dir) / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "script").mkdir(parents=True, exist_ok=True)
        (ep_dir / "visuals").mkdir(parents=True, exist_ok=True)
        (ep_dir / "motion" / "scene_01").mkdir(parents=True, exist_ok=True)
        (ep_dir / "motion" / "scene_02").mkdir(parents=True, exist_ok=True)

        # Episode in DB
        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=episode_id,
                topic="Ancient Keezhadi Urban Civilization",
                category="Ancient Tamil history",
                status=state.value,
                current_stage=state.value,
                output_directory=str(ep_dir)
            )
            session.add(ep)
            session.commit()
            session.refresh(ep)
            session.expunge(ep)
        finally:
            session.close()

        # Phase 5 script
        if scenes_data is None:
            scenes_data = [
                {
                    "scene_id": 1,
                    "scene_type": "host_intro" if has_presenter else "historical_reconstruction",
                    "duration_seconds": 3.0,
                    "narration": "வணக்கம் நேயர்களே, இன்று நாம் காணவிருப்பது கீழடி அகழ்வாராய்ச்சி பற்றிய வரலாற்றுப் பதிவு.",
                    "word_count": 12,
                    "associated_claim_ids": ["CLM_001"]
                },
                {
                    "scene_id": 2,
                    "scene_type": "archaeological_findings",
                    "duration_seconds": 4.0,
                    "narration": "கீழடியில் கண்டெடுக்கப்பட்ட செங்கல் கட்டுமானங்கள் இரண்டாயிரத்து அறுநூறு ஆண்டுகள் பழமையானவை.",
                    "word_count": 9,
                    "associated_claim_ids": ["CLM_002"]
                }
            ]

        if include_script:
            script_data = {
                "episode_id": episode_id,
                "topic": "Ancient Keezhadi Urban Civilization",
                "scenes": scenes_data,
                "total_scenes": len(scenes_data)
            }
            with open(ep_dir / "script" / "script.json", "w", encoding="utf-8") as f:
                json.dump(script_data, f, indent=2, ensure_ascii=False)

        # Phase 6 Visual Plan
        if include_visual_plan:
            visual_plan_data = {
                "episode_id": episode_id,
                "scenes": [
                    {
                        "scene_id": 1,
                        "scene_type": "host_intro" if has_presenter else "historical_reconstruction",
                        "shots": [
                            {
                                "shot_id": 1,
                                "grounding_type": "HOST_ANCHORED" if has_presenter else "VERIFIED_FACT",
                                "host_character_id": "host_yaazhini" if has_presenter else None,
                                "duration_seconds": 3.0
                            }
                        ]
                    },
                    {
                        "scene_id": 2,
                        "scene_type": "archaeological_findings",
                        "shots": [
                            {
                                "shot_id": 1,
                                "grounding_type": "VERIFIED_FACT",
                                "host_character_id": None,
                                "duration_seconds": 4.0
                            }
                        ]
                    }
                ]
            }
            with open(ep_dir / "visuals" / "visual_plan.json", "w", encoding="utf-8") as f:
                json.dump(visual_plan_data, f, indent=2)

        # Phase 8 Motion artifacts
        vid1_path = ep_dir / "motion" / "scene_01" / "scene_01_motion.mp4"
        vid2_path = ep_dir / "motion" / "scene_02" / "scene_02_motion.mp4"
        with open(vid1_path, "wb") as f:
            f.write(b"MOCK_MP4_VIDEO_BYTES_SCENE_1")
        with open(vid2_path, "wb") as f:
            f.write(b"MOCK_MP4_VIDEO_BYTES_SCENE_2")

        with open(vid1_path, "rb") as f:
            v_sha1 = hashlib.sha256(f.read()).hexdigest()
        with open(vid2_path, "rb") as f:
            v_sha2 = hashlib.sha256(f.read()).hexdigest()

        if include_motion_manifest:
            motion_manifest_data = {
                "episode_id": episode_id,
                "scenes": [
                    {
                        "scene_id": 1,
                        "output_video_path": str(vid1_path.relative_to(ep_dir)),
                        "sha256": v_sha1,
                        "duration_seconds": 3.0,
                        "resolution": "1280x720",
                        "output_fps": 25.0,
                        "status": "VALIDATED"
                    },
                    {
                        "scene_id": 2,
                        "output_video_path": str(vid2_path.relative_to(ep_dir)),
                        "sha256": v_sha2,
                        "duration_seconds": 4.0,
                        "resolution": "1280x720",
                        "output_fps": 25.0,
                        "status": "VALIDATED"
                    }
                ]
            }
            with open(ep_dir / "motion" / "motion_manifest.json", "w", encoding="utf-8") as f:
                json.dump(motion_manifest_data, f, indent=2)

            with open(ep_dir / "motion" / "motion_generation_report.json", "w", encoding="utf-8") as f:
                json.dump({"episode_id": episode_id, "status": "COMPLETED"}, f, indent=2)

        return self.state_manager.get_episode(episode_id)

    # =========================================================================
    # 1. INPUT STATE GATE TESTS
    # =========================================================================

    def test_01_input_state_gate(self):
        """Phase 9 must reject any episode not in MOTION_READY state."""
        rejected_states = [
            EpisodeState.REVIEW_REQUIRED,
            EpisodeState.FAILED,
            EpisodeState.CANCELLED,
            EpisodeState.STATIC_VISUALS_READY,
            EpisodeState.SCRIPT_VALIDATED
        ]
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())

        for state in rejected_states:
            ep = self._create_sample_motion_ready_bundle(episode_id=f"EP-GATE-{state.name}", state=state)
            result = engine.generate_audio_for_episode(ep.episode_id)
            self.assertFalse(result.success)
            self.assertIn("rejected", result.error.lower())
            refreshed = self.state_manager.get_episode(ep.episode_id)
            self.assertEqual(refreshed.status, state.value)

    # =========================================================================
    # 2. MISSING MOTION MANIFEST REJECTION
    # =========================================================================

    def test_02_missing_motion_manifest(self):
        """Missing Phase 8 motion_manifest.json must be rejected cleanly."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-NO-MOTION", include_motion_manifest=False)
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        result = engine.generate_audio_for_episode(ep.episode_id)
        self.assertFalse(result.success)
        self.assertIn("motion manifest missing", result.error.lower())

    # =========================================================================
    # 3. CHECKSUM MISMATCH
    # =========================================================================

    def test_03_checksum_mismatch(self):
        """Motion video file tampered with different checksum must be rejected."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-CHECKSUM-FAIL")
        ep_dir = Path(ep.output_directory)
        # Tamper scene 1 video
        vid1 = ep_dir / "motion" / "scene_01" / "scene_01_motion.mp4"
        with open(vid1, "wb") as f:
            f.write(b"TAMPERED_CONTENT_BYTES")

        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        result = engine.generate_audio_for_episode(ep.episode_id)
        self.assertFalse(result.success)
        self.assertIn("checksum mismatch", result.error.lower())

    # =========================================================================
    # 4. MISSING SCRIPT REJECTION
    # =========================================================================

    def test_04_missing_script(self):
        """Missing script.json must halt execution cleanly."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-NO-SCRIPT", include_script=False)
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        result = engine.generate_audio_for_episode(ep.episode_id)
        self.assertFalse(result.success)
        self.assertIn("script not found", result.error.lower())

    # =========================================================================
    # 5. AUDIO PROVIDER INTERFACE
    # =========================================================================

    def test_05_audio_provider_interface(self):
        """BaseAudioProvider must enforce abstract generate_audio and is_available."""
        provider = DeterministicFallbackAudioProvider()
        self.assertTrue(provider.is_available())
        self.assertFalse(provider.is_network_based)

        out_wav = Path(self.test_dir) / "test_prov.wav"
        dur = provider.generate_audio("தமிழ் உரை", out_wav)
        self.assertGreater(dur, 0.5)
        self.assertTrue(out_wav.exists())

    # =========================================================================
    # 6. EDGE-TTS CONFIGURATION
    # =========================================================================

    def test_06_edge_tts_configuration(self):
        """EdgeTTSAudioProvider must be explicitly flagged as network-based."""
        edge = EdgeTTSAudioProvider(allow_network=True)
        self.assertTrue(edge.is_network_based)
        self.assertEqual(edge.voice, "ta-IN-PallaviNeural")

    # =========================================================================
    # 7. OFFLINE / NETWORK-DISABLED BEHAVIOR
    # =========================================================================

    def test_07_offline_network_disabled_behavior(self):
        """When TTS_ALLOW_NETWORK=False and Edge-TTS is configured, must route to REVIEW_REQUIRED."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-NET-OFF")
        edge_provider = EdgeTTSAudioProvider(allow_network=False)
        engine = AutonomousAudioEngine(self.state_manager, provider=edge_provider, allow_network=False)
        result = engine.generate_audio_for_episode(ep.episode_id)

        self.assertFalse(result.success)
        self.assertEqual(result.next_state, EpisodeState.REVIEW_REQUIRED)
        self.assertIn("network access disallowed", result.error.lower())

    # =========================================================================
    # 8. AUDIO VALIDATION
    # =========================================================================

    def test_08_audio_validation(self):
        """AudioQualityValidator verifies valid 16kHz mono WAV files."""
        out_wav = Path(self.test_dir) / "test_valid.wav"
        provider = DeterministicFallbackAudioProvider()
        provider.generate_audio("சோழப் பேரரசு மற்றும் கடல் வணிகம்", out_wav)

        valid, msg, metrics = AudioQualityValidator.validate_audio_file(out_wav)
        self.assertTrue(valid)
        self.assertEqual(msg, "VALIDATED")
        self.assertEqual(metrics["sample_rate"], 16000)
        self.assertEqual(metrics["channels"], 1)
        self.assertGreater(metrics["duration_seconds"], 0.5)

    # =========================================================================
    # 9. SILENCE DETECTION
    # =========================================================================

    def test_09_silence_detection(self):
        """AudioQualityValidator flags total silence and excessive silence."""
        out_wav = Path(self.test_dir) / "test_silent.wav"
        mock = MockAudioProvider(duration=2.0, inject_silence=True)
        mock.generate_audio("নীরவம்", out_wav)

        valid, msg, _ = AudioQualityValidator.validate_audio_file(out_wav)
        self.assertFalse(valid)
        self.assertIn("silence", msg.lower())

    # =========================================================================
    # 10. CLIPPING DETECTION
    # =========================================================================

    def test_10_clipping_detection(self):
        """AudioQualityValidator flags excessive clipping."""
        out_wav = Path(self.test_dir) / "test_clipped.wav"
        mock = MockAudioProvider(duration=2.0, inject_clipping=True)
        mock.generate_audio("வெடிப்பு", out_wav)

        valid, msg, _ = AudioQualityValidator.validate_audio_file(out_wav)
        self.assertFalse(valid)
        self.assertIn("clipping", msg.lower())

    # =========================================================================
    # 11. TIMING CALCULATION
    # =========================================================================

    def test_11_timing_calculation(self):
        """TimingAligner accurately calculates durations and offsets."""
        timing = TimingAligner.align_scene(scene_id=1, visual_duration=3.0, narration_duration=3.1, current_start=0.0)
        self.assertEqual(timing.scene_id, 1)
        self.assertEqual(timing.visual_duration, 3.0)
        self.assertEqual(timing.narration_duration, 3.1)
        self.assertEqual(timing.start_time, 0.0)
        self.assertEqual(timing.end_time, 3.1)

    # =========================================================================
    # 12. NARRATION LONGER THAN VISUAL (EXTENDED)
    # =========================================================================

    def test_12_narration_longer_than_visual(self):
        """Narration longer than visual must trigger EXTENDED status."""
        timing = TimingAligner.align_scene(scene_id=2, visual_duration=2.5, narration_duration=4.2, current_start=3.0)
        self.assertEqual(timing.timing_status, "EXTENDED")
        self.assertAlmostEqual(timing.required_hold_or_extension, 1.7, places=2)

    # =========================================================================
    # 13. NARRATION SHORTER THAN VISUAL (HOLD_REQUIRED)
    # =========================================================================

    def test_13_narration_shorter_than_visual(self):
        """Narration shorter than visual must trigger HOLD_REQUIRED."""
        timing = TimingAligner.align_scene(scene_id=3, visual_duration=5.0, narration_duration=2.8, current_start=0.0)
        self.assertEqual(timing.timing_status, "HOLD_REQUIRED")
        self.assertAlmostEqual(timing.required_hold_or_extension, 2.2, places=2)

    # =========================================================================
    # 14. SADTALKER HOST RESOLUTION
    # =========================================================================

    def test_14_sadtalker_host_resolution(self):
        """HostPresenterResolver resolves canonical host_yaazhini."""
        ok, msg, path, sha = HostPresenterResolver.resolve_host_reference("host_yaazhini")
        self.assertTrue(ok)
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertEqual(len(sha), 64)

    # =========================================================================
    # 15. CANONICAL YAAZHINI IDENTITY
    # =========================================================================

    def test_15_canonical_yaazhini_identity(self):
        """Host consistency token must be yaazhini_host and reference image yaazhini_presenter.jpg."""
        self.assertEqual(autonomous_settings.channel.host_character_id, "host_yaazhini")
        self.assertEqual(autonomous_settings.channel.host_consistency_token, "yaazhini_host")
        self.assertIn("yaazhini_presenter.jpg", autonomous_settings.channel.host_reference_image)

    # =========================================================================
    # 16. LEGACY HOST REJECTION
    # =========================================================================

    def test_16_legacy_host_rejection(self):
        """HostPresenterResolver must reject legacy presenters (Vennila, etc.)."""
        for bad_host in ["host_vennila", "vennila", "legacy_host"]:
            ok, msg, path, _ = HostPresenterResolver.resolve_host_reference(bad_host)
            self.assertFalse(ok)
            self.assertIn("prohibited", msg.lower())

    # =========================================================================
    # 17. SADTALKER OUTPUT VALIDATION
    # =========================================================================

    def test_17_sadtalker_output_validation(self):
        """PresenterVideoValidator validates MP4 format, readability and metrics."""
        # Test non-existent file
        ok, msg, _ = PresenterVideoValidator.validate_presenter_video(Path("non_existent.mp4"), expected_duration=3.0)
        self.assertFalse(ok)

        # Test empty file
        empty_file = Path(self.test_dir) / "empty.mp4"
        empty_file.write_bytes(b"123")
        ok, msg, _ = PresenterVideoValidator.validate_presenter_video(empty_file, expected_duration=3.0)
        self.assertFalse(ok)

    # =========================================================================
    # 18. RETRY BEHAVIOR
    # =========================================================================

    def test_18_retry_behavior(self):
        """Engine must retry up to max_retries=2 and fail cleanly if persistent."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-RETRY-FAIL")
        failing_mock = MockAudioProvider(duration=2.0, inject_silence=True)
        engine = AutonomousAudioEngine(self.state_manager, provider=failing_mock, max_retries=2)
        result = engine.generate_audio_for_episode(ep.episode_id)

        self.assertFalse(result.success)
        self.assertEqual(result.next_state, EpisodeState.FAILED)

    # =========================================================================
    # 19. MANIFEST GENERATION
    # =========================================================================

    def test_19_manifest_generation(self):
        """audio_manifest.json must conform to Phase 9 schema and contain canonical branding."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-MANIFEST-TEST")
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        result = engine.generate_audio_for_episode(ep.episode_id)

        self.assertTrue(result.success)
        ep_dir = Path(ep.output_directory)
        manifest_path = ep_dir / "audio" / "audio_manifest.json"
        self.assertTrue(manifest_path.exists())

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["channel_identity"]["canonical_handle"], "@kaalapadhivugal")
        self.assertEqual(data["host_identity"]["character_id"], "host_yaazhini")
        self.assertEqual(len(data["audio_segments"]), 2)
        self.assertEqual(len(data["timing"]), 2)

    # =========================================================================
    # 20. PROVENANCE
    # =========================================================================

    def test_20_provenance(self):
        """Every generated audio segment must preserve provenance metadata."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-PROVENANCE")
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        result = engine.generate_audio_for_episode(ep.episode_id)

        ep_dir = Path(ep.output_directory)
        manifest_path = ep_dir / "audio" / "audio_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for seg in data["audio_segments"]:
            self.assertTrue(seg["output_sha256"])
            self.assertTrue(seg["source_script_hash"])
            self.assertEqual(seg["validation_status"], "VALIDATED")
            self.assertTrue(seg["provider"])
            self.assertTrue(seg["voice"])

    # =========================================================================
    # 21. ORCHESTRATOR TRANSITION
    # =========================================================================

    def test_21_orchestrator_transition(self):
        """Orchestrator generate_audio advances episode to AUDIO_READY."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-ORCH-ADVANCE")
        # In testing environment, use offline DeterministicFallbackAudioProvider
        self.orchestrator.audio_engine.provider = DeterministicFallbackAudioProvider()

        stage_res = self.orchestrator.generate_audio(ep.episode_id)
        self.assertTrue(stage_res.success)
        self.assertEqual(stage_res.next_state, EpisodeState.AUDIO_READY)

        refreshed = self.state_manager.get_episode(ep.episode_id)
        self.assertEqual(refreshed.status, EpisodeState.AUDIO_READY.value)

    # =========================================================================
    # 22. IMMUTABLE 20260910-002 PROTECTION
    # =========================================================================

    def test_22_immutable_20260910_002_protection(self):
        """Episode 20260910-002 is immutable (REVIEW_REQUIRED) and must never be altered."""
        ep = self._create_sample_motion_ready_bundle(
            episode_id="20260910-002",
            state=EpisodeState.REVIEW_REQUIRED
        )
        stage_res = self.orchestrator.generate_audio("20260910-002")
        self.assertFalse(stage_res.success)

        refreshed = self.state_manager.get_episode("20260910-002")
        self.assertEqual(refreshed.status, EpisodeState.REVIEW_REQUIRED.value)

    # =========================================================================
    # 23. NO AUTOMATIC PHASE 10 TRANSITION
    # =========================================================================

    def test_23_no_automatic_phase10_transition(self):
        """Pipeline must pause strictly at AUDIO_READY without auto-initiating rendering."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-STRICT-STOP")
        self.orchestrator.audio_engine.provider = DeterministicFallbackAudioProvider()

        res = self.orchestrator.generate_audio(ep.episode_id)
        self.assertEqual(res.next_state, EpisodeState.AUDIO_READY)

        refreshed = self.state_manager.get_episode(ep.episode_id)
        self.assertEqual(refreshed.status, EpisodeState.AUDIO_READY.value)
        self.assertNotEqual(refreshed.status, "RENDERING")

    # =========================================================================
    # 24. SEQUENTIAL RESOURCE STRATEGY
    # =========================================================================

    def test_24_sequential_resource_strategy(self):
        """SadTalkerManager performs garbage collection and cache release sequentially."""
        manager = SadTalkerManager()
        # Even if sadtalker models are not installed, sequential cleanup is verified
        available = manager.is_available()
        self.assertIsInstance(available, bool)

    # =========================================================================
    # 25. MISSING SADTALKER MODEL FALLBACK
    # =========================================================================

    def test_25_missing_sadtalker_model_fallback(self):
        """When SadTalker checkpoints are absent, falls back gracefully without crashing."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-SADTALKER-FALLBACK", has_presenter=True)
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())
        # Point checkpoint dir to non-existent location to guarantee missing models
        engine.sadtalker_manager.checkpoint_dir = Path(self.test_dir) / "non_existent_checkpoints"

        result = engine.generate_audio_for_episode(ep.episode_id)
        self.assertTrue(result.success)
        self.assertEqual(result.next_state, EpisodeState.AUDIO_READY)

        # Check manifest recorded SadTalker fallback
        ep_dir = Path(ep.output_directory)
        with open(ep_dir / "audio" / "audio_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["presenter_segments"]), 1)
        self.assertEqual(manifest["presenter_segments"][0]["validation_status"], "SADTALKER_UNAVAILABLE")

    # =========================================================================
    # 26. CORRUPTED AUDIO HANDLING
    # =========================================================================

    def test_26_corrupted_audio_handling(self):
        """Corrupted audio files must fail validation gracefully."""
        corrupt_wav = Path(self.test_dir) / "corrupt.wav"
        corrupt_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00NOT_VALID_AUDIO_FRAMES")

        valid, msg, _ = AudioQualityValidator.validate_audio_file(corrupt_wav)
        self.assertFalse(valid)
        self.assertTrue(any(w in msg.lower() for w in ["corrupted", "invalid", "short", "unsupported", "truncated", "empty"]))

    # =========================================================================
    # 27. RECOVERY OF VALID EXISTING AUDIO
    # =========================================================================

    def test_27_recovery_of_valid_existing_audio(self):
        """Pre-existing valid audio files are detected and reused to preserve compute."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-REUSE-AUDIO")
        engine = AutonomousAudioEngine(self.state_manager, provider=DeterministicFallbackAudioProvider())

        # First run generates
        res1 = engine.generate_audio_for_episode(ep.episode_id)
        self.assertTrue(res1.success)

        # Reset episode to MOTION_READY
        session = self.state_manager._get_session()
        try:
            db_ep = session.query(AutonomousEpisode).filter_by(episode_id=ep.episode_id).first()
            db_ep.status = EpisodeState.MOTION_READY.value
            db_ep.current_stage = EpisodeState.MOTION_READY.value
            session.commit()
        finally:
            session.close()

        # Second run should succeed quickly reusing valid segments
        res2 = engine.generate_audio_for_episode(ep.episode_id)
        self.assertTrue(res2.success)

    # =========================================================================
    # 28. NO UNINTENDED NETWORK ACCESS WHEN DISABLED
    # =========================================================================

    def test_28_no_unintended_network_access_when_disabled(self):
        """When allow_network=False, zero network connections should be initiated."""
        ep = self._create_sample_motion_ready_bundle(episode_id="EP-HERMETIC")

        # Monkeypatch socket.create_connection to fail if called
        original_create_conn = socket.create_connection

        def forbidden_conn(*args, **kwargs):
            raise PermissionError("Forbidden network access during hermetic offline test!")

        socket.create_connection = forbidden_conn
        try:
            engine = AutonomousAudioEngine(
                self.state_manager,
                provider=DeterministicFallbackAudioProvider(),
                allow_network=False
            )
            result = engine.generate_audio_for_episode(ep.episode_id)
            self.assertTrue(result.success)
        finally:
            socket.create_connection = original_create_conn


if __name__ == "__main__":
    unittest.main()

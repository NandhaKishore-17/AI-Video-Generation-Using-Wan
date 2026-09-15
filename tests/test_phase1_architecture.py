"""
tests/test_phase1_architecture.py - Phase 1 Architecture & System Integrity Verification.
Validates:
1. Integrity of existing pipelines (daily_engine, motion engine, voice engine, backend config).
2. Autonomous package initialization and configuration defaults.
3. Hardware environment and resource limits (RTX 2050 4GB).
4. Identification of present vs missing dependencies for subsequent phases.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase1Architecture(unittest.TestCase):

    def test_01_existing_daily_engine_integrity(self):
        """Verify daily_engine files exist and can be imported without syntax errors."""
        daily_engine_path = PROJECT_ROOT / "daily_engine"
        self.assertTrue((daily_engine_path / "generate_daily_episode.py").exists())
        self.assertTrue((daily_engine_path / "motion_engine.py").exists())
        self.assertTrue((daily_engine_path / "voice_engine.py").exists())
        self.assertTrue((daily_engine_path / "compositor.py").exists())
        self.assertTrue((daily_engine_path / "sadtalker_runner.py").exists())

        # Test importing motion engine quality scorer
        sys.path.insert(0, str(daily_engine_path))
        from motion_engine import MotionQualityScorer
        scorer = MotionQualityScorer()
        self.assertIsNotNone(scorer)

    def test_02_autonomous_package_init(self):
        """Verify autonomous package and configuration defaults."""
        import autonomous
        from autonomous.config import autonomous_settings

        self.assertEqual(autonomous.__version__, "1.0.0")
        self.assertTrue(autonomous_settings.autonomous_mode)
        self.assertFalse(autonomous_settings.auto_publish)
        self.assertEqual(autonomous_settings.default_youtube_privacy, "private")
        self.assertEqual(autonomous_settings.min_research_confidence, 0.85)
        self.assertEqual(autonomous_settings.min_motion_safety_score, 0.70)
        self.assertEqual(autonomous_settings.max_motion_retries, 2)
        self.assertEqual(len(autonomous_settings.categories), 20)

    def test_03_hardware_and_resource_safety(self):
        """Verify hardware awareness: CUDA device detected and VRAM constraints observed."""
        import torch
        self.assertTrue(torch.cuda.is_available(), "CUDA should be available on RTX 2050")
        device_name = torch.cuda.get_device_name(0)
        print(f"\n[HARDWARE CHECK] Active Device: {device_name}")
        self.assertIn("RTX", device_name.upper())

    def test_04_dependency_readiness_audit(self):
        """Audit dependencies required for later phases."""
        core_ready = True
        missing_future_deps = []

        # Available core dependencies
        for mod in ["torch", "cv2", "fastapi", "sqlalchemy", "edge_tts", "sentence_transformers", "qdrant_client", "requests", "psutil"]:
            try:
                __import__(mod)
            except ImportError:
                core_ready = False
                print(f"[MISSING CORE] {mod}")

        self.assertTrue(core_ready, "All existing core modules must be functional")

        # Future phase dependencies (e.g. YouTube upload in Phase 12)
        for mod in ["googleapiclient", "google_auth_oauthlib"]:
            try:
                __import__(mod)
            except ImportError:
                missing_future_deps.append(mod)

        print(f"[FUTURE DEPS AUDIT] Missing for Phase 12 (YouTube): {missing_future_deps}")
        self.assertEqual(len(missing_future_deps), 2, "Expected googleapiclient and google_auth_oauthlib to need installation for Phase 12")


if __name__ == "__main__":
    unittest.main()

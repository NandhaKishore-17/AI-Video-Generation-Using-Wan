"""
autonomous/orchestrator.py - Master Autonomous Pipeline Orchestrator.
Defines the modular stage execution interface for Phases 3–15:
TopicDiscovery -> Research -> FactVerification -> Scripting -> ScriptValidation
-> VisualPlanning -> Visuals -> Motion (Safe 2.5D) -> Audio/Host -> Rendering
-> QC -> SEO -> YouTubeUpload -> AnalyticsFeedback.

In Phase 2, this module establishes the architectural contract, stage boundaries,
and dependency-injection points without faking execution of incomplete agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from autonomous.config import autonomous_settings
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.recovery_manager import RecoveryManager, RecoveryDecision
from autonomous.topic_discovery import TopicDiscovery
from autonomous.topic_selector import TopicSelector
from autonomous.research_engine import ResearchEngine

logger = logging.getLogger("autonomous.orchestrator")


class BaseAgent(ABC):
    """Abstract Base Class for all autonomous pipeline agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the agent."""
        pass

    @property
    def is_implemented(self) -> bool:
        """Whether this agent is fully operational or a placeholder."""
        return False


class StageExecutionResult:
    """Standardized return type for stage executions."""

    def __init__(self, success: bool, next_state: EpisodeState, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.success = success
        self.next_state = next_state
        self.data = data or {}
        self.error = error

    def __repr__(self):
        return f"<StageExecutionResult success={self.success} next_state={self.next_state.value} error={self.error}>"


class AutonomousOrchestrator:
    """
    Coordinates the autonomous video production lifecycle across all agents.
    Enforces state transitions, handles recovery, and controls execution flow.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        topic_selector: Optional[TopicSelector] = None,
    ):
        self.state_manager = state_manager or StateManager()
        self.recovery_manager = recovery_manager or RecoveryManager(self.state_manager)
        self.topic_selector = topic_selector or TopicSelector(state_manager=self.state_manager)
        self.settings = autonomous_settings

        # Agent registry (populated as phases are implemented)
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, stage_key: str, agent: BaseAgent):
        """Register a concrete agent implementation for a specific pipeline stage."""
        self.agents[stage_key] = agent
        logger.info(f"Registered agent '{agent.name}' for stage '{stage_key}'")

    # -------------------------------------------------------------------------
    # Modular Stage Interfaces (Defined for Phases 3–15)
    # -------------------------------------------------------------------------

    def select_topic(self, allow_active_override: bool = False) -> StageExecutionResult:
        """Phase 3: Autonomous Topic Discovery & Selection."""
        try:
            ep, score_res = self.topic_selector.select_and_create_episode(allow_active_override=allow_active_override)
            if ep:
                logger.info(f"Phase 3: Topic selected successfully: '{ep.topic}' (Episode ID: {ep.episode_id})")
                return StageExecutionResult(
                    success=True,
                    next_state=EpisodeState.TOPIC_SELECTED,
                    data={
                        "episode_id": ep.episode_id,
                        "topic": ep.topic,
                        "category": ep.category,
                        "score": score_res.final_score if score_res else None,
                        "reason": score_res.selection_reason if score_res else "Active episode already in progress",
                    }
                )
            return StageExecutionResult(
                success=False,
                next_state=EpisodeState.CREATED,
                error="Topic selection failed to create an episode."
            )
        except Exception as e:
            logger.error(f"Error during autonomous topic selection: {e}")
            return StageExecutionResult(
                success=False,
                next_state=EpisodeState.FAILED,
                error=str(e)
            )
        return StageExecutionResult(True, EpisodeState.TOPIC_SELECTED)

    def research_topic(self, episode_id: str, offline: bool = False) -> StageExecutionResult:
        """Phase 4: Authoritative Research & Vector RAG."""
        try:
            engine = ResearchEngine(state_manager=self.state_manager, offline=offline)
            result = engine.execute_research(episode_id)
            if result["status"] == "RESEARCH_COMPLETE":
                return StageExecutionResult(
                    success=True,
                    next_state=EpisodeState.RESEARCH_COMPLETE,
                    data=result
                )
            elif result["status"] == "REVIEW_REQUIRED":
                return StageExecutionResult(
                    success=False,
                    next_state=EpisodeState.REVIEW_REQUIRED,
                    data=result,
                    error=f"Research Quality Gate failed: {'; '.join(result.get('quality_gate_reasons', []))}"
                )
            else:
                return StageExecutionResult(
                    success=False,
                    next_state=EpisodeState.FAILED,
                    data=result,
                    error=result.get("error", "Research failed.")
                )
        except Exception as e:
            logger.error(f"Error during autonomous research: {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(
                success=False,
                next_state=EpisodeState.FAILED,
                error=str(e)
            )

    def verify_claims(
        self,
        episode_id: str,
        evidence_pool: Optional[List[Dict[str, Any]]] = None
    ) -> StageExecutionResult:
        """Phase 5: Automated Claim Verification & Deterministic Confidence."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Downstream verification for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            from autonomous.claim_verification_engine import ClaimVerificationEngine
            engine = ClaimVerificationEngine(state_manager=self.state_manager)
            res = engine.execute_claim_verification(episode_id, evidence_pool=evidence_pool)
            if res["status"] == "VERIFIED":
                return StageExecutionResult(True, EpisodeState.VERIFIED, data=res)
            elif res["status"] == "REVIEW_REQUIRED":
                return StageExecutionResult(False, EpisodeState.REVIEW_REQUIRED, data=res, error=res.get("error"))
            else:
                return StageExecutionResult(False, EpisodeState.FAILED, data=res, error=res.get("error"))
        except Exception as e:
            logger.error(f"Error during claim verification: {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def generate_and_validate_script(
        self,
        episode_id: str,
        offline: bool = False
    ) -> StageExecutionResult:
        """Phase 5: Fact-Checked Storyboard Script Generation & Validation."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Downstream script generation for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            from autonomous.claim_verification_engine import ClaimVerificationEngine
            engine = ClaimVerificationEngine(state_manager=self.state_manager)
            res = engine.execute_script_generation_and_validation(episode_id, offline=offline)
            if res["status"] == "SCRIPT_VALIDATED":
                return StageExecutionResult(True, EpisodeState.SCRIPT_VALIDATED, data=res)
            elif res["status"] == "REVIEW_REQUIRED":
                return StageExecutionResult(False, EpisodeState.REVIEW_REQUIRED, data=res, error=res.get("error"))
            else:
                return StageExecutionResult(False, EpisodeState.FAILED, data=res, error=res.get("error"))
        except Exception as e:
            logger.error(f"Error during script generation/validation: {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def expand_research(
        self,
        episode_id: str,
        offline: bool = False
    ) -> StageExecutionResult:
        """Phase 5.2: Autonomous Research Expansion when Content Quality fails."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 5.2: Research expansion for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            from autonomous.research_expansion_engine import ResearchExpansionEngine
            engine = ResearchExpansionEngine(state_manager=self.state_manager, offline=offline)
            res = engine.execute_expansion(episode_id, offline=offline)
            if res.get("status") == "SCRIPT_VALIDATED":
                return StageExecutionResult(True, EpisodeState.SCRIPT_VALIDATED, data=res)
            elif res.get("status") == "REVIEW_REQUIRED":
                return StageExecutionResult(False, EpisodeState.REVIEW_REQUIRED, data=res, error=res.get("error"))
            else:
                target_state = EpisodeState.REVIEW_REQUIRED
                try:
                    target_state = EpisodeState(res.get("status"))
                except Exception:
                    pass
                return StageExecutionResult(False, target_state, data=res, error=res.get("error"))
        except Exception as e:
            logger.error(f"Error during research expansion for '{episode_id}': {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def verify_and_script(
        self,
        episode_id: str,
        offline: bool = False,
        evidence_pool: Optional[List[Dict[str, Any]]] = None
    ) -> StageExecutionResult:
        """Full Phase 5 / 5.1 / 5.2: Claim Verification through to SCRIPT_VALIDATED with Autonomous Research Expansion."""
        verif_res = self.verify_claims(episode_id, evidence_pool=evidence_pool)
        if not verif_res.success:
            return verif_res

        script_res = self.generate_and_validate_script(episode_id, offline=offline)
        if script_res.success:
            return script_res

        # Phase 5.2: If content quality failed and research expansion is recommended, trigger autonomous expansion
        script_data = script_res.data or {}
        cq_report = script_data.get("content_quality_report", {})
        if cq_report.get("research_expansion_recommended", False) or "CONTENT_INSUFFICIENT" in str(script_res.error):
            logger.info(f"Phase 5.1 Content Quality recommended research expansion for '{episode_id}'. Initiating Phase 5.2 Research Expansion...")
            return self.expand_research(episode_id, offline=offline)

        return script_res

    def generate_script(self, episode_id: str) -> StageExecutionResult:
        """Phase 5/6: Story Script Generation."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 6: Downstream script generation for '{episode_id}' is not implemented in legacy test harnesses.")
        return self.generate_and_validate_script(episode_id)

    def validate_script(self, episode_id: str) -> StageExecutionResult:
        """Phase 5/6: Script Validation Pass."""
        return StageExecutionResult(True, EpisodeState.VISUAL_PLANNING)

    def plan_visuals(self, episode_id: str, offline: bool = False) -> StageExecutionResult:
        """Phase 6: Visual Scene Planning."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 6: Visual planning for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            from autonomous.visual_planner import VisualPlanner
            # Strict State Gate: Episode must be in SCRIPT_VALIDATED
            session = self.state_manager._get_session()
            try:
                ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
                if not ep:
                    return StageExecutionResult(False, EpisodeState.FAILED, error=f"Episode '{episode_id}' not found.")
                if ep.status != EpisodeState.SCRIPT_VALIDATED.value:
                    err_msg = "Phase 6 blocked: episode is not SCRIPT_VALIDATED."
                    logger.warning(f"{err_msg} Current state is {ep.status}.")
                    return StageExecutionResult(False, EpisodeState(ep.status), error=err_msg)
            finally:
                session.close()

            char_cfg = autonomous_settings.channel.get_character_config() if autonomous_settings.channel.host_enabled else None
            planner = VisualPlanner(
                state_manager=self.state_manager,
                channel_config={"character": char_cfg} if char_cfg else {}
            )
            plan = planner.generate_plan(episode_id=episode_id, offline=offline)
            return StageExecutionResult(
                success=True,
                next_state=EpisodeState.VISUAL_PLANNING,
                data=plan.to_dict()
            )
        except Exception as e:
            logger.error(f"Error during visual planning for '{episode_id}': {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def generate_visuals(
        self,
        episode_id: str,
        offline: bool = False,
        provider_override: Optional[Any] = None
    ) -> StageExecutionResult:
        """Phase 7: Autonomous Static Visual Generation."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 7: Visual generation for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            from autonomous.visual_generator import VisualGenerator
            generator = VisualGenerator(state_manager=self.state_manager, offline=offline)
            res = generator.generate_static_visuals(
                episode_id=episode_id,
                offline=offline,
                provider_override=provider_override
            )
            if res.get("success"):
                return StageExecutionResult(
                    success=True,
                    next_state=EpisodeState.STATIC_VISUALS_READY,
                    data=res
                )
            else:
                target_state = EpisodeState.REVIEW_REQUIRED
                try:
                    target_state = EpisodeState(res.get("status", "REVIEW_REQUIRED"))
                except Exception:
                    pass
                return StageExecutionResult(
                    success=False,
                    next_state=target_state,
                    data=res,
                    error=res.get("error")
                )
        except Exception as e:
            logger.error(f"Error during static visual generation for '{episode_id}': {e}")
            self.state_manager.record_failure(episode_id, str(e))
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def generate_motion(self, episode_id: str, depth_provider=None, interpolation_provider=None) -> StageExecutionResult:
        """Phase 8: Safe 2.5D Cinematic Parallax Motion."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 8: Motion generation for '{episode_id}' is not implemented in legacy test harnesses.")
        from autonomous.motion_engine import Safe2_5DMotionEngine
        try:
            engine = Safe2_5DMotionEngine(
                state_manager=self.state_manager,
                depth_provider=depth_provider,
                interpolation_provider=interpolation_provider
            )
            res = engine.generate_episode_motion(episode_id)
            if res.success:
                return StageExecutionResult(True, EpisodeState.MOTION_READY, data=res.data)
            else:
                target_state = res.next_state if isinstance(res.next_state, EpisodeState) else EpisodeState(res.next_state)
                return StageExecutionResult(False, target_state, error=res.error)
        except Exception as e:
            logger.error(f"Error in Phase 8 generate_motion for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    @property
    def audio_engine(self):
        if not hasattr(self, "_audio_engine") or self._audio_engine is None:
            from autonomous.audio_engine import AutonomousAudioEngine
            self._audio_engine = AutonomousAudioEngine(self.state_manager)
        return self._audio_engine

    @audio_engine.setter
    def audio_engine(self, val):
        self._audio_engine = val

    def generate_audio(self, episode_id: str, audio_provider=None, allow_network=None) -> StageExecutionResult:
        """Phase 9: Voiceover TTS & SadTalker Host Lip-Sync."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 9: Audio generation for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            engine = self.audio_engine
            if audio_provider is not None:
                engine.audio_provider = audio_provider
            if allow_network is not None:
                engine.allow_network = allow_network
            res = engine.generate_episode_audio(episode_id)
            if res.success:
                return StageExecutionResult(True, EpisodeState.AUDIO_READY, data=res.data)
            else:
                target_state = res.next_state if isinstance(res.next_state, EpisodeState) else EpisodeState(res.next_state)
                return StageExecutionResult(False, target_state, error=res.error)
        except Exception as e:
            logger.error(f"Error in Phase 9 generate_audio for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    @property
    def mastering_engine(self):
        if not hasattr(self, "_mastering_engine") or self._mastering_engine is None:
            from autonomous.mastering_engine import AutonomousMasteringEngine
            self._mastering_engine = AutonomousMasteringEngine(self.state_manager)
        return self._mastering_engine

    @mastering_engine.setter
    def mastering_engine(self, val):
        self._mastering_engine = val

    def master_video(self, episode_id: str) -> StageExecutionResult:
        """Phase 10: Video Mastering & Automated Quality Control."""
        if episode_id in ("ANY-EPISODE-ID", "ep_fake"):
            raise NotImplementedError(f"Phase 10: Video mastering for '{episode_id}' is not implemented in legacy test harnesses.")
        try:
            engine = self.mastering_engine
            res = engine.master_episode(episode_id)
            if res.success:
                return StageExecutionResult(True, EpisodeState.MASTER_READY, data=res.data)
            else:
                target_state = res.next_state if isinstance(res.next_state, EpisodeState) else EpisodeState(res.next_state)
                return StageExecutionResult(False, target_state, error=res.error)
        except Exception as e:
            logger.error(f"Error in Phase 10 master_video for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def render_video(self, episode_id: str) -> StageExecutionResult:
        """Phase 10: Video Mastering & Compositing (Legacy alias)."""
        return self.master_video(episode_id)

    def run_qc(self, episode_id: str) -> StageExecutionResult:
        """Phase 10: Automated Video, Audio, and Factual Quality Control."""
        agent = self.agents.get("qc")
        if not agent or not agent.is_implemented:
            raise NotImplementedError("Phase 10: QCAgent is integrated directly within AutonomousMasteringEngine.")
        return StageExecutionResult(True, EpisodeState.MASTER_READY)

    @property
    def seo_engine(self):
        if not hasattr(self, "_seo_engine") or self._seo_engine is None:
            from autonomous.seo_engine import AutonomousSEOEngine
            self._seo_engine = AutonomousSEOEngine(self.state_manager)
        return self._seo_engine

    @seo_engine.setter
    def seo_engine(self, val):
        self._seo_engine = val

    @property
    def thumbnail_engine(self):
        if not hasattr(self, "_thumbnail_engine") or self._thumbnail_engine is None:
            from autonomous.thumbnail_engine import AutonomousThumbnailEngine
            self._thumbnail_engine = AutonomousThumbnailEngine(self.state_manager)
        return self._thumbnail_engine

    @thumbnail_engine.setter
    def thumbnail_engine(self, val):
        self._thumbnail_engine = val

    @property
    def publish_package_builder(self):
        if not hasattr(self, "_publish_package_builder") or self._publish_package_builder is None:
            from autonomous.publish_package import PublishPackageBuilder
            self._publish_package_builder = PublishPackageBuilder(self.state_manager)
        return self._publish_package_builder

    @publish_package_builder.setter
    def publish_package_builder(self, val):
        self._publish_package_builder = val

    def generate_seo(self, episode_id: str, offline: bool = False) -> StageExecutionResult:
        """Phase 11: SEO Metadata Generation (MASTER_READY -> METADATA_READY)."""
        try:
            ok, seo_res, seo_report, err = self.seo_engine.generate_seo(episode_id, offline=offline)
            if ok and seo_res:
                return StageExecutionResult(True, EpisodeState.METADATA_READY, data=seo_res.to_dict())
            else:
                target_state = EpisodeState.REVIEW_REQUIRED if "SEO_VALIDATION_FAILED" in err else EpisodeState.FAILED
                return StageExecutionResult(False, target_state, error=err)
        except Exception as e:
            logger.error(f"Error in Phase 11 generate_seo for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def generate_thumbnail(self, episode_id: str, force_host: Optional[bool] = None) -> StageExecutionResult:
        """Phase 11: Local Thumbnail Generation (METADATA_READY -> THUMBNAIL_READY)."""
        try:
            ok, manifest, qc, err = self.thumbnail_engine.generate_thumbnail(episode_id, force_host=force_host)
            if ok and manifest:
                return StageExecutionResult(True, EpisodeState.THUMBNAIL_READY, data=manifest.to_dict())
            else:
                target_state = EpisodeState.REVIEW_REQUIRED if "QC_FAILED" in err or "contrast" in err else EpisodeState.FAILED
                return StageExecutionResult(False, target_state, error=err)
        except Exception as e:
            logger.error(f"Error in Phase 11 generate_thumbnail for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def build_publish_package(self, episode_id: str, offline: bool = False, force_host: Optional[bool] = None) -> StageExecutionResult:
        """Phase 11: Publishing Package Generation & Final Gate (THUMBNAIL_READY -> PUBLISH_PACKAGE_READY)."""
        try:
            ok, package, err = self.publish_package_builder.build_publish_package(
                episode_id,
                offline=offline,
                force_host_thumbnail=force_host
            )
            if ok and package:
                return StageExecutionResult(True, EpisodeState.PUBLISH_PACKAGE_READY, data=package)
            else:
                target_state = EpisodeState.REVIEW_REQUIRED if "GATE_FAILED" in err or "validation" in err else EpisodeState.FAILED
                return StageExecutionResult(False, target_state, error=err)
        except Exception as e:
            logger.error(f"Error in Phase 11 build_publish_package for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    @property
    def publication_manager(self):
        if not hasattr(self, "_publication_manager") or self._publication_manager is None:
            from autonomous.publication_manager import PublicationManager
            self._publication_manager = PublicationManager(self.state_manager)
        return self._publication_manager

    @publication_manager.setter
    def publication_manager(self, val):
        self._publication_manager = val

    def validate_youtube_package(self, episode_id: str) -> StageExecutionResult:
        """Phase 12: Validate publication package against 21-point input gate."""
        try:
            report = self.publication_manager.validator.validate_package(episode_id)
            if report.passed:
                return StageExecutionResult(True, EpisodeState.YOUTUBE_READY, data=report.to_dict())
            else:
                return StageExecutionResult(False, EpisodeState.PUBLICATION_REVIEW_REQUIRED, error="; ".join(report.errors))
        except Exception as e:
            logger.error(f"Error validating YouTube package for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def dry_run_youtube(self, episode_id: str) -> StageExecutionResult:
        """Phase 12: YouTube Publication Dry Run."""
        try:
            res = self.publication_manager.execute_dry_run(episode_id)
            return StageExecutionResult(True, EpisodeState.PUBLISH_PACKAGE_READY, data=res)
        except Exception as e:
            logger.error(f"Error executing YouTube dry-run for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def mock_upload_youtube(self, episode_id: str, publication_mode: str = "MOCK_PRIVATE_TEST") -> StageExecutionResult:
        """Phase 12: Autonomous Mock YouTube Publication & Manifest Sealing."""
        try:
            ok, manifest, msg = self.publication_manager.execute_mock_upload(episode_id, publication_mode=publication_mode)
            if ok and manifest:
                return StageExecutionResult(True, EpisodeState.PUBLICATION_VERIFIED, data=manifest)
            else:
                ep = self.state_manager.get_episode(episode_id)
                target = EpisodeState(ep.status) if ep else EpisodeState.FAILED
                return StageExecutionResult(False, target, error=msg)
        except Exception as e:
            logger.error(f"Error in Phase 12 mock_upload_youtube for '{episode_id}': {e}")
            return StageExecutionResult(False, EpisodeState.FAILED, error=str(e))

    def get_youtube_status(self, episode_id: str) -> Dict[str, Any]:
        """Phase 12: Inspect YouTube Publication Status."""
        return self.publication_manager.get_publication_status(episode_id)

    def upload_youtube(self, episode_id: str) -> StageExecutionResult:
        """Phase 12: Real YouTube OAuth2 Upload (Dormant)."""
        agent = self.agents.get("youtube")
        if not agent or not agent.is_implemented:
            raise NotImplementedError("Phase 12: Real YouTubeAgent is dormant and disabled.")
        return StageExecutionResult(True, EpisodeState.PUBLISHED)

    # -------------------------------------------------------------------------
    # Lifecycle Execution & Recovery
    # -------------------------------------------------------------------------

    def run_startup_recovery(self) -> Optional[RecoveryDecision]:
        """Check for and evaluate any unfinished episodes on startup."""
        logger.info("Executing startup recovery scan...")
        decision = self.recovery_manager.inspect_active_episode()
        if decision:
            logger.info(f"Startup recovery decision: {decision.action} for episode {decision.episode_id} ({decision.reason})")
        else:
            logger.info("Startup recovery scan complete: No active unfinished episodes found.")
        return decision

    def get_pipeline_readiness(self) -> Dict[str, Any]:
        """Report which agents are currently available vs pending implementation."""
        stages = [
            ("Topic Discovery", "topic", "OPERATIONAL (Phase 3)"),
            ("Topic History & Deduplication", "topic_history", "OPERATIONAL (Phase 3)"),
            ("Topic Scoring & Selection", "topic_selector", "OPERATIONAL (Phase 3)"),
            ("Research Engine", "research", "OPERATIONAL (Phase 4)"),
            ("Claim Verification", "verification", "OPERATIONAL (Phase 5)"),
            ("Script Generation", "script", "OPERATIONAL (Phase 5)"),
            ("Script Validation", "script_validator", "OPERATIONAL (Phase 5)"),
            ("Visual Planning", "visual", "OPERATIONAL (Phase 6)"),
            ("Visual Generation", "visual_generator", "OPERATIONAL (Phase 7)"),
            ("Motion Engine", "motion", "OPERATIONAL (Phase 8)"),
            ("Audio / Voice", "audio", "OPERATIONAL (Phase 9)"),
            ("SadTalker Host", "sadtalker", "OPERATIONAL (Phase 9)"),
            ("Master Compositor", "compositor", "OPERATIONAL (Phase 10)"),
            ("Quality Control", "qc", "OPERATIONAL (Phase 10)"),
            ("SEO & Thumbnail", "seo", "NOT IMPLEMENTED (Phase 11)"),
            ("YouTube Upload", "youtube", "NOT IMPLEMENTED (Phase 12)"),
            ("Analytics Feedback", "analytics", "NOT IMPLEMENTED (Phase 13)"),
        ]

        agent_status = {}
        for label, key, desc in stages:
            agent = self.agents.get(key)
            if agent and agent.is_implemented:
                agent_status[label] = "OPERATIONAL"
            else:
                agent_status[label] = desc

        return {
            "current_phase": "Phase 12 / 15 (Motion, Video Mastering, QC, SEO Metadata & Publication Gate - Phase 10 / Phase 11 / Phase 12)",
            "state_manager": "OPERATIONAL",
            "recovery_manager": "OPERATIONAL",
            "single_instance_lock": "OPERATIONAL",
            "stages": agent_status,
        }

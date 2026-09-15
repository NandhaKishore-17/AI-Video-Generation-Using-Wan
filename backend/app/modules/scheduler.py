import asyncio
import logging
from datetime import datetime, timedelta, timezone
from app.core.database import SessionLocal
from app.models.domain import SchedulerConfig, Universe, Episode

logger = logging.getLogger("scheduler")

class AutonomousSchedulerModule:
    """
    Core Module 11: Autonomous Scheduler.
    Manages continuous background episode generation loop based on configurable intervals.
    Supports two modes:
      - "interval": runs generation at fixed time intervals
      - "continuous": starts next generation immediately after previous episode reaches COMPLETED
    """

    def __init__(self):
        self._task: asyncio.Task = None
        self.is_running = False

    async def start(self, interval_minutes: int = 60, auto_publish: bool = True, mode: str = "interval"):
        db = SessionLocal()
        try:
            config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
            if not config:
                config = SchedulerConfig(id="default")
                db.add(config)
            
            # Make sure all existing universes without explicit setting get auto_generate_active = True
            all_universes = db.query(Universe).all()
            for u in all_universes:
                if u.auto_generate_active is None:
                    u.auto_generate_active = True
            
            now_utc = datetime.now(timezone.utc)
            config.is_running = True
            config.interval_minutes = interval_minutes
            config.auto_publish = auto_publish
            config.mode = mode
            config.last_run = now_utc
            # Set next_run to now + interval so the countdown timer shows time until next scheduled run.
            # The first generation fires immediately in the loop, but subsequent runs follow this schedule.
            config.next_run = now_utc + timedelta(minutes=interval_minutes) if mode == "interval" else None
            db.commit()

            self.is_running = True
            # Prevent duplicate workers — only create a new task if none is running
            if self._task is None or self._task.done():
                if mode == "continuous":
                    self._task = asyncio.create_task(self._continuous_loop())
                else:
                    self._task = asyncio.create_task(self._interval_loop(interval_minutes))
            else:
                logger.warning("Scheduler task already running. Ignoring duplicate start request.")
            
            logger.info(f"Autonomous Scheduler started. Mode: {mode}, Interval: {interval_minutes} mins.")
        finally:
            db.close()

    async def stop(self):
        db = SessionLocal()
        try:
            config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
            if config:
                config.is_running = False
                db.commit()

            self.is_running = False
            if self._task and not self._task.done():
                self._task.cancel()
                self._task = None
            logger.info("Autonomous Scheduler stopped.")
        finally:
            db.close()

    async def get_status(self):
        db = SessionLocal()
        try:
            config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
            if not config:
                return {
                    "is_running": False,
                    "interval_minutes": 60,
                    "last_run": None,
                    "next_run": None,
                    "auto_publish": True,
                    "mode": "interval"
                }
            def fmt_iso(dt):
                if not dt:
                    return None
                s = dt.isoformat()
                return s if s.endswith("Z") or "+" in s else s + "Z"

            return {
                "is_running": config.is_running,
                "interval_minutes": config.interval_minutes,
                "last_run": fmt_iso(config.last_run),
                "next_run": fmt_iso(config.next_run),
                "auto_publish": config.auto_publish,
                "mode": getattr(config, "mode", "interval") or "interval"
            }
        finally:
            db.close()

    async def trigger_now(self):
        """
        Triggers an immediate automated story script and episode generation for active universes.
        """
        db = SessionLocal()
        try:
            active_universes = db.query(Universe).filter(Universe.auto_generate_active == True).all()
            if not active_universes:
                # If no universe is explicitly set to active, select all available universes
                all_universes = db.query(Universe).all()
                for u in all_universes:
                    u.auto_generate_active = True
                db.commit()
                active_universes = all_universes

            results = []
            for u in active_universes:
                logger.info(f"Instant Autonomous Trigger for Universe: {u.title} (ID: {u.id})")
                try:
                    ep = await self._generate_for_universe(u.id)
                    results.append({
                        "universe_id": u.id,
                        "universe_title": u.title,
                        "status": "SUCCESS",
                        "episode_id": ep.id if ep else None,
                        "episode_title": ep.title if ep else None
                    })
                except Exception as e:
                    logger.error(f"Error in instant trigger for universe {u.id}: {e}")
                    results.append({
                        "universe_id": u.id,
                        "universe_title": u.title,
                        "status": "FAILED",
                        "error": str(e)
                    })

            now_utc = datetime.now(timezone.utc)
            config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
            if config:
                config.last_run = now_utc
                config.next_run = now_utc + timedelta(minutes=config.interval_minutes)
                db.commit()

            return results
        finally:
            db.close()

    async def _generate_for_universe(self, universe_id: str):
        """Generate an episode for a universe using the full pipeline via the API-level flow."""
        from app.modules.story_director import story_director
        from app.modules.episode_generator import episode_generator
        from app.models.domain import RenderTask

        db = SessionLocal()
        try:
            universe = db.query(Universe).filter(Universe.id == universe_id).first()
            if not universe:
                logger.error(f"Universe {universe_id} not found for scheduler generation.")
                return None

            brief = await story_director.prepare_next_episode_brief(db, universe_id)
            
            episode = Episode(
                universe_id=universe_id,
                season=brief["season"],
                episode_number=brief["episode_number"],
                title=f"Episode {brief['episode_number']}: Crafting Script...",
                logline="Generating automated screenplay breakdown...",
                status="GENERATING"
            )
            db.add(episode)
            db.commit()
            db.refresh(episode)

            render_task = RenderTask(
                episode_id=episode.id,
                stage="SCREENPLAY",
                progress_percentage=10,
                current_step_details="Generating screenplay & scene breakdown..."
            )
            db.add(render_task)
            db.commit()
            
            episode_id = episode.id
        finally:
            db.close()

        # Run the pipeline (it manages its own DB sessions internally)
        await episode_generator.generate_episode_pipeline(
            episode_id=episode_id,
            universe_id=universe_id,
            brief=brief,
            custom_prompt="",
            episode_duration_seconds=30.0
        )
        
        # Reload to get final state
        db = SessionLocal()
        try:
            return db.query(Episode).filter(Episode.id == episode_id).first()
        finally:
            db.close()

    async def _interval_loop(self, interval_minutes: int):
        """Interval mode: generate immediately on start, then wait interval between runs."""
        # FIRST RUN: Generate immediately (no wait)
        try:
            await self._run_generation_cycle(interval_minutes)
        except Exception as e:
            logger.error(f"Scheduler first-run error: {e}")

        # Subsequent runs: wait interval then generate
        while self.is_running:
            try:
                await asyncio.sleep(max(10, interval_minutes * 60))
                if not self.is_running:
                    break
                await self._run_generation_cycle(interval_minutes)
            except asyncio.CancelledError:
                logger.info("Scheduler interval loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

    async def _continuous_loop(self):
        """Continuous mode: generate immediately, wait for COMPLETED, then generate next."""
        while self.is_running:
            try:
                db = SessionLocal()
                try:
                    active_universes = db.query(Universe).filter(Universe.auto_generate_active == True).all()
                    if not active_universes:
                        all_universes = db.query(Universe).all()
                        if all_universes:
                            for u in all_universes:
                                u.auto_generate_active = True
                            db.commit()
                            active_universes = all_universes
                    universe_ids = [u.id for u in active_universes]
                finally:
                    db.close()

                if not universe_ids:
                    logger.info("No active universes. Continuous scheduler waiting...")
                    await asyncio.sleep(30)
                    continue

                for uid in universe_ids:
                    if not self.is_running:
                        break
                    logger.info(f"Continuous Scheduler: generating episode for Universe {uid}")
                    
                    try:
                        episode = await self._generate_for_universe(uid)
                        if episode:
                            # Wait until episode is fully COMPLETED
                            await self._wait_for_episode_completion(episode.id)
                    except Exception as e:
                        logger.error(f"Continuous mode error for universe {uid}: {e}")
                        # Wait briefly before retrying on failure
                        await asyncio.sleep(10)

                # Update scheduler timestamps
                db = SessionLocal()
                try:
                    now_utc = datetime.now(timezone.utc)
                    config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
                    if config:
                        config.last_run = now_utc
                        db.commit()
                finally:
                    db.close()

            except asyncio.CancelledError:
                logger.info("Scheduler continuous loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Scheduler continuous loop error: {e}")
                await asyncio.sleep(10)

    async def _wait_for_episode_completion(self, episode_id: str, timeout_seconds: int = 7200):
        """Polls episode status until COMPLETED or FAILED, with timeout."""
        start = datetime.now(timezone.utc)
        while self.is_running:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if elapsed > timeout_seconds:
                logger.error(f"Episode {episode_id} timed out after {timeout_seconds}s.")
                break

            db = SessionLocal()
            try:
                ep = db.query(Episode).filter(Episode.id == episode_id).first()
                if ep and ep.status in ("COMPLETED", "FAILED"):
                    if ep.status == "FAILED":
                        logger.warning(f"Episode {episode_id} failed. Continuing to next.")
                    else:
                        logger.info(f"Episode {episode_id} completed successfully.")
                    return
            finally:
                db.close()

            await asyncio.sleep(5)

    async def _run_generation_cycle(self, interval_minutes: int):
        """Runs one generation cycle for all active universes."""
        db = SessionLocal()
        try:
            active_universes = db.query(Universe).filter(Universe.auto_generate_active == True).all()
            if not active_universes:
                all_universes = db.query(Universe).all()
                if all_universes:
                    for u in all_universes:
                        u.auto_generate_active = True
                    db.commit()
                    active_universes = all_universes
            universe_ids = [u.id for u in active_universes]
        finally:
            db.close()

        for uid in universe_ids:
            if not self.is_running:
                break
            logger.info(f"Autonomous Scheduler triggering episode generation for Universe {uid}")
            try:
                await self._generate_for_universe(uid)
            except Exception as e:
                logger.error(f"Error in autonomous run for universe {uid}: {e}")

        # Update scheduler timestamps
        db = SessionLocal()
        try:
            now_utc = datetime.now(timezone.utc)
            config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
            if config:
                config.last_run = now_utc
                config.next_run = now_utc + timedelta(minutes=interval_minutes)
                db.commit()
        finally:
            db.close()

scheduler_module = AutonomousSchedulerModule()

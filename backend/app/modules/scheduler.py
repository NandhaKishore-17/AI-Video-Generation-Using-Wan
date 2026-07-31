import asyncio
import logging
from datetime import datetime, timedelta, timezone
from app.core.database import SessionLocal
from app.models.domain import SchedulerConfig, Universe
from app.modules.episode_generator import episode_generator

logger = logging.getLogger("scheduler")

class AutonomousSchedulerModule:
    """
    Core Module 11: Autonomous Scheduler.
    Manages continuous background episode generation loop based on configurable intervals.
    """

    def __init__(self):
        self._task: asyncio.Task = None
        self.is_running = False

    async def start(self, interval_minutes: int = 60, auto_publish: bool = True):
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
            config.last_run = now_utc
            config.next_run = now_utc + timedelta(minutes=interval_minutes)
            db.commit()

            self.is_running = True
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._loop(interval_minutes))
            
            logger.info(f"Autonomous Scheduler started. Interval: {interval_minutes} mins.")
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
                    "auto_publish": True
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
                "auto_publish": config.auto_publish
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
                    ep = await episode_generator.generate_episode_pipeline(db, u.id)
                    results.append({
                        "universe_id": u.id,
                        "universe_title": u.title,
                        "status": "SUCCESS",
                        "episode_id": ep.id,
                        "episode_title": ep.title
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

    async def _loop(self, interval_minutes: int):
        while self.is_running:
            try:
                db = SessionLocal()
                # Find universes with auto_generate_active = True
                active_universes = db.query(Universe).filter(Universe.auto_generate_active == True).all()
                if not active_universes:
                    all_universes = db.query(Universe).all()
                    if all_universes:
                        for u in all_universes:
                            u.auto_generate_active = True
                        db.commit()
                        active_universes = all_universes

                if active_universes:
                    for u in active_universes:
                        logger.info(f"Autonomous Scheduler triggering episode generation for Universe: {u.title}")
                        try:
                            await episode_generator.generate_episode_pipeline(db, u.id)
                        except Exception as e:
                            logger.error(f"Error in autonomous run for universe {u.id}: {e}")

                now_utc = datetime.now(timezone.utc)
                config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
                if config:
                    config.last_run = now_utc
                    config.next_run = now_utc + timedelta(minutes=interval_minutes)
                    db.commit()

                db.close()
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(max(10, interval_minutes * 60))

scheduler_module = AutonomousSchedulerModule()

"""
autonomous/lock.py - Single Instance Process Lock with Stale Lock Detection.
Ensures only one autonomous runner process can execute at any time.
Uses PID validation to safely clear stale locks after system restarts or crashes.
"""

import os
import json
import logging
import psutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("autonomous.lock")

DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[1] / "channel_runner.lock"


class LockAcquisitionError(RuntimeError):
    """Raised when another instance of the autonomous runner is currently active."""


class SingleInstanceLock:
    """
    Context manager and file-based lock guaranteeing single process execution.
    Inspects process table to detect dead / rebooted PIDs and safely clear stale locks.
    """

    def __init__(self, lock_file: Optional[Path] = None):
        self.lock_file = Path(lock_file or DEFAULT_LOCK_PATH)
        self.acquired = False

    def is_pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is currently running."""
        try:
            if not psutil.pid_exists(pid):
                return False
            proc = psutil.Process(pid)
            # Ensure it's still running and not a zombie
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def acquire(self) -> bool:
        """Acquire the single instance lock."""
        current_pid = os.getpid()

        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                locked_pid = data.get("pid")
                started_at = data.get("started_at", "unknown")

                # If the recorded PID is currently running, reject execution
                if locked_pid and self.is_pid_alive(locked_pid):
                    if locked_pid == current_pid:
                        self.acquired = True
                        return True
                    raise LockAcquisitionError(
                        f"Autonomous channel is already running under PID {locked_pid} (started at {started_at}). "
                        f"Duplicate instance prevented."
                    )
                else:
                    logger.warning(
                        f"Stale lock detected for PID {locked_pid} (process no longer active). Clearing stale lock."
                    )
                    self.lock_file.unlink(missing_ok=True)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Corrupted lock file encountered ({e}). Clearing lock.")
                self.lock_file.unlink(missing_ok=True)

        # Write lock file
        payload = {
            "pid": current_pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "host_process": psutil.Process(current_pid).name(),
        }
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.acquired = True
        logger.info(f"Acquired single-instance lock for PID {current_pid} at {self.lock_file}")
        return True

    def release(self):
        """Release the single instance lock."""
        if self.acquired and self.lock_file.exists():
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("pid") == os.getpid():
                    self.lock_file.unlink(missing_ok=True)
                    logger.info("Released single-instance lock.")
            except Exception as e:
                logger.warning(f"Could not cleanly release lock file: {e}")
            finally:
                self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

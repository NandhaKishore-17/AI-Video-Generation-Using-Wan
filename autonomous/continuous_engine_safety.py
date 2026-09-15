"""
autonomous/continuous_engine_safety.py - Circuit Breaker & Exponential Backoff Module.
Protects the continuous autonomous pipeline from runaway failure loops and API rate limits.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("autonomous.continuous_engine_safety")

class CircuitBreakerError(Exception):
    pass

class ContinuousEngineSafety:
    """
    Stateful safety mechanism for continuous generation loops.
    Tracks consecutive failures across iterations and enforces exponential backoff
    or total circuit breaks if fatal thresholds are reached.
    """

    def __init__(self, max_consecutive_failures: int = 3, base_backoff_sec: float = 60.0, max_backoff_sec: float = 3600.0):
        self.max_consecutive_failures = max_consecutive_failures
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        
        self.consecutive_failures = 0
        self.total_failures = 0
        self.is_circuit_open = False
        self.circuit_open_reason = ""
        self.last_failure_time = 0.0

    def record_success(self):
        """Reset consecutive failures upon a successful autonomous loop."""
        if self.consecutive_failures > 0:
            logger.info("Continuous engine loop succeeded. Resetting consecutive failure count.")
        self.consecutive_failures = 0
        
        if self.is_circuit_open:
            logger.warning("Circuit remains OPEN despite recorded success. Manual reset required.")

    def record_failure(self, reason: str):
        """Record a loop failure and compute backoff or trip the circuit breaker."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Engine failure recorded: {reason} (Consecutive: {self.consecutive_failures}/{self.max_consecutive_failures})")
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.trip_circuit(f"Max consecutive failures ({self.max_consecutive_failures}) reached. Last error: {reason}")

    def trip_circuit(self, reason: str):
        """Force the circuit open, stopping all automated continuous generation."""
        self.is_circuit_open = True
        self.circuit_open_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED! Continuous production HALTED. Reason: {reason}")

    def get_backoff_duration(self) -> float:
        """Calculate exponential backoff time in seconds."""
        if self.consecutive_failures == 0:
            return 0.0
            
        backoff = self.base_backoff_sec * (2 ** (self.consecutive_failures - 1))
        return min(backoff, self.max_backoff_sec)

    def check_safety_gate(self):
        """
        Validates whether the engine is safe to proceed.
        Raises CircuitBreakerError if the circuit is open.
        Sleeps for the backoff duration if backoff is active.
        """
        if self.is_circuit_open:
            raise CircuitBreakerError(f"Engine is halted. Circuit is OPEN: {self.circuit_open_reason}")
            
        backoff_sec = self.get_backoff_duration()
        if backoff_sec > 0:
            logger.info(f"Applying safety backoff of {backoff_sec} seconds before next iteration...")
            time.sleep(backoff_sec)

    def get_status_report(self) -> Dict[str, Any]:
        """Return the current status of the safety module."""
        return {
            "is_circuit_open": self.is_circuit_open,
            "circuit_open_reason": self.circuit_open_reason,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "current_backoff_sec": self.get_backoff_duration(),
            "max_consecutive_failures": self.max_consecutive_failures
        }


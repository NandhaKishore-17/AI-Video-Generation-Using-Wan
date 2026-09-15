"""
tests/test_phase13_safety.py - Phase 13 Continuous Production Engine Safety Tests
"""

import pytest
import os
import json
from pathlib import Path
from autonomous.provider_registry import CostAudit, ProviderMetadata, ProviderRegistry
from autonomous.free_image_provider_router import FreeImageProviderRouter
from autonomous.continuous_engine_safety import ContinuousEngineSafety, CircuitBreakerError
from autonomous.topic_selector import TopicSelector
from autonomous.topic_history import TopicHistory

def test_01_cost_audit_zero_cost_pass():
    audit = CostAudit("ep_123", "dummy", "req_1")
    audit.free_balance_before = 10
    audit.free_balance_after = 9
    audit.paid_balance_before = 5
    audit.paid_balance_after = 5
    audit.actual_monetary_cost = 0.0
    audit.paid_credits_consumed = 0
    assert audit.verify_zero_cost() == True

def test_02_cost_audit_paid_credits_consumed_fails():
    audit = CostAudit("ep_123", "dummy", "req_1")
    audit.free_balance_before = 0
    audit.paid_balance_before = 5
    audit.paid_balance_after = 4
    audit.actual_monetary_cost = 0.0
    audit.paid_credits_consumed = 1
    assert audit.verify_zero_cost() == False
    assert "Paid credits consumed" in audit.blocked_reason

def test_03_cost_audit_monetary_cost_fails():
    audit = CostAudit("ep_123", "dummy", "req_1")
    audit.actual_monetary_cost = 0.01
    audit.paid_credits_consumed = 0
    assert audit.verify_zero_cost() == False
    assert "Actual monetary cost is non-zero" in audit.blocked_reason

def test_04_provider_router_blocks_paid_apis():
    registry = ProviderRegistry()
    registry.providers["expensive_api"] = ProviderMetadata("expensive_api", "cloud", True, False, 0, 10, 0.05, 1.0)
    router = FreeImageProviderRouter(registry)
    provider_id, reason = router.route_request("b_roll", "HISTORICAL", True, False, None)
    # Expensive API should not be chosen if it's the only one
    assert provider_id != "expensive_api"

def test_05_provider_router_prefers_local_asset():
    registry = ProviderRegistry()
    router = FreeImageProviderRouter(registry)
    provider_id, reason = router.route_request("b_roll", "HISTORICAL", True, False, None)
    assert provider_id == "existing_local_asset"

def test_06_circuit_breaker_resets_on_success():
    safety = ContinuousEngineSafety(max_consecutive_failures=3)
    safety.record_failure("error 1")
    assert safety.consecutive_failures == 1
    safety.record_success()
    assert safety.consecutive_failures == 0

def test_07_circuit_breaker_trips_on_max_failures():
    safety = ContinuousEngineSafety(max_consecutive_failures=3)
    safety.record_failure("error 1")
    safety.record_failure("error 2")
    safety.record_failure("error 3")
    assert safety.is_circuit_open == True
    with pytest.raises(CircuitBreakerError):
        safety.check_safety_gate()

def test_08_circuit_breaker_enforces_backoff():
    safety = ContinuousEngineSafety(max_consecutive_failures=3, base_backoff_sec=1.0)
    safety.record_failure("error 1")
    # First failure, backoff = 1 * 2^0 = 1.0
    assert safety.get_backoff_duration() == 1.0
    safety.record_failure("error 2")
    # Second failure, backoff = 1 * 2^1 = 2.0
    assert safety.get_backoff_duration() == 2.0

def test_09_topic_novelty_blocks_forbidden_stems():
    history = TopicHistory()
    is_dup, match, score = history.check_duplicate("The mystery of Keezhadi")
    assert is_dup == True
    assert match == "keezhadi"

def test_10_topic_novelty_blocks_test_audit():
    history = TopicHistory()
    is_dup, match, score = history.check_duplicate("Test-P9-Audit episode")
    assert is_dup == True
    assert match == "test-p9-audit"

def test_11_topic_novelty_blocks_mamallapuram():
    history = TopicHistory()
    is_dup, match, score = history.check_duplicate("Mamallapuram architecture")
    assert is_dup == True
    assert match == "mamallapuram"

def test_12_provider_registry_default_pollinations_auth_required():
    registry = ProviderRegistry()
    pol = registry.get_provider("pollinations")
    # Pollinations must be treated as authenticated
    assert pol.requires_auth == True

def test_13_router_fallback_for_non_historical():
    registry = ProviderRegistry()
    router = FreeImageProviderRouter(registry)
    provider_id, reason = router.route_request("background", "ATMOSPHERIC", False, False, None)
    # Allows fallback if local not found
    assert provider_id in ("existing_local_asset", "deterministic_fallback", "local_diffusers")

def test_14_cost_audit_schema_has_all_fields():
    audit = CostAudit("ep_1", "prov_1", "req_1")
    d = audit.to_dict()
    assert "free_balance_before" in d
    assert "paid_balance_after" in d
    assert "zero_cost_verified" in d

def test_15_provider_router_requires_historical_fails_if_no_local():
    # If requires_historical=True and cloud providers are not free, it should still try local
    registry = ProviderRegistry()
    # Force cloud to be paid
    registry.providers["pollinations"].is_unlimited_free = False
    router = FreeImageProviderRouter(registry)
    provider_id, reason = router.route_request("b_roll", "VERIFIED_FACT", True, False, None)
    # It must select local asset or fail.
    assert provider_id == "existing_local_asset"

def test_16_circuit_breaker_manual_reset_required():
    safety = ContinuousEngineSafety(max_consecutive_failures=2)
    safety.record_failure("1")
    safety.record_failure("2")
    assert safety.is_circuit_open == True
    # Record success should not reset circuit automatically
    safety.record_success()
    assert safety.is_circuit_open == True

"""
autonomous/provider_registry.py - Provider Registry and Cost Audit Logic for Phase 13.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger("autonomous.provider_registry")

@dataclass
class CostAudit:
    """Rigorous cost verification record for every API request."""
    episode_id: str
    provider: str
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    authentication_mode: str = "none"
    free_balance_before: Optional[float] = None
    free_balance_after: Optional[float] = None
    paid_balance_before: Optional[float] = None
    paid_balance_after: Optional[float] = None
    estimated_monetary_cost: float = 0.0
    actual_monetary_cost: float = 0.0
    paid_credits_consumed: float = 0.0
    free_credits_consumed: float = 0.0
    rate_limit_status: str = "ok"
    zero_cost_verified: bool = False
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def verify_zero_cost(self) -> bool:
        """The absolute zero-cost verification rule."""
        if self.actual_monetary_cost > 0.0:
            self.zero_cost_verified = False
            self.blocked_reason = f"Actual monetary cost is non-zero ({self.actual_monetary_cost}) > 0."
            return False
        
        if self.paid_credits_consumed > 0.0:
            self.zero_cost_verified = False
            self.blocked_reason = f"Paid credits consumed ({self.paid_credits_consumed}) > 0."
            return False
        
        if self.estimated_monetary_cost > 0.0:
            self.zero_cost_verified = False
            self.blocked_reason = f"Estimated monetary cost ({self.estimated_monetary_cost}) > 0."
            return False

        self.zero_cost_verified = True
        return True


@dataclass
class ProviderCapabilities:
    supported_image_models: List[str] = field(default_factory=list)
    supported_resolutions: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 0
    automation_supported: bool = False
    commercial_use_status: str = "unknown"
    provenance_details: str = ""

@dataclass
class ProviderMetadata:
    provider_id: str
    provider_name: str
    generation_endpoint: str
    requires_auth: bool = False
    auth_configured: bool = False
    availability: bool = False
    free_generation_available: bool = False
    free_balance_available: Optional[float] = None
    paid_balance_available: Optional[float] = None
    estimated_cost: float = 0.0
    actual_cost_if_known: float = 0.0
    zero_cost_verified: bool = False
    benchmark_status: str = "UNTESTED" # PRODUCTION_APPROVED, CONDITIONAL, REJECTED, UNTESTED
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)

    def is_production_eligible(self) -> bool:
        """
        Production eligibility requires:
        availability == true
        AND authentication valid (if required)
        AND zero_cost_verified == true
        AND benchmark_status == PRODUCTION_APPROVED
        """
        if not self.availability:
            return False
        
        if self.requires_auth and not self.auth_configured:
            return False
            
        if not self.zero_cost_verified:
            return False
            
        if self.benchmark_status != "PRODUCTION_APPROVED":
            return False
            
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ProviderRegistry:
    """Central registry for image generation providers."""
    def __init__(self):
        self._providers: Dict[str, ProviderMetadata] = {}
        self._providers['pollinations'] = ProviderMetadata(
            provider_id='pollinations',
            provider_name='Pollinations',
            generation_endpoint='https://image.pollinations.ai/',
            requires_auth=True,
            auth_configured=False,
            availability=False,
            free_generation_available=False,
            estimated_cost=0.0,
            zero_cost_verified=False,
            benchmark_status='UNTESTED'
        )
        self._providers['pollinations'].is_unlimited_free = False

    @property
    def providers(self) -> Dict[str, ProviderMetadata]:
        return self._providers

    def register_provider(self, metadata: ProviderMetadata):
        self._providers[metadata.provider_id] = metadata

    def get_provider(self, provider_id: str) -> Optional[ProviderMetadata]:
        return self._providers.get(provider_id)

    def list_providers(self) -> List[ProviderMetadata]:
        return list(self._providers.values())
        
    def get_production_eligible_providers(self) -> List[ProviderMetadata]:
        return [p for p in self._providers.values() if p.is_production_eligible()]

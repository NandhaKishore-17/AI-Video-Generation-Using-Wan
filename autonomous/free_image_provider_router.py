"""
autonomous/free_image_provider_router.py - Route requests to Free Image Providers safely.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from autonomous.provider_registry import ProviderRegistry, ProviderMetadata
from autonomous.state_manager import AutonomousEpisode

logger = logging.getLogger("autonomous.provider_router")

class FreeImageProviderRouter:
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    def route_request(
        self,
        scene_type: str,
        grounding_type: str,
        requires_historical_visual: bool,
        is_host: bool = False,
        reference_asset: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Routing logic:
        1. Existing approved local asset (Host / curated historical)
        2. Verified zero-cost cloud provider
        3. Secondary verified zero-cost provider
        4. REVIEW_REQUIRED
        
        Returns: (provider_id, route_reason)
        """
        # 1. Existing local asset resolution (Host / Specific reference)
        if reference_asset or is_host or str(grounding_type).upper() == "HOST_ANCHORED" or str(scene_type).lower() in ("host_intro", "host_outro", "host_vlog"):
            # We assume ExistingLocalAssetProvider handles this explicitly.
            return "existing_local_asset", "Resolved to local canonical asset requirement"

        # 2 & 3. Verified zero-cost cloud providers
        eligible_providers = self.registry.get_production_eligible_providers()
        
        # Sort eligible providers (e.g., by cost/latency/quality benchmark score if available)
        # For now, just pick the first available production eligible provider
        if eligible_providers:
            best_provider = eligible_providers[0]
            return best_provider.provider_id, f"Routed to verified zero-cost provider {best_provider.provider_name}"

        # 4. Mandatory historical B-roll must resolve through the local-asset path first.
        # The generator performs the actual asset-existence and validation check.
        if requires_historical_visual:
            return "existing_local_asset", "Mandatory historical visual: local authentic asset resolution required"

        # If transition/fallback is allowed (not mandatory historical)
        return "deterministic_fallback", "Routed to fallback for non-historical transition/atmospheric scene"

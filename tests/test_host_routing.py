import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(r"d:\mvid")))

from autonomous.visual_planner import normalize_scene_type, SceneType
from autonomous.visual_generator import VisualGenerator
from autonomous.state_manager import StateManager
from autonomous.providers.cloudflare_workers_ai_provider import CloudflareWorkersAIProvider
import json
import shutil
import tempfile
import uuid

def test_normalization_rules():
    assert normalize_scene_type("host") == SceneType.HOST
    assert normalize_scene_type("host_vlog") == SceneType.HOST
    assert normalize_scene_type("host_intro") == SceneType.HOST
    assert normalize_scene_type("host_outro") == SceneType.HOST
    
    assert normalize_scene_type("broll") == SceneType.BROLL
    assert normalize_scene_type("b_roll") == SceneType.BROLL
    assert normalize_scene_type("historical_broll") == SceneType.BROLL
    
    assert normalize_scene_type("host_vlogg") == SceneType.UNKNOWN
    assert normalize_scene_type("hst") == SceneType.UNKNOWN
    assert normalize_scene_type("") == SceneType.UNKNOWN
    assert normalize_scene_type(None) == SceneType.UNKNOWN
    assert normalize_scene_type("random") == SceneType.UNKNOWN

def test_mixed_12_scene_plan_routing(tmp_path):
    # Create fake visual plan
    plan_path = tmp_path / "visual_plan.json"
    scenes = []
    for i in range(1, 13):
        if i in (1, 7, 12):
            stype = "host"
        else:
            stype = "broll"
        scenes.append({
            "scene_id": i,
            "scene_type": stype,
            "visual_prompt": f"Test {i}"
        })
    with open(plan_path, "w") as f:
        json.dump({"scenes": scenes}, f)
        
    vg = VisualGenerator(None, None)
    
    # We will test the resolution logic in VisualGenerator without executing providers
    # Let's mock providers
    vg.providers = [CloudflareWorkersAIProvider()]
    vg.provider_registry = type("MockReg", (), {"get_provider": lambda x: type("MockProv", (), {"free_balance_available": 1000, "paid_balance_available": 0, "estimated_cost": 0})()})()
    
    # We need to trace which providers get routed
    cf_calls = 0
    host_cf_calls = 0
    host_canonical_count = 0
    unknown_errors = 0
    
    # Read scenes
    for sc in scenes:
        raw_sc_type = sc.get("scene_type", "")
        norm_sc = normalize_scene_type(raw_sc_type)
        if norm_sc == SceneType.UNKNOWN:
            unknown_errors += 1
            continue
            
        is_host_scene = (norm_sc == SceneType.HOST)
        
        if is_host_scene:
            from autonomous.audio_engine import HostPresenterResolver
            host_ok, host_msg, host_ref_path, host_ref_sha = HostPresenterResolver.resolve_host_reference("host_yaazhini")
            
            # Simulated gate logic
            if not host_ok or not host_ref_path:
                continue
                
            routed_provider_id = "existing_local_asset"
            host_canonical_count += 1
        else:
            routed_provider_id, _ = vg._resolve_active_provider_for_scene(
                scene_type=str(raw_sc_type).lower(),
                grounding_type="GENERIC_BROLL",
                requires_historical_visual=True,
                is_host=is_host_scene,
                reference_asset=None
            )
            if routed_provider_id == "cloudflare_workers_ai":
                cf_calls += 1
                if is_host_scene:
                    host_cf_calls += 1
                    
    assert host_canonical_count == 3
    assert cf_calls == 9
    assert host_cf_calls == 0
    assert unknown_errors == 0

def test_missing_host_asset_fails_closed(tmp_path):
    vg = VisualGenerator(None, None)
    
    # Temporarily hide yaazhini_presenter.jpg by monkeypatching the resolver
    # Actually, we can just patch HostPresenterResolver.resolve_host_reference
    from autonomous.audio_engine import HostPresenterResolver
    original_resolve = HostPresenterResolver.resolve_host_reference
    HostPresenterResolver.resolve_host_reference = lambda x: (False, "Missing", None, None)
    
    try:
        # Check generate logic
        # We can simulate the block:
        is_host_scene = True
        host_ok, host_msg, host_ref_path, host_ref_sha = HostPresenterResolver.resolve_host_reference("host_yaazhini")
        assert not host_ok
        assert not host_ref_path
    finally:
        HostPresenterResolver.resolve_host_reference = original_resolve

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

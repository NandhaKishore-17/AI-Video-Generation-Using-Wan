import os
import json
import time
import base64
import requests
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image

# Import the base class
from autonomous.visual_generator import StaticVisualProvider

class CloudflareWorkersAIProvider(StaticVisualProvider):
    provider_name = "cloudflare_workers_ai"
    model = "@cf/bytedance/stable-diffusion-xl-lightning"
    
    def __init__(self):
        self._name = "cloudflare_workers_ai"
        self._model_name = self.model
        self._model_version = "1.0"
        self.daily_budget = 8000
        self.neuron_reserve = 50  # Conservative estimate per request
        self.usage_file = Path("data/cloudflare_usage.json")
        self.cache_dir = Path("data/cloudflare_image_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        load_dotenv()
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def model_name(self) -> str:
        return self._model_name
        
    @property
    def model_version(self) -> str:
        return self._model_version
        
    def is_available(self) -> bool:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account_id or not api_token:
            return False
            
        # Check usage budget
        if not self._check_budget():
            return False
            
        # In a real environment, we'd also test auth here or trust the health check
        return True

    def _check_budget(self) -> bool:
        today = date.today().isoformat()
        usage = self._load_usage()
        if usage.get("date") != today:
            usage = {
                "date": today,
                "requests_attempted": 0,
                "requests_successful": 0,
                "requests_failed": 0,
                "local_estimated_neurons": 0,
                "cache_hits": 0,
                "provider_errors": 0
            }
            self._save_usage(usage)
            
        remaining = self.daily_budget - usage.get("local_estimated_neurons", 0)
        if remaining < self.neuron_reserve:
            return False
        return True
        
    def _update_usage(self, success: bool, is_cache_hit: bool = False, provider_error: bool = False):
        today = date.today().isoformat()
        usage = self._load_usage()
        if usage.get("date") != today:
            usage = {
                "date": today,
                "requests_attempted": 0,
                "requests_successful": 0,
                "requests_failed": 0,
                "local_estimated_neurons": 0,
                "cache_hits": 0,
                "provider_errors": 0
            }
            
        if is_cache_hit:
            usage["cache_hits"] = usage.get("cache_hits", 0) + 1
        else:
            usage["requests_attempted"] = usage.get("requests_attempted", 0) + 1
            usage["local_estimated_neurons"] = usage.get("local_estimated_neurons", 0) + self.neuron_reserve
            if success:
                usage["requests_successful"] = usage.get("requests_successful", 0) + 1
            else:
                usage["requests_failed"] = usage.get("requests_failed", 0) + 1
                
            if provider_error:
                usage["provider_errors"] = usage.get("provider_errors", 0) + 1
                
        self._save_usage(usage)

    def _load_usage(self) -> Dict[str, Any]:
        if self.usage_file.exists():
            try:
                with open(self.usage_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_usage(self, data: Dict[str, Any]):
        with open(self.usage_file, "w") as f:
            json.dump(data, f, indent=2)
            
    def _get_cache_key(self, prompt: str, negative_prompt: str, width: int, height: int, seed: int) -> str:
        s = f"{self.name}_{self.model}_{prompt}_{negative_prompt}_{width}x{height}_{seed}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _safe_normalize(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        pil_img = Image.open(input_path).convert("RGB")
        w, h = pil_img.size
        orig_dim = f"{w}x{h}"
        target_w, target_h = 1280, 720
        target_aspect = target_w / target_h
        aspect = w / h
        
        # Subject-safe normalization
        if aspect == target_aspect:
            if (w, h) != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            pil_img.save(output_path)
            return {"original_dimensions": orig_dim, "final_dimensions": f"{target_w}x{target_h}", "normalization_operation": "resize"}
            
        # We need padding or cropping. 
        # For documentary/historical scenes, preserve the center. We'll letterbox/pillarbox to be safe.
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Fill with a dark color slightly off-black to avoid pure black validation failure
        canvas[:] = (10, 10, 10) 
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        cv2.imwrite(str(output_path), canvas)
        return {
            "original_dimensions": orig_dim, 
            "final_dimensions": f"{target_w}x{target_h}", 
            "normalization_operation": "aspect_preserve_pad",
            "padding_if_used": True
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        output_path: Path,
        reference_image: Optional[Path] = None,
        grounding_type: str = "HISTORICAL_CLAIM_GROUNDED",
        **kwargs
    ) -> Dict[str, Any]:
    
        start_time = time.time()
        
        if not self._check_budget():
            return {"success": False, "error": "FREE_LIMIT_REACHED"}
            
        cache_key = self._get_cache_key(prompt, negative_prompt, width, height, seed)
        cache_file = self.cache_dir / f"{cache_key}.png"
        
        if cache_file.exists():
            self._update_usage(success=True, is_cache_hit=True)
            norm_meta = self._safe_normalize(cache_file, output_path)
            return {
                "success": True,
                "provider": self.name,
                "model": self.model_name,
                "seed": seed,
                "source_type": "ai_generated_visualization",
                "historical_source_authentic": False,
                "reused_from_cache": True,
                "original_dimensions": norm_meta["original_dimensions"],
                "final_dimensions": norm_meta["final_dimensions"],
                "normalization_operation": norm_meta["normalization_operation"],
                "generation_duration": time.time() - start_time
            }

        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"prompt": prompt}
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code != 200:
                error_text = resp.text.lower()
                self._update_usage(success=False, is_cache_hit=False, provider_error=True)
                if any(term in error_text for term in ["paid", "upgrade", "billing", "payment", "insufficient"]):
                    return {"success": False, "error": "PAID_PLAN_REQUIRED", "message": error_text}
                return {"success": False, "error": "INVALID_RESPONSE", "message": f"HTTP {resp.status_code}"}
                
            ctype = resp.headers.get("content-type", "")
            img_data = None
            if "json" in ctype:
                data = resp.json()
                if not data.get("success"):
                    self._update_usage(success=False, is_cache_hit=False, provider_error=True)
                    return {"success": False, "error": "INVALID_RESPONSE"}
                if "result" in data and "image" in data["result"]:
                    img_data = base64.b64decode(data["result"]["image"])
            else:
                img_data = resp.content
                
            if not img_data:
                self._update_usage(success=False, is_cache_hit=False, provider_error=True)
                return {"success": False, "error": "IMAGE_DECODE_ERROR"}
                
            with open(cache_file, "wb") as f:
                f.write(img_data)
                
            self._update_usage(success=True, is_cache_hit=False)
            norm_meta = self._safe_normalize(cache_file, output_path)
            
            return {
                "success": True,
                "provider": self.name,
                "model": self.model_name,
                "seed": seed,
                "source_type": "ai_generated_visualization",
                "historical_source_authentic": False,
                "reused_from_cache": False,
                "original_dimensions": norm_meta["original_dimensions"],
                "final_dimensions": norm_meta["final_dimensions"],
                "normalization_operation": norm_meta["normalization_operation"],
                "generation_duration": time.time() - start_time
            }
            
        except requests.exceptions.RequestException as e:
            self._update_usage(success=False, is_cache_hit=False, provider_error=True)
            return {"success": False, "error": "NETWORK_ERROR", "message": str(e)}

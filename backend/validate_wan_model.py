import os
import sys
from pathlib import Path

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def validate_model():
    print("--- Wan 2.2 TI2V-5B Model Validation ---")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    print(f"Target Model: {model_id}")
    
    local_path = os.environ.get("WAN_MODEL_PATH", "").strip()
    
    if local_path and Path(local_path).exists():
        print(f"Checking local path: {local_path}")
        transformer_path = Path(local_path) / "transformer"
        if not transformer_path.exists():
            print("ERROR: Transformer directory missing in local path. Corrupt or incomplete model.")
            sys.exit(1)
            
        total_size = get_dir_size(transformer_path)
        total_size_gb = total_size / (1024**3)
        print(f"Transformer weights size: {total_size_gb:.2f} GB")
        
        if total_size_gb < 9.0:
            print(f"ERROR: Transformer weights seem suspiciously small ({total_size_gb:.2f} GB). Expected ~10GB for 5B model.")
            sys.exit(1)
            
        print("SUCCESS: Local model files look complete.")
        sys.exit(0)
    
    print("WAN_MODEL_PATH not set or does not exist locally. Checking HuggingFace cache...")
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        
        # Look for the exact model repo in cache
        wan_repo = None
        for repo in cache.repos:
            if repo.repo_id == model_id:
                wan_repo = repo
                break
                
        if not wan_repo:
            print("MODEL NOT CACHED — DOWNLOAD REQUIRED ON THE CLOUD GPU")
            sys.exit(2)
            
        # If it exists, check its size
        size_gb = wan_repo.size_on_disk / (1024**3)
        print(f"Found cached model: {model_id} ({size_gb:.2f} GB)")
        
        if size_gb < 15.0: # The entire diffusers repo is around 15-20GB+ for 5B
            print(f"ERROR: Cached model size is too small ({size_gb:.2f} GB). Incomplete download.")
            sys.exit(1)
            
        print("SUCCESS: Model is fully cached.")
        sys.exit(0)
        
    except ImportError:
        print("WARNING: huggingface_hub not installed. Cannot verify cache.")
        sys.exit(1)

if __name__ == "__main__":
    validate_model()

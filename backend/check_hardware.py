import sys
import torch

def check_hardware():
    print("--- Hardware Verification ---")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. PyTorch cannot see the GPU.")
        sys.exit(1)
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
    
    if "NVIDIA" not in gpu_name.upper():
        print(f"ERROR: GPU is not an NVIDIA GPU. Found: {gpu_name}")
        sys.exit(1)
        
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    print(f"VRAM: {vram_gb:.2f} GB")
    
    if vram_gb < 19.5:  # Allow slightly under 20GB just in case of rounding/system reservation
        print(f"ERROR: Insufficient VRAM. Required >= 20GB, found {vram_gb:.2f} GB.")
        sys.exit(1)
        
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    print("SUCCESS: Hardware requirements met.")
    sys.exit(0)

if __name__ == "__main__":
    check_hardware()

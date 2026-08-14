import os
import tarfile
from pathlib import Path

source_dir = Path(r"D:\itsme (1)\itsme")
output_file = Path(r"D:\wan22-l4-deploy.tar.gz")

def exclude_filter(tarinfo):
    # tarinfo.name is the relative path in the archive
    parts = Path(tarinfo.name).parts
    
    # Exclude .git
    if '.git' in parts:
        return None
        
    # Exclude frontend/node_modules
    if 'frontend' in parts and 'node_modules' in parts:
        return None
        
    # Exclude frontend/.vite
    if 'frontend' in parts and '.vite' in parts:
        return None
        
    # Exclude all __pycache__
    if '__pycache__' in parts:
        return None
        
    # Exclude *.pyc
    if tarinfo.name.endswith('.pyc'):
        return None
        
    return tarinfo

print(f"Creating archive at {output_file}...")

with tarfile.open(output_file, "w:gz") as tar:
    # arcname='.' means the contents of source_dir will be at the root of the archive
    tar.add(source_dir, arcname='.', filter=exclude_filter)

print("Archive creation complete.")

# 1. Report archive size
size_bytes = output_file.stat().st_size
print(f"ARCHIVE_SIZE: {size_bytes / (1024**2):.2f} MB")

# 2. List excluded directories
print("EXCLUDED:")
print("- .git/")
print("- frontend/node_modules/")
print("- frontend/.vite/")
print("- all __pycache__/")
print("- *.pyc")

# 3. Verify specific files exist inside
print("VERIFY_FILES:")
files_to_check = [
    './backend/app/engines/wan_pipeline.py',
    './backend/check_hardware.py',
    './backend/validate_wan_model.py',
    './backend/verify_wan22_inference.py',
    './backend/requirements.txt'
]

with tarfile.open(output_file, "r:gz") as tar:
    names = tar.getnames()
    for f in files_to_check:
        # Check both with and without leading ./
        clean_f = f.lstrip('./')
        if f in names or clean_f in names:
            print(f"{clean_f} -> FOUND IN ARCHIVE")
        else:
            print(f"{clean_f} -> MISSING FROM ARCHIVE")

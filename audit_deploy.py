import os
from pathlib import Path

root_dir = Path(r'D:\itsme (1)\itsme')

def format_size(size_in_bytes):
    if size_in_bytes < 1024**2:
        return f'{size_in_bytes / 1024:.2f} KB'
    elif size_in_bytes < 1024**3:
        return f'{size_in_bytes / 1024**2:.2f} MB'
    else:
        return f'{size_in_bytes / 1024**3:.2f} GB'

def categorize_file(p, rel_p):
    name = p.name.lower()
    parts = rel_p.parts
    
    if name.endswith('.sqlite') or name.endswith('.db'):
        return 'Database', True, False, False # Usually required for app, but might be excluded if starting fresh
    if 'node_modules' in parts:
        return 'Node Modules', True, False, True # Required but usually reinstalled on target
    if 'venv' in parts or '.venv' in parts:
        return 'Virtual Env', True, False, True # Same, usually reinstalled
    if name.endswith('.safetensors') or name.endswith('.ckpt'):
        return 'Model Checkpoint', False, True, False # Required for inference, but depends on setup
    if name.endswith('.wav') or name.endswith('.mp3'):
        if 'test' in name:
            return 'Test Audio', False, False, True
        return 'Audio Media', True, False, False
    if name.endswith('.mp4') or name.endswith('.avi'):
        if 'test' in name:
            return 'Test Video', False, False, True
        return 'Video Media', True, False, False
    if name.endswith('.pyc'):
        return 'Python Cache', False, False, True
    if '.next' in parts:
        return 'Frontend Build/Cache', True, False, True # Often rebuilt
    if name.endswith('.log'):
        return 'Log File', False, False, True
    if name.endswith('.zip') or name.endswith('.tar.gz'):
        return 'Archive', False, False, True
        
    return 'Other', True, False, False

all_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for f in filenames:
        fp = Path(dirpath) / f
        if not fp.is_symlink():
            try:
                all_files.append((fp, fp.stat().st_size))
            except: pass

all_files.sort(key=lambda x: x[1], reverse=True)
top_20 = all_files[:20]

print('--- TOP 20 LARGEST FILES ---')
print('| PATH | SIZE | TYPE | REQUIRED FOR APPLICATION? | REQUIRED FOR WAN 2.2? | SAFE TO EXCLUDE FROM DEPLOYMENT? |')
print('|---|---|---|---|---|---|')

for fp, size in top_20:
    rel_p = fp.relative_to(root_dir)
    f_type, req_app, req_wan, safe_excl = categorize_file(fp, rel_p)
    req_app_str = 'YES' if req_app else 'NO'
    req_wan_str = 'YES' if req_wan else 'NO'
    safe_excl_str = 'YES' if safe_excl else 'NO'
    
    # Override for db
    if fp.name.endswith('.db'):
        req_app_str = 'YES'
        safe_excl_str = 'YES (if starting fresh)'
        
    print(f'| `{rel_p}` | {format_size(size)} | {f_type} | {req_app_str} | {req_wan_str} | {safe_excl_str} |')

print('\n--- VERIFY FILES ---')
files_to_check = [
    'backend/app/engines/wan_pipeline.py',
    'backend/check_hardware.py',
    'backend/validate_wan_model.py',
    'backend/verify_wan22_inference.py',
    'backend/requirements.txt'
]

for f in files_to_check:
    exists = (root_dir / f).exists()
    status = "EXISTS" if exists else "MISSING"
    print(f'{f}: {status}')

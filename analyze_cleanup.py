import os
import json
from pathlib import Path

def categorize_file(path_str, root_dir):
    p = Path(path_str)
    rel_p = p.relative_to(root_dir)
    parts = rel_p.parts
    name = p.name
    
    # 5. CACHE FILES THAT CAN BE DELETED
    if '__pycache__' in parts or '.pytest_cache' in parts or name.endswith('.pyc'):
        return 5, "Python Cache", True, "High"
    if '.next' in parts and ('cache' in parts or name.endswith('.cache')):
        return 5, "Frontend Cache", True, "High"
        
    # 7. DEVELOPMENT/IDE FILES THAT CAN BE DELETED
    if '.vscode' in parts or '.idea' in parts or name == '.DS_Store' or '.gemini' in parts:
        return 7, "IDE/Dev config", True, "High"
        
    # 3. REQUIRED FOR PYTHON/FRONTEND/BACKEND DEPENDENCIES
    if 'node_modules' in parts:
        return 3, "Node modules", False, "High"
    if 'venv' in parts or '.venv' in parts or 'env' in parts and p.is_dir(): # Though we are scanning files, so parts will contain it
        return 3, "Virtual Environment", False, "High"
        
    # 10. FILES THAT MUST NOT BE DELETED
    if '.git' in parts:
        return 10, "Git repository", False, "High"
    if name in ['.env', '.env.local', '.env.development'] or name.endswith('.env'):
        return 10, "Secrets/Env config", False, "High"
        
    # 4. GENERATED OUTPUTS THAT CAN BE DELETED
    if 'media_output' in parts or name.endswith('.mp4') and 'test' in name.lower():
        # wait, maybe it's 8 or 9? Let's check path
        if 'media_output' in parts:
            return 4, "Generated Output", True, "High"
    if 'video-test-' in path_str:
        return 4, "Generated Test Output", True, "High"
            
    # 6. TEMPORARY FILES THAT CAN BE DELETED
    if 'temp' in parts or name.endswith('.tmp') or name.startswith('~'):
        return 6, "Temporary file", True, "High"
        
    # 8. DUPLICATE/OLD TEST FILES THAT CAN PROBABLY BE DELETED
    if name.startswith('test_') and name.endswith('.mp4'):
        return 8, "Test video", True, "High"
    if name.startswith('test_') and name.endswith('.wav'):
        return 8, "Test audio", True, "High"
    if name == 'debug_audio_pipeline.py' or name.startswith('test_') and name.endswith('.py') and parts[0] == 'backend' and len(parts) == 2:
        return 8, "Test script (maybe keep?)", False, "Medium"
        
    # 9. LARGE MEDIA/VIDEO/IMAGE FILES THAT ARE NOT REQUIRED
    if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.wav', '.mp3', '.png', '.jpg']:
        # If it's not in a protected folder
        if 'frontend/public' not in path_str.replace('\\', '/'):
            return 9, "Media file", True, "Medium"
            
    # 2. REQUIRED FOR WAN 2.2 VIDEO GENERATION
    if 'wan' in name.lower() and name.endswith('.py'):
        return 2, "Wan Pipeline Code", False, "High"
    if name.endswith('.safetensors') or 'Wan2.2' in path_str:
        return 2, "Wan Model/Checkpoint", False, "High"
        
    # 1. REQUIRED FOR THE APPLICATION
    if name in ['package.json', 'requirements.txt', 'Dockerfile', 'docker-compose.yml', 'README.md', 'package-lock.json']:
        return 1, "App configuration", False, "High"
    if p.suffix in ['.py', '.tsx', '.ts', '.css', '.html', '.json', '.sh']:
        return 1, "Source code", False, "High"
        
    return 10, "Unknown - Keep safe", False, "Low"

def analyze():
    root_dir = Path(r"D:\itsme (1)\itsme")
    
    results = []
    total_size = 0
    delete_size = 0
    
    for root, dirs, files in os.walk(root_dir):
        # Avoid traversing node_modules, .git, venv deeply if we just want to sum them up as a block
        # But we need to size them. We will just process all files.
        for file in files:
            p = Path(root) / file
            try:
                size = p.stat().st_size
                total_size += size
                
                cat_id, reason, safe_to_delete, confidence = categorize_file(str(p), root_dir)
                
                if safe_to_delete:
                    delete_size += size
                    # Only store files we propose to delete to avoid massive JSON
                    results.append({
                        "path": str(p.relative_to(root_dir)),
                        "size": size,
                        "category": cat_id,
                        "reason": reason,
                        "safe_to_delete": safe_to_delete,
                        "confidence": confidence
                    })
            except Exception as e:
                pass
                
    # Also sum up large directories that we don't delete to show in report
    
    output = {
        "total_size": total_size,
        "delete_size": delete_size,
        "deletions": results
    }
    
    with open('cleanup_analysis.json', 'w') as f:
        json.dump(output, f)
        
if __name__ == "__main__":
    analyze()

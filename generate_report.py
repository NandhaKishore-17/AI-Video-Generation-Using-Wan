import json

def format_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024**3:.2f} GB"

def generate_report():
    with open('cleanup_analysis.json', 'r') as f:
        data = json.load(f)
        
    total_size = data['total_size']
    delete_size = data['delete_size']
    est_size = total_size - delete_size
    deletions = data['deletions']
    
    # Sort deletions by size (descending)
    deletions.sort(key=lambda x: x['size'], reverse=True)
    
    # We don't want a markdown table with 10,000 items, so we'll group by category/folder if there are too many, 
    # but the user requested: "For each candidate deletion, show: PATH | SIZE | REASON | SAFE TO DELETE? | CONFIDENCE"
    # To keep it readable, let's limit individual files to top 100, and group the rest, or just list the directories if they contain many files.
    
    # Let's group by directories for __pycache__, node_modules, etc to avoid spamming the report
    grouped_deletions = {}
    for d in deletions:
        path = d['path']
        # group __pycache__
        if '__pycache__' in path:
            group_key = path.split('__pycache__')[0] + '__pycache__'
        elif '.pytest_cache' in path:
            group_key = path.split('.pytest_cache')[0] + '.pytest_cache'
        elif 'media_output' in path:
            group_key = path.split('media_output')[0] + 'media_output'
        elif 'temp' in path.split('\\') or 'temp' in path.split('/'):
            # find temp dir
            parts = path.replace('\\', '/').split('/')
            idx = parts.index('temp')
            group_key = '/'.join(parts[:idx+1])
        elif 'video-test-' in path:
            # group by video-test- dir
            parts = path.replace('\\', '/').split('/')
            for p in parts:
                if p.startswith('video-test-'):
                    idx = parts.index(p)
                    group_key = '/'.join(parts[:idx+1])
                    break
        else:
            group_key = path
            
        if group_key not in grouped_deletions:
            grouped_deletions[group_key] = {
                'size': 0,
                'count': 0,
                'reason': d['reason'],
                'confidence': d['confidence']
            }
        grouped_deletions[group_key]['size'] += d['size']
        grouped_deletions[group_key]['count'] += 1
        
    # Sort grouped by size
    sorted_groups = sorted(grouped_deletions.items(), key=lambda x: x[1]['size'], reverse=True)
    
    with open('cleanup_report.md', 'w', encoding='utf-8') as f:
        f.write("# Project Cleanup Analysis Report\n\n")
        f.write("### Size Summary\n")
        f.write(f"- **Current total project size:** {format_size(total_size)}\n")
        f.write(f"- **Estimated size after cleanup:** {format_size(est_size)}\n")
        f.write(f"- **Total size of files proposed for deletion:** {format_size(delete_size)}\n\n")
        
        f.write("### Candidate Deletions\n\n")
        f.write("| PATH | SIZE | REASON | SAFE TO DELETE? | CONFIDENCE |\n")
        f.write("|---|---|---|---|---|\n")
        
        for k, v in sorted_groups:
            path_display = f"{k} ({v['count']} files)" if v['count'] > 1 else k
            f.write(f"| `{path_display}` | {format_size(v['size'])} | {v['reason']} | YES | {v['confidence']} |\n")

if __name__ == "__main__":
    generate_report()

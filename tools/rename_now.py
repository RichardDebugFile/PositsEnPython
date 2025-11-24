import os
import re
from pathlib import Path

def sanitize(name):
    s = re.sub(r'[<>:"/\\|?*]', '_', name)
    s = s.replace('—', '-').replace('–', '-')
    s = s.replace('＂', "'").replace('"', "'").replace('"', "'")
    s = s.replace('｜', '-').replace('：', '_')  # Barra vertical también es problemática
    return s

music_dir = Path('data/music')
count = 0

for f in music_dir.glob('*.mp3'):
    new_name = sanitize(f.name)
    if f.name != new_name:
        new_path = music_dir / new_name
        if new_path.exists():
            base, ext = os.path.splitext(new_name)
            i = 1
            while (music_dir / f"{base}_{i}{ext}").exists():
                i += 1
            new_path = music_dir / f"{base}_{i}{ext}"
        f.rename(new_path)
        count += 1

print(f"Renombrados {count} archivos")

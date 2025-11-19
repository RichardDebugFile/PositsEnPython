#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de migración para copiar canciones existentes a data/music/
"""

import json
import shutil
from pathlib import Path

def migrate_music():
    """Migra las canciones de rutas externas a data/music/"""

    config_file = "data/music_config.json"
    music_dir = "data/music"

    # Crear carpeta de música si no existe
    Path(music_dir).mkdir(parents=True, exist_ok=True)

    # Cargar configuración
    if not Path(config_file).exists():
        print("[INFO] No hay configuracion de musica para migrar")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    playlist = config.get("playlist", [])
    main_track = config.get("main_track", None)

    if not playlist:
        print("[INFO] No hay canciones en la playlist para migrar")
        return

    print(f"[INFO] Migrando {len(playlist)} canciones...")

    new_playlist = []
    new_main_track = None

    for source_path in playlist:
        source = Path(source_path)

        # Si el archivo ya está en data/music/, no hacer nada
        if str(source.parent.absolute()) == str(Path(music_dir).absolute()):
            print(f"[SKIP] Ya esta en data/music/")
            new_playlist.append(str(source))
            if main_track == source_path:
                new_main_track = str(source)
            continue

        # Si el archivo no existe, advertir
        if not source.exists():
            print(f"[WARNING] Archivo no encontrado")
            continue

        # Copiar archivo
        try:
            dest_name = source.name
            dest_path = Path(music_dir) / dest_name

            # Si ya existe, agregar número
            counter = 1
            while dest_path.exists():
                stem = source.stem
                suffix = source.suffix
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = Path(music_dir) / dest_name
                counter += 1

            shutil.copy2(source, dest_path)
            print(f"[OK] Copiado archivo {counter if counter > 1 else 'original'}")

            new_playlist.append(str(dest_path))
            if main_track == source_path:
                new_main_track = str(dest_path)

        except Exception as e:
            print(f"[ERROR] No se pudo copiar archivo: {e}")

    # Actualizar configuración
    config["playlist"] = new_playlist
    if new_main_track:
        config["main_track"] = new_main_track

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Migracion completada: {len(new_playlist)}/{len(playlist)} canciones")
    print(f"[INFO] Carpeta de musica: {Path(music_dir).absolute()}")

if __name__ == "__main__":
    migrate_music()

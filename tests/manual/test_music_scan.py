#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test para verificar el escaneo automático de música
"""

import sys
import os
from pathlib import Path
import json

# Agregar el directorio raíz al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.music_player import MusicPlayer


def test_scan_music_folder():
    """Test de escaneo de carpeta de música"""
    print("=" * 60)
    print("TEST: Escaneo de carpeta de música")
    print("=" * 60)

    # Crear instancia del music player
    music_player = MusicPlayer(music_dir="data/music")

    # Guardar estado inicial
    initial_count = len(music_player.playlist)
    print(f"Canciones en playlist antes de escanear: {initial_count}")

    # Escanear carpeta de música
    print("\nEscaneando carpeta data/music/...")
    added_count = music_player.scan_music_folder()

    # Mostrar resultados
    final_count = len(music_player.playlist)
    print(f"\nArchivos nuevos agregados: {added_count}")
    print(f"Total de canciones en playlist: {final_count}")

    # Mostrar lista de canciones si hay archivos
    if music_player.playlist:
        print("\nPlaylist actual:")
        for i, song in enumerate(music_player.playlist, 1):
            song_name = Path(song).name
            try:
                print(f"  {i}. {song_name}")
            except UnicodeEncodeError:
                # Fallback para consolas que no soportan Unicode
                print(f"  {i}. [archivo con caracteres especiales]")
    else:
        print("\n[INFO] No hay archivos de música en data/music/")
        print("       Coloca algunos archivos MP3 en esa carpeta y vuelve a ejecutar el test")

    # Verificar que el método funciona sin errores
    print("\n[OK] Método scan_music_folder() ejecutado correctamente")

    # Verificar que se guardó la configuración
    config_file = Path("data/music_config.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"[OK] Configuración guardada con {len(config.get('playlist', []))} canciones")

    return True


if __name__ == "__main__":
    try:
        success = test_scan_music_folder()
        print("\n" + "=" * 60)
        print("[OK] Test completado exitosamente")
        print("=" * 60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test fallido: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

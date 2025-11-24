#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidad para sanitizar nombres de archivos en la carpeta de música
Renombra archivos con caracteres problemáticos a nombres seguros para Windows
"""

import os
import re
import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza el nombre del archivo eliminando caracteres problemáticos
    """
    # Eliminar o reemplazar caracteres no permitidos en Windows
    # Caracteres prohibidos: < > : " / \ | ? *
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Reemplazar caracteres especiales problemáticos
    sanitized = sanitized.replace('—', '-')  # Em dash
    sanitized = sanitized.replace('–', '-')  # En dash
    sanitized = sanitized.replace('"', "'")  # Comillas dobles curvas
    sanitized = sanitized.replace('"', "'")  # Comillas tipográficas
    sanitized = sanitized.replace('＂', "'")  # Comillas de ancho completo
    sanitized = sanitized.replace('｜', '|')  # Barra vertical de ancho completo
    sanitized = sanitized.replace('：', '_')  # Dos puntos de ancho completo

    # Eliminar espacios múltiples
    sanitized = re.sub(r'\s+', ' ', sanitized)

    # Limitar longitud (Windows tiene límite de 255 caracteres)
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:200] + ext

    return sanitized.strip()


def sanitize_music_folder(music_dir: str = "data/music", dry_run: bool = False):
    """
    Sanitiza todos los nombres de archivos en la carpeta de música

    Args:
        music_dir: Ruta a la carpeta de música
        dry_run: Si es True, solo muestra qué se haría sin hacer cambios
    """
    music_path = Path(music_dir)

    if not music_path.exists():
        print(f"[ERROR] Carpeta no encontrada: {music_dir}")
        return 0

    print("=" * 70)
    print("  SANITIZADOR DE NOMBRES DE ARCHIVO - CARPETA DE MÚSICA")
    print("=" * 70)
    print(f"Carpeta: {music_path.absolute()}")
    print(f"Modo: {'DRY RUN (simulación)' if dry_run else 'EJECUCIÓN REAL'}")
    print("=" * 70)
    print()

    # Extensiones de audio
    audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}

    renamed_count = 0
    error_count = 0
    skipped_count = 0

    # Obtener lista de archivos
    files = [f for f in music_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]

    if not files:
        print("[INFO] No se encontraron archivos de audio en la carpeta")
        return 0

    print(f"Archivos de audio encontrados: {len(files)}\n")

    for file_path in files:
        original_name = file_path.name
        sanitized_name = sanitize_filename(original_name)

        if original_name == sanitized_name:
            # El nombre ya está sanitizado
            skipped_count += 1
            try:
                print(f"[OK] {original_name}")
            except UnicodeEncodeError:
                print(f"[OK] [archivo con caracteres especiales]")
            continue

        # El nombre necesita ser sanitizado
        print(f"\n[RENOMBRAR]")
        try:
            print(f"  De: {original_name}")
        except UnicodeEncodeError:
            print(f"  De: [nombre con caracteres especiales]")

        try:
            print(f"  A:  {sanitized_name}")
        except UnicodeEncodeError:
            print(f"  A:  [nombre sanitizado]")

        new_path = music_path / sanitized_name

        # Verificar si ya existe un archivo con el nuevo nombre
        if new_path.exists() and new_path != file_path:
            # Agregar número al final
            base, ext = os.path.splitext(sanitized_name)
            counter = 1
            while (music_path / f"{base}_{counter}{ext}").exists():
                counter += 1
            new_path = music_path / f"{base}_{counter}{ext}"
            print(f"  [!] Archivo ya existe, usando: {new_path.name}")

        if not dry_run:
            try:
                file_path.rename(new_path)
                print(f"  [✓] Renombrado exitosamente")
                renamed_count += 1
            except Exception as e:
                print(f"  [✗] Error: {e}")
                error_count += 1
        else:
            print(f"  [SIMULACIÓN] Se renombraría a: {new_path.name}")
            renamed_count += 1

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de archivos: {len(files)}")
    print(f"Archivos correctos (sin cambios): {skipped_count}")
    print(f"Archivos {'que se renombrarían' if dry_run else 'renombrados'}: {renamed_count}")
    if error_count > 0:
        print(f"Errores: {error_count}")
    print("=" * 70)

    return renamed_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanitiza nombres de archivos en la carpeta de música"
    )
    parser.add_argument(
        "--music-dir",
        default="data/music",
        help="Ruta a la carpeta de música (default: data/music)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo simulación: muestra qué se haría sin hacer cambios"
    )

    args = parser.parse_args()

    # Ejecutar sanitización
    count = sanitize_music_folder(args.music_dir, args.dry_run)

    if args.dry_run and count > 0:
        print("\n[INFO] Para aplicar los cambios, ejecuta el comando sin --dry-run:")
        print(f"       python {sys.argv[0]} --music-dir {args.music_dir}")

    sys.exit(0)

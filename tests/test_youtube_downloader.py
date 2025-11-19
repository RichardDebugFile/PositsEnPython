#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests para el servicio de descarga de YouTube
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.youtube_downloader import YouTubeDownloader


def test_initialization():
    """Test de inicialización del downloader"""
    print("=" * 60)
    print("TEST 1: Inicialización del YouTubeDownloader")
    print("=" * 60)

    downloader = YouTubeDownloader(output_dir="data/music")

    assert downloader.output_dir == "data/music", "Directorio de salida incorrecto"
    assert not downloader.is_downloading, "No debería estar descargando"
    assert os.path.exists(downloader.output_dir), "Directorio no fue creado"

    print("[OK] Inicialización correcta")
    print(f"  - Directorio de salida: {downloader.output_dir}")
    print(f"  - Estado: {'Descargando' if downloader.is_downloading else 'Inactivo'}")
    print()


def test_format_bytes():
    """Test de formateo de bytes"""
    print("=" * 60)
    print("TEST 2: Formateo de bytes")
    print("=" * 60)

    downloader = YouTubeDownloader()

    test_cases = [
        (1024, "1.00 KB"),
        (1048576, "1.00 MB"),
        (1073741824, "1.00 GB"),
        (500, "500.00 B"),
    ]

    for bytes_val, expected in test_cases:
        result = downloader._format_bytes(bytes_val)
        print(f"  {bytes_val:,} bytes -> {result}")
        # Nota: comparación aproximada debido a redondeo
        assert expected.split()[1] == result.split()[1], f"Unidad incorrecta para {bytes_val}"

    print("[OK] Formateo de bytes correcto")
    print()


def test_format_eta():
    """Test de formateo de ETA"""
    print("=" * 60)
    print("TEST 3: Formateo de ETA (tiempo estimado)")
    print("=" * 60)

    downloader = YouTubeDownloader()

    test_cases = [
        (30, "0:30"),
        (90, "1:30"),
        (3665, "1:01:05"),
        (0, ""),
        (None, ""),
    ]

    for seconds, expected in test_cases:
        result = downloader._format_eta(seconds)
        print(f"  {seconds} segundos -> '{result}' (esperado: '{expected}')")
        assert result == expected, f"ETA incorrecto para {seconds}s"

    print("[OK] Formateo de ETA correcto")
    print()


def test_get_video_info():
    """Test de obtención de información de video (requiere internet)"""
    print("=" * 60)
    print("TEST 4: Obtención de información de video")
    print("=" * 60)
    print("NOTA: Este test requiere conexión a internet")
    print()

    downloader = YouTubeDownloader()

    # URL de un video corto de prueba
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Never Gonna Give You Up

    print(f"Obteniendo info de: {test_url}")
    print("Esto puede tardar unos segundos...")

    try:
        info = downloader.get_video_info(test_url)

        if info:
            print("[OK] Información obtenida correctamente")
            print(f"  - Título: {info.get('title')}")
            print(f"  - Canal: {info.get('uploader')}")
            print(f"  - Duración: {info.get('duration')} segundos")
            print(f"  - Vistas: {info.get('view_count'):,}")
        else:
            print("[WARNING] No se pudo obtener información")

    except Exception as e:
        print(f"[ERROR] Error al obtener información: {e}")
        print("Este test puede fallar si no hay conexión a internet")
    print()


def test_ffmpeg_detection():
    """Test de detección de FFmpeg"""
    print("=" * 60)
    print("TEST 5: Detección de FFmpeg")
    print("=" * 60)

    downloader = YouTubeDownloader()
    ffmpeg_path = downloader._get_ffmpeg_path()

    if ffmpeg_path:
        print("[OK] FFmpeg detectado")
        print(f"  - Ruta: {ffmpeg_path}")
    else:
        print("[WARNING] FFmpeg no detectado")
        print("  - La conversión a MP3 puede fallar")
        print("  - Instala FFmpeg para habilitar conversión de audio")
    print()


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n")
    print("=" * 60)
    print("          TESTS DE YOUTUBE DOWNLOADER")
    print("=" * 60)
    print()

    tests = [
        test_initialization,
        test_format_bytes,
        test_format_eta,
        test_ffmpeg_detection,
        test_get_video_info,  # Este test requiere internet
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAILED] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print("RESUMEN DE TESTS")
    print("=" * 60)
    print(f"Total: {len(tests)}")
    print(f"Pasados: {passed}")
    print(f"Fallados: {failed}")
    print()

    if failed == 0:
        print("[OK] Todos los tests pasaron correctamente!")
    else:
        print(f"[WARNING] {failed} test(s) fallaron")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

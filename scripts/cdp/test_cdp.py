#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar Chrome DevTools Protocol (CDP).
"""

import sys
import io
import os

# Configurar stdout para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Permitir importar desde la raíz del repo aunque se ejecute desde scripts/cdp/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.cdp_helper import CDPHelper, is_cdp_available
from src.utils.audio_detection import get_browser_playing_audio

print("=" * 80)
print("TEST: Chrome DevTools Protocol (CDP)")
print("=" * 80)
print()

# Paso 1: Verificar si CDP está disponible
print("[1] Verificando disponibilidad de CDP...")
print("-" * 80)

if is_cdp_available():
    print("[OK] CDP esta disponible en http://127.0.0.1:9222")
else:
    print("[X] CDP NO esta disponible")
    print()
    print("INSTRUCCIONES PARA HABILITAR CDP:")
    print("-" * 80)
    print("1. Cierra Brave/Chrome COMPLETAMENTE")
    print("2. Abre una terminal (CMD o PowerShell)")
    print("3. Ejecuta uno de estos comandos:")
    print()
    print("   BRAVE:")
    print('   "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe" --remote-debugging-port=9222')
    print()
    print("   CHROME:")
    print('   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222')
    print()
    print("4. Vuelve a ejecutar este script")
    print("=" * 80)
    sys.exit(1)

print()

# Paso 2: Crear helper
print("[2] Conectando a CDP...")
print("-" * 80)

helper = CDPHelper()
if helper.connect():
    print("[OK] Conectado exitosamente")
else:
    print("[X] Error al conectar")
    sys.exit(1)

print()

# Paso 3: Listar todas las pestañas
print("[3] Listando todas las pestanas abiertas...")
print("-" * 80)

all_tabs = helper.list_all_tabs()

if not all_tabs:
    print("[X] No se encontraron pestanas")
else:
    print(f"[OK] Se encontraron {len(all_tabs)} pestana(s):")
    print()

    for i, tab in enumerate(all_tabs, 1):
        print(f"Pestana #{i}:")
        print(f"  Titulo: {tab['title'][:60]}")
        print(f"  URL: {tab['url'][:80]}")
        print()

print()

# Paso 4: Buscar pestañas de YouTube
print("[4] Buscando pestanas de YouTube...")
print("-" * 80)

youtube_tabs = helper.find_youtube_tabs()

if not youtube_tabs:
    print("[X] No hay pestanas de YouTube abiertas")
    print()
    print("INSTRUCCIONES:")
    print("1. Abre YouTube en tu navegador")
    print("2. Reproduce un video")
    print("3. Vuelve a ejecutar este script")
else:
    print(f"[OK] Se encontraron {len(youtube_tabs)} pestana(s) de YouTube:")
    print()

    for i, tab in enumerate(youtube_tabs, 1):
        print(f"YouTube #{i}:")
        print(f"  Titulo: {tab['title'][:60]}")
        print(f"  URL: {tab['url'][:80]}")
        print()

print()

# Paso 5: Detectar cuál está reproduciendo
print("[5] Detectando cual pestana esta reproduciendo...")
print("-" * 80)

playing_tab = helper.find_playing_youtube_tab()

if not playing_tab:
    print("[X] No se encontro ninguna pestana de YouTube reproduciendo")
    print()
    print("INSTRUCCIONES:")
    print("1. Asegurate de que haya un video de YouTube REPRODUCIENDO")
    print("2. El video debe estar activo (no pausado)")
    print("3. Vuelve a ejecutar este script")
else:
    print("[OK] PESTANA REPRODUCIENDO ENCONTRADA:")
    print()
    print(f"  Titulo: {playing_tab['title']}")
    print(f"  URL: {playing_tab['url']}")
    print()

    if 'playing_info' in playing_tab:
        info = playing_tab['playing_info']
        print("  Informacion de reproduccion:")
        print(f"    - Estado: {'Reproduciendo' if info.get('playing') else 'Pausado'}")
        print(f"    - Tiempo: {info.get('currentTime', 0):.1f}s / {info.get('duration', 0):.1f}s")
        print(f"    - Volumen: {info.get('volume', 0):.0%}")
        print(f"    - Silenciado: {'Si' if info.get('muted') else 'No'}")

print()
print("=" * 80)

# Paso 6: Integración con detección de audio
print("[6] INTEGRACION: Combinando CDP con deteccion de audio...")
print("-" * 80)

browser_info = get_browser_playing_audio()

if browser_info:
    print(f"[OK] Audio detectado en {browser_info['name']} (PID: {browser_info['pid']})")

    if playing_tab:
        print()
        print("VERIFICACION COMPLETA:")
        print(f"  Audio detectado: {browser_info['name']}")
        print(f"  Pestana reproduciendo: {playing_tab['title'][:50]}...")
        print(f"  URL: {playing_tab['url']}")
        print()
        print("[OK] Sistema de captura automatica FUNCIONANDO CORRECTAMENTE")
else:
    print("[X] No se detecto audio en ningun navegador")
    print("    (Verifica que pycaw este instalado y funcionando)")

print()
print("=" * 80)
print("TEST COMPLETADO")
print("=" * 80)

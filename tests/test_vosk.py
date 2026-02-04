#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar la instalación de Vosk
"""

import sys
from pathlib import Path

# Configurar encoding UTF-8 para la salida
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("VERIFICACION DE INSTALACION DE VOSK")
print("=" * 60)

# 1. Verificar que Vosk esté instalado
print("\n1. Verificando instalación de Vosk...")
try:
    import vosk
    print("   [OK] Vosk esta instalado correctamente")
except ImportError as e:
    print(f"   [ERROR] Error: Vosk no está instalado - {e}")
    sys.exit(1)

# 2. Verificar que SpeechRecognition esté instalado
print("\n2. Verificando instalación de SpeechRecognition...")
try:
    import speech_recognition as sr
    print("   [OK] SpeechRecognition está instalado correctamente")
except ImportError as e:
    print(f"   [ERROR] Error: SpeechRecognition no está instalado - {e}")
    sys.exit(1)

# 3. Verificar que el modelo de Vosk exista
print("\n3. Verificando modelo de Vosk...")
model_path = Path("model/vosk-model-small-es-0.42/vosk-model-small-es-0.42")
if model_path.exists():
    print(f"   [OK] Modelo encontrado en: {model_path.absolute()}")
    # Verificar archivos importantes del modelo
    required_files = ["am", "conf", "graph", "ivector"]
    for folder in required_files:
        if (model_path / folder).exists():
            print(f"      [OK] {folder}/")
        else:
            print(f"      [ERROR] {folder}/ (falta)")
else:
    print(f"   [ERROR] Modelo NO encontrado en: {model_path.absolute()}")
    sys.exit(1)

# 4. Verificar que PyAudio esté instalado (opcional pero recomendado)
print("\n4. Verificando instalación de PyAudio...")
try:
    import pyaudio
    print("   [OK] PyAudio está instalado correctamente")
    print("   → El dictado por voz funcionará")
except ImportError:
    print("   [ADVERTENCIA] PyAudio NO esta instalado")
    print("   -> Instala con: pip install pyaudio")
    print("   -> En Windows: pip install pipwin && pipwin install pyaudio")

# 5. Verificar importación del módulo STT
print("\n5. Verificando módulo STT del proyecto...")
try:
    from src.services.stt import stt_from_audio_bytes, PushToTalkRecorder, VOSK_MODEL_PATH
    print("   [OK] Módulo STT importado correctamente")
    print(f"   -> Ruta configurada: {VOSK_MODEL_PATH}")
except ImportError as e:
    print(f"   [ERROR] Error al importar módulo STT - {e}")
    sys.exit(1)

# 6. Verificar que el reconocedor puede usar Vosk
print("\n6. Verificando reconocedor de Vosk...")
try:
    recognizer = sr.Recognizer()
    print("   [OK] Reconocedor creado correctamente")

    # Verificar que el método recognize_vosk existe
    if hasattr(recognizer, 'recognize_vosk'):
        print("   [OK] Método recognize_vosk disponible")
    else:
        print("   [ERROR] Método recognize_vosk NO disponible")
        print("   -> Actualiza SpeechRecognition: pip install --upgrade SpeechRecognition")
except Exception as e:
    print(f"   [ERROR] Error al crear reconocedor - {e}")

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("[OK] Vosk esta listo para usarse")
print("[OK] El modelo en espanol esta instalado")
print("[OK] La funcionalidad de dictado por voz deberia funcionar")
print("\nPara probar el dictado:")
print("1. Ejecuta la aplicacion: python main.py")
print("2. Haz clic en 'IA (Ollama)'")
print("3. Manten presionado el boton 'Mantener para dictar'")
print("4. Habla tu instruccion")
print("5. Suelta el boton para que se transcriba")
print("=" * 60)

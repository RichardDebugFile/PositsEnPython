#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de Speech-to-Text (dictado por voz)
"""

import json
from pathlib import Path
from tkinter import messagebox
from ..utils.logger import logger

# Ruta al modelo de Vosk
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOSK_MODEL_PATH = str(PROJECT_ROOT / "models" / "vosk-model-small-es-0.42")


def clean_user_freeform(text: str) -> str:
    """
    Limpia texto que podría venir como JSON {"text": "..."}
    Extrae el campo 'text' si es JSON, o devuelve el texto original
    """
    if not text:
        return ""

    s = text.strip()
    if s.startswith("{") and '"text"' in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"]).strip()
        except Exception:
            pass
    return s


def stt_from_audio_bytes(raw_bytes: bytes, sample_rate: int = 16000) -> tuple[str | None, str | None]:
    """
    Transcribe audio a texto usando motores locales (Faster-Whisper o Vosk).

    Args:
        raw_bytes: Bytes PCM del audio grabado
        sample_rate: Frecuencia de muestreo (16000 Hz recomendado)

    Returns:
        Tupla (texto_transcrito | None, mensaje_error | None)
    """
    try:
        import speech_recognition as sr
    except Exception:
        return None, "Instala SpeechRecognition: pip install SpeechRecognition"

    audio = sr.AudioData(raw_bytes, sample_rate, 2)
    r = sr.Recognizer()

    # 1) Intentar Faster-Whisper
    try:
        text = r.recognize_faster_whisper(audio, language="es")
        if text and text.strip():
            logger.debug(f"STT (Faster-Whisper): {text}")
            return text.strip(), None
    except Exception:
        pass

    # 2) Intentar Vosk
    try:
        # Verificar si existe el modelo
        if not Path(VOSK_MODEL_PATH).exists():
            error_msg = f"Modelo de Vosk no encontrado en: {VOSK_MODEL_PATH}\nDescarga el modelo desde https://alphacephei.com/vosk/models"
            return None, error_msg

        text = r.recognize_vosk(audio, language=VOSK_MODEL_PATH)
        if text is not None:
            cleaned = clean_user_freeform(text).strip()
            if cleaned:
                logger.debug(f"STT (Vosk): {cleaned}")
                return cleaned, None
    except Exception as e:
        error_msg = f"Error al transcribir con Vosk. Verifica que el modelo esté instalado en ./models/vosk-model-small-es-0.42\nError: {e}"
        logger.error(error_msg)
        return None, error_msg

    return None, "No fue posible transcribir con los motores locales (Faster-Whisper/Vosk)."


class PushToTalkRecorder:
    """
    Grabador de audio tipo push-to-talk (mantener presionado).
    Graba mientras el usuario mantiene presionado y retorna los bytes al soltar.
    """

    def __init__(self, owner_widget, rate: int = 16000, chunk: int = 2048):
        self.owner = owner_widget
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.running = False
        self.stream = None
        self.pa = None

    def start(self) -> bool:
        """Inicia la grabación"""
        try:
            import pyaudio
        except Exception:
            messagebox.showwarning(
                "Dictado",
                "Instala PyAudio (en Windows: pip install pipwin && pipwin install pyaudio)"
            )
            return False

        try:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            messagebox.showwarning("Dictado", f"No se pudo abrir el micrófono.\n{e}")
            return False

        self.frames = []
        self.running = True
        self._loop()
        return True

    def _loop(self):
        """Loop interno de grabación"""
        if not self.running:
            return
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
        except Exception:
            pass

        # Reprogramar
        self.owner.after(10, self._loop)

    def stop(self) -> bytes | None:
        """Detiene la grabación y retorna los bytes grabados"""
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pa:
                self.pa.terminate()
        except Exception:
            pass

        return b"".join(self.frames) if self.frames else None

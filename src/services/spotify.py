#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integración ligera con Spotify (sin credenciales ni Premium).

- Abre Spotify y reproduce una playlist mediante URIs `spotify:...`
  (usa el handler de protocolo del Spotify instalado).
- Controla la reproducción (play/pausa, siguiente, anterior) enviando las
  teclas de MEDIA del sistema, que Spotify obedece cuando está reproduciendo.
- Guarda una lista de playlists configurable por el usuario.
"""

import os
import re
import json
import ctypes
import shutil
import webbrowser
from pathlib import Path

from ..config import DATA_DIR
from ..utils.logger import logger

PLAYLISTS_FILE = DATA_DIR / "spotify_playlists.json"

# Playlists por defecto (editoriales de Spotify, IDs estables). El usuario
# puede añadir/quitar las suyas desde la app.
_DEFAULT_PLAYLISTS = [
    {"name": "Lofi Beats", "uri": "spotify:playlist:0vvXsWCC9xrXsKd4FyS8kM"},
    {"name": "Deep Focus", "uri": "spotify:playlist:37i9dQZF1DWZeKCadgRdKQ"},
    {"name": "Peaceful Piano", "uri": "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO"},
]

# Virtual-key codes de las teclas de media (Windows)
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_KEYEVENTF_KEYUP = 0x0002


def normalize_spotify_uri(text):
    """
    Convierte una URL o URI de Spotify a formato ``spotify:tipo:id``.
    Acepta ``spotify:...`` o ``https://open.spotify.com/tipo/id?...``.
    Devuelve None si no reconoce el formato.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if re.fullmatch(r"spotify:(playlist|track|album|artist):[A-Za-z0-9]+", text):
        return text
    match = re.search(
        r"open\.spotify\.com/(playlist|track|album|artist)/([A-Za-z0-9]+)", text
    )
    if match:
        return f"spotify:{match.group(1)}:{match.group(2)}"
    return None


def spotify_exe_path():
    """Ruta del ejecutable de Spotify si se encuentra (Windows), o None."""
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "Spotify" / "Spotify.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "Spotify.exe",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            continue
    return shutil.which("spotify")


def is_spotify_available():
    """True si parece que Spotify está instalado."""
    return spotify_exe_path() is not None


def open_uri(uri):
    """Abre una URI de Spotify (lanza Spotify y empieza a reproducir)."""
    if not uri:
        return False
    try:
        os.startfile(uri)  # Windows: ShellExecute maneja el protocolo spotify:
        logger.info(f"Spotify: abriendo {uri}")
        return True
    except Exception as e:
        logger.warning(f"No se pudo abrir Spotify con {uri}: {e}")
        # Fallback: web player en el navegador
        match = re.match(r"spotify:(playlist|track|album|artist):([A-Za-z0-9]+)", uri)
        if match:
            try:
                webbrowser.open(
                    f"https://open.spotify.com/{match.group(1)}/{match.group(2)}"
                )
                return True
            except Exception:
                pass
        return False


def _send_media_key(vk):
    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        return True
    except Exception as e:
        logger.warning(f"No se pudo enviar la tecla de media: {e}")
        return False


def play_pause():
    """Alterna reproducir/pausar en el reproductor de media activo (Spotify)."""
    return _send_media_key(_VK_MEDIA_PLAY_PAUSE)


def next_track():
    """Siguiente pista."""
    return _send_media_key(_VK_MEDIA_NEXT_TRACK)


def previous_track():
    """Pista anterior."""
    return _send_media_key(_VK_MEDIA_PREV_TRACK)


def load_playlists():
    """Carga las playlists guardadas, o las de por defecto."""
    try:
        if PLAYLISTS_FILE.exists():
            with open(PLAYLISTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return [p for p in data if p.get("name") and p.get("uri")]
    except Exception as e:
        logger.warning(f"No se pudieron cargar playlists de Spotify: {e}")
    return list(_DEFAULT_PLAYLISTS)


def save_playlists(playlists):
    """Guarda la lista de playlists."""
    try:
        with open(PLAYLISTS_FILE, "w", encoding="utf-8") as f:
            json.dump(playlists, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning(f"No se pudieron guardar playlists de Spotify: {e}")
        return False

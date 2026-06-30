#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades (sin UI) para el flujo "Capturar" del panel de música.

Reúne helpers puros y de sistema, separados de la interfaz para poder
testearlos de forma aislada:

- sanitize_youtube_url: normaliza una URL de YouTube dejando solo el video ID.
- clean_music_title:    limpia un título para mejorar la búsqueda en YouTube.
- get_windows_media_info: lee la sesión de medios global de Windows (WinRT).

Acompaña a ``cdp_helper`` y ``audio_detection`` dentro de ``utils``.
"""

import re
import json
import subprocess
from urllib.parse import urlparse, parse_qs

from .logger import logger


# Formatos de URL de YouTube de los que se puede extraer el video ID (11 chars).
_YOUTUBE_ID_PATTERNS = [
    r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
    r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
]

_YOUTUBE_HOSTS = ['www.youtube.com', 'youtube.com', 'm.youtube.com']


def sanitize_youtube_url(url):
    """
    Limpia una URL de YouTube eliminando parámetros innecesarios.
    Mantiene solo el video ID para evitar errores con playlists, índices, etc.

    Ejemplos:
    - https://www.youtube.com/watch?v=H2kUfHhAL3M&list=RDMM&index=2
      → https://www.youtube.com/watch?v=H2kUfHhAL3M
    - https://youtu.be/H2kUfHhAL3M?si=xyz123
      → https://youtu.be/H2kUfHhAL3M
    """
    if not url or not isinstance(url, str):
        return url

    url = url.strip()

    # Si no es una URL de YouTube, devolverla tal cual
    if not ("youtube.com" in url or "youtu.be" in url):
        return url

    try:
        video_id = None
        for pattern in _YOUTUBE_ID_PATTERNS:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break

        if video_id:
            # Retornar URL limpia con solo el video ID
            return f"https://www.youtube.com/watch?v={video_id}"

        # Si no se pudo extraer el ID, intentar con parse_qs
        parsed = urlparse(url)
        if parsed.hostname in _YOUTUBE_HOSTS:
            params = parse_qs(parsed.query)
            if 'v' in params:
                video_id = params['v'][0]
                return f"https://www.youtube.com/watch?v={video_id}"

        # Si todo falla, devolver URL original
        return url

    except Exception as e:
        logger.warning(f"Error sanitizando URL: {e}")
        return url


def clean_music_title(text: str) -> str:
    """
    Limpia el título de música para mejorar la búsqueda en YouTube.
    Elimina nombres de canales, símbolos extra y texto innecesario.
    """
    if not text:
        return ""

    # Eliminar texto después de símbolos comunes de separación
    separators = ["|", "//", "【", "】", "[", "]", "(Official", "(official"]
    for sep in separators:
        if sep in text:
            text = text.split(sep)[0]

    # Si el título empieza con "Nombre - ", eliminar el nombre del canal
    # Ejemplo: "Sheer's HD Video & Music - ZERO" -> "ZERO"
    if " - " in text:
        parts = text.split(" - ", 1)
        # Si la primera parte parece un nombre de canal (más de 20 caracteres o contiene palabras clave)
        channel_keywords = ["video", "music", "official", "records", "entertainment", "media", "hd", "vevo"]
        if len(parts[0]) > 20 or any(keyword in parts[0].lower() for keyword in channel_keywords):
            text = parts[1]

    # Eliminar sufijos comunes
    suffixes = [
        "- Topic", "- Official", "VEVO", "(Audio)", "(Video)", "(Lyrics)",
        "(Official Music Video)", "(Official Video)", "(Official Audio)"
    ]
    for suffix in suffixes:
        text = re.sub(re.escape(suffix), "", text, flags=re.IGNORECASE)

    # Limpiar espacios múltiples y caracteres extra
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(" -/|#")

    return text


# Script de PowerShell que consulta la sesión de medios global de Windows (WinRT).
_MEDIA_SESSION_PS = """
# Configurar salida UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

Function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

[Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager,Windows.Media.Control,ContentType=WindowsRuntime] | Out-Null
$sessionManager = Await ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])

$session = $sessionManager.GetCurrentSession()
if ($session) {
    $mediaProperties = Await ($session.TryGetMediaPropertiesAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties])

    $result = @{
        title = $mediaProperties.Title
        artist = $mediaProperties.Artist
        album = $mediaProperties.AlbumTitle
    }

    $result | ConvertTo-Json -Compress
}
"""


def get_windows_media_info():
    """
    Obtiene información de la reproducción actual desde la Windows Media Session.

    Returns:
        dict con ``title``/``artist``/``album`` si hay una sesión activa con
        datos, o ``None`` si no hay reproducción, PowerShell falla o no existe.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _MEDIA_SESSION_PS],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        if result.returncode == 0 and result.stdout.strip():
            media_info = json.loads(result.stdout.strip())

            # Verificar que tenga datos válidos
            if media_info.get("title") or media_info.get("artist"):
                return media_info
            return None
        return None

    except subprocess.TimeoutExpired:
        logger.warning("Timeout al obtener información de media")
        return None
    except json.JSONDecodeError:
        logger.warning("Error al parsear respuesta de PowerShell")
        return None
    except FileNotFoundError:
        logger.warning("No se pudo encontrar PowerShell en el sistema")
        return None
    except Exception as e:
        logger.warning(f"Error obteniendo media info: {e}")
        return None

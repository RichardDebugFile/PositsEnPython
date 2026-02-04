#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de descarga de música desde YouTube
Integrado con el reproductor de música de Posits Virtuales
"""

import os
import re
import threading
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Callable
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadCancelled

from ..utils.logger import logger


class YouTubeDownloader:
    """
    Descargador de música desde YouTube con integración al music player
    """

    def __init__(self, output_dir: str = "data/music"):
        self.output_dir = output_dir
        self.cancel_event = threading.Event()
        self.is_downloading = False
        self.current_thread: Optional[threading.Thread] = None
        self.downloaded_file = None  # Guardar el archivo descargado

        # Crear carpeta de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)

        # Callbacks
        self.on_progress: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitiza el nombre del archivo eliminando caracteres problemáticos
        """
        # Eliminar o reemplazar caracteres no permitidos en Windows
        # Caracteres prohibidos: < > : " / \ | ? *
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Reemplazar caracteres especiales problemáticos
        sanitized = sanitized.replace('—', '-')  # Em dash
        sanitized = sanitized.replace('–', '-')  # En dash
        sanitized = sanitized.replace('"', "'")  # Comillas dobles
        sanitized = sanitized.replace('"', "'")  # Comillas tipográficas

        # Eliminar espacios múltiples
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Limitar longitud (Windows tiene límite de 255 caracteres)
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200] + ext

        return sanitized.strip()

    def _get_ffmpeg_path(self) -> Optional[str]:
        """Intenta encontrar FFmpeg en el sistema"""
        return shutil.which("ffmpeg")

    def _sha256sum(self, path: str) -> str:
        """Calcula el hash SHA-256 de un archivo"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _format_bytes(self, n: Optional[int]) -> str:
        """Formatea bytes a formato legible"""
        if n is None:
            return "?"
        units = ["B", "KB", "MB", "GB"]
        i = 0
        n = float(n)
        while n >= 1024 and i < len(units) - 1:
            n /= 1024
            i += 1
        return f"{n:.2f} {units[i]}"

    def _format_eta(self, seconds: Optional[float]) -> str:
        """Formatea segundos a tiempo estimado"""
        if not seconds:
            return ""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def _progress_hook(self, d: dict):
        """Hook de progreso para yt-dlp"""
        if self.cancel_event.is_set():
            raise DownloadCancelled("Cancelado por el usuario")

        status = d.get("status")

        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            speed = d.get("speed")
            eta = d.get("eta")

            progress_info = {
                "status": "downloading",
                "downloaded": downloaded,
                "total": total,
                "speed": speed,
                "eta": eta,
                "percentage": (downloaded * 100.0 / total) if total else 0
            }

            if self.on_progress:
                self.on_progress(progress_info)

        elif status == "finished":
            if self.on_progress:
                self.on_progress({"status": "processing"})

    def download_audio(
        self,
        url: str,
        quality: str = "192",
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> bool:
        """
        Descarga audio desde YouTube y lo convierte a MP3

        Args:
            url: URL del video de YouTube
            quality: Calidad de audio en kbps (128, 192, 256, 320)
            on_progress: Callback para actualizaciones de progreso
            on_complete: Callback cuando se completa la descarga
            on_error: Callback cuando hay un error

        Returns:
            True si la descarga se inició correctamente
        """
        if self.is_downloading:
            logger.warning("Ya hay una descarga en progreso")
            return False

        # Configurar callbacks
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        # Iniciar descarga en thread separado
        self.current_thread = threading.Thread(
            target=self._download_thread,
            args=(url, quality),
            daemon=True
        )
        self.current_thread.start()
        return True

    def _download_thread(self, url: str, quality: str):
        """Thread de descarga"""
        self.is_downloading = True
        self.cancel_event.clear()
        self.downloaded_file = None

        try:
            # Primero obtener info del video
            with YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                original_title = info['title']
                sanitized_title = self._sanitize_filename(original_title)
                logger.info(f"Título original: {original_title}")
                logger.info(f"Título sanitizado: {sanitized_title}")

            # Configurar opciones de yt-dlp con nombre sanitizado
            output_template = os.path.join(self.output_dir, f"{sanitized_title}.%(ext)s")

            ydl_opts = {
                "outtmpl": output_template,
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": quality,
                }],
                "progress_hooks": [self._progress_hook],
                "quiet": True,
                "no_warnings": True,
                "prefer_ffmpeg": True,
                "keepvideo": False,
                "retries": 10,
                "fragment_retries": 10,
                "restrictfilenames": False,  # No restringir caracteres (ya lo hacemos nosotros)
                # Configuración para evitar bloqueos de YouTube
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "web"],  # Usar cliente Android primero, luego web
                        "skip": ["hls", "dash"]  # Evitar formatos problemáticos
                    }
                },
                # Headers adicionales para evitar detección
                "http_headers": {
                    "User-Agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            }

            # Verificar FFmpeg
            ffmpeg_path = self._get_ffmpeg_path()
            if ffmpeg_path:
                logger.info(f"FFmpeg encontrado en: {ffmpeg_path}")
                ydl_opts["ffmpeg_location"] = ffmpeg_path
            else:
                logger.warning("FFmpeg no encontrado, la conversión puede fallar")

            # Descargar
            logger.info(f"Iniciando descarga: {url}")
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

            # Buscar el archivo MP3 descargado
            expected_filepath = os.path.join(self.output_dir, f"{sanitized_title}.mp3")
            actual_filepath = None

            # Si el archivo esperado no existe, buscar en la carpeta por el más reciente
            if not os.path.exists(expected_filepath):
                logger.warning(f"Archivo esperado no encontrado: {expected_filepath}")
                logger.info("Buscando archivo MP3 más reciente en la carpeta...")

                # Listar archivos MP3 en la carpeta
                mp3_files = [f for f in Path(self.output_dir).glob("*.mp3")]
                if mp3_files:
                    # Obtener el más reciente
                    actual_filepath = str(max(mp3_files, key=lambda p: p.stat().st_mtime))
                    logger.info(f"Archivo encontrado: {actual_filepath}")
                else:
                    raise FileNotFoundError(f"No se encontró ningún archivo MP3 en {self.output_dir}")
            else:
                actual_filepath = expected_filepath

            # Verificar que el archivo existe
            if not os.path.exists(actual_filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {actual_filepath}")

            # Si el nombre del archivo tiene caracteres problemáticos, renombrarlo
            actual_filename = os.path.basename(actual_filepath)
            sanitized_filename = self._sanitize_filename(actual_filename)

            if actual_filename != sanitized_filename:
                logger.info(f"Renombrando archivo con caracteres problemáticos...")
                logger.info(f"  De: {actual_filename}")
                logger.info(f"  A:  {sanitized_filename}")

                new_filepath = os.path.join(self.output_dir, sanitized_filename)

                # Si ya existe un archivo con el nombre sanitizado, agregar número
                if os.path.exists(new_filepath):
                    base, ext = os.path.splitext(sanitized_filename)
                    counter = 1
                    while os.path.exists(os.path.join(self.output_dir, f"{base}_{counter}{ext}")):
                        counter += 1
                    new_filepath = os.path.join(self.output_dir, f"{base}_{counter}{ext}")
                    logger.info(f"  Archivo ya existe, usando: {os.path.basename(new_filepath)}")

                try:
                    os.rename(actual_filepath, new_filepath)
                    actual_filepath = new_filepath
                    logger.info("Archivo renombrado exitosamente")
                except Exception as e:
                    logger.error(f"Error al renombrar archivo: {e}")
                    # Continuar con el nombre original si falla el renombrado

            # Guardar la ruta del archivo descargado (renombrado si fue necesario)
            self.downloaded_file = actual_filepath

            # Calcular hash y tamaño
            size = os.path.getsize(actual_filepath)
            sha = self._sha256sum(actual_filepath)

            filename = os.path.basename(actual_filepath)
            logger.info(f"Descarga completada: {filename}")
            logger.info(f"Tamaño: {self._format_bytes(size)}")
            logger.info(f"SHA-256: {sha}")

            # Callback de completado
            if self.on_complete:
                self.on_complete({
                    "filename": filename,
                    "filepath": actual_filepath,
                    "size": size,
                    "sha256": sha,
                    "title": os.path.splitext(filename)[0],  # Título sin extensión
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader")
                })

        except DownloadCancelled:
            logger.info("Descarga cancelada por el usuario")
            if self.on_error:
                self.on_error("Descarga cancelada")

        except Exception as e:
            logger.error(f"Error en descarga: {e}", exc_info=True)
            if self.on_error:
                self.on_error(str(e))

        finally:
            self.is_downloading = False
            self.cancel_event.clear()

    def cancel(self):
        """Cancela la descarga actual"""
        if self.is_downloading:
            logger.info("Cancelando descarga...")
            self.cancel_event.set()

    def get_video_info(self, url: str) -> Optional[dict]:
        """
        Obtiene información del video sin descargarlo

        Returns:
            Diccionario con info del video o None si hay error
        """
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                return {
                    "title": info.get("title"),
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader"),
                    "thumbnail": info.get("thumbnail"),
                    "description": info.get("description"),
                    "view_count": info.get("view_count"),
                }

        except Exception as e:
            logger.error(f"Error obteniendo info del video: {e}")
            return None

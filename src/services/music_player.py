#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de reproducción de música de fondo
Soporta MP3, WAV, OGG, FLAC con playlist
Los archivos se copian a data/music/ para independencia de ubicación
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List, Callable
import pygame

logger = logging.getLogger(__name__)


class MusicPlayer:
    """
    Gestor de reproducción de música de fondo con playlist y persistencia
    """

    def __init__(self, config_file: str = "data/music_config.json", music_dir: str = "data/music"):
        self.config_file = config_file
        self.music_dir = music_dir
        self.playlist: List[str] = []  # Lista de archivos en la playlist
        self.current_index: int = 0  # Índice de la canción actual
        self.main_track: Optional[str] = None  # Canción principal (loop infinito)
        self.mode: str = "main"  # "main" o "playlist"
        self.volume: float = 0.5  # 0.0 a 1.0
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.on_track_change: Optional[Callable] = None  # Callback cuando cambia la canción

        # Crear carpeta de música si no existe
        os.makedirs(self.music_dir, exist_ok=True)

        # Inicializar pygame mixer
        try:
            pygame.mixer.init()
            # Configurar evento cuando termina una canción
            pygame.mixer.music.set_endevent(pygame.USEREVENT + 1)
            print("[INFO] Reproductor de musica inicializado")
        except Exception as e:
            print(f"[ERROR] No se pudo inicializar pygame mixer: {e}")

        # Cargar configuración guardada
        self.load_config()

        # Escanear carpeta de música para agregar archivos nuevos
        self.scan_music_folder()

    def load_config(self):
        """Carga la configuración de música guardada"""
        if not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Cargar playlist y filtrar archivos que ya no existen
            loaded_playlist = config.get("playlist", [])
            self.playlist = []
            removed_count = 0

            for file_path in loaded_playlist:
                if os.path.exists(file_path):
                    self.playlist.append(file_path)
                else:
                    logger.warning(f"Archivo en playlist no encontrado, removiendo: {file_path}")
                    removed_count += 1

            if removed_count > 0:
                logger.info(f"Se removieron {removed_count} archivo(s) inexistente(s) de la playlist")

            self.current_index = config.get("current_index", 0)
            self.main_track = config.get("main_track", None)

            # Validar que main_track exista
            if self.main_track and not os.path.exists(self.main_track):
                logger.warning(f"Main track no encontrado, reseteando: {self.main_track}")
                self.main_track = None

            self.mode = config.get("mode", "main")
            self.volume = config.get("volume", 0.5)
            was_playing = config.get("was_playing", False)

            # Validar que el índice esté dentro del rango
            if self.current_index >= len(self.playlist):
                self.current_index = 0

            # Restaurar volumen
            pygame.mixer.music.set_volume(self.volume)

            # Si estaba reproduciéndose, reiniciar automáticamente
            if was_playing:
                if self.mode == "main" and self.main_track and os.path.exists(self.main_track):
                    self._play_main_track(auto_resume=True)
                elif self.mode == "playlist" and self.playlist and self.current_index < len(self.playlist):
                    current_file = self.playlist[self.current_index]
                    if os.path.exists(current_file):
                        self._play_track(self.current_index, auto_resume=True)

        except Exception as e:
            print(f"[WARNING] Error al cargar configuracion de musica: {e}")

    def save_config(self):
        """Guarda la configuración actual de música"""
        config = {
            "playlist": self.playlist,
            "current_index": self.current_index,
            "main_track": self.main_track,
            "mode": self.mode,
            "volume": self.volume,
            "was_playing": self.is_playing and not self.is_paused
        }

        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] No se pudo guardar configuracion de musica: {e}")

    def _play_track(self, index: int, auto_resume: bool = False):
        """
        Reproduce una canción de la playlist por índice (método interno)

        Args:
            index: Índice de la canción en la playlist
            auto_resume: Si es True, está reanudando automáticamente
        """
        if not self.playlist or index >= len(self.playlist):
            return False

        file_path = self.playlist[index]
        if not os.path.exists(file_path):
            if not auto_resume:
                print(f"[ERROR] Archivo no encontrado: {file_path}")
            return False

        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play(0)  # 0 = una sola vez
            self.current_index = index
            self.mode = "playlist"
            self.is_playing = True
            self.is_paused = False
            self.save_config()

            # Notificar cambio de canción
            if self.on_track_change:
                self.on_track_change()

            return True
        except Exception as e:
            if not auto_resume:
                print(f"[ERROR] No se pudo reproducir el archivo: {e}")
            return False

    def _play_main_track(self, auto_resume: bool = False):
        """
        Reproduce la canción principal en loop infinito (método interno)

        Args:
            auto_resume: Si es True, está reanudando automáticamente
        """
        if not self.main_track or not os.path.exists(self.main_track):
            return False

        try:
            pygame.mixer.music.load(self.main_track)
            pygame.mixer.music.play(-1)  # -1 = loop infinito
            self.mode = "main"
            self.is_playing = True
            self.is_paused = False
            self.save_config()

            # Notificar cambio de canción
            if self.on_track_change:
                self.on_track_change()

            return True
        except Exception as e:
            if not auto_resume:
                print(f"[ERROR] No se pudo reproducir la cancion principal: {e}")
            return False

    def play(self):
        """Reproduce la música según el modo actual"""
        if self.mode == "main":
            if not self.main_track:
                print("[WARNING] No hay cancion principal configurada")
                return False
            return self._play_main_track()
        else:  # playlist mode
            if not self.playlist:
                print("[WARNING] No hay canciones en la playlist")
                return False
            return self._play_track(self.current_index)

    def pause(self):
        """Pausa la reproducción"""
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.save_config()
            print("[INFO] Música pausada")

    def resume(self):
        """Reanuda la reproducción después de pausar"""
        if self.is_playing and self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.save_config()
            print("[INFO] Música reanudada")

    def stop(self):
        """Detiene la reproducción completamente"""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.save_config()
        print("[INFO] Música detenida")

    def set_volume(self, volume: float):
        """
        Ajusta el volumen

        Args:
            volume: Volumen de 0.0 (silencio) a 1.0 (máximo)
        """
        self.volume = max(0.0, min(1.0, volume))  # Clamp entre 0 y 1
        pygame.mixer.music.set_volume(self.volume)
        self.save_config()
        print(f"[INFO] Volumen ajustado a {int(self.volume * 100)}%")

    def get_volume(self) -> float:
        """Retorna el volumen actual (0.0 a 1.0)"""
        return self.volume

    def get_current_file(self) -> Optional[str]:
        """Retorna la ruta del archivo actual"""
        if self.playlist and self.current_index < len(self.playlist):
            return self.playlist[self.current_index]
        return None

    def get_current_filename(self) -> str:
        """Retorna el nombre del archivo actual (sin ruta)"""
        current = self.get_current_file()
        if current:
            return Path(current).name
        return "Sin música"

    def is_music_playing(self) -> bool:
        """Retorna True si hay música reproduciéndose"""
        return self.is_playing and not self.is_paused

    def toggle_pause(self):
        """Alterna entre pausa y reproducción"""
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    # ======================== MÉTODOS DE PLAYLIST ========================

    def _copy_music_file(self, source_path: str) -> Optional[str]:
        """
        Copia un archivo de música a la carpeta local data/music/

        Args:
            source_path: Ruta original del archivo

        Returns:
            Ruta del archivo copiado, o None si hubo error
        """
        try:
            source = Path(source_path)
            if not source.exists():
                return None

            # Crear nombre único si ya existe
            dest_name = source.name
            dest_path = Path(self.music_dir) / dest_name

            # Si ya existe el archivo con el mismo nombre, agregar número
            counter = 1
            while dest_path.exists():
                stem = source.stem
                suffix = source.suffix
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = Path(self.music_dir) / dest_name
                counter += 1

            # Copiar archivo
            shutil.copy2(source, dest_path)
            print(f"[INFO] Archivo copiado: {dest_name}")
            return str(dest_path)

        except Exception as e:
            print(f"[ERROR] No se pudo copiar el archivo: {e}")
            return None

    def add_to_playlist(self, file_path: str) -> bool:
        """
        Agrega una canción a la playlist (copiándola a data/music/)

        Args:
            file_path: Ruta al archivo de audio original

        Returns:
            True si se agregó correctamente
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] Archivo no encontrado: {file_path}")
            return False

        # Verificar si ya existe por nombre (no por ruta completa)
        filename = Path(file_path).name
        for existing in self.playlist:
            if Path(existing).name == filename:
                print(f"[WARNING] Ya existe una cancion con ese nombre en la playlist")
                return False

        # Copiar archivo a carpeta local
        copied_path = self._copy_music_file(file_path)
        if not copied_path:
            return False

        self.playlist.append(copied_path)
        self.save_config()
        print(f"[INFO] Agregado a playlist: {Path(copied_path).name}")
        return True

    def remove_from_playlist(self, index: int) -> bool:
        """
        Elimina una canción de la playlist por índice y borra el archivo físico

        Args:
            index: Índice de la canción a eliminar

        Returns:
            True si se eliminó correctamente
        """
        if index < 0 or index >= len(self.playlist):
            print(f"[ERROR] Indice fuera de rango: {index}")
            return False

        removed = self.playlist.pop(index)
        print(f"[INFO] Eliminado de playlist: {Path(removed).name}")

        # Eliminar archivo físico si está en la carpeta de música
        try:
            removed_path = Path(removed)
            if removed_path.exists() and str(removed_path.parent) == str(Path(self.music_dir).absolute()):
                # Solo eliminar si no está siendo usado como main_track
                if self.main_track != removed:
                    removed_path.unlink()
                    print(f"[INFO] Archivo eliminado: {removed_path.name}")
                else:
                    print(f"[WARNING] No se elimino el archivo porque es la cancion principal")
        except Exception as e:
            print(f"[WARNING] No se pudo eliminar el archivo: {e}")

        # Ajustar el índice actual si es necesario
        if self.current_index >= len(self.playlist) and self.playlist:
            self.current_index = len(self.playlist) - 1
        elif not self.playlist:
            self.current_index = 0
            self.stop()

        self.save_config()
        return True

    def next_track(self) -> bool:
        """
        Salta a la siguiente canción en la playlist

        Returns:
            True si se cambió de canción
        """
        if not self.playlist:
            return False

        self.current_index = (self.current_index + 1) % len(self.playlist)
        return self._play_track(self.current_index)

    def previous_track(self) -> bool:
        """
        Vuelve a la canción anterior en la playlist

        Returns:
            True si se cambió de canción
        """
        if not self.playlist:
            return False

        self.current_index = (self.current_index - 1) % len(self.playlist)
        return self._play_track(self.current_index)

    def play_track_at(self, index: int) -> bool:
        """
        Reproduce una canción específica de la playlist

        Args:
            index: Índice de la canción a reproducir

        Returns:
            True si se reprodujo correctamente
        """
        if index < 0 or index >= len(self.playlist):
            return False

        return self._play_track(index)

    def set_main_track(self, file_path: str) -> bool:
        """
        Establece una canción como canción principal (loop infinito)

        Args:
            file_path: Ruta al archivo de audio

        Returns:
            True si se estableció correctamente
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] Archivo no encontrado: {file_path}")
            return False

        self.main_track = file_path
        self.save_config()
        print(f"[INFO] Cancion principal establecida: {Path(file_path).name}")
        return True

    def play_main_track(self) -> bool:
        """Cambia al modo principal y reproduce la canción principal"""
        was_playing = self.is_music_playing()
        if was_playing:
            self.stop()

        self.mode = "main"
        if self.main_track:
            return self._play_main_track()
        else:
            print("[WARNING] No hay cancion principal configurada")
            return False

    def play_playlist(self) -> bool:
        """Cambia al modo playlist y reproduce la playlist"""
        was_playing = self.is_music_playing()
        if was_playing:
            self.stop()

        self.mode = "playlist"
        if self.playlist:
            return self.play()
        else:
            print("[WARNING] No hay canciones en la playlist")
            return False

    def get_mode(self) -> str:
        """Retorna el modo actual ('main' o 'playlist')"""
        return self.mode

    def get_main_track(self) -> Optional[str]:
        """Retorna la ruta de la canción principal"""
        return self.main_track

    def get_main_track_filename(self) -> str:
        """Retorna el nombre de la canción principal (sin ruta)"""
        if self.main_track:
            return Path(self.main_track).name
        return "Sin canción principal"

    def get_playlist(self) -> List[str]:
        """Retorna la lista completa de canciones"""
        return self.playlist.copy()

    def get_playlist_filenames(self) -> List[str]:
        """Retorna los nombres de archivo de la playlist (sin rutas)"""
        return [Path(f).name for f in self.playlist]

    def get_current_index(self) -> int:
        """Retorna el índice de la canción actual"""
        return self.current_index

    def clear_playlist(self):
        """Limpia toda la playlist"""
        self.stop()
        self.playlist.clear()
        self.current_index = 0
        self.save_config()
        print("[INFO] Playlist limpiada")

    def scan_music_folder(self):
        """
        Escanea la carpeta de música y agrega todos los archivos de audio
        que no estén ya en la playlist

        Returns:
            Número de archivos nuevos agregados
        """
        added_count = 0

        try:
            # Extensiones de audio soportadas
            audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}

            # Obtener lista de archivos actuales en playlist (solo nombres)
            current_filenames = {Path(f).name for f in self.playlist}

            # Escanear carpeta de música
            music_path = Path(self.music_dir)
            if not music_path.exists():
                return 0

            # Buscar archivos de audio
            for file_path in music_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    # Solo agregar si no está ya en la playlist
                    if file_path.name not in current_filenames:
                        self.playlist.append(str(file_path))
                        added_count += 1
                        logger.info(f"Agregado a playlist: {file_path.name}")

            if added_count > 0:
                self.save_config()
                logger.info(f"Se agregaron {added_count} archivo(s) nuevos a la playlist")

        except Exception as e:
            logger.error(f"Error al escanear carpeta de música: {e}")

        return added_count

    def check_and_play_next(self):
        """
        Verifica si la canción actual terminó y reproduce la siguiente.
        Solo funciona en modo playlist (en modo main la canción hace loop infinito).
        Este método debe ser llamado periódicamente desde la UI.
        """
        if self.mode == "playlist" and self.is_playing and not self.is_paused:
            if not pygame.mixer.music.get_busy():
                # La canción terminó, reproducir la siguiente
                self.next_track()

    def cleanup(self):
        """Limpia recursos al cerrar"""
        self.save_config()
        pygame.mixer.music.stop()
        pygame.mixer.quit()

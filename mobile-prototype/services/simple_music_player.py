"""
Reproductor de música simple para Pomodoro usando Kivy Audio
"""

from pathlib import Path
from typing import List, Optional
import random


class SimpleMusicPlayer:
    """Reproductor de música simple para modo Pomodoro"""

    def __init__(self, music_dir: str = "../data/music"):
        self.music_dir = Path(music_dir)
        self.playlist: List[Path] = []
        self.current_index: int = 0
        self.current_sound = None
        self.is_playing: bool = False
        self.volume: float = 0.3

        # Escanear carpeta de música
        self.scan_music_folder()

    def scan_music_folder(self):
        """Escanea la carpeta de música para encontrar archivos MP3"""
        if not self.music_dir.exists():
            print(f"[WARN] Carpeta de música no encontrada: {self.music_dir}")
            return

        # Buscar archivos MP3
        mp3_files = list(self.music_dir.glob("*.mp3"))
        self.playlist = sorted(mp3_files)

        print(f"[INFO] Encontradas {len(self.playlist)} canciones")

    def get_playlist_names(self) -> List[str]:
        """Devuelve los nombres de las canciones en la playlist"""
        return [track.stem for track in self.playlist]

    def play(self, index: Optional[int] = None):
        """Reproduce una canción por índice"""
        if not self.playlist:
            print("[WARN] No hay canciones en la playlist")
            return

        if index is not None:
            self.current_index = index % len(self.playlist)

        try:
            # Intentar importar pygame como alternativa
            try:
                import pygame.mixer as mixer

                # Inicializar si no está inicializado
                if not mixer.get_init():
                    mixer.init()

                # Detener canción actual si existe
                if mixer.music.get_busy():
                    mixer.music.stop()

                # Cargar y reproducir nueva canción
                track_path = str(self.playlist[self.current_index])
                mixer.music.load(track_path)
                mixer.music.set_volume(self.volume)
                mixer.music.play(-1)  # -1 = loop infinito
                self.is_playing = True
                print(f"[INFO] Reproduciendo: {self.playlist[self.current_index].name}")

            except ImportError:
                print("[WARN] pygame no disponible, reproductor de música deshabilitado")
                print("[INFO] Instala pygame: pip install pygame")
                self.is_playing = False

        except Exception as e:
            print(f"[ERROR] Error reproduciendo música: {e}")

    def stop(self):
        """Detiene la reproducción"""
        try:
            import pygame.mixer as mixer
            if mixer.get_init() and mixer.music.get_busy():
                mixer.music.stop()
            self.is_playing = False
        except:
            pass

    def pause(self):
        """Pausa la reproducción"""
        try:
            import pygame.mixer as mixer
            if mixer.get_init() and mixer.music.get_busy():
                mixer.music.pause()
            self.is_playing = False
        except:
            pass

    def resume(self):
        """Reanuda la reproducción"""
        try:
            import pygame.mixer as mixer
            if mixer.get_init():
                mixer.music.unpause()
            self.is_playing = True
        except:
            pass

    def next_track(self):
        """Siguiente canción"""
        if not self.playlist:
            return

        self.current_index = (self.current_index + 1) % len(self.playlist)
        self.play()

    def previous_track(self):
        """Canción anterior"""
        if not self.playlist:
            return

        self.current_index = (self.current_index - 1) % len(self.playlist)
        self.play()

    def set_volume(self, volume: float):
        """Ajusta el volumen (0.0 a 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        try:
            import pygame.mixer as mixer
            if mixer.get_init():
                mixer.music.set_volume(self.volume)
        except:
            pass

    def get_current_track_name(self) -> str:
        """Devuelve el nombre de la canción actual"""
        if self.playlist and 0 <= self.current_index < len(self.playlist):
            return self.playlist[self.current_index].stem
        return "Sin cancion"

    def shuffle(self):
        """Mezcla aleatoriamente la playlist"""
        if self.playlist:
            random.shuffle(self.playlist)
            self.current_index = 0

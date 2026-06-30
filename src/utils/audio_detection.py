"""
Módulo para detectar procesos que están reproduciendo audio en Windows.
Utiliza la API Core Audio de Windows a través de la librería pycaw.
"""

import logging
from typing import Optional, Dict, List
from pycaw.pycaw import AudioUtilities

logger = logging.getLogger(__name__)


class AudioProcessDetector:
    """
    Detector de procesos que están reproduciendo audio activamente.
    """

    # Navegadores soportados (nombres de procesos en minúsculas)
    SUPPORTED_BROWSERS = ["chrome", "firefox", "msedge", "brave", "opera"]

    @staticmethod
    def get_active_audio_sessions() -> List[Dict]:
        """
        Obtiene todas las sesiones de audio activas en el sistema.

        Returns:
            Lista de diccionarios con información de cada sesión:
            {
                'pid': int,               # Process ID
                'name': str,              # Nombre del proceso (ej: 'chrome.exe')
                'volume': float,          # Volumen (0.0 - 1.0)
                'state': str              # Estado de la sesión
            }
        """
        sessions_info = []

        try:
            sessions = AudioUtilities.GetAllSessions()

            for session in sessions:
                if session.Process:
                    session_data = {
                        'pid': session.Process.pid,
                        'name': session.Process.name(),
                        'volume': session.SimpleAudioVolume.GetMasterVolume(),
                        'state': session.State
                    }
                    sessions_info.append(session_data)

        except Exception as e:
            logger.error(f"Error obteniendo sesiones de audio: {e}")

        return sessions_info

    @staticmethod
    def get_browser_playing_audio() -> Optional[Dict]:
        """
        Detecta qué navegador está reproduciendo audio actualmente.

        Returns:
            Diccionario con información del navegador reproduciendo:
            {
                'pid': int,
                'name': str,
                'volume': float
            }

            None si no hay ningún navegador reproduciendo.
        """
        try:
            sessions = AudioUtilities.GetAllSessions()

            for session in sessions:
                if session.Process and session.Process.name():
                    process_name = session.Process.name().lower()

                    # Verificar si es un navegador soportado
                    for browser in AudioProcessDetector.SUPPORTED_BROWSERS:
                        if browser in process_name:
                            pid = session.Process.pid
                            volume = session.SimpleAudioVolume.GetMasterVolume()

                            logger.info(f"Navegador reproduciendo detectado: {process_name} (PID: {pid})")

                            return {
                                'pid': pid,
                                'name': session.Process.name(),
                                'volume': volume
                            }

            logger.warning("No se detectó ningún navegador reproduciendo audio")
            return None

        except Exception as e:
            logger.error(f"Error detectando navegador reproduciendo: {e}")
            return None

    @staticmethod
    def is_process_playing_audio(pid: int) -> bool:
        """
        Verifica si un proceso específico está reproduciendo audio.

        Args:
            pid: Process ID a verificar

        Returns:
            True si el proceso está reproduciendo audio, False en caso contrario.
        """
        try:
            sessions = AudioUtilities.GetAllSessions()

            for session in sessions:
                if session.Process and session.Process.pid == pid:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error verificando proceso {pid}: {e}")
            return False

    @staticmethod
    def get_all_browser_processes() -> List[Dict]:
        """
        Obtiene todos los procesos de navegadores que están reproduciendo audio.

        Returns:
            Lista de diccionarios con información de cada navegador:
            [
                {'pid': 12345, 'name': 'chrome.exe', 'volume': 0.8},
                {'pid': 67890, 'name': 'firefox.exe', 'volume': 1.0}
            ]
        """
        browser_processes = []

        try:
            sessions = AudioUtilities.GetAllSessions()

            for session in sessions:
                if session.Process and session.Process.name():
                    process_name = session.Process.name().lower()

                    # Verificar si es un navegador
                    for browser in AudioProcessDetector.SUPPORTED_BROWSERS:
                        if browser in process_name:
                            browser_processes.append({
                                'pid': session.Process.pid,
                                'name': session.Process.name(),
                                'volume': session.SimpleAudioVolume.GetMasterVolume()
                            })
                            break

        except Exception as e:
            logger.error(f"Error obteniendo procesos de navegadores: {e}")

        return browser_processes


# Función de conveniencia para uso rápido
def get_browser_playing_audio() -> Optional[Dict]:
    """
    Función helper para obtener el navegador que está reproduciendo audio.

    Returns:
        Diccionario con {'pid': int, 'name': str, 'volume': float} o None
    """
    return AudioProcessDetector.get_browser_playing_audio()

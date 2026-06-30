"""
Módulo para interactuar con navegadores usando Chrome DevTools Protocol (CDP).
Permite listar pestañas, detectar cuál está reproduciendo audio y activarlas.
"""

import logging
from typing import Optional, Dict, List
import pychrome

logger = logging.getLogger(__name__)


class CDPHelper:
    """
    Helper para Chrome DevTools Protocol.
    Permite controlar navegadores Chromium (Chrome, Brave, Edge, etc.)
    """

    DEFAULT_PORT = 9222
    DEFAULT_HOST = "127.0.0.1"

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Inicializa el helper CDP.

        Args:
            host: Host del servidor CDP (default: 127.0.0.1)
            port: Puerto del servidor CDP (default: 9222)
        """
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.browser = None

    def connect(self) -> bool:
        """
        Intenta conectar al navegador via CDP.

        Returns:
            True si la conexión fue exitosa, False en caso contrario.
        """
        try:
            self.browser = pychrome.Browser(url=self.url)
            # Verificar que podemos listar tabs
            _ = self.browser.list_tab()
            logger.info(f"✓ Conectado a CDP en {self.url}")
            return True
        except Exception as e:
            logger.warning(f"No se pudo conectar a CDP: {e}")
            self.browser = None
            return False

    def is_available(self) -> bool:
        """
        Verifica si CDP está disponible.

        Returns:
            True si CDP está disponible, False en caso contrario.
        """
        if self.browser:
            return True
        return self.connect()

    def list_all_tabs(self) -> List[Dict]:
        """
        Lista todas las pestañas abiertas en el navegador.

        Returns:
            Lista de diccionarios con información de cada pestaña:
            [
                {
                    'id': str,
                    'url': str,
                    'title': str,
                    'type': str
                },
                ...
            ]
        """
        if not self.browser:
            if not self.connect():
                return []

        try:
            tabs = self.browser.list_tab()
            result = []

            for tab in tabs:
                if hasattr(tab, 'type') and tab.type == 'page':
                    result.append({
                        'id': tab.id,
                        'url': tab.url,
                        'title': tab.title,
                        'type': tab.type,
                        '_tab_object': tab  # Guardar referencia al objeto original
                    })

            return result
        except Exception as e:
            logger.error(f"Error listando pestañas: {e}")
            return []

    def find_youtube_tabs(self) -> List[Dict]:
        """
        Encuentra todas las pestañas de YouTube (que contengan /watch en la URL).

        Returns:
            Lista de pestañas de YouTube.
        """
        all_tabs = self.list_all_tabs()
        youtube_tabs = [
            tab for tab in all_tabs
            if 'youtube.com/watch' in tab['url'].lower()
        ]

        logger.info(f"Encontradas {len(youtube_tabs)} pestaña(s) de YouTube")
        return youtube_tabs

    def check_if_tab_playing(self, tab_info: Dict) -> Optional[Dict]:
        """
        Verifica si una pestaña está reproduciendo audio/video.

        Args:
            tab_info: Diccionario con información de la pestaña

        Returns:
            Diccionario con información de reproducción si está activa:
            {
                'playing': bool,
                'currentTime': float,
                'duration': float,
                'title': str,
                'paused': bool
            }
            None si no está reproduciendo o hay error.
        """
        tab = tab_info.get('_tab_object')
        if not tab:
            return None

        try:
            # Iniciar sesión con la pestaña
            tab.start()

            # Habilitar Runtime para ejecutar JavaScript
            tab.call_method("Runtime.enable")

            # JavaScript para verificar estado del video
            js_code = """
            (function() {
                const video = document.querySelector('video');
                if (!video) {
                    return { error: 'No video element found' };
                }

                return {
                    playing: !video.paused && !video.ended && video.readyState > 2,
                    paused: video.paused,
                    ended: video.ended,
                    currentTime: video.currentTime,
                    duration: video.duration,
                    volume: video.volume,
                    muted: video.muted,
                    title: document.title
                };
            })()
            """

            # Ejecutar JavaScript
            result = tab.call_method("Runtime.evaluate", expression=js_code)

            # Procesar resultado
            if result.get('result', {}).get('type') == 'object':
                value = result['result'].get('value', {})

                # Verificar si hay error
                if 'error' in value:
                    logger.debug(f"No hay video en pestaña: {tab_info['url']}")
                    return None

                # Verificar si está reproduciendo
                if value.get('playing'):
                    logger.info(f"✓ Pestaña reproduciendo encontrada: {value['title'][:50]}...")
                    return value

            return None

        except Exception as e:
            logger.debug(f"Error verificando pestaña: {e}")
            return None
        finally:
            try:
                tab.stop()
            except:
                pass

    def find_playing_youtube_tab(self) -> Optional[Dict]:
        """
        Encuentra la pestaña de YouTube que está reproduciendo audio/video.

        Returns:
            Diccionario con información de la pestaña reproduciendo:
            {
                'id': str,
                'url': str,
                'title': str,
                'playing_info': {...}
            }
            None si no se encontró ninguna pestaña reproduciendo.
        """
        youtube_tabs = self.find_youtube_tabs()

        if not youtube_tabs:
            logger.warning("No hay pestañas de YouTube abiertas")
            return None

        logger.info(f"Verificando {len(youtube_tabs)} pestaña(s) de YouTube...")

        for tab_info in youtube_tabs:
            playing_info = self.check_if_tab_playing(tab_info)

            if playing_info and playing_info.get('playing'):
                tab_info['playing_info'] = playing_info
                return tab_info

        logger.warning("No se encontró ninguna pestaña de YouTube reproduciendo")
        return None

    def activate_tab(self, tab_info: Dict) -> bool:
        """
        Activa una pestaña específica (la trae al frente).

        Args:
            tab_info: Diccionario con información de la pestaña

        Returns:
            True si se activó correctamente, False en caso contrario.
        """
        if not self.browser:
            return False

        tab = tab_info.get('_tab_object')
        if not tab:
            return False

        try:
            self.browser.activate_tab(tab)
            logger.info(f"✓ Pestaña activada: {tab_info['title'][:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error activando pestaña: {e}")
            return False

    def get_youtube_playing_url(self) -> Optional[str]:
        """
        Obtiene la URL de la pestaña de YouTube que está reproduciendo.
        Función de conveniencia que combina find_playing_youtube_tab() y activate_tab().

        Returns:
            URL de la pestaña reproduciendo, o None si no se encontró.
        """
        playing_tab = self.find_playing_youtube_tab()

        if not playing_tab:
            return None

        # Activar la pestaña
        self.activate_tab(playing_tab)

        return playing_tab['url']


# Funciones de conveniencia para uso rápido
def is_cdp_available(host: str = "127.0.0.1", port: int = 9222) -> bool:
    """
    Verifica si CDP está disponible en el host y puerto especificados.

    Args:
        host: Host del servidor CDP
        port: Puerto del servidor CDP

    Returns:
        True si CDP está disponible, False en caso contrario.
    """
    helper = CDPHelper(host, port)
    return helper.is_available()


def get_youtube_url_via_cdp(host: str = "127.0.0.1", port: int = 9222) -> Optional[str]:
    """
    Obtiene la URL de YouTube que está reproduciendo usando CDP.

    Args:
        host: Host del servidor CDP
        port: Puerto del servidor CDP

    Returns:
        URL de YouTube reproduciendo, o None si no se encontró o CDP no está disponible.
    """
    helper = CDPHelper(host, port)

    if not helper.connect():
        return None

    return helper.get_youtube_playing_url()

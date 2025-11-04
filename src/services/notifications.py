#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de notificaciones del sistema
Monitorea tareas próximas a vencer y muestra alertas
"""

import threading
import time
from datetime import date
from typing import Callable, Optional

from ..config import NOTIFICATION_CONFIG
from ..utils.dates import today_date, days_until
from ..utils.logger import logger


class NotificationService:
    """
    Servicio que monitorea tareas y envía notificaciones cuando:
    - Una tarea está próxima a vencer
    - Una tarea está vencida
    - Se completaron todas las misiones diarias
    """

    def __init__(self, get_tasks_callback: Callable):
        """
        Args:
            get_tasks_callback: Función que retorna lista de tareas actuales
        """
        self.get_tasks_callback = get_tasks_callback
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.notified_tasks = set()  # IDs de tareas ya notificadas hoy

    def start(self):
        """Inicia el servicio de notificaciones en segundo plano"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Servicio de notificaciones iniciado")

    def stop(self):
        """Detiene el servicio de notificaciones"""
        self.running = False
        logger.info("Servicio de notificaciones detenido")

    def _monitor_loop(self):
        """Loop de monitoreo en segundo plano"""
        check_interval_seconds = NOTIFICATION_CONFIG["check_interval_minutes"] * 60

        while self.running:
            try:
                self._check_tasks()
            except Exception as e:
                logger.error(f"Error en monitor de notificaciones: {e}")

            # Esperar hasta el próximo chequeo
            time.sleep(check_interval_seconds)

    def _check_tasks(self):
        """Revisa tareas y envía notificaciones si es necesario"""
        if not NOTIFICATION_CONFIG["enable_system_notifications"]:
            return

        tasks = self.get_tasks_callback()
        today = today_date()

        for task in tasks:
            # Solo notificar tareas pendientes
            if task.done:
                continue

            # Resetear notificaciones si cambió el día
            if task.id not in self.notified_tasks or self._should_notify_again(task, today):
                days_left = days_until(task.due)

                # Verificar si debe notificar
                if days_left in NOTIFICATION_CONFIG["days_before_alert"]:
                    self._send_notification(task, days_left)
                    self.notified_tasks.add(task.id)

    def _should_notify_again(self, task, today: date) -> bool:
        """Determina si debe volver a notificar sobre una tarea"""
        # Si ya venció y sigue pendiente, notificar cada vez
        return task.due < today

    def _send_notification(self, task, days_left: int):
        """Envía una notificación al sistema operativo"""
        if days_left < 0:
            title = "⚠️ Tarea Vencida"
            message = f"La tarea '{task.title}' está vencida desde hace {abs(days_left)} día(s)"
        elif days_left == 0:
            title = "📅 Tarea Vence Hoy"
            message = f"La tarea '{task.title}' vence hoy"
        elif days_left == 1:
            title = "⏰ Tarea Vence Mañana"
            message = f"La tarea '{task.title}' vence mañana"
        else:
            title = "📌 Tarea Próxima"
            message = f"La tarea '{task.title}' vence en {days_left} días"

        logger.info(f"Notificación: {title} - {message}")

        # Intentar usar notificaciones nativas del sistema
        try:
            self._show_system_notification(title, message)
        except Exception as e:
            logger.error(f"Error mostrando notificación: {e}")

    def _show_system_notification(self, title: str, message: str):
        """Muestra notificación nativa del sistema operativo"""
        import platform

        system = platform.system()

        if system == "Windows":
            self._show_windows_notification(title, message)
        elif system == "Darwin":  # macOS
            self._show_macos_notification(title, message)
        elif system == "Linux":
            self._show_linux_notification(title, message)
        else:
            logger.warning(f"Notificaciones no soportadas en {system}")

    def _show_windows_notification(self, title: str, message: str):
        """Notificación en Windows usando win10toast"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                duration=10,
                threaded=True,
                icon_path=None
            )
        except ImportError:
            logger.warning("win10toast no está instalado. Usa: pip install win10toast")
        except Exception as e:
            logger.error(f"Error en notificación Windows: {e}")

    def _show_macos_notification(self, title: str, message: str):
        """Notificación en macOS usando osascript"""
        import subprocess
        try:
            subprocess.run([
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"'
            ])
        except Exception as e:
            logger.error(f"Error en notificación macOS: {e}")

    def _show_linux_notification(self, title: str, message: str):
        """Notificación en Linux usando notify-send"""
        import subprocess
        try:
            subprocess.run(["notify-send", title, message])
        except Exception as e:
            logger.error(f"Error en notificación Linux: {e}")

    def notify_level_up(self, new_level: int):
        """Notificación especial cuando el usuario sube de nivel"""
        title = "🎉 ¡NIVEL UP!"
        message = f"¡Felicidades! Ahora eres nivel {new_level} de Productividad"
        try:
            self._show_system_notification(title, message)
        except Exception as e:
            logger.error(f"Error en notificación de level up: {e}")

    def notify_all_missions_completed(self):
        """Notificación cuando se completan todas las misiones diarias"""
        title = "🏆 ¡Misiones Completadas!"
        message = "¡Has completado todas las misiones diarias! Sigue así 💪"
        try:
            self._show_system_notification(title, message)
        except Exception as e:
            logger.error(f"Error en notificación de misiones: {e}")

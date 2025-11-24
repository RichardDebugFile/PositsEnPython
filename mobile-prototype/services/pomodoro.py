"""
Timer Pomodoro - Versión móvil
"""

from datetime import datetime, timedelta
from typing import Callable, Optional, List


class PomodoroTimer:
    """
    Timer Pomodoro: 25min trabajo + 5min descanso
    """

    def __init__(
        self,
        work_duration: int = 25,  # minutos
        break_duration: int = 5,  # minutos
        long_break_duration: int = 15,  # minutos
        sessions_until_long_break: int = 4
    ):
        self.work_duration = work_duration * 60  # convertir a segundos
        self.break_duration = break_duration * 60
        self.long_break_duration = long_break_duration * 60
        self.sessions_until_long_break = sessions_until_long_break

        self.is_running = False
        self.is_paused = False
        self.is_work_time = True
        self.sessions_completed = 0
        self.time_remaining = self.work_duration
        self.start_time: Optional[datetime] = None

        # Cola de tareas
        self.tasks_in_queue: List[str] = []  # IDs de tareas en cola

        # Callbacks
        self.on_tick: Optional[Callable] = None  # Llamado cada segundo
        self.on_work_complete: Optional[Callable] = None
        self.on_break_complete: Optional[Callable] = None
        self.on_session_complete: Optional[Callable] = None

    def start(self):
        """Inicia el timer"""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.start_time = datetime.now()

    def pause(self):
        """Pausa el timer"""
        if self.is_running and not self.is_paused:
            self.is_paused = True

    def resume(self):
        """Resume el timer"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.start_time = datetime.now()

    def stop(self):
        """Detiene el timer completamente"""
        self.is_running = False
        self.is_paused = False
        self.start_time = None

    def reset(self):
        """Resetea el timer al inicio"""
        self.stop()
        self.is_work_time = True
        self.time_remaining = self.work_duration
        self.sessions_completed = 0

    def skip_to_break(self):
        """Salta al descanso"""
        self._complete_work_session()

    def skip_break(self):
        """Salta el descanso"""
        self._complete_break()

    def update(self):
        """
        Actualiza el timer (llamar cada segundo o frame)
        Retorna True si el timer cambió de fase
        """
        if not self.is_running or self.is_paused:
            return False

        if self.start_time is None:
            return False

        # Calcular tiempo transcurrido
        elapsed = (datetime.now() - self.start_time).total_seconds()

        # Actualizar tiempo restante
        self.time_remaining -= elapsed
        self.start_time = datetime.now()

        # Callback de tick
        if self.on_tick:
            self.on_tick(self.get_status())

        # Verificar si completó la fase actual
        if self.time_remaining <= 0:
            if self.is_work_time:
                self._complete_work_session()
            else:
                self._complete_break()
            return True

        return False

    def _complete_work_session(self):
        """Completa una sesión de trabajo"""
        self.sessions_completed += 1

        # Callback de trabajo completado
        if self.on_work_complete:
            self.on_work_complete(self.sessions_completed)

        # Determinar duración del descanso
        if self.sessions_completed % self.sessions_until_long_break == 0:
            self.time_remaining = self.long_break_duration
        else:
            self.time_remaining = self.break_duration

        self.is_work_time = False
        self.start_time = datetime.now()

    def _complete_break(self):
        """Completa un descanso"""
        # Callback de descanso completado
        if self.on_break_complete:
            self.on_break_complete()

        # Callback de sesión completa
        if self.on_session_complete:
            self.on_session_complete(self.sessions_completed)

        # Volver a trabajo
        self.time_remaining = self.work_duration
        self.is_work_time = True
        self.start_time = datetime.now()

    def get_status(self) -> dict:
        """Obtiene el estado actual del timer"""
        minutes = int(self.time_remaining // 60)
        seconds = int(self.time_remaining % 60)

        total_duration = self.work_duration if self.is_work_time else (
            self.long_break_duration if self.sessions_completed % self.sessions_until_long_break == 0
            else self.break_duration
        )

        progress = ((total_duration - self.time_remaining) / total_duration) * 100

        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_work_time": self.is_work_time,
            "sessions_completed": self.sessions_completed,
            "time_remaining": self.time_remaining,
            "minutes": minutes,
            "seconds": seconds,
            "progress": progress,
            "phase": "work" if self.is_work_time else "break",
            "next_is_long_break": (self.sessions_completed + 1) % self.sessions_until_long_break == 0
        }

    def get_time_formatted(self) -> str:
        """Retorna el tiempo en formato MM:SS"""
        status = self.get_status()
        return f"{status['minutes']:02d}:{status['seconds']:02d}"

    def add_task_to_queue(self, task_id: str):
        """Agrega una tarea a la cola de Pomodoro"""
        if task_id not in self.tasks_in_queue:
            self.tasks_in_queue.append(task_id)

    def remove_task_from_queue(self, task_id: str):
        """Remueve una tarea de la cola de Pomodoro"""
        if task_id in self.tasks_in_queue:
            self.tasks_in_queue.remove(task_id)

    def get_tasks_in_queue(self) -> List[str]:
        """Obtiene la lista de IDs de tareas en cola"""
        return self.tasks_in_queue.copy()

    def clear_queue(self):
        """Limpia la cola de tareas"""
        self.tasks_in_queue.clear()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TaskStore: Persistencia de tareas con soporte de ordenamiento y filtrado
"""

import json
from pathlib import Path
from datetime import date
from typing import Optional, Callable
from tkinter import messagebox

from .task import Task
from .gamification import GamificationManager
from ..config import DATA_FILE, LOCAL_DATA_FILE, APP_NAME
from ..utils.dates import fmt_date, parse_date
from ..utils.logger import logger


class TaskStore:
    """
    Gestor de persistencia de tareas con soporte de:
    - Guardado automático con throttling
    - Ordenamiento por múltiples criterios
    - Filtrado avanzado
    - Integración con sistema de gamificación
    """

    def __init__(self, file_path: Path = DATA_FILE):
        self.file_path = file_path
        self.tasks: list[Task] = []
        self._after: Optional[Callable] = None
        self._save_pending = False
        self.gamification = GamificationManager()
        self.load()

    def set_after(self, after_callable: Callable):
        """Configura la función de throttling (típicamente tk.after)"""
        self._after = after_callable

    def load(self):
        """Carga tareas desde el archivo JSON"""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tasks_data = data.get("tasks", [])
                    self.tasks = [Task.from_dict(t) for t in tasks_data]
                    logger.debug(f"Cargadas {len(self.tasks)} tareas")
            except json.JSONDecodeError:
                messagebox.showwarning(APP_NAME, "Archivo de notas dañado; se reinicia.")
                self.tasks = []
        else:
            # Intentar migrar desde archivo legacy
            self._migrate_from_legacy()

    def _migrate_from_legacy(self):
        """Migra datos del archivo legacy tasks.json si existe"""
        if LOCAL_DATA_FILE.exists() and not self.file_path.exists():
            try:
                with open(LOCAL_DATA_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.tasks = [Task.from_dict(t) for t in data]
                        self.save()
                        logger.info(f"Migradas {len(self.tasks)} tareas desde {LOCAL_DATA_FILE}")
            except Exception as e:
                logger.error(f"Error migrando tareas legacy: {e}")

    def save(self):
        """Guarda tareas inmediatamente al archivo JSON"""
        try:
            data = {"tasks": [t.to_dict() for t in self.tasks]}
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error guardando tareas: {e}")

    def save_throttled(self, delay_ms: int = 400):
        """Guarda con throttling para evitar múltiples escrituras rápidas"""
        if not self._after:
            return self.save()

        if self._save_pending:
            return

        self._save_pending = True

        def _flush():
            self._save_pending = False
            self.save()

        self._after(delay_ms, _flush)

    def add(self, title: str, desc: str, due: date, priority: str, color: Optional[str] = None) -> Task:
        """Crea y agrega una nueva tarea"""
        task = Task(
            title=title,
            desc=desc,
            due=due,
            priority=priority,
            color=color or "Sunshine",
        )
        self.tasks.append(task)
        self.save()

        # Notificar al sistema de gamificación
        self.gamification.on_task_created()
        logger.info(f"Tarea creada: {task.title}")

        return task

    def delete(self, index: int):
        """Elimina una tarea por índice"""
        if 0 <= index < len(self.tasks):
            task = self.tasks.pop(index)
            self.save()
            logger.info(f"Tarea eliminada: {task.title}")

    def toggle_done(self, index: int):
        """Marca/desmarca una tarea como completada"""
        if 0 <= index < len(self.tasks):
            task = self.tasks[index]
            task.done = not task.done
            self.save()

            # Si se completó (no se descompletó), notificar gamificación
            if task.done:
                result = self.gamification.on_task_completed(priority_level=task.priority)
                logger.info(f"Tarea completada: {task.title} (+{result['xp_gained']} XP)")
                return result
            return None

    def toggle_priority(self, index: int):
        """Cicla entre los niveles de prioridad: low -> medium -> high -> urgent -> low"""
        if 0 <= index < len(self.tasks):
            task = self.tasks[index]
            # Convertir si es booleano antiguo
            if isinstance(task.priority, bool):
                task.priority = "high" if task.priority else "medium"

            # Ciclar entre niveles
            priority_cycle = ["low", "medium", "high", "urgent"]
            try:
                current_index = priority_cycle.index(task.priority)
                next_index = (current_index + 1) % len(priority_cycle)
                task.priority = priority_cycle[next_index]
            except ValueError:
                # Si no está en la lista, poner medium por defecto
                task.priority = "medium"

            self.save()

    def index_by_id(self, task_id: str) -> int:
        """Encuentra el índice de una tarea por su ID"""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                return i
        return -1

    def get_by_id(self, task_id: str) -> Optional[Task]:
        """Obtiene una tarea por su ID"""
        idx = self.index_by_id(task_id)
        if idx >= 0:
            return self.tasks[idx]
        return None

    # ------------------------ Ordenamiento y Filtrado ------------------------

    def sort_tasks(self, by: str = "due", reverse: bool = False):
        """
        Ordena las tareas según criterio.

        Criterios disponibles:
        - "due": Por fecha de vencimiento
        - "created": Por fecha de creación
        - "priority": Prioritarias primero
        - "title": Alfabético por título
        - "color": Por color
        - "done": Completadas al final
        """
        if by == "due":
            self.tasks.sort(key=lambda t: t.due, reverse=reverse)
        elif by == "created":
            self.tasks.sort(key=lambda t: t.start, reverse=reverse)
        elif by == "priority":
            self.tasks.sort(key=lambda t: t.priority, reverse=not reverse)  # True primero
        elif by == "title":
            self.tasks.sort(key=lambda t: t.title.lower(), reverse=reverse)
        elif by == "color":
            self.tasks.sort(key=lambda t: t.color, reverse=reverse)
        elif by == "done":
            self.tasks.sort(key=lambda t: t.done, reverse=reverse)

        logger.debug(f"Tareas ordenadas por: {by} (reverse={reverse})")

    def get_filtered_tasks(
        self,
        only_pending: bool = False,
        color_filter: Optional[str] = None,
        overdue_only: bool = False,
        today_only: bool = False,
    ) -> list[Task]:
        """
        Retorna tareas filtradas según criterios.

        Args:
            only_pending: Solo tareas no completadas
            color_filter: Filtrar por color específico
            overdue_only: Solo tareas vencidas
            today_only: Solo tareas que vencen hoy
        """
        from ..utils.dates import today_date, is_overdue, is_today

        filtered = self.tasks

        if only_pending:
            filtered = [t for t in filtered if not t.done]

        if color_filter and color_filter != "Todos":
            filtered = [t for t in filtered if t.color == color_filter]

        if overdue_only:
            filtered = [t for t in filtered if is_overdue(t.due) and not t.done]

        if today_only:
            filtered = [t for t in filtered if is_today(t.due)]

        return filtered

    def get_statistics(self) -> dict:
        """Retorna estadísticas generales"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.done)
        pending = total - completed
        overdue = sum(1 for t in self.tasks if not t.done and t.due < date.today())

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "overdue": overdue,
        }

    # ------------------------ Consultas por Fecha (Calendario) ------------------------

    def get_tasks_in_date_range(self, start_date: date, end_date: date) -> list[Task]:
        """
        Retorna tareas cuya fecha de vencimiento está en el rango especificado.

        Args:
            start_date: Fecha inicial del rango (inclusive)
            end_date: Fecha final del rango (inclusive)

        Returns:
            Lista de tareas en el rango
        """
        return [t for t in self.tasks if start_date <= t.due <= end_date]

    def get_tasks_for_date(self, target_date: date) -> list[Task]:
        """
        Retorna tareas que vencen en una fecha específica.

        Args:
            target_date: Fecha a buscar

        Returns:
            Lista de tareas que vencen ese día
        """
        return [t for t in self.tasks if t.due == target_date]

    def get_tasks_starting_on(self, target_date: date) -> list[Task]:
        """
        Retorna tareas que inician en una fecha específica.

        Args:
            target_date: Fecha a buscar

        Returns:
            Lista de tareas que inician ese día
        """
        return [t for t in self.tasks if t.start == target_date]

    def get_tasks_by_month(self, year: int, month: int) -> dict[str, list[Task]]:
        """
        Retorna tareas agrupadas por día para un mes específico.

        Args:
            year: Año
            month: Mes (1-12)

        Returns:
            Dict {fecha_str: [lista de tareas]}
        """
        from calendar import monthrange

        _, last_day = monthrange(year, month)
        start_date = date(year, month, 1)
        end_date = date(year, month, last_day)

        result = {}
        for task in self.tasks:
            if start_date <= task.due <= end_date:
                date_str = fmt_date(task.due)
                if date_str not in result:
                    result[date_str] = []
                result[date_str].append(task)

        return result

    def update_task_date(self, task_id: str, new_due: date) -> bool:
        """
        Actualiza la fecha de vencimiento de una tarea.

        Args:
            task_id: ID de la tarea
            new_due: Nueva fecha de vencimiento

        Returns:
            True si se actualizó, False si no se encontró
        """
        task = self.get_by_id(task_id)
        if task:
            task.due = new_due
            self.save()
            logger.info(f"Fecha de tarea actualizada: {task.title} -> {fmt_date(new_due)}")
            return True
        return False

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modelo de datos: Task (Tarea)
"""

import uuid
from datetime import date, timedelta
from typing import Optional
from ..utils.dates import fmt_date, today_date


class Task:
    """
    Representa una tarea/nota en la aplicación.

    Atributos:
        id: Identificador único
        title: Título de la tarea
        desc: Descripción detallada
        start: Fecha de inicio
        due: Fecha de vencimiento
        priority: Si es tarea prioritaria
        done: Si está completada
        color: Color del posit (clave de VIBRANT_COLORS)
        pinned: Si está anclada (always on top)
        alpha: Transparencia de la ventana (0.0-1.0)
        geometry: Geometría de la ventana (ej: "560x520+200+200")
        open: Si la ventana está abierta
        attachments: Lista de rutas de archivos adjuntos
    """

    def __init__(
        self,
        id: Optional[str] = None,
        title: str = "",
        desc: str = "",
        start: Optional[date] = None,
        due: Optional[date] = None,
        priority: bool = False,
        done: bool = False,
        color: str = "Sunshine",
        pinned: bool = False,
        alpha: float = 0.98,
        geometry: str = "560x520+200+200",
        open: bool = False,
        attachments: Optional[list[str]] = None,
    ):
        self.id = id or uuid.uuid4().hex[:8]
        self.title = title
        self.desc = desc
        self.start = start or today_date()
        self.due = due or (today_date() + timedelta(days=3))
        self.priority = priority
        self.done = done
        self.color = color
        self.pinned = pinned
        self.alpha = alpha
        self.geometry = geometry
        self.open = open
        self.attachments = attachments or []

    def to_dict(self) -> dict:
        """Convierte la tarea a diccionario para JSON"""
        return {
            "id": self.id,
            "title": self.title,
            "desc": self.desc,
            "start": fmt_date(self.start),
            "due": fmt_date(self.due),
            "priority": self.priority,
            "done": self.done,
            "color": self.color,
            "pinned": self.pinned,
            "alpha": self.alpha,
            "geometry": self.geometry,
            "open": self.open,
            "attachments": self.attachments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Crea una tarea desde un diccionario"""
        from ..utils.dates import parse_date

        return cls(
            id=data.get("id"),
            title=data.get("title", ""),
            desc=data.get("desc", ""),
            start=parse_date(data.get("start", fmt_date(today_date()))),
            due=parse_date(data.get("due", fmt_date(today_date() + timedelta(days=3)))),
            priority=data.get("priority", False),
            done=data.get("done", False),
            color=data.get("color", "Sunshine"),
            pinned=data.get("pinned", False),
            alpha=data.get("alpha", 0.98),
            geometry=data.get("geometry", "560x520+200+200"),
            open=data.get("open", False),
            attachments=data.get("attachments", []),
        )

    def __repr__(self) -> str:
        return f"Task(id={self.id}, title={self.title!r}, done={self.done})"

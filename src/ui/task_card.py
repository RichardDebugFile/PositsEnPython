#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Renderizado de tarjetas de tareas en la lista principal
"""

import tkinter as tk
from datetime import date
from ..config import MODERN_COLORS, GRADIENTS, VIBRANT_COLORS
from ..utils.dates import parse_date, today_date
from ..utils.colors import urgency_color
from .components import PillButton


class TaskCardRenderer:
    """Renderiza tarjetas de tareas con estilos modernos"""

    def __init__(self, parent_frame, app_instance):
        self.parent_frame = parent_frame
        self.app = app_instance

    def render_task(self, task, now: date):
        """Renderiza una tarjeta de tarea"""
        tid = task.id
        border_color = self._get_task_border_color(task, now)

        outer = tk.Frame(self.parent_frame, bg=border_color)
        outer.pack(fill="x", pady=4, padx=4)

        card = tk.Frame(outer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        card.pack(fill="x", padx=2, pady=2)

        # Franja de color lateral
        stripe_color = VIBRANT_COLORS.get(task.color, "#CCCCCC")
        stripe = tk.Frame(card, bg=stripe_color, width=10)
        stripe.pack(side="left", fill="y")

        body = tk.Frame(card, bg=GRADIENTS["Card"][0])
        body.pack(side="left", fill="both", expand=True)

        # Header con checkbox
        header = tk.Frame(body, bg=GRADIENTS["Card"][0])
        header.pack(fill="x", padx=12, pady=8)

        var_done = tk.BooleanVar(value=task.done)
        tk.Checkbutton(
            header,
            variable=var_done,
            command=lambda t=tid: self.app.toggle_done_by_id(t),
            bg=GRADIENTS["Card"][0],
            font=("Segoe UI", 12)
        ).pack(side="left")

        # Título
        title_font = ("Segoe UI", 12, "bold")
        title_color = MODERN_COLORS["Text"] if not task.done else MODERN_COLORS["TextLight"]

        title_frame = tk.Frame(header, bg=GRADIENTS["Card"][0])
        title_frame.pack(side="left", fill="x", expand=True, padx=8)

        lbl_title = tk.Label(
            title_frame,
            text=task.title,
            bg=GRADIENTS["Card"][0],
            fg=title_color,
            font=title_font,
            anchor="w",
            wraplength=460,
            justify="left",
            cursor="hand2"
        )
        lbl_title.pack(fill="x")
        lbl_title.bind("<Double-Button-1>", lambda e, t=tid: self.app.open_quick_sticky_by_id(t))

        # Badge de prioridad
        if task.priority:
            tk.Label(
                header,
                text="⭐ PRIORIDAD",
                bg=GRADIENTS["Warning"][0],
                fg="white",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=2
            ).pack(side="right")

        # Descripción
        content = tk.Frame(body, bg=GRADIENTS["Card"][0])
        content.pack(fill="x", padx=12, pady=(0, 8))

        if task.desc:
            desc_font = ("Segoe UI", 10, "italic" if task.done else "normal")
            desc_fg = MODERN_COLORS["TextLight"] if task.done else MODERN_COLORS["Text"]
            tk.Label(
                content,
                text=task.desc,
                bg=GRADIENTS["Card"][0],
                fg=desc_fg,
                font=desc_font,
                anchor="w",
                wraplength=460,
                justify="left"
            ).pack(fill="x", pady=(0, 8))

        # Footer con fecha y acciones
        footer = tk.Frame(body, bg=GRADIENTS["Card"][0])
        footer.pack(fill="x", padx=12, pady=(0, 8))

        try:
            due_d = task.due if isinstance(task.due, date) else parse_date(str(task.due))
        except Exception:
            due_d = now

        due_txt = f"📅 {task.due if isinstance(task.due, str) else str(task.due)}"
        overdue = (due_d <= now)
        due_fg = MODERN_COLORS["Danger"] if overdue and not task.done else MODERN_COLORS["TextLight"]

        tk.Label(
            footer,
            text=due_txt,
            bg=GRADIENTS["Card"][0],
            fg=due_fg,
            font=("Segoe UI", 9)
        ).pack(side="left")

        # Botones de acción
        actions = tk.Frame(footer, bg=GRADIENTS["Card"][0])
        actions.pack(side="right")

        priority_icon = "⭐" if task.priority else "☆"
        self._create_small_button(actions, priority_icon, lambda t=tid: self.app.toggle_priority_by_id(t), "Warning")
        self._create_small_button(actions, "📝", lambda t=tid: self.app.open_note_by_id(t), "Primary")
        self._create_small_button(actions, "🗑️", lambda t=tid: self.app.delete_task_by_id(t), "Danger")

    def _create_small_button(self, parent, icon, command, color):
        """Crea un botón pequeño de acción"""
        btn = tk.Label(
            parent,
            text=icon,
            bg=GRADIENTS[color][0],
            fg="white",
            font=("Segoe UI", 10),
            padx=6,
            pady=2,
            relief="flat",
            bd=0,
            takefocus=1,
            cursor="hand2"
        )
        btn.pack(side="left", padx=2)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Return>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(bg=GRADIENTS[color][1]))
        btn.bind("<Leave>", lambda e: btn.configure(bg=GRADIENTS[color][0]))

    def _get_task_border_color(self, task, now: date) -> str:
        """Calcula el color del borde según el estado de la tarea"""
        if task.done:
            return MODERN_COLORS["Success"]
        if task.priority:
            return MODERN_COLORS["Warning"]

        try:
            start_d = task.start if isinstance(task.start, date) else parse_date(str(task.start))
            due_d = task.due if isinstance(task.due, date) else parse_date(str(task.due))
            return urgency_color(start_d, due_d, now)
        except Exception:
            return MODERN_COLORS["TextLight"]

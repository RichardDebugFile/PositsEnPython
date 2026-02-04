#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mini widget de calendario flotante.
Muestra una vista compacta del mes actual con indicadores de tareas.
"""

import tkinter as tk
from datetime import date, timedelta

from ..config import MODERN_COLORS, GRADIENTS, CALENDAR_CONFIG
from ..utils.dates import today_date, fmt_date


class MiniCalendarWidget(tk.Toplevel):
    """
    Pequeño calendario flotante siempre visible.
    Muestra el mes actual con indicadores de tareas pendientes.
    """

    def __init__(self, master, store, on_date_click=None):
        super().__init__(master)
        self.store = store
        self.on_date_click = on_date_click
        self.current_date = today_date()

        # Configurar ventana
        self.title("")
        self.overrideredirect(True)  # Sin bordes de ventana
        self.attributes("-topmost", True)  # Siempre arriba
        self.configure(bg=GRADIENTS["Card"][0])

        # Posicionar en esquina inferior derecha
        self._position_widget()

        # Variables para drag
        self._drag_start_x = 0
        self._drag_start_y = 0

        # Crear UI
        self._create_ui()

        # Render inicial
        self.refresh()

    def _position_widget(self):
        """Posiciona el widget según configuración"""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        width = 220
        height = 200

        position = CALENDAR_CONFIG.get("mini_calendar_position", "bottom-right")

        if position == "bottom-right":
            x = screen_width - width - 20
            y = screen_height - height - 60
        elif position == "bottom-left":
            x = 20
            y = screen_height - height - 60
        elif position == "top-right":
            x = screen_width - width - 20
            y = 40
        elif position == "top-left":
            x = 20
            y = 40
        else:
            x = screen_width - width - 20
            y = screen_height - height - 60

        self.geometry(f"{width}x{height}+{x}+{y}")

    def _create_ui(self):
        """Crea la interfaz del mini calendario"""
        # Frame principal con borde
        main_frame = tk.Frame(self, bg=MODERN_COLORS["Primary"], relief="flat", bd=0)
        main_frame.pack(fill="both", expand=True, padx=1, pady=1)

        inner_frame = tk.Frame(main_frame, bg=GRADIENTS["Card"][0])
        inner_frame.pack(fill="both", expand=True, padx=1, pady=1)

        # Header con título y controles
        header = tk.Frame(inner_frame, bg=GRADIENTS["Primary"][0])
        header.pack(fill="x")

        # Botón anterior
        btn_prev = tk.Label(
            header,
            text="◀",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8),
            cursor="hand2",
            padx=4
        )
        btn_prev.pack(side="left")
        btn_prev.bind("<Button-1>", lambda e: self._prev_month())

        # Título del mes (arrastrable)
        self.month_label = tk.Label(
            header,
            text="",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 10, "bold"),
            cursor="fleur"
        )
        self.month_label.pack(side="left", expand=True)

        # Bind para arrastrar
        self.month_label.bind("<Button-1>", self._start_drag)
        self.month_label.bind("<B1-Motion>", self._on_drag)

        # Botón siguiente
        btn_next = tk.Label(
            header,
            text="▶",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8),
            cursor="hand2",
            padx=4
        )
        btn_next.pack(side="left")
        btn_next.bind("<Button-1>", lambda e: self._next_month())

        # Botón cerrar
        close_btn = tk.Label(
            header,
            text="✕",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8),
            cursor="hand2",
            padx=4
        )
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: self.withdraw())

        # Días de la semana
        days_header = tk.Frame(inner_frame, bg=GRADIENTS["Card"][0])
        days_header.pack(fill="x", padx=4, pady=(4, 0))

        day_names = CALENDAR_CONFIG.get("day_names_short", ["L", "M", "X", "J", "V", "S", "D"])
        for i, dia in enumerate(day_names):
            tk.Label(
                days_header,
                text=dia,
                bg=GRADIENTS["Card"][0],
                fg=MODERN_COLORS["TextLight"],
                font=("Segoe UI", 8, "bold"),
                width=3
            ).grid(row=0, column=i)
            days_header.grid_columnconfigure(i, weight=1)

        # Grilla de días
        self.days_grid = tk.Frame(inner_frame, bg=GRADIENTS["Card"][0])
        self.days_grid.pack(fill="both", expand=True, padx=4, pady=4)

        # Footer con resumen
        self.footer = tk.Label(
            inner_frame,
            text="",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 8),
            anchor="w",
            padx=6,
            pady=2
        )
        self.footer.pack(fill="x")

    def _start_drag(self, event):
        """Inicia el arrastre del widget"""
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        """Mueve el widget mientras se arrastra"""
        x = self.winfo_x() + event.x - self._drag_start_x
        y = self.winfo_y() + event.y - self._drag_start_y
        self.geometry(f"+{x}+{y}")

    def _prev_month(self):
        """Navega al mes anterior"""
        year = self.current_date.year
        month = self.current_date.month - 1
        if month < 1:
            month = 12
            year -= 1
        self.current_date = date(year, month, 1)
        self.refresh()

    def _next_month(self):
        """Navega al mes siguiente"""
        year = self.current_date.year
        month = self.current_date.month + 1
        if month > 12:
            month = 1
            year += 1
        self.current_date = date(year, month, 1)
        self.refresh()

    def refresh(self):
        """Refresca el calendario"""
        # Actualizar título
        month_names_short = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        self.month_label.configure(
            text=f"{month_names_short[self.current_date.month - 1]} {self.current_date.year}"
        )

        # Limpiar grilla
        for widget in self.days_grid.winfo_children():
            widget.destroy()

        # Obtener datos del mes
        year = self.current_date.year
        month = self.current_date.month
        first_day = date(year, month, 1)

        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        first_weekday = first_day.weekday()

        # Obtener tareas del mes
        tasks_by_date = {}
        pending_count = 0
        for task in self.store.tasks:
            if first_day <= task.due <= last_day:
                date_str = fmt_date(task.due)
                if date_str not in tasks_by_date:
                    tasks_by_date[date_str] = 0
                tasks_by_date[date_str] += 1
                if not task.done:
                    pending_count += 1

        # Renderizar días
        row = 0
        col = first_weekday
        current_day = first_day

        while current_day <= last_day:
            self._render_mini_day(current_day, row, col, tasks_by_date)

            col += 1
            if col > 6:
                col = 0
                row += 1
            current_day += timedelta(days=1)

        # Configurar columnas
        for i in range(7):
            self.days_grid.grid_columnconfigure(i, weight=1)

        # Actualizar footer
        if pending_count > 0:
            self.footer.configure(text=f"📋 {pending_count} tareas pendientes")
        else:
            self.footer.configure(text="✨ Sin tareas pendientes")

    def _render_mini_day(self, day_date, row, col, tasks_by_date):
        """Renderiza un día en miniatura"""
        date_str = fmt_date(day_date)
        has_tasks = date_str in tasks_by_date
        task_count = tasks_by_date.get(date_str, 0)
        is_today = day_date == today_date()

        # Colores
        if is_today:
            bg = MODERN_COLORS["Primary"]
            fg = "white"
        elif has_tasks:
            bg = MODERN_COLORS["Warning"]
            fg = MODERN_COLORS["Text"]
        else:
            bg = GRADIENTS["Card"][0]
            fg = MODERN_COLORS["Text"]

        # Label del día
        day_label = tk.Label(
            self.days_grid,
            text=str(day_date.day),
            bg=bg,
            fg=fg,
            font=("Segoe UI", 8, "bold" if is_today or has_tasks else "normal"),
            width=3,
            height=1,
            cursor="hand2" if has_tasks or self.on_date_click else ""
        )
        day_label.grid(row=row, column=col, padx=1, pady=1)

        # Tooltip con número de tareas
        if has_tasks:
            def show_tooltip(e, count=task_count):
                tooltip = tk.Toplevel(self)
                tooltip.overrideredirect(True)
                tooltip.geometry(f"+{e.x_root + 10}+{e.y_root + 10}")
                tk.Label(
                    tooltip,
                    text=f"{count} tarea{'s' if count > 1 else ''}",
                    bg=MODERN_COLORS["Dark"],
                    fg="white",
                    font=("Segoe UI", 8),
                    padx=4,
                    pady=2
                ).pack()
                day_label._tooltip = tooltip

            def hide_tooltip(e):
                if hasattr(day_label, '_tooltip'):
                    day_label._tooltip.destroy()

            day_label.bind("<Enter>", show_tooltip)
            day_label.bind("<Leave>", hide_tooltip)

        # Click para notificar
        if self.on_date_click:
            day_label.bind("<Button-1>", lambda e, d=day_date: self.on_date_click(d))

    def show(self):
        """Muestra el widget"""
        self.deiconify()
        self.refresh()

    def toggle(self):
        """Muestra/oculta el widget"""
        if self.winfo_viewable():
            self.withdraw()
        else:
            self.show()

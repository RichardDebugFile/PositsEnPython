#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel de calendario interactivo con heatmap de productividad.
Permite visualizar tareas por fecha y cambiar fechas con drag & drop.
"""

import tkinter as tk
from tkinter import ttk
from datetime import date, timedelta
from typing import Optional

from ..config import (
    MODERN_COLORS,
    GRADIENTS,
    VIBRANT_COLORS,
    CALENDAR_CONFIG,
    PRIORITY_LEVELS,
)
from ..utils.dates import today_date, fmt_date, parse_date
from .components import PillButton

# Importar TkinterDnD si está disponible
try:
    from tkinterdnd2 import DND_ALL
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


class CalendarPanel(tk.Frame):
    """
    Panel de calendario con:
    - Vista mensual con navegación
    - Heatmap de productividad basado en XP ganado
    - Celdas interactivas con indicador de tareas
    - Panel de detalle para tareas del día seleccionado
    - Soporte para drag & drop de tareas
    """

    def __init__(self, parent, store, gamification_manager, app=None):
        super().__init__(parent, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        self.store = store
        self.gm = gamification_manager
        self.app = app

        # Estado del calendario
        self.current_date = today_date()
        self.selected_date = today_date()
        self.day_cells = {}  # {date_str: frame_widget}

        # Colores del heatmap
        self.HEATMAP_COLORS = CALENDAR_CONFIG["heatmap_colors"]

        # Crear UI
        self._create_header()

        # Usar PanedWindow para dividir calendario y panel de detalle (ajustable)
        self.paned = tk.PanedWindow(self, orient=tk.VERTICAL, bg=GRADIENTS["Card"][0], sashwidth=4)
        self.paned.pack(fill="both", expand=True)

        # Container para el calendario
        self.calendar_container = tk.Frame(self.paned, bg=GRADIENTS["Card"][0])
        self.paned.add(self.calendar_container, minsize=200)

        # Container para el panel de detalle
        self.detail_container = tk.Frame(self.paned, bg=MODERN_COLORS["Light"])
        self.paned.add(self.detail_container, minsize=120)

        self._create_calendar_grid()
        self._create_task_detail_panel()

        # Render inicial
        self.refresh()

    def _create_header(self):
        """Crea el header con navegación de meses y leyenda del heatmap"""
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0])
        header.pack(fill="x")

        # Navegación
        nav_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0])
        nav_frame.pack(fill="x", padx=16, pady=12)

        # Botón mes anterior
        btn_prev = PillButton(nav_frame, "◀", self._prev_month, "Secondary", "small")
        btn_prev.pack(side="left")

        # Título mes/año
        self.month_label = tk.Label(
            nav_frame,
            text="",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 16, "bold")
        )
        self.month_label.pack(side="left", expand=True)

        # Botón mes siguiente
        btn_next = PillButton(nav_frame, "▶", self._next_month, "Secondary", "small")
        btn_next.pack(side="left")

        # Botón Hoy
        btn_today = PillButton(nav_frame, "Hoy", self._go_today, "Success", "small")
        btn_today.pack(side="left", padx=(12, 0))

        # Leyenda del heatmap
        legend_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0])
        legend_frame.pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(
            legend_frame,
            text="Productividad:",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 9)
        ).pack(side="left")

        tk.Label(
            legend_frame,
            text="Menos",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8)
        ).pack(side="left", padx=(8, 4))

        # Escala de colores
        for color in self.HEATMAP_COLORS:
            tk.Frame(legend_frame, bg=color, width=16, height=12).pack(side="left", padx=1)

        tk.Label(
            legend_frame,
            text="Más",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8)
        ).pack(side="left", padx=(4, 0))

    def _create_calendar_grid(self):
        """Crea la grilla del calendario con scroll automático"""
        # Container principal (dentro de calendar_container)
        container = tk.Frame(self.calendar_container, bg=GRADIENTS["Card"][0])
        container.pack(fill="both", expand=True, padx=8, pady=8)

        # Canvas con scrollbar para el calendario
        self.calendar_canvas = tk.Canvas(
            container,
            bg=GRADIENTS["Card"][0],
            highlightthickness=0,
            bd=0
        )
        self.calendar_scrollbar = ttk.Scrollbar(
            container,
            orient="vertical",
            command=self.calendar_canvas.yview
        )

        # Frame de scroll (va dentro del canvas)
        scroll_frame = tk.Frame(self.calendar_canvas, bg=GRADIENTS["Card"][0])

        # Frame centrador con expansión horizontal
        center_wrapper = tk.Frame(scroll_frame, bg=GRADIENTS["Card"][0])
        center_wrapper.pack(fill="x", expand=True)

        # Frame del calendario (contenido real, sin expansión)
        self.calendar_frame = tk.Frame(center_wrapper, bg=GRADIENTS["Card"][0])
        self.calendar_frame.pack(anchor="center")  # Centrado dentro del wrapper

        # Configurar scroll
        scroll_frame.bind(
            "<Configure>",
            lambda e: self._on_calendar_configure()
        )

        # Crear window del scroll_frame en el canvas
        self.calendar_window = self.calendar_canvas.create_window(
            0, 0,
            window=scroll_frame,
            anchor="nw"
        )
        self.calendar_canvas.configure(yscrollcommand=self.calendar_scrollbar.set)

        # Pack canvas (scrollbar se muestra solo si es necesario)
        self.calendar_canvas.pack(side="left", fill="both", expand=True)

        # Bind para ajustar el ancho del scroll_frame al ancho del canvas
        self.calendar_canvas.bind("<Configure>", lambda e: self._on_canvas_resize(e))

        # Bind mousewheel
        def _on_mousewheel(e):
            self.calendar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def _on_enter(e):
            self.calendar_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_leave(e):
            self.calendar_canvas.unbind_all("<MouseWheel>")

        self.calendar_canvas.bind("<Enter>", _on_enter)
        self.calendar_canvas.bind("<Leave>", _on_leave)

        # Header de días de la semana
        self.weekday_frame = tk.Frame(self.calendar_frame, bg=GRADIENTS["Card"][0])
        self.weekday_frame.pack(pady=(0, 4))

        days = CALENDAR_CONFIG["day_names"]
        for i, dia in enumerate(days):
            lbl = tk.Label(
                self.weekday_frame,
                text=dia,
                bg=GRADIENTS["Primary"][0],
                fg="white",
                font=("Segoe UI", 10, "bold"),
                width=14,
                pady=4
            )
            lbl.grid(row=0, column=i, padx=2)
            self.weekday_frame.grid_columnconfigure(i, minsize=110, uniform="cols")

        # Frame para días del mes
        self.days_frame = tk.Frame(self.calendar_frame, bg=GRADIENTS["Card"][0])
        self.days_frame.pack()

    def _on_calendar_configure(self):
        """Configurar scroll region y mostrar scrollbar solo si es necesario"""
        self.calendar_canvas.configure(scrollregion=self.calendar_canvas.bbox("all"))

        # Mostrar/ocultar scrollbar según si es necesario
        canvas_height = self.calendar_canvas.winfo_height()
        # Obtener altura del scroll_frame completo
        bbox = self.calendar_canvas.bbox("all")
        content_height = bbox[3] - bbox[1] if bbox else 0

        if content_height > canvas_height:
            # Contenido más grande que canvas, mostrar scrollbar
            self.calendar_scrollbar.pack(side="right", fill="y")
        else:
            # Contenido cabe, ocultar scrollbar
            self.calendar_scrollbar.pack_forget()

    def _on_canvas_resize(self, event):
        """Ajusta el ancho del frame de scroll al ancho del canvas"""
        canvas_width = event.width
        # Hacer que el scroll_frame tenga el mismo ancho que el canvas
        # para que el center_wrapper pueda centrar correctamente
        self.calendar_canvas.itemconfig(self.calendar_window, width=canvas_width)

    def _create_task_detail_panel(self):
        """Crea el panel inferior para mostrar tareas del día seleccionado"""
        # Crear dentro de detail_container
        self.detail_panel = tk.Frame(self.detail_container, bg=MODERN_COLORS["Light"])
        self.detail_panel.pack(fill="both", expand=True, padx=8, pady=8)

        # Header del panel de detalle
        header = tk.Frame(self.detail_panel, bg=MODERN_COLORS["Light"])
        header.pack(fill="x", padx=12, pady=8)

        self.detail_date_label = tk.Label(
            header,
            text="Selecciona un día",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 12, "bold")
        )
        self.detail_date_label.pack(side="left")

        # Botón para agregar tarea en la fecha seleccionada
        PillButton(
            header,
            "Añadir tarea",
            self._add_task_for_selected_date,
            "Success",
            "small",
            "➕"
        ).pack(side="left", padx=12)

        self.detail_xp_label = tk.Label(
            header,
            text="",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9)
        )
        self.detail_xp_label.pack(side="right")

        # Container scrollable para tareas
        tasks_container = tk.Frame(self.detail_panel, bg=MODERN_COLORS["Light"])
        tasks_container.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.detail_canvas = tk.Canvas(
            tasks_container,
            bg=MODERN_COLORS["Light"],
            highlightthickness=0,
            bd=0
        )
        scrollbar = ttk.Scrollbar(
            tasks_container,
            orient="vertical",
            command=self.detail_canvas.yview
        )

        self.detail_tasks_frame = tk.Frame(self.detail_canvas, bg=MODERN_COLORS["Light"])
        self.detail_tasks_frame.bind(
            "<Configure>",
            lambda e: self.detail_canvas.configure(scrollregion=self.detail_canvas.bbox("all"))
        )

        self.detail_canvas.create_window((0, 0), window=self.detail_tasks_frame, anchor="nw")
        self.detail_canvas.configure(yscrollcommand=scrollbar.set)

        self.detail_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel para scroll
        def _on_mousewheel(e):
            self.detail_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        self.detail_canvas.bind("<Enter>", lambda e: self.detail_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.detail_canvas.bind("<Leave>", lambda e: self.detail_canvas.unbind_all("<MouseWheel>"))

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

    def _go_today(self):
        """Navega al mes actual y selecciona hoy"""
        self.current_date = today_date()
        self.selected_date = today_date()
        self.refresh()

    def refresh(self):
        """Refresca todo el calendario"""
        # Actualizar título
        month_names = CALENDAR_CONFIG["month_names"]
        self.month_label.configure(
            text=f"{month_names[self.current_date.month - 1]} {self.current_date.year}"
        )

        # Limpiar días existentes
        for widget in self.days_frame.winfo_children():
            widget.destroy()
        self.day_cells.clear()

        # Obtener datos del mes
        year = self.current_date.year
        month = self.current_date.month

        # Primer día del mes y total de días
        first_day = date(year, month, 1)
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Día de la semana del primer día (0=Lunes, 6=Domingo)
        first_weekday = first_day.weekday()

        # Obtener tareas del mes agrupadas por fecha
        tasks_by_date = self.store.get_tasks_by_month(year, month)

        # Obtener XP por día para heatmap
        xp_by_date = self.gm.get_xp_for_date_range(first_day, last_day)
        max_xp = max(xp_by_date.values()) if xp_by_date else 1

        # Renderizar días
        current_row = 0
        current_col = first_weekday

        current_day = first_day
        while current_day <= last_day:
            date_str = fmt_date(current_day)
            tasks = tasks_by_date.get(date_str, [])
            xp = xp_by_date.get(date_str, 0)

            self._render_day_cell(
                current_day,
                current_row,
                current_col,
                tasks,
                xp,
                max_xp
            )

            current_col += 1
            if current_col > 6:
                current_col = 0
                current_row += 1
            current_day += timedelta(days=1)

        # Configurar columnas con ancho fijo (sin weight para mantener tamaño natural)
        for col in range(7):
            self.days_frame.grid_columnconfigure(col, minsize=110, uniform="cols")

        # Actualizar panel de detalle
        self._update_detail_panel(self.selected_date)

    def _render_day_cell(self, day_date, row, col, tasks, xp, max_xp):
        """Renderiza una celda de día con heatmap"""
        # Calcular color de heatmap
        if xp == 0:
            heatmap_color = self.HEATMAP_COLORS[0]
        else:
            # Normalizar XP a índice de color (1-4)
            level = min(4, max(1, int((xp / max(max_xp, 1)) * 4)))
            heatmap_color = self.HEATMAP_COLORS[level]

        # Determinar si es hoy o está seleccionado
        is_today = day_date == today_date()
        is_selected = day_date == self.selected_date

        # Color del borde
        if is_today:
            border_color = MODERN_COLORS["Primary"]
            border_width = 3
        elif is_selected:
            border_color = MODERN_COLORS["Warning"]
            border_width = 2
        else:
            border_color = heatmap_color
            border_width = 1

        # Frame exterior (borde)
        outer = tk.Frame(self.days_frame, bg=border_color, relief="flat", bd=0)
        outer.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")

        # Frame interior (contenido)
        cell = tk.Frame(outer, bg=heatmap_color, cursor="hand2")
        cell.pack(fill="both", expand=True, padx=border_width, pady=border_width)

        # Número del día
        text_color = "white" if xp > 0 else MODERN_COLORS["Text"]
        day_num = tk.Label(
            cell,
            text=str(day_date.day),
            bg=heatmap_color,
            fg=text_color,
            font=("Segoe UI", 12, "bold" if is_today else "normal")
        )
        day_num.pack(anchor="nw", padx=6, pady=4)

        # Indicador de tareas
        if tasks:
            task_count = len(tasks)
            pending_count = sum(1 for t in tasks if not t.done)

            # Crear frame para indicadores
            indicators_frame = tk.Frame(cell, bg=heatmap_color)
            indicators_frame.pack(fill="x", padx=4, pady=2)

            # Mostrar cantidad de tareas
            task_text = f"{task_count} tarea{'s' if task_count > 1 else ''}"
            if pending_count < task_count:
                task_text += f" ({pending_count} pend.)"

            tasks_label = tk.Label(
                indicators_frame,
                text=task_text,
                bg=heatmap_color,
                fg=text_color,
                font=("Segoe UI", 8)
            )
            tasks_label.pack(anchor="w")

        # XP del día
        if xp > 0:
            xp_label = tk.Label(
                cell,
                text=f"+{xp} XP",
                bg=heatmap_color,
                fg="white",
                font=("Segoe UI", 8, "bold")
            )
            xp_label.pack(anchor="se", padx=4, pady=2)

        # Guardar referencia
        date_str = fmt_date(day_date)
        self.day_cells[date_str] = cell

        # Bindings para click - aplicar a todos los widgets de la celda
        def on_click(e, d=day_date):
            self.selected_date = d
            self.refresh()

        # Lista de todos los widgets que necesitan el binding
        clickable_widgets = [cell, day_num]

        # Agregar labels de tareas si existen
        if tasks:
            clickable_widgets.append(indicators_frame)
            clickable_widgets.append(tasks_label)

        # Agregar label de XP si existe
        if xp > 0:
            clickable_widgets.append(xp_label)

        # Aplicar binding a todos los widgets
        for widget in clickable_widgets:
            widget.bind("<Button-1>", on_click)

        # Configurar como drop target para DnD
        if DND_AVAILABLE:
            try:
                cell.drop_target_register(DND_ALL)
                cell.dnd_bind("<<Drop>>", lambda e, d=day_date: self._on_task_drop(e, d))
            except Exception:
                pass

        # Configurar fila con altura fija (sin weight para mantener tamaño natural)
        self.days_frame.grid_rowconfigure(row, minsize=80)

    def _on_task_drop(self, event, target_date):
        """Maneja el drop de una tarea en una celda de día"""
        try:
            task_id = event.data.strip()
            # Limpiar caracteres extra que puede agregar TkinterDnD
            if task_id.startswith("{"):
                task_id = task_id[1:]
            if task_id.endswith("}"):
                task_id = task_id[:-1]

            success = self.store.update_task_date(task_id, target_date)

            if success:
                # Refrescar vistas
                self.refresh()
                if self.app:
                    self.app.render_tasks()
        except Exception as e:
            print(f"[ERROR] Error al mover tarea: {e}")

    def _update_detail_panel(self, day_date):
        """Actualiza el panel de detalle con las tareas del día seleccionado"""
        # Limpiar tareas anteriores
        for widget in self.detail_tasks_frame.winfo_children():
            widget.destroy()

        # Actualizar labels
        month_names = CALENDAR_CONFIG["month_names"]
        self.detail_date_label.configure(
            text=f"{day_date.day} de {month_names[day_date.month - 1]}, {day_date.year}"
        )

        # Obtener XP del día
        xp_data = self.gm.get_xp_for_date_range(day_date, day_date)
        xp = xp_data.get(fmt_date(day_date), 0)
        self.detail_xp_label.configure(text=f"XP ganado: {xp}" if xp > 0 else "")

        # Obtener tareas del día
        tasks = self.store.get_tasks_for_date(day_date)

        if not tasks:
            tk.Label(
                self.detail_tasks_frame,
                text="No hay tareas para este día",
                bg=MODERN_COLORS["Light"],
                fg=MODERN_COLORS["TextLight"],
                font=("Segoe UI", 10, "italic")
            ).pack(pady=20)
            return

        # Indicación de interacción
        hint_label = tk.Label(
            self.detail_tasks_frame,
            text="💡 Doble clic en título para abrir como posit | 📌 = Abrir posit",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 8, "italic")
        )
        hint_label.pack(anchor="w", pady=(0, 4))

        # Renderizar tareas
        for task in tasks:
            self._render_task_in_detail(task)

    def _render_task_in_detail(self, task):
        """Renderiza una tarea en el panel de detalle"""
        color = VIBRANT_COLORS.get(task.color, "#CCCCCC")
        priority_info = PRIORITY_LEVELS.get(task.priority, PRIORITY_LEVELS["medium"])

        # Frame de la tarea
        task_frame = tk.Frame(
            self.detail_tasks_frame,
            bg=MODERN_COLORS["Card"],
            relief="solid",
            bd=1
        )
        task_frame.pack(fill="x", pady=3, padx=4)

        # Stripe de color
        stripe = tk.Frame(task_frame, bg=color, width=5)
        stripe.pack(side="left", fill="y")

        # Contenido principal
        content = tk.Frame(task_frame, bg=MODERN_COLORS["Card"])
        content.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        # Fila superior: checkbox + título
        top_row = tk.Frame(content, bg=MODERN_COLORS["Card"])
        top_row.pack(fill="x")

        # Checkbox
        var_done = tk.BooleanVar(value=task.done)
        chk = tk.Checkbutton(
            top_row,
            variable=var_done,
            command=lambda t=task: self._toggle_task_done(t.id),
            bg=MODERN_COLORS["Card"],
            activebackground=MODERN_COLORS["Card"]
        )
        chk.pack(side="left")

        # Emoji de prioridad
        tk.Label(
            top_row,
            text=priority_info["emoji"],
            bg=MODERN_COLORS["Card"],
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(0, 4))

        # Título (clickeable para abrir como posit)
        title_style = "overstrike" if task.done else "normal"
        title_fg = MODERN_COLORS["TextLight"] if task.done else MODERN_COLORS["Text"]
        title_label = tk.Label(
            top_row,
            text=task.title[:50] + ("..." if len(task.title) > 50 else ""),
            bg=MODERN_COLORS["Card"],
            fg=title_fg,
            font=("Segoe UI", 10, title_style),
            anchor="w",
            cursor="hand2"
        )
        title_label.pack(side="left", fill="x", expand=True)

        # Doble clic para abrir como posit
        title_label.bind("<Double-Button-1>", lambda e, t=task: self._open_as_posit(t.id))

        # Efecto hover en el título
        def on_enter(e):
            title_label.configure(fg=MODERN_COLORS["Primary"])
        def on_leave(e):
            title_label.configure(fg=title_fg)
        title_label.bind("<Enter>", on_enter)
        title_label.bind("<Leave>", on_leave)

        # Descripción (si existe y es corta)
        if task.desc and len(task.desc) <= 100:
            tk.Label(
                content,
                text=task.desc[:100],
                bg=MODERN_COLORS["Card"],
                fg=MODERN_COLORS["TextLight"],
                font=("Segoe UI", 9),
                anchor="w",
                wraplength=400,
                justify="left"
            ).pack(fill="x", pady=(2, 0))

        # Botones de acción
        actions_frame = tk.Frame(task_frame, bg=MODERN_COLORS["Card"])
        actions_frame.pack(side="right", padx=8, pady=6)

        # Botón editar
        edit_btn = tk.Label(
            actions_frame,
            text="✏️",
            bg=MODERN_COLORS["Card"],
            cursor="hand2",
            font=("Segoe UI", 10)
        )
        edit_btn.pack(side="left", padx=2)
        edit_btn.bind("<Button-1>", lambda e, t=task: self._open_task_editor(t.id))

        # Botón Pomodoro
        pomo_btn = tk.Label(
            actions_frame,
            text="🍅",
            bg=MODERN_COLORS["Card"],
            cursor="hand2",
            font=("Segoe UI", 10)
        )
        pomo_btn.pack(side="left", padx=2)
        pomo_btn.bind("<Button-1>", lambda e, t=task: self._add_to_pomodoro(t.id))

        # Botón abrir como Posit
        posit_btn = tk.Label(
            actions_frame,
            text="📌",
            bg=MODERN_COLORS["Card"],
            cursor="hand2",
            font=("Segoe UI", 10)
        )
        posit_btn.pack(side="left", padx=2)
        posit_btn.bind("<Button-1>", lambda e, t=task: self._open_as_posit(t.id))

    def _toggle_task_done(self, task_id: str):
        """Marca/desmarca una tarea como completada"""
        idx = self.store.index_by_id(task_id)
        if idx >= 0:
            self.store.toggle_done(idx)
            self.refresh()
            if self.app:
                self.app.render_tasks()
                self.app._update_statistics()

    def _open_task_editor(self, task_id: str):
        """Abre el editor de la tarea"""
        if self.app:
            self.app.open_note_by_id(task_id)

    def _add_to_pomodoro(self, task_id: str):
        """Agrega la tarea a la cola de Pomodoro"""
        if self.app and hasattr(self.app, 'pomodoro_manager'):
            self.app.pomodoro_manager.add_task_to_queue(task_id)

    def _open_as_posit(self, task_id: str):
        """Abre la tarea como un posit flotante (QuickStickyWindow)"""
        if self.app:
            self.app.open_quick_sticky_by_id(task_id)

    def _add_task_for_selected_date(self):
        """Abre el diálogo de agregar tarea con la fecha seleccionada preestablecida"""
        if self.app and hasattr(self.app, 'open_add_dialog_with_date'):
            self.app.open_add_dialog_with_date(self.selected_date)

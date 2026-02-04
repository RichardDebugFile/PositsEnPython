#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aplicación principal de Posits Virtuales
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date

# Importar tkinterdnd2 si está disponible
try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None
    DND_AVAILABLE = False

from .config import APP_NAME, MODERN_COLORS, GRADIENTS, COLOR_LABELS, LABEL_TO_KEY, VIBRANT_COLORS, PRIORITY_LEVELS
from .models import TaskStore, PomodoroManager
from .utils.dates import today_date, fmt_date, parse_date
from .utils.logger import logger
from .utils.colors import urgency_color
from .ui import (
    PillButton,
    create_centered_row,
    create_stat_card,
    TaskCardRenderer,
    ModernNoteWindow,
    QuickStickyWindow,
    GamificationPanel,
    PomodoroWindow,
    CalendarPanel,
    MiniCalendarWidget,
)
from .dialogs import ModernAddTaskDialog, OllamaCaptureDialog
from .services import NotificationService
from .services.music_player import MusicPlayer


def update_statistics(app):
    """Actualiza las estadísticas en el header de la app"""
    stats = app.store.get_statistics()
    if hasattr(app, 'total_label'):
        app.total_label.configure(text=str(stats["total"]))
    if hasattr(app, 'completed_label'):
        app.completed_label.configure(text=str(stats["completed"]))
    if hasattr(app, 'pending_label'):
        app.pending_label.configure(text=str(stats["pending"]))
    if hasattr(app, 'footer_label'):
        app.footer_label.configure(
            text=f"✨ {stats['total']} tareas • {stats['completed']} completadas"
        )


class ModernStickyApp(TkinterDnD.Tk if DND_AVAILABLE else tk.Tk):
    BG = MODERN_COLORS["Background"]

    def __init__(self, store: TaskStore):
        super().__init__()
        self.store = store
        self.store.set_after(self.after)
        self.title(APP_NAME)
        self.configure(bg=self.BG)

        # Inicializar reproductor de música
        try:
            self.music_player = MusicPlayer()
        except (ImportError, OSError, RuntimeError) as e:
            print(f"[WARNING] No se pudo inicializar el reproductor de música: {e}")
            self.music_player = None

        # Inicializar gestor de Pomodoro
        self.pomodoro_manager = PomodoroManager()
        self.pomodoro_window = None

        self.geometry("1450x850")  # Tamaño aumentado para mejor visualización (tamaño estándar)
        self.minsize(1280, 750)  # Tamaño mínimo de la app
        self.resizable(True, True)

        self._create_header()
        self._create_toolbar()

        # Frame principal con 2 columnas: tabs (tareas/calendario) + gamificación
        main_container = tk.Frame(self, bg=self.BG)
        main_container.pack(fill="both", expand=True, padx=8, pady=8)

        # Columna izquierda: Notebook con tabs (65%)
        self.left_column = tk.Frame(main_container, bg=self.BG)
        self.left_column.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Columna derecha: panel de gamificación (35% - más espacio)
        self.gamification_column = tk.Frame(main_container, bg=self.BG, width=350)
        self.gamification_column.pack(side="right", fill="y", padx=(8, 8))
        self.gamification_column.pack_propagate(False)

        # Crear sistema de tabs
        self._create_tabs_notebook()
        self._create_content_area()  # Crea canvas en tasks_column (dentro del tab de tareas)
        self._create_calendar_panel()  # Crea panel de calendario en el segundo tab
        self._create_gamification_panel()
        self._create_footer()

        # Mini calendario flotante (inicialmente oculto)
        self.mini_calendar = None

        self.note_windows: dict[str, ModernNoteWindow] = {}
        self.quick_windows: dict[str, QuickStickyWindow] = {}

        # Iniciar servicio de notificaciones
        self.notification_service = NotificationService(lambda: self.store.tasks)
        self.notification_service.start()

        self.render_tasks()
        self.after(200, self.reopen_notes)
        self.after(400, self.open_daily_mission_posits)  # Abrir posits de misiones diarias

    def _create_tabs_notebook(self):
        """Crea el sistema de tabs personalizado (más visible que ttk.Notebook)"""
        # Frame para los tabs (header)
        self.tabs_header = tk.Frame(self.left_column, bg=self.BG, height=50)
        self.tabs_header.pack(fill="x", pady=(0, 4))
        self.tabs_header.pack_propagate(False)

        # Variable para rastrear tab activo
        self.current_tab = -1  # -1 para permitir que el primer _switch_tab(0) se ejecute correctamente

        # Crear botones de tabs personalizados
        self.tab_buttons = []

        # Tab 1: Tareas
        btn_tasks = tk.Frame(self.tabs_header, bg="#CCCCCC", relief="raised", bd=2, cursor="hand2")
        btn_tasks.pack(side="left", fill="both", expand=True, padx=(0, 2))

        lbl_tasks = tk.Label(
            btn_tasks,
            text="📋 Tareas",
            bg="#CCCCCC",
            fg="#000000",
            font=("Segoe UI", 12, "bold"),
            cursor="hand2"
        )
        lbl_tasks.pack(expand=True, pady=12)
        lbl_tasks.bind("<Button-1>", lambda e: self._switch_tab(0))
        btn_tasks.bind("<Button-1>", lambda e: self._switch_tab(0))

        self.tab_buttons.append((btn_tasks, lbl_tasks))

        # Tab 2: Calendario
        btn_calendar = tk.Frame(self.tabs_header, bg="#CCCCCC", relief="raised", bd=2, cursor="hand2")
        btn_calendar.pack(side="left", fill="both", expand=True, padx=(2, 0))

        lbl_calendar = tk.Label(
            btn_calendar,
            text="📅 Calendario",
            bg="#CCCCCC",
            fg="#000000",
            font=("Segoe UI", 12, "bold"),
            cursor="hand2"
        )
        lbl_calendar.pack(expand=True, pady=12)
        lbl_calendar.bind("<Button-1>", lambda e: self._switch_tab(1))
        btn_calendar.bind("<Button-1>", lambda e: self._switch_tab(1))

        self.tab_buttons.append((btn_calendar, lbl_calendar))

        # Container para el contenido de los tabs
        self.tabs_container = tk.Frame(self.left_column, bg=self.BG)
        self.tabs_container.pack(fill="both", expand=True)

        # Tab 1: Tareas
        self.tasks_column = tk.Frame(self.tabs_container, bg=self.BG)

        # Tab 2: Calendario
        self.calendar_column = tk.Frame(self.tabs_container, bg=self.BG)

        # Activar el primer tab
        self._switch_tab(0)

    def _switch_tab(self, tab_index):
        """Cambia entre tabs"""
        if tab_index == self.current_tab:
            return

        # Ocultar tab actual
        if self.current_tab == 0:
            self.tasks_column.pack_forget()
        else:
            self.calendar_column.pack_forget()

        # Actualizar estilos de botones
        for i, (btn_frame, btn_label) in enumerate(self.tab_buttons):
            if i == tab_index:
                # Tab seleccionado: Azul brillante
                btn_frame.config(bg="#2196F3", relief="solid", bd=3)
                btn_label.config(bg="#2196F3", fg="#FFFFFF")
            else:
                # Tab no seleccionado: Gris
                btn_frame.config(bg="#CCCCCC", relief="raised", bd=2)
                btn_label.config(bg="#CCCCCC", fg="#000000")

        # Mostrar nuevo tab
        if tab_index == 0:
            self.tasks_column.pack(fill="both", expand=True)
        else:
            self.calendar_column.pack(fill="both", expand=True)
            # Refrescar calendario al cambiar a él
            if hasattr(self, 'calendar_panel'):
                self.calendar_panel.refresh()

        self.current_tab = tab_index

    def _create_calendar_panel(self):
        """Crea el panel de calendario interactivo"""
        self.calendar_panel = CalendarPanel(
            self.calendar_column,
            self.store,
            self.store.gamification,
            app=self
        )
        self.calendar_panel.pack(fill="both", expand=True)


    def toggle_mini_calendar(self):
        """Muestra/oculta el mini calendario flotante"""
        if self.mini_calendar is None:
            self.mini_calendar = MiniCalendarWidget(
                self,
                self.store,
                on_date_click=self._on_mini_calendar_date_click
            )
        else:
            self.mini_calendar.toggle()

    def _on_mini_calendar_date_click(self, clicked_date):
        """Callback cuando se hace click en una fecha del mini calendario"""
        # Cambiar al tab de calendario
        self._switch_tab(1)
        # Seleccionar la fecha y navegar al mes
        if hasattr(self, 'calendar_panel'):
            self.calendar_panel.selected_date = clicked_date
            self.calendar_panel.current_date = clicked_date
            self.calendar_panel.refresh()

    def _create_header(self):
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0], relief="flat", bd=0)
        header.pack(fill="x")
        title_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0])
        title_frame.pack(fill="x", padx=16, pady=12)
        tk.Label(
            title_frame,
            text=APP_NAME,
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        stats_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0])
        stats_frame.pack(fill="x", padx=16, pady=(0, 12))
        _, self.total_label = create_stat_card(
            stats_frame, "Total", 0, "Primary", "📋"
        )
        _, self.completed_label = create_stat_card(
            stats_frame, "Completadas", 0, "Success", "✅"
        )
        _, self.pending_label = create_stat_card(
            stats_frame, "Pendientes", 0, "Warning", "⏳"
        )
        update_statistics(self)

    def _create_toolbar(self):
        toolbar = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        toolbar.pack(fill="x", padx=8, pady=8)
        center = create_centered_row(toolbar)

        PillButton(
            center, "Nueva Tarea", self.open_add_dialog, "Primary", "normal", "➕"
        ).pack(side="left", padx=6)

        self.var_only_pending = tk.BooleanVar(value=False)
        tk.Checkbutton(
            center,
            text="👁️ Solo Pendientes",
            variable=self.var_only_pending,
            command=self.render_tasks,
            bg=center.cget("bg"),
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=8)

        PillButton(
            center, "Abrir Notas", self.reopen_notes, "Secondary", "normal", "📝"
        ).pack(side="left", padx=6)

        # --- Botón IA (Ollama) ---
        PillButton(
            center, "IA (Ollama)", self.open_ollama_dialog, "Success", "normal", "🤖"
        ).pack(side="left", padx=6)

        # --- Botón Descargar Música ---
        PillButton(
            center, "Descargar Música", self.open_music_downloader, "Info", "normal", "🎵"
        ).pack(side="left", padx=6)

        # --- Botón Pomodoro ---
        PillButton(
            center, "Pomodoro", self.open_pomodoro, "Warning", "normal", "🍅"
        ).pack(side="left", padx=6)

        # --- Botón Mini Calendario ---
        PillButton(
            center, "Mini Cal", self.toggle_mini_calendar, "Primary", "normal", "📅"
        ).pack(side="left", padx=6)

        # Ordenamiento
        sort_frame = tk.Frame(center, bg=center.cget("bg"))
        sort_frame.pack(side="left", padx=8)
        tk.Label(sort_frame, text="Ordenar:", bg=center.cget("bg"),
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0,4))
        self.var_sort = tk.StringVar(value="Fecha")
        sort_options = ["Fecha", "Prioridad", "Título", "Color"]
        self.cb_sort = ttk.Combobox(sort_frame, textvariable=self.var_sort,
                                    state="readonly", width=10, values=sort_options)
        self.cb_sort.pack(side="left")
        self.cb_sort.bind("<<ComboboxSelected>>", lambda e: self._on_sort_change())

        # Filtro de color
        self.color_filter_frame = tk.Frame(center, bg=center.cget("bg"))
        self.color_filter_frame.pack(side="left", padx=8)
        tk.Label(self.color_filter_frame, text="Color:", bg=center.cget("bg"),
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0,4))
        self.var_color_filter = tk.StringVar(value="Todos")
        self.cb_color_filter = ttk.Combobox(self.color_filter_frame, textvariable=self.var_color_filter,
                                            state="readonly", width=12, values=["Todos"])
        self.cb_color_filter.pack(side="left")
        self.cb_color_filter.bind(
            "<<ComboboxSelected>>", lambda e: self.render_tasks()
        )

        self.var_topmost = tk.BooleanVar(value=True)

        def _toggle_topmost():
            self.attributes("-topmost", self.var_topmost.get())
        tk.Checkbutton(
            center,
            text="Siempre arriba",
            variable=self.var_topmost,
            command=_toggle_topmost,
            bg=center.cget("bg"),
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=8)
        _toggle_topmost()

    def _create_content_area(self):
        """Crea el área de contenido con scroll para las tareas"""
        content_frame = tk.Frame(self.tasks_column, bg=self.BG)
        content_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            content_frame, bg=self.BG, highlightthickness=0, relief="flat", bd=0
        )
        self.scrollbar = ttk.Scrollbar(
            content_frame, orient="vertical", command=self.canvas.yview
        )

        # Frame de scroll (va dentro del canvas)
        scroll_frame = tk.Frame(self.canvas, bg=self.BG)

        # Frame centrador con grid para mejor control
        center_wrapper = tk.Frame(scroll_frame, bg=self.BG)
        center_wrapper.pack(fill="both", expand=True)

        # Configurar grid: columnas laterales con peso 1, columna central sin peso
        center_wrapper.grid_columnconfigure(0, weight=1)  # Espacio izquierdo
        center_wrapper.grid_columnconfigure(1, weight=0)  # Columna de tareas (ancho fijo)
        center_wrapper.grid_columnconfigure(2, weight=1)  # Espacio derecho

        # Frame de tareas (contenido real, en columna central)
        self.task_frame = tk.Frame(center_wrapper, bg=self.BG)
        self.task_frame.grid(row=0, column=1, sticky="n", padx=20)  # sticky="n" para arriba, padx para márgenes

        scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Crear window del scroll_frame en el canvas
        self.task_window = self.canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind para ajustar el ancho del scroll_frame al ancho del canvas
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.task_window, width=e.width))

        self._bind_mousewheel()

    def _bind_mousewheel(self):
        """Vincula el scroll del mouse solo cuando está sobre el canvas"""
        def _on_mousewheel(e):
            delta = e.delta
            if delta == 0:
                return
            self.canvas.yview_scroll(int(-1*(delta/120)), "units")

        def _on_mousewheel_linux_up(e):
            self.canvas.yview_scroll(-3, "units")

        def _on_mousewheel_linux_down(e):
            self.canvas.yview_scroll(3, "units")

        def _on_enter(e):
            """Cuando el mouse entra al canvas, activar scroll"""
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            self.canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)

        def _on_leave(e):
            """Cuando el mouse sale del canvas, desactivar scroll"""
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        # Vincular eventos de entrada/salida
        self.canvas.bind("<Enter>", _on_enter)
        self.canvas.bind("<Leave>", _on_leave)

    def _create_gamification_panel(self):
        """Crea el panel de gamificación con XP, nivel y misiones diarias"""
        self.gamification_panel = GamificationPanel(
            self.gamification_column,
            self.store.gamification,
            app=self,  # Pasar referencia a la app
            music_player=self.music_player  # Pasar reproductor de música
        )
        self.gamification_panel.pack(fill="both", expand=True)

    def _create_footer(self):
        footer = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        footer.pack(fill="x", padx=8, pady=8)
        self.footer_label = tk.Label(
            footer,
            text="✨ 0 tareas • 0 completadas",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9)
        )
        self.footer_label.pack(pady=4)

    # ---------- Diálogo agregar ----------
    def open_add_dialog(self):
        ModernAddTaskDialog(self, self._on_add)

    def open_add_dialog_with_date(self, due: date):
        """Abre el diálogo de agregar tarea con una fecha preseleccionada"""
        ModernAddTaskDialog(
            self,
            self._on_add,
            preset={"due": due}
        )

    def open_add_dialog_prefilled(
        self, title: str, desc: str, due: date, priority: str, color: str
    ):
        ModernAddTaskDialog(
            self,
            self._on_add,
            preset={
                "title": title,
                "desc": desc,
                "due": due,
                "priority": priority,
                "color": color
            }
        )

    def _on_add(
        self,
        title: str,
        desc: str,
        due: date,
        priority: str,
        color: str | None = None
    ):
        self.store.add(title, desc, due, priority, color=color)
        # Asegúrate de que la nueva tarea sea visible aunque haya un filtro activo:
        if hasattr(self, "var_color_filter"):
            self.var_color_filter.set("Todos")
        self.render_tasks()
        update_statistics(self)


    # ---------- IA (Ollama) ----------
    def open_ollama_dialog(self):
        OllamaCaptureDialog(self, self._on_add)

    # ---------- Descargador de Música ----------
    def open_music_downloader(self):
        """Abre diálogo de descarga de música desde YouTube"""
        from .dialogs import MusicDownloaderDialog
        MusicDownloaderDialog(self, self.music_player)

    def open_pomodoro(self):
        """Abre la ventana de Pomodoro"""
        if self.pomodoro_window is None or not self.pomodoro_window.winfo_exists():
            self.pomodoro_window = PomodoroWindow(
                self,
                self.pomodoro_manager,
                self.store,
                self.music_player
            )
        else:
            # Si ya existe, traerla al frente
            self.pomodoro_window.lift()
            self.pomodoro_window.focus()

    # ---------- Filtro de color dinámico ----------
    def _refresh_color_filter(self):
        colors_present = sorted({t.color for t in self.store.tasks})
        if len(self.store.tasks) == 0:
            self.color_filter_frame.pack_forget()
            self.var_color_filter.set("Todos")
            return
        labels = ["Todos"] + [COLOR_LABELS.get(k, k) for k in colors_present]
        current = self.var_color_filter.get()
        self.cb_color_filter["values"] = labels
        if not self.color_filter_frame.winfo_ismapped():
            self.color_filter_frame.pack(side="left", padx=8)
        if current not in labels:
            self.var_color_filter.set("Todos")

    # ---------- Ordenamiento ----------
    def _on_sort_change(self):
        """Maneja el cambio de criterio de ordenamiento"""
        sort_map = {
            "Fecha": "date",
            "Prioridad": "priority",
            "Título": "title",
            "Color": "color"
        }
        sort_by = sort_map.get(self.var_sort.get(), "date")
        self.store.sort_tasks(sort_by)
        self.render_tasks()

    # ---------- Render ----------
    def render_tasks(self):
        self._refresh_color_filter()
        logger.debug("Render: only_pending=%s color_filter=%s",
                    getattr(self, "var_only_pending", tk.BooleanVar(value=False)).get(),
                    getattr(self, "var_color_filter", tk.StringVar(value="Todos")).get())
        for w in self.task_frame.winfo_children():
            w.destroy()

        now = today_date()
        selected_label = self.var_color_filter.get()
        selected_key = LABEL_TO_KEY.get(selected_label) if selected_label and selected_label != "Todos" else None

        for task in self.store.tasks:
            if self.var_only_pending.get() and task.done:
                continue
            if selected_key and task.color != selected_key:
                continue
            self._render_modern_task_card(task, now)

        update_statistics(self)

        # Actualizar panel de gamificación
        if hasattr(self, 'gamification_panel'):
            self.gamification_panel.refresh()

        # Actualizar calendario si está visible
        if hasattr(self, 'calendar_panel') and hasattr(self, 'current_tab'):
            if self.current_tab == 1:  # Tab de calendario activo
                self.calendar_panel.refresh()

        # Actualizar mini calendario si está visible
        if hasattr(self, 'mini_calendar') and self.mini_calendar and self.mini_calendar.winfo_exists():
            try:
                if self.mini_calendar.winfo_viewable():
                    self.mini_calendar.refresh()
            except Exception:
                pass

    def _render_modern_task_card(self, task: dict, now: date):
        tid = task.id
        border_color = self._get_task_border_color(task, now)

        # Container con borde de color (ancho de 750px para consistencia)
        outer = tk.Frame(self.task_frame, bg=border_color, width=750)
        outer.pack(pady=4)
        outer.pack_propagate(True)  # Permitir que se ajuste a la altura del contenido

        card = tk.Frame(outer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        card.pack(fill="both", expand=True, padx=2, pady=2)

        stripe_color = VIBRANT_COLORS.get(task.color, "#CCCCCC")
        stripe = tk.Frame(card, bg=stripe_color, width=10)
        stripe.pack(side="left", fill="y")

        body = tk.Frame(card, bg=GRADIENTS["Card"][0])
        body.pack(side="left", fill="both", expand=True)

        header = tk.Frame(body, bg=GRADIENTS["Card"][0])
        header.pack(fill="x", padx=12, pady=8)

        var_done = tk.BooleanVar(value=task.done)
        tk.Checkbutton(
            header,
            variable=var_done,
            command=lambda t=tid: self._toggle_done_with_update_by_id(t),
            bg=GRADIENTS["Card"][0],
            font=("Segoe UI", 12)
        ).pack(side="left")

        title_font = ("Segoe UI", 12, "bold")
        title_color = (
            MODERN_COLORS["Text"] if not task.done else MODERN_COLORS["TextLight"]
        )

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
        lbl_title.bind(
            "<Double-Button-1>", lambda e, t=tid: self.open_quick_sticky_by_id(t)
        )
        # Botón extra para abrir editor completo sigue en acciones

        # Mostrar nivel de prioridad
        priority_key = task.priority if isinstance(task.priority, str) else ("high" if task.priority else "medium")
        if priority_key in PRIORITY_LEVELS:
            priority_info = PRIORITY_LEVELS[priority_key]
            tk.Label(
                header,
                text=f"{priority_info['emoji']} {priority_info['label'].upper()}",
                bg=priority_info['color'],
                fg="white",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=2
            ).pack(side="right")

        content = tk.Frame(body, bg=GRADIENTS["Card"][0])
        content.pack(fill="x", padx=12, pady=(0, 8))
        if task.desc:
            desc_font = ("Segoe UI", 10, "italic" if task.done else "normal")
            desc_fg = (
                MODERN_COLORS["TextLight"] if task.done else MODERN_COLORS["Text"]
            )
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

        footer = tk.Frame(body, bg=GRADIENTS["Card"][0])
        footer.pack(fill="x", padx=12, pady=(0, 8))
        try:
            due_d = (
                task.due if isinstance(task.due, date) else parse_date(task.due)
            )
        except Exception:
            due_d = now

        due_txt = f"📅 {fmt_date(due_d)}"
        overdue = (due_d <= now)
        due_fg = (
            MODERN_COLORS["Danger"]
            if overdue and not task.done
            else MODERN_COLORS["TextLight"]
        )
        tk.Label(
            footer,
            text=due_txt,
            bg=GRADIENTS["Card"][0],
            fg=due_fg,
            font=("Segoe UI", 9)
        ).pack(side="left")

        actions = tk.Frame(footer, bg=GRADIENTS["Card"][0])
        actions.pack(side="right")
        priority_icon = "⭐" if task.priority else "☆"
        self._create_small_button(
            actions,
            priority_icon,
            lambda t=tid: self._toggle_priority_with_update_by_id(t),
            "Warning"
        )
        self._create_small_button(
            actions, "🍅", lambda t=tid: self._add_task_to_pomodoro(t), "Warning"
        )
        self._create_small_button(
            actions, "📝", lambda t=tid: self.open_note_by_id(t), "Primary"
        )
        self._create_small_button(
            actions, "🗑️", lambda t=tid: self._delete_task_with_update_by_id(t), "Danger"
        )

    def _create_small_button(self, parent, icon, command, color):
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

    def _toggle_done_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0:
            return
        self.store.toggle_done(idx)
        self.render_tasks()
        update_statistics(self)
        # Refrescar panel de gamificación para actualizar rachas y misiones
        if hasattr(self, 'gamification_panel'):
            self.gamification_panel.refresh()

    def _toggle_priority_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0:
            return
        self.store.toggle_priority(idx)
        self.render_tasks()
        update_statistics(self)

    def _delete_task_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0:
            return
        self.delete_task_by_index(idx, tid)
        update_statistics(self)

    def _add_task_to_pomodoro(self, tid: str):
        """Agrega una tarea a la cola de Pomodoro"""
        try:
            # Agregar tarea a la cola
            self.pomodoro_manager.add_task_to_queue(tid)

            # Abrir ventana de Pomodoro si no está abierta
            if self.pomodoro_window is None or not self.pomodoro_window.winfo_exists():
                self.open_pomodoro()
            else:
                # Si ya está abierta, solo refrescar la lista
                self.pomodoro_window._refresh_task_list()
                self.pomodoro_window.lift()

            # Obtener tarea para mostrar nombre
            task = self.store.get_by_id(tid)
            if task:
                print(f"[INFO] Tarea agregada al Pomodoro: {task.title}")
        except Exception as e:
            print(f"[ERROR] No se pudo agregar al Pomodoro: {e}")
            messagebox.showerror("Error", f"No se pudo agregar al Pomodoro:\n{e}")

    def _get_task_border_color(self, task, now: date) -> str:
        if task.done:
            return MODERN_COLORS["Success"]
        if task.priority:
            return MODERN_COLORS["Warning"]
        try:
            due_d = (
                task.due if isinstance(task.due, date) else parse_date(task.due)
            )
            start_d = (
                task.start if isinstance(task.start, date) else parse_date(task.start)
            )
            return urgency_color(start_d, due_d, now)
        except Exception:
            return MODERN_COLORS["TextLight"]

    def delete_task_by_index(self, idx: int, tid: str):
        win = self.note_windows.pop(tid, None)
        if win is not None and win.winfo_exists():
            try:
                win.destroy()
            except Exception:
                pass
        qwin = self.quick_windows.pop(tid, None)
        if qwin is not None and qwin.winfo_exists():
            try:
                qwin.destroy()
            except Exception:
                pass
        self.store.delete(idx)
        self.render_tasks()

    # ---------- Notas flotantes (editor) ----------
    def open_note_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0:
            return
        task = self.store.tasks[idx]
        win = self.note_windows.get(tid)
        if win and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        task.open = True
        self.store.save_throttled()
        win = ModernNoteWindow(self, tid, task)
        self.note_windows[tid] = win

    # ---------- Posit rápido (overlay) ----------
    def open_quick_sticky_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0:
            return
        task = self.store.tasks[idx]
        qwin = self.quick_windows.get(tid)
        if qwin and qwin.winfo_exists():
            qwin.deiconify()
            qwin.lift()
            return
        qwin = QuickStickyWindow(self, task)
        self.quick_windows[tid] = qwin

    def reopen_notes(self):
        for t in self.store.tasks:
            if t.open or t.pinned:
                try:
                    self.open_note_by_id(t.id)
                except Exception:
                    pass

    def open_daily_mission_posits(self):
        """Abre automáticamente posits de las misiones diarias al iniciar"""
        print(
            "[DEBUG] ===== Iniciando apertura de posits de misiones diarias ====="
        )
        import json
        import os
        from .models.task import Task
        from .utils.dates import today_date, fmt_date

        # Cargar misiones guardadas
        missions_file = os.path.join("data", "daily_missions.json")
        if not os.path.exists(missions_file):
            print(f"[WARNING] No existe el archivo de misiones: {missions_file}")
            return

        print(f"[DEBUG] Archivo de misiones encontrado: {missions_file}")

        try:
            with open(missions_file, "r", encoding="utf-8") as f:
                missions_data = json.load(f)

            # Cargar posiciones guardadas
            positions_file = os.path.join("data", "posit_positions.json")
            saved_positions = {}
            if os.path.exists(positions_file):
                try:
                    with open(positions_file, "r", encoding="utf-8") as f:
                        saved_positions = json.load(f)
                    print(f"[DEBUG] Posiciones cargadas: {saved_positions}")
                except Exception as e:
                    print(f"[WARNING] Error al cargar posiciones: {e}")
                    saved_positions = {}
            else:
                print(f"[INFO] No hay archivo de posiciones guardadas")

            # Posiciones por defecto para los posits (fuera de la pantalla principal, en el escritorio)
            default_positions = [
                (self.winfo_screenwidth() - 250, 100),  # Esquina superior derecha
                (self.winfo_screenwidth() - 250, 300),  # Medio derecha
                (self.winfo_screenwidth() - 250, 500),  # Inferior derecha
            ]

            idx = 0
            for mission_key, mission_data in missions_data.items():
                task_id = f"mission_{mission_key}"

                # Buscar si ya existe esta tarea en el store
                existing_task = self.store.get_by_id(task_id)

                if not existing_task:
                    # Crear tarea real en el TaskStore si no existe
                    # Convertir "Alta" a nivel de prioridad
                    priority_str = "high" if mission_data.get("priority") == "Alta" else "medium"

                    task = Task(
                        id=task_id,
                        title=mission_data.get("title", "Misión Diaria"),
                        desc=mission_data.get("desc", ""),
                        priority=priority_str,
                        color=mission_data.get("color", "Ocean"),
                        start=today_date(),
                        due=today_date()
                    )

                    # Agregar al store
                    self.store.tasks.append(task)
                    self.store.save()
                    print(f"[INFO] Misión diaria creada como tarea real: {task.title}")
                    existing_task = task
                else:
                    print(f"[INFO] Misión diaria ya existe: {existing_task.title}")

                # Crear QuickStickyWindow para la misión usando la tarea real
                qwin = QuickStickyWindow(self, existing_task, skip_centering=True)

                # Usar posición guardada si existe, sino usar posición por defecto
                if task_id in saved_positions:
                    x = saved_positions[task_id]["x"]
                    y = saved_positions[task_id]["y"]
                    print(f"[DEBUG] Usando posición guardada para {task_id}: ({x}, {y})")
                else:
                    # Usar posición por defecto si no hay guardada
                    if idx < len(default_positions):
                        x, y = default_positions[idx]
                    else:
                        x, y = (self.winfo_screenwidth() - 250, 100)
                    print(f"[DEBUG] Usando posición por defecto para {task_id}: ({x}, {y})")

                # Posicionar el posit
                qwin.update_idletasks()  # Actualizar geometría antes de posicionar
                qwin.geometry(f"+{x}+{y}")
                print(f"[DEBUG] Posit {task_id} posicionado en ({x}, {y})")

                # Guardar referencia
                self.quick_windows[task_id] = qwin
                idx += 1

        except (OSError, ValueError, KeyError) as e:
            print(f"[WARNING] No se pudieron abrir posits de misiones: {e}")


    # ==================== Métodos auxiliares para interacción desde task_card y note_window ====================

    def toggle_done_by_id(self, task_id: str):
        """Marca/desmarca tarea como completada por ID"""
        idx = self.store.index_by_id(task_id)
        if idx < 0:
            return

        result = self.store.toggle_done(idx)

        # Notificar si hubo level up
        if result and result.get("leveled_up"):
            self.notification_service.notify_level_up(self.store.gamification.data["level"])
            messagebox.showinfo("¡NIVEL UP!", f"¡Ahora eres nivel {self.store.gamification.data['level']}! 🎉")

        self.render_tasks()
        update_statistics(self)

    def toggle_priority_by_id(self, task_id: str):
        """Marca/desmarca tarea como prioritaria por ID"""
        idx = self.store.index_by_id(task_id)
        if idx >= 0:
            self.store.toggle_priority(idx)
            self.render_tasks()

    def delete_task_by_id(self, task_id: str):
        """Elimina tarea por ID"""
        idx = self.store.index_by_id(task_id)
        if idx >= 0:
            self.delete_task_by_index(idx, task_id)

    def _update_statistics(self):
        """Actualiza estadísticas (wrapper local)"""
        update_statistics(self)

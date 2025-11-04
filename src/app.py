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

from .config import APP_NAME, MODERN_COLORS, GRADIENTS, COLOR_LABELS, LABEL_TO_KEY, VIBRANT_COLORS
from .models import TaskStore
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
        except Exception as e:
            print(f"[WARNING] No se pudo inicializar el reproductor de música: {e}")
            self.music_player = None
        self.geometry("1150x750")  # Más ancho para panel de gamificación expandido
        self.minsize(1150, 700)  # Tamaño mínimo aumentado para ver todo
        self.resizable(True, True)

        self._create_header()
        self._create_toolbar()

        # Frame principal con 2 columnas: tareas + gamificación
        main_container = tk.Frame(self, bg=self.BG)
        main_container.pack(fill="both", expand=True, padx=8, pady=8)

        # Columna izquierda: área de tareas (65%)
        self.tasks_column = tk.Frame(main_container, bg=self.BG)
        self.tasks_column.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Columna derecha: panel de gamificación (35% - más espacio)
        self.gamification_column = tk.Frame(main_container, bg=self.BG, width=350)
        self.gamification_column.pack(side="right", fill="y", padx=(8, 8))
        self.gamification_column.pack_propagate(False)

        self._create_content_area()  # Crea canvas en tasks_column
        self._create_gamification_panel()  # NUEVO: Panel de gamificación
        self._create_footer()

        self.note_windows: dict[str, ModernNoteWindow] = {}
        self.quick_windows: dict[str, QuickStickyWindow] = {}

        # Iniciar servicio de notificaciones
        self.notification_service = NotificationService(self.store)
        self.notification_service.start()

        self.render_tasks()
        self.after(200, self.reopen_notes)
        self.after(400, self.open_daily_mission_posits)  # Abrir posits de misiones diarias

    def _create_header(self):
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0], relief="flat", bd=0); header.pack(fill="x")
        title_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0]); title_frame.pack(fill="x", padx=16, pady=12)
        tk.Label(title_frame, text=APP_NAME, bg=GRADIENTS["Primary"][0], fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        stats_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0]); stats_frame.pack(fill="x", padx=16, pady=(0,12))
        _, self.total_label = create_stat_card(stats_frame, "Total", 0, "Primary", "📋")
        _, self.completed_label = create_stat_card(stats_frame, "Completadas", 0, "Success", "✅")
        _, self.pending_label = create_stat_card(stats_frame, "Pendientes", 0, "Warning", "⏳")
        update_statistics(self)

    def _create_toolbar(self):
        toolbar = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0); toolbar.pack(fill="x", padx=8, pady=8)
        center = create_centered_row(toolbar)

        PillButton(center, "Nueva Tarea", self.open_add_dialog, "Primary", "normal", "➕").pack(side="left", padx=6)

        self.var_only_pending = tk.BooleanVar(value=False)
        tk.Checkbutton(center, text="👁️ Solo Pendientes", variable=self.var_only_pending,
                       command=self.render_tasks, bg=center.cget("bg"),
                       font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)

        PillButton(center, "Abrir Notas", self.reopen_notes, "Secondary", "normal", "📝").pack(side="left", padx=6)

        # --- Botón IA (Ollama) ---
        PillButton(center, "IA (Ollama)", self.open_ollama_dialog, "Success", "normal", "🤖").pack(side="left", padx=6)

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
        self.cb_color_filter.bind("<<ComboboxSelected>>", lambda e: self.render_tasks())

        self.var_topmost = tk.BooleanVar(value=True)
        def _toggle_topmost(): self.attributes("-topmost", self.var_topmost.get())
        tk.Checkbutton(center, text="Siempre arriba", variable=self.var_topmost,
                       command=_toggle_topmost, bg=center.cget("bg"),
                       font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)
        _toggle_topmost()

    def _create_content_area(self):
        """Crea el área de contenido con scroll para las tareas"""
        content_frame = tk.Frame(self.tasks_column, bg=self.BG)
        content_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(content_frame, bg=self.BG, highlightthickness=0, relief="flat", bd=0)
        self.scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=self.canvas.yview)
        self.task_frame = tk.Frame(self.canvas, bg=self.BG)
        self.task_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.task_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel()

    def _bind_mousewheel(self):
        def _on_mousewheel(e):
            delta = e.delta
            if delta == 0: return
            self.canvas.yview_scroll(int(-1*(delta/120)), "units")
        def _on_mousewheel_linux_up(e): self.canvas.yview_scroll(-3, "units")
        def _on_mousewheel_linux_down(e): self.canvas.yview_scroll(3, "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)

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
        footer = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0); footer.pack(fill="x", padx=8, pady=8)
        self.footer_label = tk.Label(footer, text="✨ 0 tareas • 0 completadas",
                                     bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["TextLight"], font=("Segoe UI", 9))
        self.footer_label.pack(pady=4)

    # ---------- Diálogo agregar ----------
    def open_add_dialog(self):
        ModernAddTaskDialog(self, self._on_add)

    def open_add_dialog_prefilled(self, title: str, desc: str, due: date, priority: bool, color: str):
        ModernAddTaskDialog(self, self._on_add, preset={"title": title, "desc": desc, "due": due, "priority": priority, "color": color})

    def _on_add(self, title: str, desc: str, due: date, priority: bool, color: str | None = None):
        self.store.add(title, desc, due, priority, color=color)
        # Asegúrate de que la nueva tarea sea visible aunque haya un filtro activo:
        if hasattr(self, "var_color_filter"):
            self.var_color_filter.set("Todos")
        self.render_tasks()
        update_statistics(self)


    # ---------- IA (Ollama) ----------
    def open_ollama_dialog(self):
        OllamaCaptureDialog(self, self._on_add)


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

    def _render_modern_task_card(self, task: dict, now: date):
        tid = task.id
        border_color = self._get_task_border_color(task, now)

        # Container para centrar con padding
        outer = tk.Frame(self.task_frame, bg=border_color)
        outer.pack(fill="x", pady=4, padx=40)  # Padding horizontal para centrar

        card = tk.Frame(outer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        card.pack(fill="both", expand=True, padx=2, pady=2)

        stripe_color = VIBRANT_COLORS.get(task.color, "#CCCCCC")
        stripe = tk.Frame(card, bg=stripe_color, width=10)
        stripe.pack(side="left", fill="y")

        body = tk.Frame(card, bg=GRADIENTS["Card"][0])
        body.pack(side="left", fill="both", expand=True)

        header = tk.Frame(body, bg=GRADIENTS["Card"][0]); header.pack(fill="x", padx=12, pady=8)

        var_done = tk.BooleanVar(value=task.done)
        tk.Checkbutton(header, variable=var_done,
                       command=lambda t=tid: self._toggle_done_with_update_by_id(t),
                       bg=GRADIENTS["Card"][0], font=("Segoe UI", 12)).pack(side="left")

        title_font = ("Segoe UI", 12, "bold")
        title_color = MODERN_COLORS["Text"] if not task.done else MODERN_COLORS["TextLight"]

        title_frame = tk.Frame(header, bg=GRADIENTS["Card"][0]); title_frame.pack(side="left", fill="x", expand=True, padx=8)
        lbl_title = tk.Label(title_frame, text=task.title, bg=GRADIENTS["Card"][0], fg=title_color,
                             font=title_font, anchor="w", wraplength=460, justify="left", cursor="hand2")
        lbl_title.pack(fill="x")
        lbl_title.bind("<Double-Button-1>", lambda e, t=tid: self.open_quick_sticky_by_id(t))  # overlay
        # Botón extra para abrir editor completo sigue en acciones

        if task.priority:
            tk.Label(header, text="⭐ PRIORIDAD", bg=GRADIENTS["Warning"][0], fg="white",
                     font=("Segoe UI", 8, "bold"), padx=6, pady=2).pack(side="right")

        content = tk.Frame(body, bg=GRADIENTS["Card"][0]); content.pack(fill="x", padx=12, pady=(0,8))
        if task.desc:
            desc_font = ("Segoe UI", 10, "italic" if task.done else "normal")
            desc_fg = MODERN_COLORS["TextLight"] if task.done else MODERN_COLORS["Text"]
            tk.Label(content, text=task.desc, bg=GRADIENTS["Card"][0], fg=desc_fg,
                     font=desc_font, anchor="w", wraplength=460, justify="left").pack(fill="x", pady=(0,8))

        footer = tk.Frame(body, bg=GRADIENTS["Card"][0]); footer.pack(fill="x", padx=12, pady=(0,8))
        try:
            start_d = task.start if isinstance(task.start, date) else parse_date(task.start)
            due_d = task.due if isinstance(task.due, date) else parse_date(task.due)
        except Exception:
            start_d = due_d = now

        due_txt = f"📅 {fmt_date(due_d)}"
        overdue = (due_d <= now)
        due_fg = MODERN_COLORS["Danger"] if overdue and not task.done else MODERN_COLORS["TextLight"]
        tk.Label(footer, text=due_txt, bg=GRADIENTS["Card"][0], fg=due_fg, font=("Segoe UI", 9)).pack(side="left")

        actions = tk.Frame(footer, bg=GRADIENTS["Card"][0]); actions.pack(side="right")
        priority_icon = "⭐" if task.priority else "☆"
        self._create_small_button(actions, priority_icon, lambda t=tid: self._toggle_priority_with_update_by_id(t), "Warning")
        self._create_small_button(actions, "📝", lambda t=tid: self.open_note_by_id(t), "Primary")
        self._create_small_button(actions, "🗑️", lambda t=tid: self._delete_task_with_update_by_id(t), "Danger")

    def _create_small_button(self, parent, icon, command, color):
        btn = tk.Label(parent, text=icon, bg=GRADIENTS[color][0], fg="white",
                       font=("Segoe UI", 10), padx=6, pady=2, relief="flat", bd=0, takefocus=1, cursor="hand2")
        btn.pack(side="left", padx=2)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Return>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(bg=GRADIENTS[color][1]))
        btn.bind("<Leave>", lambda e: btn.configure(bg=GRADIENTS[color][0]))

    def _toggle_done_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.store.toggle_done(idx)
        self.render_tasks(); update_statistics(self)

    def _toggle_priority_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.store.toggle_priority(idx)
        self.render_tasks(); update_statistics(self)

    def _delete_task_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.delete_task_by_index(idx, tid)
        update_statistics(self)

    def _get_task_border_color(self, task, now: date) -> str:
        if task.done: return MODERN_COLORS["Success"]
        if task.priority: return MODERN_COLORS["Warning"]
        try:
            start_d = task.start if isinstance(task.start, date) else parse_date(task.start)
            due_d = task.due if isinstance(task.due, date) else parse_date(task.due)
            return urgency_color(start_d, due_d, now)
        except Exception:
            return MODERN_COLORS["TextLight"]

    def delete_task_by_index(self, idx: int, tid: str):
        win = self.note_windows.pop(tid, None)
        if win is not None and win.winfo_exists():
            try: win.destroy()
            except Exception: pass
        qwin = self.quick_windows.pop(tid, None)
        if qwin is not None and qwin.winfo_exists():
            try: qwin.destroy()
            except Exception: pass
        self.store.delete(idx)
        self.render_tasks()

    # ---------- Notas flotantes (editor) ----------
    def open_note_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        task = self.store.tasks[idx]
        win = self.note_windows.get(tid)
        if win and win.winfo_exists():
            win.deiconify(); win.lift(); return
        task.open = True
        self.store.save_throttled()
        win = ModernNoteWindow(self, tid, task)
        self.note_windows[tid] = win

    # ---------- Posit rápido (overlay) ----------
    def open_quick_sticky_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        task = self.store.tasks[idx]
        qwin = self.quick_windows.get(tid)
        if qwin and qwin.winfo_exists():
            qwin.deiconify(); qwin.lift(); return
        qwin = QuickStickyWindow(self, task)
        self.quick_windows[tid] = qwin

    def reopen_notes(self):
        for t in self.store.tasks:
            if t.open or t.pinned:
                try: self.open_note_by_id(t.id)
                except Exception: pass

    def open_daily_mission_posits(self):
        """Abre automáticamente posits de las misiones diarias al iniciar"""
        print("[DEBUG] ===== Iniciando apertura de posits de misiones diarias =====")
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
                # Crear un objeto Task temporal para cada misión
                task_id = f"mission_{mission_key}"
                fake_task = Task(
                    id=task_id,
                    title=mission_data.get("title", "Misión Diaria"),
                    desc=mission_data.get("desc", ""),
                    priority=mission_data.get("priority") == "Alta",
                    color=mission_data.get("color", "Ocean"),
                    start=fmt_date(today_date()),
                    due=fmt_date(today_date())
                )

                # Crear QuickStickyWindow para la misión (skip_centering=True para usar posiciones guardadas)
                qwin = QuickStickyWindow(self, fake_task, skip_centering=True)

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

        except Exception as e:
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

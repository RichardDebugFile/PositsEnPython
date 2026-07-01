#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel de gamificación: muestra nivel, XP, misiones diarias y rachas
"""

import tkinter as tk
from tkinter import ttk
from ..config import MODERN_COLORS, GRADIENTS


class GamificationPanel(tk.Frame):
    """
    Panel que muestra información de gamificación:
    - Nivel y barra de progreso
    - Misiones diarias
    - Racha actual
    """

    def __init__(self, parent, gamification_manager, app=None, music_player=None):
        super().__init__(parent, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        self.gm = gamification_manager
        self.app = app  # Referencia a la app principal
        self.music_player = music_player  # Reproductor de música

        # Header (fijo)
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0])
        header.pack(fill="x")

        tk.Label(
            header,
            text="🎮 Productividad",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).pack(padx=12, pady=8)

        # Canvas con scrollbar para contenido
        canvas_container = tk.Frame(self, bg=GRADIENTS["Card"][0])
        canvas_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_container, bg=GRADIENTS["Card"][0], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg=GRADIENTS["Card"][0])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Contenido dentro del frame scrollable
        content = tk.Frame(self.scrollable_frame, bg=GRADIENTS["Card"][0])
        content.pack(fill="both", expand=True, padx=12, pady=12)

        # Nivel y XP
        self._create_level_section(content)

        # Misiones diarias
        self._create_missions_section(content)

        # Estadísticas
        self._create_stats_section(content)

        # Panel de música (si está disponible)
        if self.music_player:
            self._create_music_section(content)

        # Bind mousewheel para scroll
        self._bind_mousewheel()

        # Actualizar datos
        self.refresh()

    def _bind_mousewheel(self):
        """Bind mousewheel para scroll en el panel solo cuando el mouse está encima"""
        def _on_mousewheel(e):
            self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")

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

    def _create_level_section(self, parent):
        """Crea la sección de nivel y XP"""
        level_frame = tk.Frame(parent, bg=GRADIENTS["Success"][0], relief="flat", bd=0)
        level_frame.pack(fill="x", pady=(0, 10))

        # Nivel actual
        self.level_label = tk.Label(
            level_frame,
            text="Nivel 1",
            bg=GRADIENTS["Success"][0],
            fg="white",
            font=("Segoe UI", 20, "bold")
        )
        self.level_label.pack(padx=12, pady=(10, 5))

        # XP y progreso
        self.xp_label = tk.Label(
            level_frame,
            text="0 / 100 XP",
            bg=GRADIENTS["Success"][0],
            fg="white",
            font=("Segoe UI", 10)
        )
        self.xp_label.pack(padx=12, pady=(0, 5))

        # Barra de progreso
        progress_container = tk.Frame(level_frame, bg=GRADIENTS["Success"][0])
        progress_container.pack(fill="x", padx=12, pady=(0, 10))

        style = ttk.Style()
        style.configure(
            "Level.Horizontal.TProgressbar",
            troughcolor=MODERN_COLORS["Light"],
            background=MODERN_COLORS["Warning"],
            borderwidth=0,
            thickness=12
        )

        self.progress_bar = ttk.Progressbar(
            progress_container,
            style="Level.Horizontal.TProgressbar",
            mode="determinate",
            maximum=100
        )
        self.progress_bar.pack(fill="x")

    def _create_missions_section(self, parent):
        """Crea la sección de misiones diarias recurrentes"""
        missions_frame = tk.Frame(parent, bg=GRADIENTS["Card"][0])
        missions_frame.pack(fill="x", pady=(10, 10))

        # Header con título
        header_frame = tk.Frame(missions_frame, bg=GRADIENTS["Card"][0])
        header_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
            header_frame,
            text="📋 Misiones Recurrentes",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(side="left")

        # Info tooltip
        info_label = tk.Label(
            header_frame,
            text="ℹ️",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9),
            cursor="hand2"
        )
        info_label.pack(side="right")
        info_label.bind("<Button-1>", lambda e: self._show_missions_info())

        # Descripción
        desc_label = tk.Label(
            missions_frame,
            text="Marca las misiones conforme las completes",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 8),
            wraplength=320,
            justify="left"
        )
        desc_label.pack(anchor="w", pady=(0, 8))

        # Container para misiones diarias con checkboxes
        self.daily_missions_container = tk.Frame(missions_frame, bg=GRADIENTS["Card"][0])
        self.daily_missions_container.pack(fill="x", pady=(0, 8))

        # Cargar misiones diarias interactivas
        self._load_daily_missions_checkboxes()

        # Botón para reset de posiciones de posits
        from .components import PillButton
        reset_btn_frame = tk.Frame(missions_frame, bg=GRADIENTS["Card"][0])
        reset_btn_frame.pack(fill="x", pady=(8, 0))
        PillButton(
            reset_btn_frame,
            "🔄 Reiniciar Posiciones",
            self._reset_posit_positions,
            "Warning",
            "small"
        ).pack(anchor="center")

    def _create_stats_section(self, parent):
        """Crea la sección de estadísticas"""
        stats_frame = tk.Frame(parent, bg=GRADIENTS["Card"][0])
        stats_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            stats_frame,
            text="📊 Estadísticas",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Grid de estadísticas
        stats_grid = tk.Frame(stats_frame, bg=GRADIENTS["Card"][0])
        stats_grid.pack(fill="x")

        # Racha actual
        self.streak_label = self._create_stat_item(
            stats_grid,
            "🔥 Racha Actual",
            "0 días",
            row=0,
            col=0
        )

        # Racha más larga
        self.longest_streak_label = self._create_stat_item(
            stats_grid,
            "⭐ Mejor Racha",
            "0 días",
            row=0,
            col=1
        )

        # Tareas totales
        self.total_tasks_label = self._create_stat_item(
            stats_grid,
            "✅ Total Completadas",
            "0",
            row=1,
            col=0
        )

    def _create_stat_item(self, parent, title, value, row, col):
        """Crea un item de estadística"""
        item = tk.Frame(parent, bg=MODERN_COLORS["Light"], relief="flat", bd=0)
        item.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        parent.grid_columnconfigure(col, weight=1)

        tk.Label(
            item,
            text=title,
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9)
        ).pack(padx=10, pady=(8, 2))

        value_label = tk.Label(
            item,
            text=value,
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 14, "bold")
        )
        value_label.pack(padx=10, pady=(0, 8))

        return value_label

    def _create_music_section(self, parent):
        """Crea la sección de música"""
        from .music_panel import MusicPanel

        music_frame = tk.Frame(parent, bg=GRADIENTS["Card"][0])
        music_frame.pack(fill="x", pady=(10, 0))

        self.music_panel = MusicPanel(music_frame, self.music_player, app=self.app, bg=GRADIENTS["Card"][0])
        self.music_panel.pack(fill="x")

    def refresh(self):
        """Actualiza todos los datos del panel"""
        stats = self.gm.get_stats()
        missions = self.gm.get_daily_missions()

        # Actualizar nivel y XP con formato "generados/necesarios"
        self.level_label.configure(text=f"Nivel {stats['level']}")
        self.xp_label.configure(text=f"{stats['xp_in_level']} / {stats['xp_needed_for_next']} XP")

        # Actualizar barra de progreso
        progress_pct = self.gm.get_progress_percentage()
        self.progress_bar['value'] = progress_pct

        # Actualizar stats
        self.streak_label.configure(text=f"{stats['current_streak']} días")
        self.longest_streak_label.configure(text=f"{stats['longest_streak']} días")
        self.total_tasks_label.configure(text=str(stats['total_tasks_completed']))

    def _load_daily_missions_checkboxes(self):
        """Carga y muestra las misiones diarias con checkboxes interactivos"""
        import json
        import os
        from ..utils.dates import today_date, fmt_date

        # Las 3 misiones diarias fijas (se repiten todos los días)
        daily_missions = [
            {
                "icon": "✅",
                "title": "Completar 3 tareas",
                "desc": "Completa 3 tareas hoy (+50 XP)",
                "priority": "Normal",
                "color": "Ocean",
                "key": "complete_3_tasks"
            },
            {
                "icon": "⭐",
                "title": "Completar 1 tarea prioritaria",
                "desc": "Completa 1 tarea con prioridad alta (+50 XP)",
                "priority": "Alta",
                "color": "Sunset",
                "key": "complete_priority"
            },
            {
                "icon": "🎯",
                "title": "Crear una nueva tarea",
                "desc": "Crea al menos 1 tarea nueva hoy (+50 XP)",
                "priority": "Normal",
                "color": "Nature",
                "key": "create_task"
            }
        ]

        # Cargar misiones personalizadas si existen
        missions_file = os.path.join("data", "daily_missions.json")
        if os.path.exists(missions_file):
            try:
                with open(missions_file, "r", encoding="utf-8") as f:
                    saved_missions = json.load(f)
                    # Sobrescribir con datos guardados
                    for mission in daily_missions:
                        if mission["key"] in saved_missions:
                            saved = saved_missions[mission["key"]]
                            mission["title"] = saved.get("title", mission["title"])
                            mission["desc"] = saved.get("desc", mission["desc"])
                            mission["priority"] = saved.get("priority", mission["priority"])
                            mission["color"] = saved.get("color", mission["color"])
            except Exception as e:
                print(f"[WARNING] No se pudieron cargar misiones guardadas: {e}")

        from ..config import VIBRANT_COLORS

        # Cargar estado de misiones completadas hoy
        missions_state_file = os.path.join("data", "daily_missions_state.json")
        missions_completed_state = {}
        today_str = fmt_date(today_date())

        if os.path.exists(missions_state_file):
            try:
                with open(missions_state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    # Solo usar el estado si es del día de hoy
                    if state_data.get("date") == today_str:
                        missions_completed_state = state_data.get("completed", {})
                    else:
                        # Es un nuevo día, resetear estado
                        print(f"[INFO] Nuevo día detectado, reseteando misiones")
            except Exception as e:
                print(f"[WARNING] No se pudo cargar estado de misiones: {e}")

        # Almacenar variables de checkbox y Entry widgets
        self.mission_vars = {}
        self.mission_entries = {}  # {key: {"title": Entry, "desc": Entry}}
        self.missions_state_file = missions_state_file
        self.today_str = today_str

        for mission in daily_missions:
            mission_card = tk.Frame(self.daily_missions_container, bg=MODERN_COLORS["Light"], relief="flat", bd=0)
            mission_card.pack(fill="x", pady=3)

            # Color stripe
            color_hex = VIBRANT_COLORS.get(mission["color"], "#CCCCCC")
            stripe = tk.Frame(mission_card, bg=color_hex, width=4)
            stripe.pack(side="left", fill="y")

            # Checkbox - inicializar con estado guardado
            is_completed = missions_completed_state.get(mission["key"], False)
            var = tk.BooleanVar(value=is_completed)
            self.mission_vars[mission["key"]] = var

            checkbox = tk.Checkbutton(
                mission_card,
                variable=var,
                command=lambda k=mission["key"]: self._on_mission_checked(k),
                bg=MODERN_COLORS["Light"],
                fg=MODERN_COLORS["Text"],
                selectcolor=MODERN_COLORS["White"],
                activebackground=MODERN_COLORS["Light"],
                font=("Segoe UI", 11)
            )
            checkbox.pack(side="left", padx=(8, 4))

            # Content editable
            content = tk.Frame(mission_card, bg=MODERN_COLORS["Light"])
            content.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=6)

            # Title editable (Entry con icono)
            title_frame = tk.Frame(content, bg=MODERN_COLORS["Light"])
            title_frame.pack(fill="x", pady=(0, 2))

            tk.Label(
                title_frame,
                text=mission["icon"],
                bg=MODERN_COLORS["Light"],
                fg=MODERN_COLORS["Text"],
                font=("Segoe UI", 9)
            ).pack(side="left", padx=(0, 4))

            title_entry = tk.Entry(
                title_frame,
                bg=MODERN_COLORS["White"],
                fg=MODERN_COLORS["Text"],
                insertbackground=MODERN_COLORS["Text"],
                font=("Segoe UI", 9, "bold"),
                relief="flat",
                bd=1
            )
            title_entry.insert(0, mission["title"])
            title_entry.pack(side="left", fill="x", expand=True)

            # Description editable (Entry)
            desc_entry = tk.Entry(
                content,
                bg=MODERN_COLORS["White"],
                fg=MODERN_COLORS["TextLight"],
                insertbackground=MODERN_COLORS["Text"],
                font=("Segoe UI", 8),
                relief="flat",
                bd=1
            )
            desc_entry.insert(0, mission["desc"])
            desc_entry.pack(fill="x", pady=(2, 0))

            # Priority selector (Combobox)
            from tkinter import ttk
            priority_frame = tk.Frame(content, bg=MODERN_COLORS["Light"])
            priority_frame.pack(fill="x", pady=(4, 0))

            tk.Label(
                priority_frame,
                text="Prioridad:",
                bg=MODERN_COLORS["Light"],
                fg=MODERN_COLORS["TextLight"],
                font=("Segoe UI", 7)
            ).pack(side="left", padx=(0, 4))

            priority_var = tk.StringVar(value=mission["priority"])
            priority_combo = ttk.Combobox(
                priority_frame,
                textvariable=priority_var,
                values=["Normal", "Alta"],
                state="readonly",
                font=("Segoe UI", 7),
                width=10
            )
            priority_combo.pack(side="left")
            priority_combo.bind("<<ComboboxSelected>>", lambda e, k=mission["key"]: self._save_mission_edits(k))

            # Guardar referencias a los Entry widgets y variables
            self.mission_entries[mission["key"]] = {
                "title": title_entry,
                "desc": desc_entry,
                "priority_var": priority_var,
                "icon": mission["icon"],
                "color": mission["color"]
            }

            # Bind para guardar cambios cuando se edita
            title_entry.bind("<FocusOut>", lambda e, k=mission["key"]: self._save_mission_edits(k))
            desc_entry.bind("<FocusOut>", lambda e, k=mission["key"]: self._save_mission_edits(k))

    def _on_mission_checked(self, mission_key):
        """Callback cuando se marca una misión"""
        import json
        import os

        if not self.app:
            return

        # Verificar si se marcó
        is_checked = self.mission_vars[mission_key].get()

        # Guardar estado de misiones
        completed_state = {key: var.get() for key, var in self.mission_vars.items()}
        state_data = {
            "date": self.today_str,
            "completed": completed_state
        }

        os.makedirs("data", exist_ok=True)
        with open(self.missions_state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        if is_checked:
            # Otorgar XP al marcar
            self.gm.add_xp(50)  # +50 XP por misión diaria
            self.gm.save()

            # Actualizar UI
            self.refresh()
            if hasattr(self.app, 'render_tasks'):
                self.app.render_tasks()

            # Mensaje de felicitación
            from tkinter import messagebox
            messagebox.showinfo(
                "¡Misión Completada!",
                f"¡Excelente! Has ganado +50 XP\n\n"
                f"Sigue así para mantener tu racha."
            )

    def _save_mission_edits(self, mission_key):
        """Guarda las ediciones de una misión"""
        import json
        import os

        if mission_key not in self.mission_entries:
            return

        # Obtener valores actuales de los Entry widgets
        entry_data = self.mission_entries[mission_key]
        title = entry_data["title"].get()
        desc = entry_data["desc"].get()
        priority = entry_data["priority_var"].get()

        # Cargar archivo de misiones personalizadas
        missions_file = os.path.join("data", "daily_missions.json")

        # Crear estructura de datos
        missions_data = {}
        if os.path.exists(missions_file):
            try:
                with open(missions_file, "r", encoding="utf-8") as f:
                    missions_data = json.load(f)
            except:
                missions_data = {}

        # Guardar datos de la misión
        missions_data[mission_key] = {
            "title": title,
            "desc": desc,
            "priority": priority,
            "icon": entry_data["icon"],
            "color": entry_data["color"]
        }

        # Guardar archivo
        os.makedirs("data", exist_ok=True)
        with open(missions_file, "w", encoding="utf-8") as f:
            json.dump(missions_data, f, indent=2, ensure_ascii=False)

    def _reset_posit_positions(self):
        """Reinicia las posiciones de los posits a valores por defecto"""
        import json
        import os
        from tkinter import messagebox

        # Eliminar archivo de posiciones
        positions_file = os.path.join("data", "posit_positions.json")
        if os.path.exists(positions_file):
            try:
                os.remove(positions_file)
                messagebox.showinfo(
                    "Posiciones Reiniciadas",
                    "Las posiciones de las misiones se han reiniciado.\n\n"
                    "Cierra y vuelve a abrir la app para ver los cambios."
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"No se pudieron reiniciar las posiciones:\n{e}"
                )
        else:
            messagebox.showinfo(
                "Info",
                "No hay posiciones guardadas para reiniciar."
            )

    def _show_missions_info(self):
        """Muestra información sobre las misiones recurrentes"""
        from tkinter import messagebox
        messagebox.showinfo(
            "Misiones Recurrentes",
            "Las misiones recurrentes son tareas especiales que:\n\n"
            "✅ Se repiten automáticamente cada día\n"
            "⭐ Mantienen las mismas prioridades\n"
            "🎮 Otorgan +50 XP al completarse\n"
            "🔄 Se crean con un solo clic\n\n"
            "Haz clic en 'Crear Misiones de Hoy' para añadirlas "
            "a tu lista de tareas del día."
        )

    def _create_daily_missions_as_tasks(self):
        """Crea las 3 misiones diarias como tareas reales en la lista"""
        if not self.app:
            return

        from datetime import date, timedelta
        from tkinter import messagebox

        # Las 3 misiones diarias fijas (se repiten todos los días)
        daily_missions = [
            {
                "title": "✅ Completar 3 tareas",
                "desc": "Completa 3 tareas hoy para cumplir esta misión diaria y ganar +50 XP",
                "priority": False,
                "color": "Ocean"
            },
            {
                "title": "⭐ Completar 1 tarea prioritaria",
                "desc": "Completa al menos 1 tarea marcada como prioritaria para ganar +50 XP",
                "priority": True,
                "color": "Sunset"
            },
            {
                "title": "🎯 Crear una nueva tarea",
                "desc": "Crea al menos una nueva tarea hoy para mantener tu productividad y ganar +50 XP",
                "priority": False,
                "color": "Nature"
            }
        ]

        # Fecha: hoy
        today = date.today()

        created_count = 0
        for mission in daily_missions:
            # Verificar si ya existe una tarea con el mismo título hoy
            exists = False
            for task in self.app.store.tasks:
                if task.title == mission["title"] and task.due == today:
                    exists = True
                    break

            if not exists:
                self.app.store.add(
                    title=mission["title"],
                    desc=mission["desc"],
                    due=today,
                    priority=mission["priority"],
                    color=mission["color"]
                )
                created_count += 1

        if created_count > 0:
            self.app.render_tasks()
            messagebox.showinfo(
                "Misiones Creadas",
                f"Se han creado {created_count} misiones diarias como tareas.\n\n"
                "Estas misiones se repiten todos los días con las mismas prioridades."
            )
        else:
            messagebox.showinfo(
                "Misiones Ya Existen",
                "Las misiones diarias ya están creadas para hoy."
            )

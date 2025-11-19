#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ventana flotante de Pomodoro con drag & drop de tareas
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable
from datetime import timedelta

# Importar TkinterDnD si está disponible
try:
    from tkinterdnd2 import DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

from ..config import MODERN_COLORS, GRADIENTS, PRIORITY_LEVELS
from .components import PillButton


class PomodoroWindow(tk.Toplevel):
    """Ventana flotante de gestión de sesiones Pomodoro"""

    def __init__(self, parent, pomodoro_manager, task_store, music_player):
        super().__init__(parent)
        self.pomodoro_manager = pomodoro_manager
        self.task_store = task_store
        self.music_player = music_player

        # Estado del timer
        self.time_remaining = pomodoro_manager.work_duration * 60  # segundos
        self.timer_running = False
        self.timer_id = None

        # Configurar ventana
        self.title("🍅 Modo Pomodoro")
        self.geometry("450x600")
        self.configure(bg=GRADIENTS["Card"][0])
        self.resizable(False, False)

        # Mantener siempre visible
        self.attributes("-topmost", True)

        # Crear UI
        self._create_ui()

        # Actualizar display inicial
        self._update_display()

        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_ui(self):
        """Crea la interfaz de la ventana Pomodoro"""

        # === HEADER con timer ===
        header = tk.Frame(self, bg=GRADIENTS["Warning"][0], height=120)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🍅 MODO POMODORO",
            bg=GRADIENTS["Warning"][0],
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(15, 5))

        # Timer
        self.timer_label = tk.Label(
            header,
            text="25:00",
            bg=GRADIENTS["Warning"][0],
            fg="white",
            font=("Segoe UI", 32, "bold")
        )
        self.timer_label.pack()

        # Barra de progreso
        progress_frame = tk.Frame(header, bg=GRADIENTS["Warning"][0])
        progress_frame.pack(fill="x", padx=20, pady=(5, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            length=380,
            maximum=100
        )
        self.progress_bar.pack(fill="x")

        # Sesión actual
        self.session_label = tk.Label(
            header,
            text="Sesión 0/4",
            bg=GRADIENTS["Warning"][0],
            fg="white",
            font=("Segoe UI", 10)
        )
        self.session_label.pack()

        # === ZONA DE TAREAS ===
        tasks_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        tasks_frame.pack(fill="both", expand=True, padx=15, pady=15)

        tk.Label(
            tasks_frame,
            text="📋 Tareas en Cola:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 10))

        # Frame con scrollbar para tareas
        tasks_scroll_frame = tk.Frame(tasks_frame, bg=GRADIENTS["Card"][0])
        tasks_scroll_frame.pack(fill="both", expand=True)

        # Canvas para scroll
        self.tasks_canvas = tk.Canvas(
            tasks_scroll_frame,
            bg=MODERN_COLORS["Light"],
            highlightthickness=0,
            height=250
        )
        scrollbar = ttk.Scrollbar(
            tasks_scroll_frame,
            orient="vertical",
            command=self.tasks_canvas.yview
        )

        self.tasks_inner_frame = tk.Frame(self.tasks_canvas, bg=MODERN_COLORS["Light"])
        self.tasks_canvas_window = self.tasks_canvas.create_window(
            (0, 0),
            window=self.tasks_inner_frame,
            anchor="nw",
            width=400
        )

        self.tasks_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tasks_canvas.configure(yscrollcommand=scrollbar.set)

        # Bind para actualizar scroll region
        self.tasks_inner_frame.bind("<Configure>", lambda e: self.tasks_canvas.configure(
            scrollregion=self.tasks_canvas.bbox("all")
        ))

        # Zona de drop
        self.drop_zone = tk.Frame(
            self.tasks_inner_frame,
            bg=MODERN_COLORS["Light"],
            relief="groove",
            bd=2,
            height=80
        )
        self.drop_zone.pack(fill="x", padx=10, pady=10)

        tk.Label(
            self.drop_zone,
            text="⬇️ Arrastra tareas aquí desde el panel o posits",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9)
        ).pack(expand=True)

        # Configurar drop zone para recibir drops (si está disponible)
        if DND_AVAILABLE:
            try:
                self.drop_zone.drop_target_register("DND_ALL")
                self.drop_zone.dnd_bind("<<Drop>>", self._on_task_drop)
            except:
                pass

        # === CONTROLES ===
        controls_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        controls_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Botón Play/Pause
        self.play_pause_btn = PillButton(
            controls_frame,
            "▶ Iniciar",
            self._toggle_timer,
            color="Success",
            size="normal"
        )
        self.play_pause_btn.pack(side="left", padx=(0, 5))

        # Botón Stop
        PillButton(
            controls_frame,
            "⏹ Detener",
            self._stop_timer,
            color="Danger",
            size="small"
        ).pack(side="left", padx=(0, 5))

        # Botón +5 min
        PillButton(
            controls_frame,
            "⏩ +5min",
            self._add_time,
            color="Secondary",
            size="small"
        ).pack(side="left", padx=(0, 5))

        # Botón Configuración
        PillButton(
            controls_frame,
            "⚙️",
            self._open_config,
            color="Secondary",
            size="small"
        ).pack(side="right")

        # === CONFIGURACIÓN DE MÚSICA ===
        music_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        music_frame.pack(fill="x", padx=15, pady=(0, 10))

        tk.Label(
            music_frame,
            text="🎵 Configuración de Música:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Botones de música
        music_btns_frame = tk.Frame(music_frame, bg=GRADIENTS["Card"][0])
        music_btns_frame.pack(fill="x")

        # Botón música de trabajo
        PillButton(
            music_btns_frame,
            "🎧 Música de Trabajo",
            self._config_work_music,
            color="Primary",
            size="small"
        ).pack(side="left", padx=(0, 5))

        # Botón música de descanso
        PillButton(
            music_btns_frame,
            "☕ Música de Descanso",
            self._config_break_music,
            color="Info",
            size="small"
        ).pack(side="left")

    def _on_task_drop(self, event):
        """Maneja el evento de soltar una tarea en la zona de drop"""
        try:
            # Obtener el ID de la tarea desde el evento
            task_id = event.data

            # Agregar tarea a la cola de Pomodoro
            self.pomodoro_manager.add_task_to_queue(task_id)

            # Actualizar visualización
            self._refresh_task_list()

            print(f"[INFO] Tarea agregada a Pomodoro: {task_id}")

        except Exception as e:
            print(f"[ERROR] Error al agregar tarea a Pomodoro: {e}")

    def _refresh_task_list(self):
        """Actualiza la lista de tareas en la UI"""
        # Limpiar tareas existentes (excepto drop zone)
        for widget in self.tasks_inner_frame.winfo_children():
            if widget != self.drop_zone:
                widget.destroy()

        # Recrear drop zone al principio
        self.drop_zone.pack_forget()
        self.drop_zone.pack(fill="x", padx=10, pady=10)

        # Agregar tareas de la cola
        for task_id in self.pomodoro_manager.tasks_in_queue:
            task = self.task_store.get_by_id(task_id)
            if task:
                self._create_task_widget(task)

    def _create_task_widget(self, task):
        """Crea un widget para una tarea en la cola"""
        task_frame = tk.Frame(
            self.tasks_inner_frame,
            bg="white",
            relief="solid",
            bd=1
        )
        task_frame.pack(fill="x", padx=10, pady=5)

        # Contenido de la tarea
        content_frame = tk.Frame(task_frame, bg="white")
        content_frame.pack(fill="x", padx=10, pady=8)

        # Prioridad y título
        priority_key = task.priority if isinstance(task.priority, str) else ("high" if task.priority else "medium")
        priority_info = PRIORITY_LEVELS.get(priority_key, PRIORITY_LEVELS["medium"])

        title_label = tk.Label(
            content_frame,
            text=f"{priority_info['emoji']} {task.title}",
            bg="white",
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        )
        title_label.pack(side="left", fill="x", expand=True)

        # XP
        xp_label = tk.Label(
            content_frame,
            text=f"+{priority_info['xp']} XP",
            bg="white",
            fg=priority_info['color'],
            font=("Segoe UI", 9, "bold")
        )
        xp_label.pack(side="right")

        # Botones
        btn_frame = tk.Frame(task_frame, bg="white")
        btn_frame.pack(fill="x", padx=10, pady=(0, 8))

        # Botón completar
        complete_btn = tk.Button(
            btn_frame,
            text="✓ Completar",
            command=lambda: self._complete_task(task.id),
            bg=GRADIENTS["Success"][0],
            fg="white",
            font=("Segoe UI", 8),
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2"
        )
        complete_btn.pack(side="left", padx=(0, 5))

        # Botón quitar
        remove_btn = tk.Button(
            btn_frame,
            text="✗ Quitar",
            command=lambda: self._remove_task(task.id),
            bg=GRADIENTS["Danger"][0],
            fg="white",
            font=("Segoe UI", 8),
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2"
        )
        remove_btn.pack(side="left")

    def _complete_task(self, task_id: str):
        """Marca una tarea como completada"""
        try:
            # Obtener tarea
            task = self.task_store.get_by_id(task_id)
            if not task:
                return

            # Obtener índice y marcar como completada
            idx = self.task_store.index_by_id(task_id)
            self.task_store.toggle_done(idx)

            # Obtener XP basado en el nivel de prioridad
            priority_key = task.priority if isinstance(task.priority, str) else ("high" if task.priority else "medium")
            xp = PRIORITY_LEVELS.get(priority_key, PRIORITY_LEVELS["medium"])["xp"]

            # Remover de cola
            self.pomodoro_manager.remove_task_from_queue(task_id)

            # Actualizar lista
            self._refresh_task_list()

            print(f"[INFO] Tarea completada: {task.title} (+{xp} XP)")

        except Exception as e:
            print(f"[ERROR] Error al completar tarea: {e}")

    def _remove_task(self, task_id: str):
        """Remueve una tarea de la cola sin completarla"""
        self.pomodoro_manager.remove_task_from_queue(task_id)
        self._refresh_task_list()

    def _toggle_timer(self):
        """Inicia o pausa el timer"""
        if self.timer_running:
            self._pause_timer()
        else:
            self._start_timer()

    def _start_timer(self):
        """Inicia el timer"""
        self.timer_running = True
        self.play_pause_btn.set_text("⏸ Pausar")

        # Iniciar sesión en el manager
        self.pomodoro_manager.start_session(is_break=False)

        # Iniciar música de trabajo
        self._start_work_music()

        # Comenzar countdown
        self._countdown()

    def _pause_timer(self):
        """Pausa el timer"""
        self.timer_running = False
        self.play_pause_btn.set_text("▶ Reanudar")

        # Cancelar timer si existe
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None

        # Pausar música
        self.music_player.pause()

        # Pausar sesión en el manager
        self.pomodoro_manager.pause_session()

    def _stop_timer(self):
        """Detiene el timer completamente"""
        if messagebox.askyesno("Detener Pomodoro", "¿Deseas detener la sesión actual?"):
            self.timer_running = False
            self.play_pause_btn.set_text("▶ Iniciar")

            # Cancelar timer
            if self.timer_id:
                self.after_cancel(self.timer_id)
                self.timer_id = None

            # Detener música
            self.music_player.stop()

            # Finalizar sesión
            self.pomodoro_manager.end_session(interrupted=True)

            # Reiniciar timer
            self.time_remaining = self.pomodoro_manager.work_duration * 60
            self._update_display()

    def _add_time(self):
        """Agrega 5 minutos al timer actual"""
        self.time_remaining += 300  # 5 minutos = 300 segundos
        self._update_display()

    def _countdown(self):
        """Función de countdown del timer"""
        if self.timer_running and self.time_remaining > 0:
            self.time_remaining -= 1
            self._update_display()
            self.timer_id = self.after(1000, self._countdown)
        elif self.timer_running and self.time_remaining <= 0:
            self._timer_finished()

    def _timer_finished(self):
        """Se llama cuando el timer llega a 0"""
        self.timer_running = False

        # Detener música
        self.music_player.stop()

        if not self.pomodoro_manager.is_break:
            # Sesión de trabajo completada
            # Determinar si es descanso largo o corto
            is_long_break = (self.pomodoro_manager.current_session % self.pomodoro_manager.sessions_per_cycle == 0)

            if is_long_break:
                # Descanso largo
                break_duration = self.pomodoro_manager.long_break_duration
                messagebox.showinfo(
                    "🍅 Ciclo Completado",
                    f"¡Has completado {self.pomodoro_manager.sessions_per_cycle} sesiones!\n\n"
                    f"Toma un descanso largo de {break_duration} minutos."
                )
            else:
                # Descanso corto
                break_duration = self.pomodoro_manager.break_duration
                messagebox.showinfo(
                    "🍅 Pomodoro Completado",
                    f"¡Sesión de trabajo completada!\n\nToma un descanso de {break_duration} minutos."
                )

            # Configurar para descanso
            self.time_remaining = break_duration * 60
            self.play_pause_btn.set_text("☕ Iniciar Descanso")
            self._update_display()

            # Auto-iniciar descanso si está habilitado
            if self.pomodoro_manager.auto_start_break:
                self.after(1000, lambda: self._start_break_session())

        else:
            # Descanso completado
            messagebox.showinfo(
                "☕ Descanso Completado",
                "¡Descanso terminado!\n\n¿Continuar con la siguiente sesión?"
            )

            # Reiniciar para trabajo
            self.time_remaining = self.pomodoro_manager.work_duration * 60
            self.play_pause_btn.set_text("▶ Iniciar")
            self._update_display()

    def _start_break_session(self):
        """Inicia una sesión de descanso"""
        self.timer_running = True
        self.play_pause_btn.set_text("⏸ Pausar")

        # Iniciar sesión de descanso en el manager
        self.pomodoro_manager.start_session(is_break=True)

        # Reproducir música de descanso
        break_music = self.pomodoro_manager.get_break_music()
        if break_music and hasattr(self.music_player, 'play_specific_track'):
            self.music_player.play_specific_track(break_music)
        else:
            self.music_player.play_main_track()

        # Comenzar countdown
        self._countdown()

    def _start_work_music(self):
        """Inicia la música configurada para modo trabajo"""
        work_tracks = self.pomodoro_manager.get_work_music()

        if work_tracks:
            # Configurar las canciones de trabajo en el reproductor
            # (esto requiere modificar el music_player para soportar playlists temporales)
            print(f"[INFO] Reproduciendo música de trabajo: {len(work_tracks)} canciones")
            # TODO: Implementar lógica de reproducción de playlist temporal
        else:
            # Usar canción principal por defecto
            self.music_player.play_main_track()

    def _config_work_music(self):
        """Configura la música para modo trabajo (múltiples canciones)"""
        from tkinter import filedialog

        # Obtener playlist actual del music player
        if hasattr(self.music_player, 'playlist') and self.music_player.playlist:
            # Mostrar ventana de selección con la playlist actual
            self._show_work_music_selector()
        else:
            messagebox.showinfo(
                "Sin Playlist",
                "No hay canciones en la playlist.\n\nAgrega canciones desde el panel de música primero."
            )

    def _show_work_music_selector(self):
        """Muestra ventana para seleccionar múltiples canciones de la playlist"""
        selector_window = tk.Toplevel(self)
        selector_window.title("Seleccionar Música de Trabajo")
        selector_window.geometry("500x400")
        selector_window.configure(bg=GRADIENTS["Card"][0])
        selector_window.transient(self)
        selector_window.grab_set()

        tk.Label(
            selector_window,
            text="Selecciona las canciones para modo trabajo:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(pady=10)

        # Frame con scrollbar para la lista
        list_frame = tk.Frame(selector_window, bg=GRADIENTS["Card"][0])
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox con selección múltiple
        listbox = tk.Listbox(
            list_frame,
            selectmode="multiple",
            yscrollcommand=scrollbar.set,
            font=("Segoe UI", 9),
            height=15
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # Agregar canciones de la playlist
        current_work_music = self.pomodoro_manager.get_work_music()
        for track_path in self.music_player.playlist:
            from pathlib import Path
            track_name = Path(track_path).stem
            listbox.insert("end", track_name)
            # Preseleccionar si ya está en work music
            if track_path in current_work_music:
                listbox.selection_set(listbox.size() - 1)

        # Botones
        btn_frame = tk.Frame(selector_window, bg=GRADIENTS["Card"][0])
        btn_frame.pack(pady=10)

        def save_selection():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Sin Selección", "Debes seleccionar al menos una canción")
                return

            selected_tracks = [self.music_player.playlist[i] for i in selected_indices]
            self.pomodoro_manager.set_work_music(selected_tracks)
            messagebox.showinfo("Guardado", f"Se configuraron {len(selected_tracks)} canciones para modo trabajo")
            selector_window.destroy()

        PillButton(
            btn_frame,
            "Guardar",
            save_selection,
            color="Success",
            size="normal"
        ).pack(side="left", padx=5)

        PillButton(
            btn_frame,
            "Cancelar",
            selector_window.destroy,
            color="Secondary",
            size="normal"
        ).pack(side="left", padx=5)

    def _config_break_music(self):
        """Configura la música para modo descanso (una sola canción)"""
        from tkinter import filedialog

        # Obtener playlist actual del music player
        if hasattr(self.music_player, 'playlist') and self.music_player.playlist:
            # Mostrar ventana de selección con la playlist actual
            self._show_break_music_selector()
        else:
            messagebox.showinfo(
                "Sin Playlist",
                "No hay canciones en la playlist.\n\nAgrega canciones desde el panel de música primero."
            )

    def _show_break_music_selector(self):
        """Muestra ventana para seleccionar una canción para descanso"""
        selector_window = tk.Toplevel(self)
        selector_window.title("Seleccionar Música de Descanso")
        selector_window.geometry("500x400")
        selector_window.configure(bg=GRADIENTS["Card"][0])
        selector_window.transient(self)
        selector_window.grab_set()

        tk.Label(
            selector_window,
            text="Selecciona UNA canción para modo descanso:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(pady=10)

        # Frame con scrollbar para la lista
        list_frame = tk.Frame(selector_window, bg=GRADIENTS["Card"][0])
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox con selección simple
        listbox = tk.Listbox(
            list_frame,
            selectmode="single",
            yscrollcommand=scrollbar.set,
            font=("Segoe UI", 9),
            height=15
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # Agregar canciones de la playlist
        current_break_music = self.pomodoro_manager.get_break_music()
        for track_path in self.music_player.playlist:
            from pathlib import Path
            track_name = Path(track_path).stem
            listbox.insert("end", track_name)
            # Preseleccionar si es la canción de descanso
            if track_path == current_break_music:
                listbox.selection_set(listbox.size() - 1)

        # Botones
        btn_frame = tk.Frame(selector_window, bg=GRADIENTS["Card"][0])
        btn_frame.pack(pady=10)

        def save_selection():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Sin Selección", "Debes seleccionar una canción")
                return

            selected_track = self.music_player.playlist[selected_indices[0]]
            self.pomodoro_manager.set_break_music(selected_track)

            from pathlib import Path
            track_name = Path(selected_track).stem
            messagebox.showinfo("Guardado", f"Canción de descanso: {track_name}")
            selector_window.destroy()

        PillButton(
            btn_frame,
            "Guardar",
            save_selection,
            color="Success",
            size="normal"
        ).pack(side="left", padx=5)

        PillButton(
            btn_frame,
            "Cancelar",
            selector_window.destroy,
            color="Secondary",
            size="normal"
        ).pack(side="left", padx=5)

    def _open_config(self):
        """Abre el panel de configuración de Pomodoro"""
        config_window = tk.Toplevel(self)
        config_window.title("Configuración de Pomodoro")
        config_window.geometry("450x500")
        config_window.configure(bg=GRADIENTS["Card"][0])
        config_window.transient(self)
        config_window.grab_set()

        tk.Label(
            config_window,
            text="⚙️ Configuración de Pomodoro",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 14, "bold")
        ).pack(pady=15)

        # Frame principal con scroll
        main_frame = tk.Frame(config_window, bg=GRADIENTS["Card"][0])
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Duración de trabajo
        tk.Label(
            main_frame,
            text="Duración de Trabajo (minutos):",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        work_duration_var = tk.IntVar(value=self.pomodoro_manager.work_duration)
        work_duration_scale = tk.Scale(
            main_frame,
            from_=5,
            to=60,
            orient="horizontal",
            variable=work_duration_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            highlightthickness=0
        )
        work_duration_scale.pack(fill="x", pady=(0, 15))

        # Duración de descanso corto
        tk.Label(
            main_frame,
            text="Duración de Descanso Corto (minutos):",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        break_duration_var = tk.IntVar(value=self.pomodoro_manager.break_duration)
        break_duration_scale = tk.Scale(
            main_frame,
            from_=1,
            to=15,
            orient="horizontal",
            variable=break_duration_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            highlightthickness=0
        )
        break_duration_scale.pack(fill="x", pady=(0, 15))

        # Duración de descanso largo
        tk.Label(
            main_frame,
            text="Duración de Descanso Largo (minutos):",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        long_break_var = tk.IntVar(value=self.pomodoro_manager.long_break_duration)
        long_break_scale = tk.Scale(
            main_frame,
            from_=10,
            to=30,
            orient="horizontal",
            variable=long_break_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            highlightthickness=0
        )
        long_break_scale.pack(fill="x", pady=(0, 15))

        # Sesiones por ciclo
        tk.Label(
            main_frame,
            text="Sesiones antes de Descanso Largo:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        sessions_var = tk.IntVar(value=self.pomodoro_manager.sessions_per_cycle)
        sessions_scale = tk.Scale(
            main_frame,
            from_=2,
            to=8,
            orient="horizontal",
            variable=sessions_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            highlightthickness=0
        )
        sessions_scale.pack(fill="x", pady=(0, 15))

        # Opciones adicionales
        auto_start_break_var = tk.BooleanVar(value=self.pomodoro_manager.auto_start_break)
        tk.Checkbutton(
            main_frame,
            text="Iniciar descansos automáticamente",
            variable=auto_start_break_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9),
            selectcolor=MODERN_COLORS["Light"]
        ).pack(anchor="w", pady=5)

        notifications_var = tk.BooleanVar(value=self.pomodoro_manager.notifications_enabled)
        tk.Checkbutton(
            main_frame,
            text="Notificaciones habilitadas",
            variable=notifications_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9),
            selectcolor=MODERN_COLORS["Light"]
        ).pack(anchor="w", pady=5)

        # Botones
        btn_frame = tk.Frame(config_window, bg=GRADIENTS["Card"][0])
        btn_frame.pack(pady=15)

        def save_config():
            self.pomodoro_manager.work_duration = work_duration_var.get()
            self.pomodoro_manager.break_duration = break_duration_var.get()
            self.pomodoro_manager.long_break_duration = long_break_var.get()
            self.pomodoro_manager.sessions_per_cycle = sessions_var.get()
            self.pomodoro_manager.auto_start_break = auto_start_break_var.get()
            self.pomodoro_manager.notifications_enabled = notifications_var.get()
            self.pomodoro_manager.save_config()

            # Actualizar timer si no está corriendo
            if not self.timer_running:
                self.time_remaining = self.pomodoro_manager.work_duration * 60
                self._update_display()

            messagebox.showinfo("Guardado", "Configuración guardada correctamente")
            config_window.destroy()

        PillButton(
            btn_frame,
            "Guardar",
            save_config,
            color="Success",
            size="normal"
        ).pack(side="left", padx=5)

        PillButton(
            btn_frame,
            "Cancelar",
            config_window.destroy,
            color="Secondary",
            size="normal"
        ).pack(side="left", padx=5)

    def _update_display(self):
        """Actualiza la visualización del timer y progreso"""
        # Actualizar timer
        minutes = self.time_remaining // 60
        seconds = self.time_remaining % 60
        self.timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")

        # Actualizar progreso
        total_time = self.pomodoro_manager.work_duration * 60
        progress = ((total_time - self.time_remaining) / total_time) * 100
        self.progress_bar["value"] = progress

        # Actualizar sesión
        self.session_label.configure(
            text=f"Sesión {self.pomodoro_manager.current_session}/{self.pomodoro_manager.sessions_per_cycle}"
        )

    def _on_close(self):
        """Maneja el cierre de la ventana"""
        if self.timer_running:
            if messagebox.askyesno("Cerrar Pomodoro", "¿Deseas cerrar? El timer se detendrá."):
                self._stop_timer()
                self.destroy()
        else:
            self.destroy()

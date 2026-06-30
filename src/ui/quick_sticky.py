#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlay rápido tipo posit (sin barra de título)
Se muestra al hacer doble clic en el título de una tarea
"""

import tkinter as tk

from ..config import MODERN_COLORS, VIBRANT_COLORS
from .components import PillButton


class QuickStickyWindow(tk.Toplevel):
    """
    Ventana overlay rápida sin decoraciones del SO.
    Muestra un resumen de la tarea en un posit flotante.
    """

    def __init__(self, master, task: dict, skip_centering=False):
        super().__init__(master)
        self.task = task
        self.skip_centering = skip_centering  # Para misiones diarias que ya tienen posición
        self.overrideredirect(True)  # Sin barra de título
        self.attributes("-topmost", True)

        color = VIBRANT_COLORS.get(getattr(task, "color", "Sunshine"), "#FFD54F")
        fg = MODERN_COLORS["Dark"]
        self.configure(bg=color)

        pad = 12

        # Título
        tk.Label(
            self,
            text=getattr(task, "title", "") or "(Sin título)",
            bg=color,
            fg=fg,
            font=("Segoe UI", 13, "bold"),
            anchor="w"
        ).pack(fill="x", padx=pad, pady=(pad, 4))

        # Descripción
        tk.Label(
            self,
            text=getattr(task, "desc", ""),
            bg=color,
            fg=fg,
            font=("Segoe UI", 10),
            anchor="nw",
            justify="left",
            wraplength=380
        ).pack(fill="both", expand=True, padx=pad, pady=(0, 6))

        # Fecha
        tk.Label(
            self,
            text=f"📅 {getattr(task, "due", "")}",
            bg=color,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        ).pack(fill="x", padx=pad, pady=(0, 4))

        # Botones
        btn_frame = tk.Frame(self, bg=color)
        btn_frame.pack(fill="x", padx=pad, pady=(0, pad))
        PillButton(btn_frame, "Pomodoro", self._add_to_pomodoro, "Warning", "small", "🍅").pack(side="left")

        # Grip para redimensionar (esquina inferior derecha del posit)
        self._grip = tk.Label(
            btn_frame, text="◢", bg=color, fg=fg,
            font=("Segoe UI", 11, "bold"), cursor="sizing"
        )
        self._grip.pack(side="right", padx=(8, 0))
        self._grip.bind("<Button-1>", self._start_resize)
        self._grip.bind("<B1-Motion>", self._on_resize)
        self._grip.bind("<ButtonRelease-1>", self._on_resize_end)

        PillButton(btn_frame, "Cerrar", self.close, "Danger", "small", "✖").pack(side="right")

        # Bindings para mover la ventana
        self.bind("<Button-1>", self._start_move)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<ButtonRelease-1>", self._on_move_end)  # Guardar al soltar
        self.bind("<Escape>", lambda e: self.close())

        # Centrar sobre la ventana principal (solo si no es una misión con posición guardada)
        if not skip_centering:
            self.after(10, self._center_over_master)
        else:
            # Misiones: la app fija la posición; aquí solo restauramos el tamaño.
            self.after(10, self._apply_saved_size)

    def _center_over_master(self):
        """Centra la ventana sobre la ventana principal"""
        self.update_idletasks()

        mw = self.master.winfo_width()
        mh = self.master.winfo_height()
        mx = self.master.winfo_rootx()
        my = self.master.winfo_rooty()

        # Usar el tamaño guardado si existe; si no, calcular uno por defecto.
        saved = self._load_saved_geometry()
        if saved and saved.get("w"):
            w = saved["w"]
        else:
            w = min(420, max(320, int(mw * 0.5)))
        if saved and saved.get("h"):
            h = saved["h"]
        else:
            h = self.winfo_height()
            if h < 200:
                h = 220

        x = mx + (mw - w) // 2
        y = my + (mh - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_saved_size(self):
        """Restaura solo el tamaño guardado (la posición la fija la app)."""
        saved = self._load_saved_geometry()
        if saved and saved.get("w") and saved.get("h"):
            self.geometry(f"{saved['w']}x{saved['h']}")

    def _start_move(self, e):
        """Inicia el movimiento de la ventana"""
        self._ox, self._oy = e.x, e.y

    def _on_move(self, e):
        """Mueve la ventana al arrastrar"""
        x = self.winfo_x() + e.x - self._ox
        y = self.winfo_y() + e.y - self._oy
        self.geometry(f"+{x}+{y}")

    def _on_move_end(self, e):
        """Se llama al soltar el botón después de mover"""
        self._save_position()

    def _start_resize(self, e):
        """Inicia el redimensionado desde el grip."""
        self._r_w = self.winfo_width()
        self._r_h = self.winfo_height()
        self._r_x = e.x_root
        self._r_y = e.y_root
        return "break"  # evita que también se dispare el movimiento de la ventana

    def _on_resize(self, e):
        """Redimensiona la ventana al arrastrar el grip."""
        new_w = max(220, self._r_w + (e.x_root - self._r_x))
        new_h = max(160, self._r_h + (e.y_root - self._r_y))
        self.geometry(f"{new_w}x{new_h}")
        return "break"

    def _on_resize_end(self, e):
        """Guarda el nuevo tamaño al soltar el grip."""
        self._save_position()
        return "break"

    def _load_saved_geometry(self):
        """Lee el dict {x, y, w, h} guardado para este posit, o None."""
        import json
        import os

        task_id = getattr(self.task, "id", None)
        if not task_id:
            return None
        positions_file = os.path.join("data", "posit_positions.json")
        if not os.path.exists(positions_file):
            return None
        try:
            with open(positions_file, "r", encoding="utf-8") as f:
                return json.load(f).get(task_id)
        except Exception:
            return None

    def _save_position(self):
        """Guarda la posición actual del posit"""
        import json
        import os

        task_id = getattr(self.task, "id", None)
        if not task_id:
            print(f"[WARNING] No se pudo guardar posición: task_id es None")
            return

        x = self.winfo_x()
        y = self.winfo_y()

        positions_file = os.path.join("data", "posit_positions.json")
        positions_data = {}

        if os.path.exists(positions_file):
            try:
                with open(positions_file, "r", encoding="utf-8") as f:
                    positions_data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Error al leer posiciones: {e}")
                positions_data = {}

        # Guardar posición y tamaño actuales
        positions_data[task_id] = {
            "x": x,
            "y": y,
            "w": self.winfo_width(),
            "h": self.winfo_height(),
        }

        try:
            os.makedirs("data", exist_ok=True)
            with open(positions_file, "w", encoding="utf-8") as f:
                json.dump(positions_data, f, indent=2)
            print(f"[DEBUG] Posición guardada para {task_id}: ({x}, {y})")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar posición: {e}")

    def _add_to_pomodoro(self):
        """Agrega esta tarea/misión a la cola de Pomodoro"""
        from tkinter import messagebox
        try:
            # Verificar si existe el pomodoro_manager
            if not hasattr(self.master, 'pomodoro_manager'):
                messagebox.showwarning("Pomodoro", "El sistema Pomodoro no está disponible")
                return

            # Obtener el ID de la tarea
            task_id = getattr(self.task, 'id', None)
            if not task_id:
                messagebox.showerror("Error", "No se pudo obtener el ID de la tarea")
                return

            # Agregar tarea a la cola
            self.master.pomodoro_manager.add_task_to_queue(task_id)

            # Abrir ventana de Pomodoro si no está abierta
            if not hasattr(self.master, 'pomodoro_window') or self.master.pomodoro_window is None or not self.master.pomodoro_window.winfo_exists():
                self.master.open_pomodoro()
            else:
                # Si ya está abierta, solo refrescar la lista
                self.master.pomodoro_window._refresh_task_list()
                self.master.pomodoro_window.lift()

            task_title = getattr(self.task, 'title', 'Tarea')
            messagebox.showinfo("Pomodoro", f"Tarea agregada al Pomodoro:\n{task_title}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo agregar al Pomodoro:\n{e}")

    def close(self):
        """Cierra la ventana"""
        try:
            self._save_position()  # Guardar posición antes de cerrar
            self.destroy()
        except Exception:
            pass

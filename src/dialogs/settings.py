#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diálogo de Ajustes: tema (claro/oscuro), Pomodoro, notificaciones y ventana/inicio.
"""

import tkinter as tk
from tkinter import messagebox

from ..config import (
    MODERN_COLORS, GRADIENTS, NOTIFICATION_CONFIG, PROJECT_ROOT,
    get_theme, load_settings, save_settings,
)
from ..ui.components import PillButton


class SettingsDialog(tk.Toplevel):
    """Ventana de preferencias de la app."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("⚙️ Ajustes")
        self.configure(bg=GRADIENTS["Card"][0])
        self.resizable(False, False)
        self.transient(master)
        self._container = None
        self._build()

    def _build(self):
        """(Re)construye el contenido con el tema activo (permite re-tintar en vivo)."""
        self.configure(bg=GRADIENTS["Card"][0])
        if self._container is not None:
            try:
                self._container.destroy()
            except Exception:
                pass
        self._container = tk.Frame(self, bg=GRADIENTS["Card"][0])
        self._container.pack(fill="both", expand=True)
        wrap = tk.Frame(self._container, bg=GRADIENTS["Card"][0])
        wrap.pack(fill="both", expand=True, padx=16, pady=16)

        # ---------- Tema ----------
        self._section(wrap, "🎨 Tema")
        self.var_theme = tk.StringVar(value=get_theme())
        theme_row = tk.Frame(wrap, bg=GRADIENTS["Card"][0])
        theme_row.pack(fill="x", pady=(0, 14))
        for value, label in (("light", "☀️ Claro"), ("dark", "🌙 Oscuro")):
            tk.Radiobutton(
                theme_row, text=label, value=value, variable=self.var_theme,
                command=self._on_theme, bg=GRADIENTS["Card"][0],
                fg=MODERN_COLORS["Text"], selectcolor=MODERN_COLORS["Light"],
                activebackground=GRADIENTS["Card"][0], font=("Segoe UI", 10, "bold")
            ).pack(side="left", padx=(0, 16))

        # ---------- Pomodoro ----------
        self._section(wrap, "🍅 Pomodoro (minutos)")
        pm = self.app.pomodoro_manager
        self.var_work = tk.IntVar(value=pm.work_duration)
        self.var_break = tk.IntVar(value=pm.break_duration)
        self.var_long = tk.IntVar(value=pm.long_break_duration)
        self.var_sessions = tk.IntVar(value=pm.sessions_per_cycle)
        grid = tk.Frame(wrap, bg=GRADIENTS["Card"][0])
        grid.pack(fill="x", pady=(0, 6))
        self._spin(grid, "Trabajo", self.var_work, 1, 120, 0)
        self._spin(grid, "Descanso", self.var_break, 1, 60, 1)
        self._spin(grid, "Descanso largo", self.var_long, 1, 60, 2)
        self._spin(grid, "Sesiones/ciclo", self.var_sessions, 1, 12, 3)
        PillButton(wrap, "Guardar Pomodoro", self._save_pomodoro, "Success", "small").pack(
            anchor="w", pady=(4, 14)
        )

        # ---------- Notificaciones ----------
        self._section(wrap, "🔔 Notificaciones")
        self.var_notif = tk.BooleanVar(value=NOTIFICATION_CONFIG["enable_system_notifications"])
        tk.Checkbutton(
            wrap, text="Avisar de tareas próximas a vencer", variable=self.var_notif,
            command=self._on_notif, bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
            selectcolor=MODERN_COLORS["Light"], activebackground=GRADIENTS["Card"][0],
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 14))

        # ---------- Ventana / inicio ----------
        self._section(wrap, "🪟 Ventana e inicio")
        self.var_topdef = tk.BooleanVar(value=bool(load_settings().get("topmost_default", False)))
        tk.Checkbutton(
            wrap, text="'Siempre arriba' por defecto", variable=self.var_topdef,
            command=self._on_topdef, bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
            selectcolor=MODERN_COLORS["Light"], activebackground=GRADIENTS["Card"][0],
            font=("Segoe UI", 10)
        ).pack(anchor="w")

        self.lbl_startup = tk.Label(
            wrap, text="", bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9)
        )
        self.lbl_startup.pack(anchor="w", pady=(6, 4))
        startup_row = tk.Frame(wrap, bg=GRADIENTS["Card"][0])
        startup_row.pack(anchor="w", pady=(0, 14))
        PillButton(startup_row, "Activar inicio", self._enable_startup, "Primary", "small").pack(
            side="left", padx=(0, 6)
        )
        PillButton(startup_row, "Desactivar", self._disable_startup, "Danger", "small").pack(side="left")
        self._refresh_startup_status()

        PillButton(wrap, "Cerrar", self.destroy, "Secondary", "normal", "✖").pack(anchor="e", pady=(6, 0))

    # ---------------------------- Helpers de UI ----------------------------
    def _section(self, parent, title):
        tk.Label(
            parent, text=title, bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(6, 4))

    def _spin(self, parent, label, var, lo, hi, row):
        tk.Label(
            parent, text=label, bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9)
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=5).grid(
            row=row, column=1, sticky="w", pady=2
        )

    # ---------------------------- Acciones ----------------------------
    def _on_theme(self):
        # Reconstruye la ventana principal en vivo y re-tinta este diálogo.
        # El rebuild se difiere para no destruir el widget que dispara el callback.
        self.app.apply_theme(self.var_theme.get())
        self.after(10, self._build)
        self.after(20, self.lift)

    def _save_pomodoro(self):
        pm = self.app.pomodoro_manager
        pm.work_duration = self.var_work.get()
        pm.break_duration = self.var_break.get()
        pm.long_break_duration = self.var_long.get()
        pm.sessions_per_cycle = self.var_sessions.get()
        try:
            pm.save_config()
        except Exception:
            pass
        messagebox.showinfo("Ajustes", "Configuración de Pomodoro guardada.")

    def _on_notif(self):
        value = self.var_notif.get()
        NOTIFICATION_CONFIG["enable_system_notifications"] = value
        settings = load_settings()
        settings["notifications_enabled"] = value
        save_settings(settings)

    def _on_topdef(self):
        value = self.var_topdef.get()
        settings = load_settings()
        settings["topmost_default"] = value
        save_settings(settings)
        # Aplicar también en la sesión actual
        try:
            self.app.var_topmost.set(value)
            self.app.attributes("-topmost", value)
        except Exception:
            pass

    # ---------------------------- Inicio automático ----------------------------
    def _startup_module(self):
        """Carga scripts/install_startup.py como módulo (usa winreg en Windows)."""
        import importlib.util
        path = PROJECT_ROOT / "scripts" / "install_startup.py"
        spec = importlib.util.spec_from_file_location("install_startup", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _refresh_startup_status(self):
        try:
            active = self._startup_module().check_startup()
        except Exception:
            active = False
        self.lbl_startup.configure(
            text="Inicio automático: ACTIVADO" if active else "Inicio automático: desactivado"
        )

    def _enable_startup(self):
        try:
            ok = self._startup_module().install_startup()
        except Exception:
            ok = False
        messagebox.showinfo("Inicio automático", "Activado." if ok else "No se pudo activar.")
        self._refresh_startup_status()

    def _disable_startup(self):
        try:
            self._startup_module().uninstall_startup()
        except Exception:
            pass
        messagebox.showinfo("Inicio automático", "Desactivado.")
        self._refresh_startup_status()

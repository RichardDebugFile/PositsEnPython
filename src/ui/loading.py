#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pantalla de carga con spinner animado
"""

import tkinter as tk
from tkinter import ttk
from ..config import MODERN_COLORS, GRADIENTS


class LoadingOverlay:
    """
    Overlay de carga con spinner que se muestra sobre la ventana principal.
    Se puede usar como context manager:

    with LoadingOverlay(parent, "Procesando..."):
        # operación larga
        pass
    """

    def __init__(self, parent, message="Cargando..."):
        self.parent = parent
        self.message = message
        self.overlay = None
        self.active = False

    def show(self):
        """Muestra el overlay de carga"""
        if self.active:
            return

        self.active = True

        # Crear overlay semi-transparente
        self.overlay = tk.Toplevel(self.parent)
        self.overlay.title("")
        self.overlay.attributes("-topmost", True)
        self.overlay.overrideredirect(True)

        # Posicionar sobre la ventana padre
        self.overlay.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()

        overlay_w = 300
        overlay_h = 150
        x = parent_x + (parent_w - overlay_w) // 2
        y = parent_y + (parent_h - overlay_h) // 2

        self.overlay.geometry(f"{overlay_w}x{overlay_h}+{x}+{y}")

        # Fondo del overlay
        bg_color = GRADIENTS["Card"][0]
        self.overlay.configure(bg=bg_color)

        # Frame principal
        main_frame = tk.Frame(self.overlay, bg=bg_color)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Spinner (usando ttk.Progressbar en modo indeterminate)
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Loading.Horizontal.TProgressbar",
            troughcolor=MODERN_COLORS["Light"],
            background=MODERN_COLORS["Primary"],
            borderwidth=0,
            thickness=8
        )

        self.progress = ttk.Progressbar(
            main_frame,
            style="Loading.Horizontal.TProgressbar",
            mode="indeterminate",
            length=260
        )
        self.progress.pack(pady=(10, 10))
        self.progress.start(10)  # Velocidad de animación

        # Mensaje
        self.label = tk.Label(
            main_frame,
            text=self.message,
            bg=bg_color,
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold"),
            wraplength=260
        )
        self.label.pack(pady=(5, 10))

        # Asegurar que se actualice la interfaz
        self.overlay.update()

    def update_message(self, new_message: str):
        """Actualiza el mensaje del loading"""
        if self.active and self.label:
            self.label.configure(text=new_message)
            self.overlay.update()

    def hide(self):
        """Oculta el overlay de carga"""
        if not self.active:
            return

        self.active = False

        if self.overlay:
            try:
                self.progress.stop()
                self.overlay.destroy()
            except Exception:
                pass
            self.overlay = None

    def __enter__(self):
        """Soporte para context manager"""
        self.show()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Soporte para context manager"""
        self.hide()
        return False

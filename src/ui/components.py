#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Componentes UI reutilizables
"""

import tkinter as tk
import tkinter.font as tkfont
from ..config import MODERN_COLORS, GRADIENTS, VIBRANT_COLORS, COLOR_LABELS


# ---------------------------- Botón Pill (Canvas) ----------------------------
class PillButton(tk.Canvas):
    """
    Botón redondeado estilo 'pill' dibujado en Canvas.
    Soporta hover, focus y diferentes tamaños.
    """

    def __init__(self, parent, text, command, color="Primary", size="normal", icon=""):
        self.base_color, self.hover_color = GRADIENTS.get(color, GRADIENTS["Primary"])
        self.text_str = f"{icon} {text}" if icon else text
        self.command = command

        # Configurar tamaño
        if size == "small":
            self.font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
            self.padx, self.pady, self.radius = 10, 6, 12
        elif size == "large":
            self.font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
            self.padx, self.pady, self.radius = 18, 10, 16
        else:  # normal
            self.font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            self.padx, self.pady, self.radius = 14, 8, 14

        # Calcular dimensiones
        tw = self.font.measure(self.text_str)
        th = self.font.metrics("linespace")
        width = tw + 2 * self.padx
        height = th + 2 * self.pady

        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            bg=parent.cget("bg"),
            takefocus=1,
            cursor="hand2"
        )

        self._draw(self.base_color)

        # Bindings
        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.base_color))
        self.bind("<Button-1>", self._on_click)
        self.bind("<Return>", self._on_click)
        self.bind("<space>", self._on_click)

    def _on_click(self, event=None):
        if callable(self.command):
            self.command()

    def _draw_rounded(self, x1, y1, x2, y2, r, fill):
        """Dibuja un rectángulo con esquinas redondeadas"""
        self.create_rectangle(x1+r, y1, x2-r, y2, outline="", fill=fill)
        self.create_rectangle(x1, y1+r, x2, y2-r, outline="", fill=fill)
        self.create_oval(x1, y1, x1+2*r, y1+2*r, outline="", fill=fill)
        self.create_oval(x2-2*r, y1, x2, y1+2*r, outline="", fill=fill)
        self.create_oval(x1, y2-2*r, x1+2*r, y2, outline="", fill=fill)
        self.create_oval(x2-2*r, y2-2*r, x2, y2, outline="", fill=fill)

    def _draw(self, bg_color):
        """Redibuja el botón con el color especificado"""
        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        self._draw_rounded(0, 0, w, h, self.radius, bg_color)
        self.create_text(w//2, h//2, text=self.text_str, fill="white", font=self.font)

    def set_text(self, new_text):
        """Cambia el texto del botón"""
        self.text_str = new_text
        self._draw(self.base_color if not self.winfo_containing(*self.winfo_pointerxy()) == self else self.hover_color)


# ---------------------------- Paleta de Colores ----------------------------
class ColorPalette(tk.Toplevel):
    """Ventana de selección de color para posits"""

    def __init__(self, master, on_pick):
        super().__init__(master)
        self.title("🎨 Paleta")
        self.resizable(False, False)
        self.on_pick = on_pick
        self.configure(bg=MODERN_COLORS["Background"])
        self.attributes("-topmost", True)

        row = 0
        col = 0

        for name, hexv in VIBRANT_COLORS.items():
            # Swatch de color
            sw = tk.Canvas(
                self,
                width=72,
                height=36,
                bg=hexv,
                highlightthickness=0,
                cursor="hand2"
            )
            sw.grid(row=row, column=col, padx=6, pady=6)
            sw.bind("<Button-1>", lambda e, n=name: self._pick(n))

            # Etiqueta
            tk.Label(
                self,
                text=COLOR_LABELS.get(name, name),
                bg=self.cget("bg"),
                fg=MODERN_COLORS["Text"],
                font=("Segoe UI", 8)
            ).grid(row=row+1, column=col)

            col += 1
            if col >= 3:
                col = 0
                row += 2

    def _pick(self, name):
        try:
            self.on_pick(name)
        finally:
            self.destroy()


# ---------------------------- Helpers de Layout ----------------------------
def create_centered_row(parent):
    """Crea un frame con contenido centrado usando grid"""
    row = tk.Frame(parent, bg=parent.cget("bg"))
    row.pack(fill="x")
    row.grid_columnconfigure(0, weight=1)
    row.grid_columnconfigure(1, weight=0)
    row.grid_columnconfigure(2, weight=1)

    tk.Frame(row, bg=parent.cget("bg")).grid(row=0, column=0, sticky="ew")
    center = tk.Frame(row, bg=parent.cget("bg"))
    center.grid(row=0, column=1)
    tk.Frame(row, bg=parent.cget("bg")).grid(row=0, column=2, sticky="ew")

    return center


def create_stat_card(parent, label, value, color="Primary", icon=""):
    """
    Crea una tarjeta de estadística.

    Returns:
        tuple (card_frame, value_label) para actualizar el valor después
    """
    card = tk.Frame(parent, bg=GRADIENTS[color][0], relief="flat", bd=0)
    card.pack(side="left", padx=4, pady=4)

    header = tk.Frame(card, bg=GRADIENTS[color][0])
    header.pack(fill="x", padx=8, pady=(4, 0))

    tk.Label(
        header,
        text=f"{icon} {label}",
        bg=GRADIENTS[color][0],
        fg="white",
        font=("Segoe UI", 9, "bold")
    ).pack(anchor="w")

    value_frame = tk.Frame(card, bg=GRADIENTS[color][0])
    value_frame.pack(fill="x", padx=8, pady=(0, 4))

    value_label = tk.Label(
        value_frame,
        text=str(value),
        bg=GRADIENTS[color][0],
        fg="white",
        font=("Segoe UI", 16, "bold")
    )
    value_label.pack(anchor="w")

    return card, value_label

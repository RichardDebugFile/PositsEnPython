#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estilos para widgets ttk (Combobox, Scrollbar, Progressbar) según el tema activo.

Los widgets tk normales toman su color de MODERN_COLORS/GRADIENTS al construirse,
pero los ttk se pintan vía ttk.Style. Este módulo aplica esos estilos con la
paleta activa; debe llamarse al iniciar y en cada cambio de tema.

Se usa el tema base 'clam' porque respeta las opciones de color (el tema nativo
de Windows ignora fieldbackground/foreground en el Combobox).
"""

from tkinter import ttk

from ..config import MODERN_COLORS


def apply_ttk_theme(root):
    """Configura los estilos ttk con los colores del tema activo."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    card = MODERN_COLORS["Card"]
    field = MODERN_COLORS["White"]
    text = MODERN_COLORS["Text"]
    light = MODERN_COLORS["Light"]
    primary = MODERN_COLORS["Primary"]

    # Combobox (campo + flecha)
    style.configure(
        "TCombobox",
        fieldbackground=field, background=field, foreground=text,
        arrowcolor=text, bordercolor=light, lightcolor=light, darkcolor=light,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", field)],
        foreground=[("readonly", text)],
        selectbackground=[("readonly", field)],
        selectforeground=[("readonly", text)],
        arrowcolor=[("active", primary)],
    )

    # Lista desplegable del Combobox (es un Listbox tk, se ajusta por option_add)
    root.option_add("*TCombobox*Listbox.background", field)
    root.option_add("*TCombobox*Listbox.foreground", text)
    root.option_add("*TCombobox*Listbox.selectBackground", primary)
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

    # Scrollbars
    for orient in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            orient, background=light, troughcolor=card,
            bordercolor=card, arrowcolor=text,
        )

    # Progressbar por defecto
    style.configure(
        "TProgressbar", background=primary, troughcolor=light, bordercolor=light,
    )

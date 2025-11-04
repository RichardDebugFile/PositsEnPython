#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para manejo de colores
"""

from datetime import date
from ..config import MODERN_COLORS, VIBRANT_COLORS, COLOR_LABELS, SPANISH_TO_KEY


def clamp(v: float, lo: float, hi: float) -> float:
    """Limita un valor entre lo y hi"""
    return max(lo, min(v, hi))


def mix_hex(c1: str, c2: str, t: float) -> str:
    """
    Mezcla dos colores hexadecimales.
    t=0 retorna c1, t=1 retorna c2
    """
    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def urgency_color(start: date, due: date, now: date) -> str:
    """
    Calcula color de urgencia basado en el progreso temporal.
    Gradiente de Text -> Danger según qué tan cerca está del vencimiento.
    """
    base = MODERN_COLORS["Text"]
    red = MODERN_COLORS["Danger"]

    if due <= now:
        return red

    total_span = (due - start).days
    if total_span <= 0:
        return red

    remaining = (due - now).days
    t = 1.0 - (remaining / total_span)
    t = clamp(t, 0.0, 1.0)
    return mix_hex(base, red, t)


def normalize_color(value: str | None) -> str | None:
    """
    Normaliza un nombre de color a su clave interna.
    Acepta claves directas, etiquetas en español, o sinónimos.
    """
    if not value:
        return None

    v = value.strip().lower()

    # Acepta clave directa (ej: "Sunshine")
    for k in VIBRANT_COLORS.keys():
        if v == k.lower():
            return k

    # Acepta etiqueta en español (ej: "Amarillo")
    for k, lbl in COLOR_LABELS.items():
        if v == lbl.lower():
            return k

    # Sinónimos
    if v in SPANISH_TO_KEY:
        return SPANISH_TO_KEY[v]

    return None

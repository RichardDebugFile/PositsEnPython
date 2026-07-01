#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests del sistema de temas (claro/oscuro) de config.py.
Aísla el archivo de settings en tmp para no tocar data/ real.
"""

import src.config as C


def test_paletas_pobladas():
    assert "Background" in C.MODERN_COLORS
    assert "Primary" in C.GRADIENTS


def test_set_theme_muta_en_el_sitio(tmp_path, monkeypatch):
    # Redirigir el archivo de settings a tmp (no tocar data/ real)
    monkeypatch.setattr(C, "SETTINGS_FILE", tmp_path / "app_settings.json")

    modern_obj = C.MODERN_COLORS
    grad_obj = C.GRADIENTS
    original = C.get_theme()
    try:
        C.set_theme("dark")
        assert C.get_theme() == "dark"
        # Debe mutar el MISMO objeto (no rebindear), para que todos los módulos
        # que ya lo importaron vean el cambio.
        assert C.MODERN_COLORS is modern_obj
        assert C.GRADIENTS is grad_obj
        assert C.MODERN_COLORS["Background"] == "#15171C"

        C.set_theme("light")
        assert C.get_theme() == "light"
        assert C.MODERN_COLORS["Background"] == "#F8F9FA"
    finally:
        C.set_theme(original)


def test_theme_invalido_cae_en_light(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SETTINGS_FILE", tmp_path / "app_settings.json")
    original = C.get_theme()
    try:
        assert C.set_theme("inexistente") == "light"
    finally:
        C.set_theme(original)

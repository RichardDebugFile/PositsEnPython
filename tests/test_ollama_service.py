#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests del health-check de Ollama (sin levantar el servicio real).
"""

from src.services.ollama import is_ollama_running


def test_is_ollama_running_false_en_puerto_cerrado():
    # Puerto cerrado -> debe devolver False rápido y sin lanzar excepción.
    assert is_ollama_running("http://127.0.0.1:1", timeout=1.0) is False

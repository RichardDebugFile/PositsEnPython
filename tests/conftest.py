#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixtures compartidas para la suite de pytest.

Aíslan todo el estado en directorios temporales (``tmp_path``) para que los
tests NO toquen los datos reales en ``data/`` (notas, gamificación, etc.).
"""

import sys
from pathlib import Path

import pytest

# Asegurar que la raíz del repo esté en sys.path al ejecutar pytest desde cualquier sitio.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import TaskStore  # noqa: E402
from src.models.gamification import GamificationManager  # noqa: E402


@pytest.fixture
def gamification(tmp_path):
    """GamificationManager aislado en un archivo temporal."""
    return GamificationManager(file_path=tmp_path / "gamification.json")


@pytest.fixture
def store(tmp_path):
    """
    TaskStore con notas y gamificación en archivos temporales.

    TaskStore construye internamente un GamificationManager con la ruta por
    defecto; aquí lo reemplazamos por uno temporal para no mutar data real.
    """
    s = TaskStore(file_path=tmp_path / "notes.json")
    s.gamification = GamificationManager(file_path=tmp_path / "gamification.json")
    return s

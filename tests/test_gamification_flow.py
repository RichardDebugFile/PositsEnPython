#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests de flujo del sistema de gamificación (XP, nivel, misiones, persistencia).
Usa la fixture `gamification` aislada en tmp.
"""

from src.models.gamification import GamificationManager


class TestEstadoInicial:
    def test_valores_por_defecto(self, gamification):
        assert gamification.data["level"] == 1
        assert gamification.data["xp"] == 0
        assert len(gamification.data["daily_missions"]) == 3


class TestMisiones:
    def test_crear_tarea_avanza_mision_de_creacion(self, gamification):
        gamification.on_task_created()
        # Misión 3 = "Crear 2 nuevas tareas"
        assert gamification.data["daily_missions"][2]["progress"] == 1

    def test_completar_tarea_otorga_xp_y_cuenta(self, gamification):
        resultado = gamification.on_task_completed("medium")
        assert resultado["xp_gained"] > 0
        assert gamification.data["total_tasks_completed"] == 1


class TestNivelYPersistencia:
    def test_add_xp_sube_de_nivel(self, gamification):
        leveled = gamification.add_xp(100_000)
        assert leveled is True
        assert gamification.data["level"] > 1

    def test_persistencia_en_disco(self, tmp_path):
        ruta = tmp_path / "gam.json"
        g1 = GamificationManager(file_path=ruta)
        g1.add_xp(50)

        g2 = GamificationManager(file_path=ruta)
        assert g2.data["xp"] == 50

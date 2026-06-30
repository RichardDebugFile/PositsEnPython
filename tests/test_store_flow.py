#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests de integración / flujo de TaskStore (alta, completar, persistencia,
prioridad, estadísticas y orden). Usa la fixture `store` aislada en tmp.
"""

from datetime import date

from src.models import TaskStore


class TestFlujoBasico:
    def test_alta_persiste_en_disco_y_se_recarga(self, store, tmp_path):
        store.add("Comprar pan", "con semillas", date.today(), "medium", color="Ocean")
        assert len(store.tasks) == 1
        notes = tmp_path / "notes.json"
        assert notes.exists()

        recargado = TaskStore(file_path=notes)
        assert len(recargado.tasks) == 1
        assert recargado.tasks[0].title == "Comprar pan"
        assert recargado.tasks[0].color == "Ocean"

    def test_completar_otorga_xp(self, store):
        store.add("tarea", "", date.today(), "high")
        xp_antes = store.gamification.data["xp"]

        resultado = store.toggle_done(0)

        assert store.tasks[0].done is True
        assert resultado["xp_gained"] > 0
        assert store.gamification.data["xp"] > xp_antes

    def test_descompletar_devuelve_a_pendiente(self, store):
        store.add("tarea", "", date.today(), "low")
        store.toggle_done(0)          # completar
        resultado = store.toggle_done(0)  # descompletar
        assert store.tasks[0].done is False
        assert resultado is None

    def test_eliminar_por_indice(self, store):
        store.add("a", "", date.today(), "low")
        store.add("b", "", date.today(), "low")
        store.delete(0)
        assert [t.title for t in store.tasks] == ["b"]


class TestPrioridadYConsultas:
    def test_ciclo_de_prioridad(self, store):
        store.add("t", "", date.today(), "low")
        esperado = ["medium", "high", "urgent", "low"]
        for valor in esperado:
            store.toggle_priority(0)
            assert store.tasks[0].priority == valor

    def test_index_y_get_by_id(self, store):
        tarea = store.add("t", "", date.today(), "low")
        assert store.index_by_id(tarea.id) == 0
        assert store.get_by_id(tarea.id) is tarea
        assert store.index_by_id("inexistente") == -1


class TestEstadisticasYOrden:
    def test_estadisticas(self, store):
        store.add("a", "", date.today(), "low")
        store.add("b", "", date.today(), "low")
        store.toggle_done(0)
        stats = store.get_statistics()
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["pending"] == 1

    def test_orden_por_titulo(self, store):
        store.add("Zeta", "", date.today(), "low")
        store.add("Alfa", "", date.today(), "low")
        store.sort_tasks("title")
        assert [t.title for t in store.tasks] == ["Alfa", "Zeta"]

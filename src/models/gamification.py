#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Gamificación: Misiones Diarias, Niveles y Puntos
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from ..config import GAMIFICATION_FILE, GAMIFICATION_CONFIG
from ..utils.dates import today_date, fmt_date, parse_date
from ..utils.logger import logger


class GamificationManager:
    """
    Gestiona el sistema de gamificación con:
    - Puntos de experiencia (XP)
    - Niveles de productividad
    - Misiones diarias (tareas que se deben completar cada día)
    - Rachas de días consecutivos
    - Estadísticas históricas
    """

    def __init__(self, file_path: Path = GAMIFICATION_FILE):
        self.file_path = file_path
        self.data = {
            "xp": 0,
            "level": 1,
            "daily_missions": [],
            "missions_completed_today": 0,
            "current_streak": 0,
            "longest_streak": 0,
            "last_mission_date": None,
            "total_tasks_completed": 0,
            "history": [],  # [{date, xp_gained, tasks_completed, missions_completed}]
        }
        self.load()

    def load(self):
        """Carga datos de gamificación desde el archivo JSON"""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
                    logger.debug(f"Gamificación cargada: Nivel {self.data['level']}, XP {self.data['xp']}")
            except Exception as e:
                logger.error(f"Error cargando gamificación: {e}")
        else:
            self._create_default_daily_missions()
            self.save()

    def save(self):
        """Guarda datos de gamificación en el archivo JSON"""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error guardando gamificación: {e}")

    def _create_default_daily_missions(self):
        """Crea misiones diarias por defecto"""
        self.data["daily_missions"] = [
            {"id": "mission_1", "title": "Completar 3 tareas", "goal": 3, "progress": 0, "completed": False},
            {"id": "mission_2", "title": "Completar 1 tarea prioritaria", "goal": 1, "progress": 0, "completed": False},
            {"id": "mission_3", "title": "Crear 2 nuevas tareas", "goal": 2, "progress": 0, "completed": False},
        ]

    def check_and_reset_daily(self):
        """
        Verifica si es un nuevo día y resetea las misiones diarias.
        Actualiza rachas si corresponde.
        """
        today = fmt_date(today_date())
        last_date = self.data.get("last_mission_date")

        if last_date != today:
            # Verificar racha
            if last_date:
                last_date_obj = parse_date(last_date)
                days_diff = (today_date() - last_date_obj).days

                # Si completó al menos UNA tarea ayer, mantener/incrementar racha
                tasks_completed_yesterday = self.data.get("tasks_completed_yesterday", 0)
                if days_diff == 1 and tasks_completed_yesterday > 0:
                    self.data["current_streak"] += 1
                    if self.data["current_streak"] > self.data["longest_streak"]:
                        self.data["longest_streak"] = self.data["current_streak"]
                    logger.info(f"¡Racha continuada! Ahora {self.data['current_streak']} días consecutivos")
                elif days_diff > 1:
                    # Se rompió la racha
                    self.data["current_streak"] = 0
                    logger.info("Racha reiniciada (días saltados)")

            # Guardar tareas completadas de hoy como "yesterday" para mañana
            self.data["tasks_completed_yesterday"] = self.data.get("tasks_completed_today", 0)
            self.data["tasks_completed_today"] = 0

            # Resetear misiones diarias
            self._create_default_daily_missions()
            self.data["missions_completed_today"] = 0
            self.data["last_mission_date"] = today
            self.save()
            logger.info("Misiones diarias reseteadas")

    def _calculate_xp_for_level(self, level: int) -> int:
        """Calcula los puntos totales necesarios para alcanzar un nivel dado"""
        base = GAMIFICATION_CONFIG["base_points_per_level"]
        multiplier = GAMIFICATION_CONFIG["level_multiplier"]

        total_xp = 0
        for lvl in range(1, level):
            # Cada nivel requiere base * (multiplier ^ (lvl-1))
            total_xp += int(base * (multiplier ** (lvl - 1)))

        return total_xp

    def add_xp(self, amount: int, reason: str = ""):
        """Agrega puntos de experiencia y sube de nivel si corresponde"""
        self.data["xp"] += amount
        old_level = self.data["level"]

        # Calcular nivel actual basado en XP total acumulado
        new_level = old_level
        while self.data["xp"] >= self._calculate_xp_for_level(new_level + 1):
            new_level += 1

        if new_level > old_level:
            self.data["level"] = new_level
            logger.info(f"¡NIVEL UP! Ahora eres nivel {new_level}")
            self.save()
            return True  # Hubo level up

        self.save()
        return False

    def on_task_completed(self, is_priority: bool = False):
        """
        Llamar cuando se completa una tarea.
        Otorga XP y actualiza progreso de misiones.
        """
        self.check_and_reset_daily()

        # XP base
        xp_gained = GAMIFICATION_CONFIG["points_per_priority_task"] if is_priority else GAMIFICATION_CONFIG["points_per_task"]

        # Bonus por racha
        if self.data["current_streak"] > 0:
            xp_gained = int(xp_gained * GAMIFICATION_CONFIG["streak_bonus_multiplier"])

        leveled_up = self.add_xp(xp_gained, "Tarea completada")

        # Actualizar contadores
        self.data["total_tasks_completed"] += 1
        self.data["tasks_completed_today"] = self.data.get("tasks_completed_today", 0) + 1

        # Iniciar racha si es la primera tarea completada
        if self.data.get("current_streak", 0) == 0 and self.data["tasks_completed_today"] == 1:
            self.data["current_streak"] = 1
            if self.data["current_streak"] > self.data.get("longest_streak", 0):
                self.data["longest_streak"] = 1
            logger.info("¡Racha iniciada! Día 1 completado")

        # Misión 1: Completar tareas
        if not self.data["daily_missions"][0]["completed"]:
            self.data["daily_missions"][0]["progress"] += 1
            if self.data["daily_missions"][0]["progress"] >= self.data["daily_missions"][0]["goal"]:
                self._complete_mission(0)

        # Misión 2: Completar tarea prioritaria
        if is_priority and not self.data["daily_missions"][1]["completed"]:
            self.data["daily_missions"][1]["progress"] += 1
            if self.data["daily_missions"][1]["progress"] >= self.data["daily_missions"][1]["goal"]:
                self._complete_mission(1)

        self.save()
        return {"xp_gained": xp_gained, "leveled_up": leveled_up}

    def on_task_created(self):
        """Llamar cuando se crea una nueva tarea"""
        self.check_and_reset_daily()

        # Misión 3: Crear nuevas tareas
        if not self.data["daily_missions"][2]["completed"]:
            self.data["daily_missions"][2]["progress"] += 1
            if self.data["daily_missions"][2]["progress"] >= self.data["daily_missions"][2]["goal"]:
                self._complete_mission(2)

        self.save()

    def _complete_mission(self, mission_index: int):
        """Marca una misión como completada y otorga bonus"""
        self.data["daily_missions"][mission_index]["completed"] = True
        self.data["missions_completed_today"] += 1

        # Bonus por completar misión
        xp_bonus = GAMIFICATION_CONFIG["points_per_daily_mission"]
        leveled_up = self.add_xp(xp_bonus, f"Misión completada: {self.data['daily_missions'][mission_index]['title']}")

        logger.info(f"¡Misión completada! +{xp_bonus} XP")

        # Si completó todas las misiones del día
        if self.data["missions_completed_today"] >= GAMIFICATION_CONFIG["daily_missions_required"]:
            logger.info("¡TODAS LAS MISIONES DIARIAS COMPLETADAS! 🎉")

    def get_daily_missions(self) -> list[dict]:
        """Retorna las misiones diarias actuales"""
        self.check_and_reset_daily()
        return self.data["daily_missions"]

    def get_stats(self) -> dict:
        """Retorna estadísticas completas"""
        self.check_and_reset_daily()

        current_level = self.data["level"]
        current_xp = self.data["xp"]

        # XP necesario para el nivel actual y siguiente
        xp_for_current_level = self._calculate_xp_for_level(current_level)
        xp_for_next_level = self._calculate_xp_for_level(current_level + 1)

        # XP dentro del nivel actual
        xp_in_current_level = current_xp - xp_for_current_level
        xp_needed_for_next = xp_for_next_level - xp_for_current_level

        return {
            "level": current_level,
            "xp": current_xp,
            "xp_in_level": xp_in_current_level,  # Puntos generados en el nivel actual
            "xp_to_next_level": xp_needed_for_next - xp_in_current_level,  # Puntos faltantes
            "xp_needed_for_next": xp_needed_for_next,  # Total necesario para subir
            "current_streak": self.data["current_streak"],
            "longest_streak": self.data["longest_streak"],
            "total_tasks_completed": self.data["total_tasks_completed"],
            "missions_today": self.data["missions_completed_today"],
            "missions_required": GAMIFICATION_CONFIG["daily_missions_required"],
        }

    def get_progress_percentage(self) -> float:
        """Retorna el progreso al siguiente nivel como porcentaje (0-100)"""
        current_level = self.data["level"]
        current_xp = self.data["xp"]

        # XP necesario para el nivel actual y siguiente
        xp_for_current_level = self._calculate_xp_for_level(current_level)
        xp_for_next_level = self._calculate_xp_for_level(current_level + 1)

        # XP dentro del nivel actual
        xp_in_current_level = current_xp - xp_for_current_level
        xp_needed_for_next = xp_for_next_level - xp_for_current_level

        if xp_needed_for_next <= 0:
            return 100.0

        return (xp_in_current_level / xp_needed_for_next) * 100

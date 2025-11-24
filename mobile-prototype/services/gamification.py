"""
Sistema de gamificación - Versión móvil simplificada
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List


class GamificationSystem:
    """Sistema de XP, niveles y misiones diarias"""

    def __init__(self, data_file: str = "data/gamification.json"):
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Carga datos de gamificación"""
        path = Path(self.data_file)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Datos por defecto
        return {
            "xp": 0,
            "level": 1,
            "missions": [],  # Usuario las crea manualmente
            "last_mission_reset": date.today().isoformat(),
            "current_streak": 0,
            "longest_streak": 0,
            "last_activity_date": None,
            "stats": {
                "tasks_completed": 0,
                "pomodoros_completed": 0,
                "tasks_completed_today": 0
            }
        }

    def _save_data(self):
        """Guarda datos de gamificación"""
        path = Path(self.data_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _check_mission_reset(self):
        """Verifica si hay que resetear el progreso de misiones diarias"""
        last_reset = self.data.get("last_mission_reset")
        today = date.today().isoformat()

        if last_reset != today:
            # Actualizar racha
            self._update_streak()

            # Resetear solo completed (mantener las misiones creadas por el usuario)
            for mission in self.data.get("missions", []):
                mission["completed"] = False

            # Resetear contador de tareas completadas hoy
            self.data["stats"]["tasks_completed_today"] = 0

            self.data["last_mission_reset"] = today
            self._save_data()

    def _update_streak(self):
        """Actualiza la racha de días consecutivos"""
        last_activity = self.data.get("last_activity_date")
        today = date.today().isoformat()

        if last_activity:
            # Calcular diferencia de días
            from datetime import datetime
            last_date = datetime.fromisoformat(last_activity).date()
            current_date = date.today()
            days_diff = (current_date - last_date).days

            # Si fue ayer y se completó al menos una tarea, incrementar racha
            if days_diff == 1 and self.data["stats"]["tasks_completed_today"] > 0:
                self.data["current_streak"] += 1
                if self.data["current_streak"] > self.data["longest_streak"]:
                    self.data["longest_streak"] = self.data["current_streak"]
            elif days_diff > 1:
                # Racha rota
                self.data["current_streak"] = 0

    def _calculate_level(self, xp: int) -> int:
        """Calcula el nivel basado en XP"""
        # Fórmula: nivel = floor(sqrt(xp / 100))
        import math
        return max(1, int(math.sqrt(xp / 100)))

    def _xp_for_next_level(self, level: int) -> int:
        """Calcula XP necesario para el siguiente nivel"""
        return (level + 1) ** 2 * 100

    def add_xp(self, amount: int, reason: str = "") -> Dict:
        """
        Agrega XP y verifica si sube de nivel

        Returns:
            Dict con info de level_up, new_level, etc.
        """
        old_xp = self.data["xp"]
        old_level = self.data["level"]

        self.data["xp"] += amount
        new_level = self._calculate_level(self.data["xp"])

        level_up = new_level > old_level
        if level_up:
            self.data["level"] = new_level

        self._save_data()

        return {
            "xp_gained": amount,
            "total_xp": self.data["xp"],
            "level_up": level_up,
            "old_level": old_level,
            "new_level": new_level,
            "reason": reason
        }

    def on_task_completed(self, priority: str = "normal") -> Dict:
        """Maneja evento de tarea completada"""
        self._check_mission_reset()

        # XP base por tarea
        xp_map = {
            "low": 15,
            "normal": 25,
            "high": 35,
            "urgent": 50
        }
        xp = xp_map.get(priority, 25)

        # Actualizar estadísticas
        self.data["stats"]["tasks_completed"] += 1
        self.data["stats"]["tasks_completed_today"] += 1
        self.data["last_activity_date"] = date.today().isoformat()

        # Iniciar racha si es la primera tarea del día
        if self.data["stats"]["tasks_completed_today"] == 1:
            last_activity = self.data.get("last_activity_date")
            if last_activity:
                from datetime import datetime
                last_date = datetime.fromisoformat(last_activity).date()
                current_date = date.today()
                days_diff = (current_date - last_date).days

                if days_diff == 1:
                    # Continúa la racha
                    self.data["current_streak"] += 1
                elif days_diff == 0:
                    # Mismo día, racha se mantiene
                    pass
                else:
                    # Racha rota, empezar de nuevo
                    self.data["current_streak"] = 1
            else:
                # Primera vez
                self.data["current_streak"] = 1

            # Actualizar récord
            if self.data["current_streak"] > self.data.get("longest_streak", 0):
                self.data["longest_streak"] = self.data["current_streak"]

        # Las misiones ya no se completan automáticamente, el usuario las marca manualmente

        result = self.add_xp(xp, f"Tarea completada (prioridad: {priority})")
        return result

    def on_pomodoro_completed(self) -> Dict:
        """Maneja evento de Pomodoro completado"""
        self._check_mission_reset()

        xp = 30
        self.data["stats"]["pomodoros_completed"] += 1
        self.data["stats"]["last_activity_date"] = date.today().isoformat()

        # Las misiones ya no se completan automáticamente

        result = self.add_xp(xp, "Pomodoro completado")
        return result

    def get_status(self) -> Dict:
        """Obtiene el estado actual de gamificación"""
        self._check_mission_reset()

        current_xp = self.data["xp"]
        current_level = self.data["level"]
        xp_for_current = current_level ** 2 * 100
        xp_for_next = self._xp_for_next_level(current_level)
        xp_progress = current_xp - xp_for_current

        return {
            "xp": current_xp,
            "level": current_level,
            "xp_for_current_level": xp_for_current,
            "xp_for_next_level": xp_for_next,
            "xp_progress": xp_progress,
            "progress_percentage": (xp_progress / (xp_for_next - xp_for_current)) * 100,
            "missions": self.data["missions"],
            "stats": self.data["stats"]
        }

    def get_missions(self) -> List[Dict]:
        """Obtiene misiones diarias"""
        self._check_mission_reset()
        return self.data["missions"]

    def add_mission(self, title: str, xp_reward: int = 50) -> Dict:
        """Agrega una nueva misión diaria (checkbox simple)"""
        mission = {
            "id": f"mission_{len(self.data['missions']) + 1}",
            "title": title,
            "xp_reward": xp_reward,
            "completed": False
        }
        self.data["missions"].append(mission)
        self._save_data()
        return mission

    def update_mission(self, mission_id: str, title: str = None, xp_reward: int = None):
        """Actualiza una misión existente"""
        for mission in self.data["missions"]:
            if mission["id"] == mission_id:
                if title is not None:
                    mission["title"] = title
                if xp_reward is not None:
                    mission["xp_reward"] = xp_reward
                self._save_data()
                return True
        return False

    def delete_mission(self, mission_id: str):
        """Elimina una misión"""
        self.data["missions"] = [m for m in self.data["missions"] if m["id"] != mission_id]
        self._save_data()

    def toggle_mission(self, mission_id: str) -> Dict:
        """Marca/desmarca una misión como completada y otorga XP si corresponde"""
        for mission in self.data["missions"]:
            if mission["id"] == mission_id:
                # Alternar estado
                mission["completed"] = not mission["completed"]

                # Si se completó, otorgar XP
                if mission["completed"]:
                    xp_reward = mission.get("xp_reward", 50)
                    result = self.add_xp(xp_reward, f"Misión completada: {mission['title']}")
                    self._save_data()
                    return {**result, "mission_completed": True}
                else:
                    # Si se desmarca, NO quitar XP (decisión de diseño)
                    self._save_data()
                    return {"mission_completed": False}
        return {}

    def get_streak(self) -> Dict:
        """Obtiene información de la racha actual"""
        return {
            "current_streak": self.data.get("current_streak", 0),
            "longest_streak": self.data.get("longest_streak", 0)
        }

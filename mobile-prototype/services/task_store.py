"""
Servicio de almacenamiento de tareas - Versión móvil
Usa SQLite para mejor rendimiento en móvil
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class TaskStore:
    """Gestor de tareas usando SQLite"""

    def __init__(self, db_path: str = "data/tasks.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Crea la base de datos y tablas si no existen"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla de tareas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT DEFAULT 'normal',
                color TEXT DEFAULT 'blue',
                completed BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                deadline TEXT,
                tags TEXT
            )
        """)

        conn.commit()
        conn.close()

    def add_task(self, title: str, description: str = "", priority: str = "normal",
                 color: str = "blue", deadline: str = None, tags: List[str] = None) -> int:
        """Agrega una nueva tarea"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tags_json = json.dumps(tags or [])
        created_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO tasks (title, description, priority, color, created_at, deadline, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, description, priority, color, created_at, deadline, tags_json))

        task_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return task_id

    def get_tasks(self, filter_completed: bool = None) -> List[Dict]:
        """Obtiene todas las tareas"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if filter_completed is None:
            cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
        else:
            cursor.execute(
                "SELECT * FROM tasks WHERE completed = ? ORDER BY created_at DESC",
                (1 if filter_completed else 0,)
            )

        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            task = dict(row)
            task['tags'] = json.loads(task['tags']) if task['tags'] else []
            tasks.append(task)

        return tasks

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Obtiene una tarea por ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            task = dict(row)
            task['tags'] = json.loads(task['tags']) if task['tags'] else []
            return task
        return None

    def complete_task(self, task_id: int) -> bool:
        """Marca una tarea como completada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completed_at = datetime.now().isoformat()
        cursor.execute("""
            UPDATE tasks SET completed = 1, completed_at = ?
            WHERE id = ?
        """, (completed_at, task_id))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def uncomplete_task(self, task_id: int) -> bool:
        """Marca una tarea como no completada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE tasks SET completed = 0, completed_at = NULL
            WHERE id = ?
        """, (task_id,))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def delete_task(self, task_id: int) -> bool:
        """Elimina una tarea"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def update_task(self, task_id: int, **kwargs) -> bool:
        """Actualiza campos de una tarea"""
        if not kwargs:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Construir query dinámicamente
        fields = []
        values = []
        for key, value in kwargs.items():
            if key == 'tags':
                value = json.dumps(value)
            fields.append(f"{key} = ?")
            values.append(value)

        values.append(task_id)
        query = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"

        cursor.execute(query, values)
        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de tareas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total de tareas
        cursor.execute("SELECT COUNT(*) FROM tasks")
        total = cursor.fetchone()[0]

        # Completadas
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE completed = 1")
        completed = cursor.fetchone()[0]

        # Pendientes
        pending = total - completed

        # Por prioridad
        cursor.execute("""
            SELECT priority, COUNT(*)
            FROM tasks
            WHERE completed = 0
            GROUP BY priority
        """)
        by_priority = dict(cursor.fetchall())

        # Completadas hoy
        today = datetime.now().date().isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM tasks
            WHERE completed = 1 AND DATE(completed_at) = ?
        """, (today,))
        completed_today = cursor.fetchone()[0]

        conn.close()

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "by_priority": by_priority,
            "completed_today": completed_today,
            "completion_rate": (completed / total * 100) if total > 0 else 0
        }

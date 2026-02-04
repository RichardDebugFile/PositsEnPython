import unittest
import sys
import os
from pathlib import Path
from datetime import date

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Task, TaskStore


class TestTaskStore(unittest.TestCase):
    def setUp(self):
        # Usar un archivo temporal para las pruebas
        self.tmp = Path("test_tasks_temp.json")
        if self.tmp.exists():
            self.tmp.unlink()
        self.store = TaskStore(file_path=self.tmp)

    def tearDown(self):
        if self.tmp.exists():
            self.tmp.unlink()

    def test_add_and_toggle_and_delete(self):
        task = Task(
            id="test-001",
            title="prueba",
            desc="",
            start=date.today(),
            due=date.today(),
            priority="medium",
            done=False,
            color="Ocean"
        )
        self.store.tasks.append(task)
        self.assertEqual(len(self.store.tasks), 1)
        self.assertFalse(self.store.tasks[0].done)

        self.store.toggle_done(0)
        self.assertTrue(self.store.tasks[0].done)

        self.store.delete(0)
        self.assertEqual(len(self.store.tasks), 0)

    def test_persistence(self):
        task = Task(
            id="test-002",
            title="persistencia",
            desc="",
            start=date.today(),
            due=date.today(),
            priority="low",
            done=False,
            color="Nature"
        )
        self.store.tasks.append(task)
        self.store.save()
        
        # Vuelve a cargar desde disco
        new_store = TaskStore(file_path=self.tmp)
        self.assertEqual(new_store.tasks[0].title, "persistencia")
        self.assertFalse(new_store.tasks[0].done)


if __name__ == "__main__":
    unittest.main()

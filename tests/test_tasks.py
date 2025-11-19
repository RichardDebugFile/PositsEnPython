import unittest
import json
from pathlib import Path
from main import TaskStore


class TestTaskStore(unittest.TestCase):
    def setUp(self):
        # Usar un archivo temporal para las pruebas
        self.tmp = Path("test_tasks_temp.json")
        if self.tmp.exists():
            self.tmp.unlink()
        self.store = TaskStore(self.tmp)

    def tearDown(self):
        if self.tmp.exists():
            self.tmp.unlink()

    def test_add_and_toggle_and_delete(self):
        self.store.add("prueba")
        self.assertEqual(len(self.store.tasks), 1)
        self.assertFalse(self.store.tasks[0]["done"])

        self.store.toggle(0)
        self.assertTrue(self.store.tasks[0]["done"])

        self.store.delete(0)
        self.assertEqual(len(self.store.tasks), 0)

    def test_persistence(self):
        self.store.add("persistencia")
        # Vuelve a cargar desde disco
        new_store = TaskStore(self.tmp)
        self.assertEqual(new_store.tasks[0]["text"], "persistencia")
        self.assertFalse(new_store.tasks[0]["done"])


if __name__ == "__main__":
    unittest.main()

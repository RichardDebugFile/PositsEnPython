#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite para verificar compatibilidad de Task objects
Identifica todos los problemas de migración de dict a objeto
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, date, timedelta
from src.models import Task, TaskStore, GamificationManager
from src.config import VIBRANT_COLORS
from pathlib import Path


def test_task_creation():
    """Test 1: Crear Task y acceder a atributos"""
    print("\n=== Test 1: Task Creation ===")
    try:
        task = Task(
            id="test-001",
            title="Test Task",
            desc="Test description",
            start=date.today(),
            due=date.today() + timedelta(days=3),
            priority=True,
            done=False,
            color="Sunshine"
        )

        # Test attribute access
        assert task.id == "test-001"
        assert task.title == "Test Task"
        assert task.desc == "Test description"
        assert task.priority == True
        assert task.done == False
        assert task.color == "Sunshine"

        # Test default values
        assert task.pinned == False
        assert task.alpha == 0.98
        assert task.open == False
        assert task.attachments == []

        print("✅ Task creation and attribute access: PASSED")
        return True
    except Exception as e:
        print(f"❌ Task creation FAILED: {e}")
        return False


def test_task_serialization():
    """Test 2: Serialización to_dict y from_dict"""
    print("\n=== Test 2: Task Serialization ===")
    try:
        original = Task(
            id="test-002",
            title="Serialize Test",
            desc="Test serialization",
            start=date.today(),
            due=date.today() + timedelta(days=5),
            priority=False,
            done=True,
            color="Ocean"
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["id"] == "test-002"
        assert data["title"] == "Serialize Test"

        # Deserialize
        restored = Task.from_dict(data)
        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.desc == original.desc
        assert restored.priority == original.priority
        assert restored.done == original.done
        assert restored.color == original.color

        print("✅ Task serialization: PASSED")
        return True
    except Exception as e:
        print(f"❌ Task serialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_store_operations():
    """Test 3: TaskStore operaciones básicas"""
    print("\n=== Test 3: TaskStore Operations ===")
    try:
        # Create temporary store
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()

        store = TaskStore(file_path=Path(temp_file.name))

        # Add task
        task = Task(
            id="test-003",
            title="Store Test",
            desc="Test store operations",
            start=date.today(),
            due=date.today() + timedelta(days=2),
            priority=False,
            done=False,
            color="Nature"
        )
        store.tasks.append(task)

        # Test index_by_id
        idx = store.index_by_id("test-003")
        assert idx >= 0
        assert store.tasks[idx].id == "test-003"

        # Test toggle_done
        result = store.toggle_done("test-003", None)
        assert result == True
        assert store.tasks[idx].done == True

        # Test sort
        store.sort_tasks("date")

        # Test filters
        all_tasks = store.get_filtered_tasks(None, None, show_done=True)
        assert len(all_tasks) > 0

        # Cleanup
        os.unlink(temp_file.name)

        print("✅ TaskStore operations: PASSED")
        return True
    except Exception as e:
        print(f"❌ TaskStore operations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_attribute_access_patterns():
    """Test 4: Patrones de acceso a atributos (no .get())"""
    print("\n=== Test 4: Attribute Access Patterns ===")
    try:
        task = Task(
            id="test-004",
            title="Access Test",
            desc="Test attribute access",
            start=date.today(),
            due=date.today() + timedelta(days=1),
            priority=True,
            done=False,
            color="Sunset"
        )

        # Test direct access
        _ = task.id
        _ = task.title
        _ = task.desc
        _ = task.priority
        _ = task.done
        _ = task.color
        _ = task.pinned
        _ = task.alpha
        _ = task.geometry
        _ = task.open
        _ = task.attachments

        # Test getattr with defaults
        color = getattr(task, "color", "Sunshine")
        assert color == "Sunset"

        title = getattr(task, "title", "")
        assert title == "Access Test"

        nonexistent = getattr(task, "nonexistent", "default")
        assert nonexistent == "default"

        print("✅ Attribute access patterns: PASSED")
        return True
    except Exception as e:
        print(f"❌ Attribute access patterns FAILED: {e}")
        return False


def test_ui_component_compatibility():
    """Test 5: Compatibilidad con componentes UI"""
    print("\n=== Test 5: UI Component Compatibility ===")
    try:
        task = Task(
            id="test-005",
            title="UI Test",
            desc="Test UI compatibility",
            start=date.today(),
            due=date.today() + timedelta(days=7),
            priority=False,
            done=False,
            color="Lavender",
            geometry="560x520+200+200",
            pinned=True,
            alpha=0.95
        )

        # Simulate note_window.py access patterns
        geom = getattr(task, 'geometry', "560x520+200+200")
        assert geom == "560x520+200+200"

        is_top = bool(getattr(task, "pinned", False))
        assert is_top == True

        alpha = float(getattr(task, "alpha", 0.98))
        assert alpha == 0.95

        # Simulate quick_sticky.py access patterns
        color = VIBRANT_COLORS.get(getattr(task, "color", "Sunshine"), "#FFD54F")
        assert color == "#BA68C8"  # Lavender

        title = getattr(task, "title", "") or "(Sin título)"
        assert title == "UI Test"

        desc = getattr(task, "desc", "")
        assert desc == "Test UI compatibility"

        # Simulate task_card.py access patterns
        t_color = task.color
        t_done = task.done
        t_priority = task.priority

        print("✅ UI component compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ UI component compatibility FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_date_handling():
    """Test 6: Manejo de fechas (date vs str)"""
    print("\n=== Test 6: Date Handling ===")
    try:
        from src.utils.dates import fmt_date, parse_date, today_date

        # Test with date object
        now = date.today()
        task1 = Task(
            id="test-006a",
            title="Date Test 1",
            desc="",
            start=now,
            due=now + timedelta(days=3),
            priority=False,
            done=False,
            color="Ocean"
        )

        # Test serialization with date
        data = task1.to_dict()
        assert isinstance(data["due"], str)

        # Test deserialization
        restored = Task.from_dict(data)
        assert isinstance(restored.due, date)

        # Test with string date
        data2 = {
            "id": "test-006b",
            "title": "Date Test 2",
            "desc": "",
            "start": fmt_date(now),
            "due": fmt_date(now + timedelta(days=5)),
            "priority": False,
            "done": False,
            "color": "Nature"
        }

        task2 = Task.from_dict(data2)
        assert isinstance(task2.due, date)

        print("✅ Date handling: PASSED")
        return True
    except Exception as e:
        print(f"❌ Date handling FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_initialization():
    """Test 7: Inicialización de app (sin GUI)"""
    print("\n=== Test 7: App Initialization (No GUI) ===")
    try:
        # Test imports
        from src.app import ModernStickyApp
        from src.ui import (
            PillButton,
            create_centered_row,
            create_stat_card,
            TaskCardRenderer,
            ModernNoteWindow,
            QuickStickyWindow,
            GamificationPanel
        )

        print("✅ App imports: PASSED")

        # Test TaskStore creation
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()

        store = TaskStore(file_path=Path(temp_file.name))

        # Add test tasks
        for i in range(3):
            task = Task(
                id=f"test-init-{i}",
                title=f"Init Test {i}",
                desc=f"Description {i}",
                start=date.today(),
                due=date.today() + timedelta(days=i+1),
                priority=(i % 2 == 0),
                done=False,
                color=list(VIBRANT_COLORS.keys())[i]
            )
            store.tasks.append(task)

        # Test filtering and sorting
        store.sort_tasks("date")
        filtered = store.get_filtered_tasks(None, None, show_done=False)
        assert len(filtered) == 3

        # Cleanup
        os.unlink(temp_file.name)

        print("✅ App initialization: PASSED")
        return True
    except Exception as e:
        print(f"❌ App initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Ejecuta todos los tests"""
    print("=" * 60)
    print("INICIANDO TEST SUITE - TASK OBJECT COMPATIBILITY")
    print("=" * 60)

    tests = [
        test_task_creation,
        test_task_serialization,
        test_task_store_operations,
        test_task_attribute_access_patterns,
        test_ui_component_compatibility,
        test_date_handling,
        test_app_initialization
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 TODOS LOS TESTS PASARON - Sistema listo para usar!")
        return True
    else:
        print("\n⚠️ HAY ERRORES - Revisa los tests fallidos arriba")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Posits Virtuales v2.0
Aplicación de gestión de tareas con IA y gamificación

Punto de entrada de la aplicación
"""

from src.app import ModernStickyApp
from src.models import TaskStore

if __name__ == "__main__":
    store = TaskStore()
    app = ModernStickyApp(store)
    app.mainloop()

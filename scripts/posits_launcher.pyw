#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher para Posits Virtuales
Este script inicia la aplicación sin mostrar la consola
"""

import sys
import os
from pathlib import Path

# Obtener el directorio del script (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Obtener el directorio raíz del proyecto (un nivel arriba)
PROJECT_ROOT = SCRIPT_DIR.parent

# Cambiar al directorio raíz del proyecto
os.chdir(PROJECT_ROOT)

# Agregar el directorio raíz al path para que encuentre 'src'
sys.path.insert(0, str(PROJECT_ROOT))

# Importar y ejecutar la aplicación
if __name__ == "__main__":
    try:
        from src.app import ModernStickyApp
        from src.models import TaskStore

        store = TaskStore()
        app = ModernStickyApp(store)
        app.mainloop()
    except Exception as e:
        # Mostrar error en caso de fallo
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error al iniciar Posits Virtuales",
            f"No se pudo iniciar la aplicación:\n\n{e}\n\nVerifica los logs en data/app.log"
        )
        root.destroy()
        sys.exit(1)

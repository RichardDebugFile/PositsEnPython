#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher para Posits Virtuales.

Inicia la aplicación sin mostrar consola. Pensado para el inicio automático
con Windows: si el intérprete actual no es el del entorno virtual (venv), se
reejecuta con el pythonw del venv para garantizar que TODAS las dependencias
estén disponibles (pygame, tkinterdnd2, yt-dlp, pychrome, pycaw, etc.).
"""

import sys
import os
import subprocess
from pathlib import Path

# Obtener el directorio del script (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Obtener el directorio raíz del proyecto (un nivel arriba)
PROJECT_ROOT = SCRIPT_DIR.parent

# Cambiar al directorio raíz del proyecto (para rutas relativas como data/)
os.chdir(PROJECT_ROOT)

# Garantizar la ejecución con el Python del venv. Si el intérprete actual no es
# el del venv, relanzar con él y salir (evita "falta dependencia" en el arranque).
_VENV_PYTHONW = PROJECT_ROOT / "venv" / "Scripts" / "pythonw.exe"
if _VENV_PYTHONW.exists() and Path(sys.executable).resolve() != _VENV_PYTHONW.resolve():
    subprocess.Popen([str(_VENV_PYTHONW), str(Path(__file__).resolve())])
    sys.exit(0)

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

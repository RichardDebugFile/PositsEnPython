#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crea un acceso directo a Posits Virtuales en el Escritorio de Windows.

El acceso directo apunta a start.bat (raiz del proyecto), que inicia la app
sin consola usando el entorno virtual. Requiere pywin32 y winshell (ya estan
en requirements.txt).

Uso:
    python scripts/crear_acceso_directo.py
    (o doble clic, si se ejecuta con el Python del venv)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "start.bat"


def crear_acceso_directo() -> bool:
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("[ERROR] Falta pywin32/winshell. Instala con: pip install pywin32 winshell")
        return False

    if not TARGET.exists():
        print(f"[ERROR] No se encontro {TARGET}")
        return False

    desktop = Path(winshell.desktop())
    lnk = desktop / "Posits Virtuales.lnk"

    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(lnk))
    shortcut.Targetpath = str(TARGET)
    shortcut.WorkingDirectory = str(ROOT)
    shortcut.Description = "Iniciar Posits Virtuales"

    # Icono: usar el pythonw del venv si existe (mejor que el generico del .bat)
    pythonw = ROOT / "venv" / "Scripts" / "pythonw.exe"
    if pythonw.exists():
        shortcut.IconLocation = str(pythonw)

    shortcut.save()
    print(f"[OK] Acceso directo creado en:\n  {lnk}")
    return True


if __name__ == "__main__":
    sys.exit(0 if crear_acceso_directo() else 1)

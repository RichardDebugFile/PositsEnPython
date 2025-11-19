#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para instalar/desinstalar el inicio automatico de Posits Virtuales
"""

import sys
import winreg
from pathlib import Path

APP_NAME = "PositsVirtuales"
SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_PATH = SCRIPT_DIR / "posits_launcher.pyw"
PYTHON_EXECUTABLE = sys.executable  # Ruta al python.exe del venv o sistema


def install_startup():
    """Instala la aplicacion en el inicio automatico de Windows"""
    try:
        # Abrir la clave de registro de inicio automatico
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )

        # Crear el comando de inicio
        # Usar pythonw.exe para que no muestre consola
        python_dir = Path(PYTHON_EXECUTABLE).parent
        pythonw_exe = python_dir / "pythonw.exe"

        if not pythonw_exe.exists():
            # Fallback a python.exe si pythonw.exe no existe
            pythonw_exe = PYTHON_EXECUTABLE

        command = f'"{pythonw_exe}" "{LAUNCHER_PATH}"'

        # Escribir el valor en el registro
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
        winreg.CloseKey(key)

        print(f"[OK] Inicio automatico instalado correctamente")
        print(f"  Nombre: {APP_NAME}")
        print(f"  Ejecutable: {pythonw_exe}")
        print(f"  Script: {LAUNCHER_PATH}")
        print(f"\nLa aplicacion se iniciara automaticamente al encender Windows")
        return True

    except WindowsError as e:
        print(f"[ERROR] Error al instalar inicio automatico: {e}")
        return False


def uninstall_startup():
    """Desinstala la aplicacion del inicio automatico de Windows"""
    try:
        # Abrir la clave de registro de inicio automatico
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )

        # Eliminar el valor del registro
        try:
            winreg.DeleteValue(key, APP_NAME)
            print(f"[OK] Inicio automatico desinstalado correctamente")
            return True
        except WindowsError:
            print(f"[ADVERTENCIA] No se encontro entrada de inicio automatico para {APP_NAME}")
            return False
        finally:
            winreg.CloseKey(key)

    except WindowsError as e:
        print(f"[ERROR] Error al desinstalar inicio automatico: {e}")
        return False


def check_startup():
    """Verifica si la aplicacion esta configurada para inicio automatico"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_READ
        )

        try:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            print(f"[OK] Inicio automatico esta ACTIVADO")
            print(f"  Comando: {value}")
            return True
        except WindowsError:
            print(f"[INFO] Inicio automatico NO esta activado")
            return False
        finally:
            winreg.CloseKey(key)

    except WindowsError as e:
        print(f"[ERROR] Error al verificar inicio automatico: {e}")
        return False


def main():
    """Menu principal"""
    print("=" * 60)
    print("CONFIGURACION DE INICIO AUTOMATICO - POSITS VIRTUALES")
    print("=" * 60)
    print()

    # Verificar estado actual
    print("Estado actual:")
    is_installed = check_startup()
    print()

    # Mostrar menu
    print("Opciones:")
    print("  1. Activar inicio automatico")
    print("  2. Desactivar inicio automatico")
    print("  3. Verificar estado")
    print("  4. Salir")
    print()

    try:
        choice = input("Selecciona una opcion (1-4): ").strip()
        print()

        if choice == "1":
            install_startup()
        elif choice == "2":
            uninstall_startup()
        elif choice == "3":
            check_startup()
        elif choice == "4":
            print("Saliendo...")
        else:
            print("Opcion no valida")

    except KeyboardInterrupt:
        print("\n\nCancelado por el usuario")
    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

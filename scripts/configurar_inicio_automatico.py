#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para configurar el inicio automático de Posits Virtuales en Windows
"""

import os
import sys
import winshell
from pathlib import Path
from win32com.client import Dispatch

def crear_atajo_inicio():
    """Crea un atajo en la carpeta de Inicio de Windows"""

    # Obtener rutas
    carpeta_inicio = winshell.startup()
    directorio_app = Path(__file__).parent.absolute()
    archivo_bat = directorio_app / "iniciar_posits.bat"

    print(f"📁 Directorio de la app: {directorio_app}")
    print(f"📁 Carpeta de Inicio: {carpeta_inicio}")
    print(f"📄 Archivo BAT: {archivo_bat}")

    # Verificar que existe el archivo BAT
    if not archivo_bat.exists():
        print(f"❌ ERROR: No se encontró el archivo {archivo_bat}")
        print("   Por favor, asegúrate de que el archivo 'iniciar_posits.bat' existe.")
        return False

    # Crear el atajo
    ruta_atajo = os.path.join(carpeta_inicio, "Posits Virtuales.lnk")

    try:
        shell = Dispatch('WScript.Shell')
        atajo = shell.CreateShortCut(ruta_atajo)
        atajo.TargetPath = str(archivo_bat)
        atajo.WorkingDirectory = str(directorio_app)
        atajo.IconLocation = str(archivo_bat)
        atajo.Description = "Posits Virtuales - Gestión de Tareas con Gamificación"
        atajo.save()

        print(f"\n✅ ¡Atajo creado exitosamente!")
        print(f"   Ubicación: {ruta_atajo}")
        print(f"\n🎉 La aplicación se iniciará automáticamente al arrancar Windows.")
        print(f"\n💡 Consejos:")
        print(f"   - Para desactivar, elimina el atajo de: {carpeta_inicio}")
        print(f"   - Para probarlo, reinicia tu PC")

        return True

    except Exception as e:
        print(f"\n❌ ERROR al crear el atajo: {e}")
        return False

def eliminar_atajo_inicio():
    """Elimina el atajo de la carpeta de Inicio"""
    carpeta_inicio = winshell.startup()
    ruta_atajo = os.path.join(carpeta_inicio, "Posits Virtuales.lnk")

    if os.path.exists(ruta_atajo):
        try:
            os.remove(ruta_atajo)
            print(f"✅ Atajo eliminado exitosamente de: {ruta_atajo}")
            print(f"   La aplicación ya NO se iniciará automáticamente.")
            return True
        except Exception as e:
            print(f"❌ ERROR al eliminar el atajo: {e}")
            return False
    else:
        print(f"ℹ️  No se encontró ningún atajo en: {ruta_atajo}")
        return False

def main():
    """Función principal"""
    print("=" * 70)
    print("   CONFIGURADOR DE INICIO AUTOMÁTICO - POSITS VIRTUALES")
    print("=" * 70)
    print()

    print("¿Qué deseas hacer?")
    print("1. Activar inicio automático")
    print("2. Desactivar inicio automático")
    print("3. Salir")
    print()

    opcion = input("Selecciona una opción (1-3): ").strip()

    if opcion == "1":
        print("\n🔧 Configurando inicio automático...")
        crear_atajo_inicio()
    elif opcion == "2":
        print("\n🔧 Desactivando inicio automático...")
        eliminar_atajo_inicio()
    elif opcion == "3":
        print("👋 ¡Hasta luego!")
    else:
        print("❌ Opción inválida")

    print()
    input("Presiona ENTER para salir...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        input("Presiona ENTER para salir...")

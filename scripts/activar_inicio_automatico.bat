@echo off
REM Script para activar el inicio automático de Posits Virtuales
echo ============================================================
echo ACTIVAR INICIO AUTOMATICO - POSITS VIRTUALES
echo ============================================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Ejecutar el script de Python con el entorno virtual
if exist "..\venv\Scripts\python.exe" (
    echo Usando Python del entorno virtual...
    ..\venv\Scripts\python.exe install_startup.py
) else (
    echo Usando Python del sistema...
    python install_startup.py
)

echo.
echo Presiona cualquier tecla para salir...
pause >nul

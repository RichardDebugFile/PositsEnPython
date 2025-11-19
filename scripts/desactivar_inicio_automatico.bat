@echo off
REM Script para desactivar el inicio automático de Posits Virtuales
echo ============================================================
echo DESACTIVAR INICIO AUTOMATICO - POSITS VIRTUALES
echo ============================================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Ejecutar el script de Python para desinstalar
if exist "..\venv\Scripts\python.exe" (
    echo Usando Python del entorno virtual...
    ..\venv\Scripts\python.exe -c "import install_startup; install_startup.uninstall_startup()"
) else (
    echo Usando Python del sistema...
    python -c "import install_startup; install_startup.uninstall_startup()"
)

echo.
echo Presiona cualquier tecla para salir...
pause >nul

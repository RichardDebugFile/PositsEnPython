@echo off
REM ============================================================
REM  Posits Virtuales - Inicio manual rapido (doble clic)
REM ============================================================
REM  Lanza la app sin consola usando el launcher, que a su vez
REM  usa el Python del entorno virtual (venv) con las dependencias.
REM  Para el inicio automatico con Windows usa:
REM    scripts\activar_inicio_automatico.bat
REM ============================================================

cd /d "%~dp0"

if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "scripts\posits_launcher.pyw"
) else (
    where pythonw.exe >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No se encontro Python. Instala Python o crea el entorno virtual.
        pause
    ) else (
        start "" pythonw.exe "scripts\posits_launcher.pyw"
    )
)

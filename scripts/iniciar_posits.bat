@echo off
REM ============================================================
REM LAUNCHER MANUAL DE POSITS VIRTUALES
REM ============================================================
REM
REM IMPORTANTE: Este es un launcher MANUAL
REM
REM Para configurar el inicio automatico usa:
REM   - activar_inicio_automatico.bat (para activar)
REM   - desactivar_inicio_automatico.bat (para desactivar)
REM
REM Este script SOLO inicia la aplicacion manualmente cuando
REM quieras abrirla (doble clic).
REM ============================================================

echo.
echo ============================================================
echo   INICIANDO POSITS VIRTUALES
echo ============================================================
echo.

REM Cambiar al directorio de la aplicacion (raiz del proyecto)
cd /d "%~dp0\.."

REM Verificar si existe el entorno virtual
if exist "venv\Scripts\pythonw.exe" (
    echo [OK] Usando Python del entorno virtual
    echo.
    echo Iniciando aplicacion en segundo plano...
    start "" venv\Scripts\pythonw.exe scripts\posits_launcher.pyw
) else (
    echo [ADVERTENCIA] No se encontro el entorno virtual
    echo Intentando usar Python del sistema...
    echo.

    REM Intentar con pythonw.exe del sistema
    where pythonw.exe >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Python encontrado en el sistema
        start "" pythonw.exe scripts\posits_launcher.pyw
    ) else (
        echo [ERROR] No se encontro Python en el sistema
        echo.
        echo Por favor, instala Python o ejecuta:
        echo   python main.py
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [OK] La aplicacion se esta iniciando en segundo plano...
echo.
echo Puedes cerrar esta ventana.
echo.

REM Esperar 3 segundos para que el usuario vea el mensaje
timeout /t 3 >nul

REM Si prefieres ver la consola de debug, usa esto en su lugar:
REM start "" venv\Scripts\python.exe main.py

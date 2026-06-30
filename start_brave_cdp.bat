@echo off
chcp 65001 >nul 2>&1

title Iniciar Brave con CDP

echo.
echo ============================================================
echo      INICIAR BRAVE BROWSER CON CDP (Puerto 9222)
echo ============================================================
echo.

:: ============================================================
:: PASO 1: Buscar instalacion de Brave
:: ============================================================

echo [1] Buscando instalacion de Brave Browser...
echo.

call :buscar_brave
if "%BRAVE_EXE%"=="" goto :no_encontrado

echo     OK - Ruta encontrada:
echo     %BRAVE_EXE%
echo.
goto :verificar_procesos

:no_encontrado
echo     ERROR: No se encontro Brave Browser.
echo.
echo     Ubicaciones buscadas:
echo     - C:\Program Files\BraveSoftware\Brave-Browser\Application\
echo     - C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\
echo     - %LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\
echo     - %USERPROFILE%\AppData\Local\BraveSoftware\Brave-Browser\Application\
echo     - %USERPROFILE%\Brave-Browser\Application\
echo.
echo     Instala Brave desde: https://brave.com/download/
echo.
pause
exit /b 1

:: ============================================================
:: PASO 2: Verificar procesos
:: ============================================================

:verificar_procesos
echo [2] Verificando si Brave esta en ejecucion...
echo.

tasklist /FI "IMAGENAME eq brave.exe" 2>nul | find /I "brave.exe" >nul
if errorlevel 1 goto :sin_procesos

echo     AVISO: Brave esta en ejecucion actualmente.
echo.
echo     Procesos activos:
echo     ----------------------------------------
tasklist /FI "IMAGENAME eq brave.exe" /FO TABLE /NH 2>nul
echo     ----------------------------------------
echo.
echo     Para usar CDP, Brave debe reiniciarse con flags especiales.
echo.
echo     Opciones:
echo       [1] Cerrar Brave automaticamente (recomendado)
echo       [2] Voy a cerrar Brave yo mismo
echo       [3] Cancelar
echo.

choice /C 123 /N /M "    Elige una opcion (1, 2 o 3): "

if errorlevel 3 goto :cancelado
if errorlevel 2 goto :cierre_manual
goto :cierre_auto

:cierre_auto
echo.
echo     Cerrando Brave automaticamente...
taskkill /IM brave.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /F /IM brave.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul
echo     Brave cerrado.
echo.
goto :iniciar_brave

:cierre_manual
echo.
echo     ============================================================
echo      INSTRUCCIONES PARA CERRAR BRAVE MANUALMENTE:
echo     ============================================================
echo      1. Cierra todas las ventanas de Brave
echo      2. Administrador de Tareas (Ctrl+Shift+Esc)
echo      3. Busca "brave.exe" y finaliza TODOS los procesos
echo      4. Presiona una tecla aqui para continuar
echo     ============================================================
echo.
pause

tasklist /FI "IMAGENAME eq brave.exe" 2>nul | find /I "brave.exe" >nul
if not errorlevel 1 (
    echo     ERROR: Brave sigue activo. Cierralo e intenta de nuevo.
    echo.
    pause
    exit /b 1
)
echo     Brave cerrado.
echo.
goto :iniciar_brave

:cancelado
echo.
echo     Operacion cancelada.
echo.
pause
exit /b 0

:sin_procesos
echo     Brave no esta en ejecucion. Continuando...
echo.

:: ============================================================
:: PASO 3: Iniciar Brave con CDP
:: ============================================================

:iniciar_brave
echo [3] Iniciando Brave con CDP en puerto 9222...
echo.
echo     Ejecutable: %BRAVE_EXE%
echo     Puerto CDP: 9222
echo.

start "" "%BRAVE_EXE%" --remote-debugging-port=9222

echo     Esperando que Brave inicie...
timeout /t 3 /nobreak >nul

tasklist /FI "IMAGENAME eq brave.exe" 2>nul | find /I "brave.exe" >nul
if errorlevel 1 (
    echo     ERROR: Brave no inicio correctamente.
    echo.
    pause
    exit /b 1
)
echo     Brave iniciado correctamente.
echo.

:: ============================================================
:: PASO 4: Verificar CDP
:: ============================================================

echo [4] Verificando conexion CDP...
echo.

powershell -NoProfile -Command "try { $null = Invoke-WebRequest 'http://127.0.0.1:9222/json' -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop; Write-Host '    CDP respondiendo correctamente en http://127.0.0.1:9222' } catch { Write-Host '    AVISO: CDP aun no responde, espera unos segundos.' }"

echo.
echo ============================================================
echo              BRAVE CON CDP LISTO
echo ============================================================
echo.
echo  Proximos pasos:
echo    1. Abre YouTube en Brave y reproduce un video
echo    2. Usa la funcion Capturar en la aplicacion
echo    3. Para verificar: python test_cdp.py
echo.
echo ============================================================
echo.
pause
exit /b 0

:: ============================================================
:: Subrutina: Buscar Brave
:: ============================================================

:buscar_brave
set "BRAVE_EXE="

if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" (
    set "BRAVE_EXE=C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    exit /b 0
)
if exist "C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe" (
    set "BRAVE_EXE=C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe"
    exit /b 0
)
if exist "%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe" (
    set "BRAVE_EXE=%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"
    exit /b 0
)
if exist "%USERPROFILE%\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe" (
    set "BRAVE_EXE=%USERPROFILE%\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"
    exit /b 0
)
if exist "%USERPROFILE%\Brave-Browser\Application\brave.exe" (
    set "BRAVE_EXE=%USERPROFILE%\Brave-Browser\Application\brave.exe"
    exit /b 0
)
exit /b 1

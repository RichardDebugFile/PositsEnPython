@echo off
chcp 65001 >nul 2>&1

title Iniciar Chrome con CDP

echo.
echo ============================================================
echo      INICIAR GOOGLE CHROME CON CDP (Puerto 9222)
echo ============================================================
echo.

:: ============================================================
:: PASO 1: Buscar instalacion de Chrome
:: ============================================================

echo [1] Buscando instalacion de Google Chrome...
echo.

:: Buscar en cada ruta posible usando goto para salir del bloque
call :buscar_chrome
if "%CHROME_EXE%"=="" goto :no_encontrado

echo     OK - Ruta encontrada:
echo     %CHROME_EXE%
echo.
goto :verificar_procesos

:no_encontrado
echo     ERROR: No se encontro Google Chrome.
echo.
echo     Ubicaciones buscadas:
echo     - C:\Program Files\Google\Chrome\Application\
echo     - C:\Program Files (x86)\Google\Chrome\Application\
echo     - %LOCALAPPDATA%\Google\Chrome\Application\
echo     - %USERPROFILE%\AppData\Local\Google\Chrome\Application\
echo.
echo     Instala Chrome desde: https://www.google.com/chrome/
echo.
pause
exit /b 1

:: ============================================================
:: PASO 2: Verificar procesos
:: ============================================================

:verificar_procesos
echo [2] Verificando si Chrome esta en ejecucion...
echo.

tasklist /FI "IMAGENAME eq chrome.exe" 2>nul | find /I "chrome.exe" >nul
if errorlevel 1 goto :sin_procesos

echo     AVISO: Chrome esta en ejecucion actualmente.
echo.
echo     Procesos activos:
echo     ----------------------------------------
tasklist /FI "IMAGENAME eq chrome.exe" /FO TABLE /NH 2>nul
echo     ----------------------------------------
echo.
echo     Para usar CDP, Chrome debe reiniciarse con flags especiales.
echo.
echo     Opciones:
echo       [1] Cerrar Chrome automaticamente (recomendado)
echo       [2] Voy a cerrar Chrome yo mismo
echo       [3] Cancelar
echo.

choice /C 123 /N /M "    Elige una opcion (1, 2 o 3): "

if errorlevel 3 goto :cancelado
if errorlevel 2 goto :cierre_manual
goto :cierre_auto

:cierre_auto
echo.
echo     Cerrando Chrome automaticamente...
taskkill /IM chrome.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /F /IM chrome.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul
echo     Chrome cerrado.
echo.
goto :iniciar_chrome

:cierre_manual
echo.
echo     ============================================================
echo      INSTRUCCIONES PARA CERRAR CHROME MANUALMENTE:
echo     ============================================================
echo      1. Cierra todas las ventanas de Chrome
echo      2. Administrador de Tareas (Ctrl+Shift+Esc)
echo      3. Busca "chrome.exe" y finaliza TODOS los procesos
echo      4. Presiona una tecla aqui para continuar
echo     ============================================================
echo.
pause

tasklist /FI "IMAGENAME eq chrome.exe" 2>nul | find /I "chrome.exe" >nul
if not errorlevel 1 (
    echo     ERROR: Chrome sigue activo. Cierralo e intenta de nuevo.
    echo.
    pause
    exit /b 1
)
echo     Chrome cerrado.
echo.
goto :iniciar_chrome

:cancelado
echo.
echo     Operacion cancelada.
echo.
pause
exit /b 0

:sin_procesos
echo     Chrome no esta en ejecucion. Continuando...
echo.

:: ============================================================
:: PASO 3: Iniciar Chrome con CDP
:: ============================================================

:iniciar_chrome
echo [3] Iniciando Chrome con CDP en puerto 9222...
echo.
echo     Ejecutable: %CHROME_EXE%
echo     Puerto CDP: 9222
echo.

start "" "%CHROME_EXE%" --remote-debugging-port=9222

echo     Esperando que Chrome inicie...
timeout /t 3 /nobreak >nul

tasklist /FI "IMAGENAME eq chrome.exe" 2>nul | find /I "chrome.exe" >nul
if errorlevel 1 (
    echo     ERROR: Chrome no inicio correctamente.
    echo.
    pause
    exit /b 1
)
echo     Chrome iniciado correctamente.
echo.

:: ============================================================
:: PASO 4: Verificar CDP
:: ============================================================

echo [4] Verificando conexion CDP...
echo.

powershell -NoProfile -Command "try { $null = Invoke-WebRequest 'http://127.0.0.1:9222/json' -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop; Write-Host '    CDP respondiendo correctamente en http://127.0.0.1:9222' } catch { Write-Host '    AVISO: CDP aun no responde, espera unos segundos.' }"

echo.
echo ============================================================
echo              CHROME CON CDP LISTO
echo ============================================================
echo.
echo  Proximos pasos:
echo    1. Abre YouTube en Chrome y reproduce un video
echo    2. Usa la funcion Capturar en la aplicacion
echo    3. Para verificar: python scripts\cdp\test_cdp.py
echo.
echo ============================================================
echo.
pause
exit /b 0

:: ============================================================
:: Subrutina: Buscar Chrome
:: ============================================================

:buscar_chrome
set "CHROME_EXE="

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=C:\Program Files\Google\Chrome\Application\chrome.exe"
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    exit /b 0
)
if exist "%USERPROFILE%\AppData\Local\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%USERPROFILE%\AppData\Local\Google\Chrome\Application\chrome.exe"
    exit /b 0
)
exit /b 1

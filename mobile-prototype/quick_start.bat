@echo off
echo ========================================
echo   Posits Mobile - Quick Start
echo ========================================
echo.

REM Usar el venv del proyecto principal (un nivel arriba)
if exist "..\venv\Scripts\python.exe" (
    echo [1/3] Usando entorno virtual del proyecto principal...
    set PYTHON_EXE=..\venv\Scripts\python.exe
    set PIP_EXE=..\venv\Scripts\pip.exe
) else (
    echo [1/3] Usando Python del sistema...
    set PYTHON_EXE=python
    set PIP_EXE=pip
)
echo.

REM Instalar dependencias
echo [2/3] Instalando dependencias de Kivy...
echo (Esto puede tardar varios minutos la primera vez)
echo.
echo Instalando Kivy desde repositorio pre-compilado...
%PIP_EXE% install kivy[base] kivymd --pre --extra-index-url https://kivy.org/downloads/simple/
echo.
echo Instalando otras dependencias...
%PIP_EXE% install pillow requests python-dateutil
echo.

REM Ejecutar app
echo [3/3] Iniciando Posits Mobile...
echo.
echo NOTA: Se abrira una ventana simulando un telefono movil (360x640)
echo.
%PYTHON_EXE% main.py

pause

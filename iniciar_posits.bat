@echo off
REM Script para iniciar Posits Virtuales al arranque de Windows
REM Cambia al directorio de la aplicación
cd /d "%~dp0"

REM Inicia la aplicación en segundo plano
start "" pythonw main.py

REM Si prefieres ver la consola de debug, usa esto en su lugar:
REM start "" python main.py

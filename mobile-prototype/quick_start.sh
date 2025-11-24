#!/bin/bash

echo "========================================"
echo "  Posits Mobile - Quick Start"
echo "========================================"
echo ""

# Verificar si existe venv
if [ ! -d "venv" ]; then
    echo "[1/4] Creando entorno virtual..."
    python3 -m venv venv
    echo ""
else
    echo "[1/4] Entorno virtual ya existe"
    echo ""
fi

# Activar venv
echo "[2/4] Activando entorno virtual..."
source venv/bin/activate
echo ""

# Instalar dependencias
echo "[3/4] Instalando dependencias..."
echo "(Esto puede tardar varios minutos la primera vez)"
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Ejecutar app
echo "[4/4] Iniciando Posits Mobile..."
echo ""
python main.py

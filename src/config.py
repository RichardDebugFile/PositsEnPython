#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración global de la aplicación
Centraliza constantes, colores, rutas y variables de entorno
"""

import os
from pathlib import Path

# ---------------------------- Información de la app ----------------------------
APP_NAME = "✨ Posits Virtuales"
APP_VERSION = "2.0.0"

# ---------------------------- Rutas ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("STICKYTASKS_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "notes.json"
LOG_FILE = DATA_DIR / "app.log"
ATTACHMENTS_DIR = DATA_DIR / "attachments"
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
GAMIFICATION_FILE = DATA_DIR / "gamification.json"

# Archivo legacy para migración
LOCAL_DATA_FILE = PROJECT_ROOT / "tasks.json"

# ---------------------------- Ollama (IA) ----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

# ---------------------------- Colores Modernos ----------------------------
MODERN_COLORS = {
    "Primary": "#2196F3",
    "Secondary": "#FF9800",
    "Success": "#4CAF50",
    "Warning": "#FFC107",
    "Danger": "#F44336",
    "Light": "#F5F5F5",
    "Dark": "#212121",
    "White": "#FFFFFF",
    "Background": "#F8F9FA",
    "Card": "#FFFFFF",
    "Text": "#2C3E50",
    "TextLight": "#95A5A6",
}

GRADIENTS = {
    "Primary": ["#2196F3", "#1976D2"],
    "Secondary": ["#FF9800", "#F57C00"],
    "Success": ["#4CAF50", "#388E3C"],
    "Warning": ["#FFC107", "#FFA000"],
    "Danger": ["#F44336", "#D32F2F"],
    "Card": ["#FFFFFF", "#F5F5F5"],
}

VIBRANT_COLORS = {
    "Sunshine": "#FFD54F",   # Amarillo
    "Ocean":    "#64B5F6",   # Azul
    "Nature":   "#81C784",   # Verde
    "Sunset":   "#FF8A65",   # Naranja
    "Lavender": "#BA68C8",   # Morado
    "Coral":    "#FF7043",   # Coral
}

COLOR_LABELS = {
    "Sunshine": "Amarillo",
    "Ocean": "Azul",
    "Nature": "Verde",
    "Sunset": "Naranja",
    "Lavender": "Morado",
    "Coral": "Coral",
}

LABEL_TO_KEY = {v: k for k, v in COLOR_LABELS.items()}

# Sinónimos en español para normalización de colores
SPANISH_TO_KEY = {
    "amarillo": "Sunshine",
    "azul": "Ocean",
    "verde": "Nature",
    "naranja": "Sunset",
    "morado": "Lavender",
    "violeta": "Lavender",
    "lila": "Lavender",
    "coral": "Coral",
    "rojo": "Coral",
    "anaranjado": "Sunset",
}

# ---------------------------- Formato de fechas ----------------------------
ISO_FMT = "%Y-%m-%d"

# ---------------------------- Gamificación ----------------------------
GAMIFICATION_CONFIG = {
    "points_per_task": 10,
    "points_per_priority_task": 25,
    "points_per_daily_mission": 50,
    "daily_missions_required": 3,
    "base_points_per_level": 150,  # Nivel 1→2 requiere 150 puntos
    "level_multiplier": 1.3,  # Cada nivel requiere 30% más que el anterior
    "streak_bonus_multiplier": 1.5,  # Bono por racha de días consecutivos
}

# ---------------------------- Notificaciones ----------------------------
NOTIFICATION_CONFIG = {
    "check_interval_minutes": 30,  # Cada cuánto revisar tareas próximas
    "days_before_alert": [0, 1, 3],  # Alertar tareas que vencen hoy, mañana, en 3 días
    "enable_sound": True,
    "enable_system_notifications": True,
}

# ---------------------------- Adjuntos ----------------------------
ALLOWED_ATTACHMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".xlsx", ".xls", ".csv",
    ".zip", ".rar", ".7z",
}

MAX_ATTACHMENT_SIZE_MB = 50

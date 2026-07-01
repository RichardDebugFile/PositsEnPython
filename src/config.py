#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración global de la aplicación
Centraliza constantes, colores, rutas y variables de entorno
"""

import os
import json
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

# ---------------------------- Temas (claro / oscuro) ----------------------------
# Los widgets leen MODERN_COLORS/GRADIENTS al construirse. Para cambiar de tema
# en vivo NO se rebindean estos nombres (otros módulos ya importaron el objeto):
# se MUTAN en el sitio con apply_theme_dicts() y se reconstruye la UI.

_LIGHT_MODERN = {
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

_LIGHT_GRADIENTS = {
    "Primary": ["#2196F3", "#1976D2"],
    "Secondary": ["#FF9800", "#F57C00"],
    "Success": ["#4CAF50", "#388E3C"],
    "Warning": ["#FFC107", "#FFA000"],
    "Danger": ["#F44336", "#D32F2F"],
    "Card": ["#FFFFFF", "#F5F5F5"],
}

_DARK_MODERN = {
    "Primary": "#42A5F5",
    "Secondary": "#FFB74D",
    "Success": "#66BB6A",
    "Warning": "#FFCA28",
    "Danger": "#EF5350",
    "Light": "#2B2F36",       # superficie secundaria (listboxes, drop zones)
    "Dark": "#0D0D0D",        # casi negro (texto sobre posit de color, tooltips)
    "White": "#20242B",       # superficie de inputs / elevada
    "Background": "#15171C",  # fondo de la app
    "Card": "#1E222A",        # tarjetas / paneles
    "Text": "#E8EAED",        # texto principal claro
    "TextLight": "#9AA0A6",   # texto secundario
}

_DARK_GRADIENTS = {
    "Primary": ["#1E88E5", "#1565C0"],
    "Secondary": ["#FB8C00", "#EF6C00"],
    "Success": ["#43A047", "#2E7D32"],
    "Warning": ["#F9A825", "#F57F17"],
    "Danger": ["#E53935", "#C62828"],
    "Card": ["#1E222A", "#15171C"],
}

_THEMES = {
    "light": (_LIGHT_MODERN, _LIGHT_GRADIENTS),
    "dark": (_DARK_MODERN, _DARK_GRADIENTS),
}

# Dicts activos que consume TODA la app (se rellenan más abajo con el tema guardado).
MODERN_COLORS = {}
GRADIENTS = {}

SETTINGS_FILE = DATA_DIR / "app_settings.json"


def load_settings() -> dict:
    """Lee las preferencias de la app (tema, notificaciones, etc.)."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_settings(data: dict):
    """Guarda las preferencias de la app."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def apply_theme_dicts(name: str):
    """Muta MODERN_COLORS/GRADIENTS en el sitio con la paleta del tema dado."""
    modern, grad = _THEMES.get(name, _THEMES["light"])
    MODERN_COLORS.clear()
    MODERN_COLORS.update(modern)
    GRADIENTS.clear()
    GRADIENTS.update(grad)


def get_theme() -> str:
    """Nombre del tema activo ('light' | 'dark')."""
    return CURRENT_THEME


def set_theme(name: str) -> str:
    """Activa un tema y lo persiste. Devuelve el nombre aplicado."""
    global CURRENT_THEME
    CURRENT_THEME = name if name in _THEMES else "light"
    apply_theme_dicts(CURRENT_THEME)
    settings = load_settings()
    settings["theme"] = CURRENT_THEME
    save_settings(settings)
    return CURRENT_THEME


# Cargar el tema guardado al importar (por defecto claro) y poblar los dicts.
_SETTINGS = load_settings()
CURRENT_THEME = _SETTINGS.get("theme", "light")
if CURRENT_THEME not in _THEMES:
    CURRENT_THEME = "light"
apply_theme_dicts(CURRENT_THEME)

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

# ---------------------------- Prioridades ----------------------------
PRIORITY_LEVELS = {
    "urgent": {
        "emoji": "🔴",
        "label": "Urgente",
        "xp": 50,
        "color": "#FF4444"
    },
    "high": {
        "emoji": "🟠",
        "label": "Alta",
        "xp": 35,
        "color": "#FF8C00"
    },
    "medium": {
        "emoji": "🟡",
        "label": "Media",
        "xp": 25,
        "color": "#FFD700"
    },
    "low": {
        "emoji": "🟢",
        "label": "Baja",
        "xp": 15,
        "color": "#4CAF50"
    }
}

# ---------------------------- Gamificación ----------------------------
GAMIFICATION_CONFIG = {
    "points_per_task": 10,
    "points_per_priority_task": 25,  # Valor base, se usa PRIORITY_LEVELS para el cálculo real
    "points_per_daily_mission": 50,
    "daily_missions_required": 3,
    "base_points_per_level": 150,  # Nivel 1→2 requiere 150 puntos
    "level_multiplier": 1.3,  # Cada nivel requiere 30% más que el anterior
    "streak_bonus_multiplier": 1.5,  # Bono por racha de días consecutivos
}

# ---------------------------- Pomodoro ----------------------------
POMODORO_CONFIG = {
    "work_duration": 25,  # minutos
    "break_duration": 5,  # minutos
    "long_break_duration": 15,  # minutos
    "sessions_per_cycle": 4,
    "auto_start_break": False,
    "auto_start_work": False,
    "notifications_enabled": True,
    "sound_enabled": True,
}

# ---------------------------- Notificaciones ----------------------------
NOTIFICATION_CONFIG = {
    "check_interval_minutes": 30,  # Cada cuánto revisar tareas próximas
    "days_before_alert": [0, 1, 3],  # Alertar tareas que vencen hoy, mañana, en 3 días
    "enable_sound": True,
    "enable_system_notifications": True,
}

# Aplicar preferencias guardadas que afectan a config de módulos
if "notifications_enabled" in _SETTINGS:
    NOTIFICATION_CONFIG["enable_system_notifications"] = bool(_SETTINGS["notifications_enabled"])

# ---------------------------- Adjuntos ----------------------------
ALLOWED_ATTACHMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".xlsx", ".xls", ".csv",
    ".zip", ".rar", ".7z",
}

MAX_ATTACHMENT_SIZE_MB = 50

# ---------------------------- Calendario ----------------------------
CALENDAR_CONFIG = {
    # Colores para el heatmap de productividad (estilo GitHub)
    "heatmap_colors": [
        "#ebedf0",  # Nivel 0: Sin actividad (gris claro)
        "#9be9a8",  # Nivel 1: Baja productividad (verde claro)
        "#40c463",  # Nivel 2: Media productividad (verde)
        "#30a14e",  # Nivel 3: Alta productividad (verde oscuro)
        "#216e39",  # Nivel 4: Muy alta productividad (verde intenso)
    ],
    # Configuración del mini widget
    "show_mini_calendar": True,
    "mini_calendar_position": "bottom-right",  # bottom-right, bottom-left, top-right, top-left
    # Nombres de meses en español
    "month_names": [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ],
    # Nombres de días en español
    "day_names": ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
    "day_names_short": ["L", "M", "X", "J", "V", "S", "D"],
}

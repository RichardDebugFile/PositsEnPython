#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de integración con Ollama (IA local)
"""

import json
import time
import base64
import shutil
import subprocess
import urllib.request
import urllib.error

from ..config import OLLAMA_URL, OLLAMA_MODEL, VIBRANT_COLORS, COLOR_LABELS
from ..utils.dates import fmt_date, today_date, valid_date
from ..utils.colors import normalize_color
from ..utils.logger import logger, log_exception


def ollama_chat_json(
    prompt: str,
    model: str = OLLAMA_MODEL,
    url_base: str = OLLAMA_URL,
    images: list[str] | None = None
) -> dict | None:
    """
    Realiza una consulta a Ollama y retorna la respuesta como JSON.

    Args:
        prompt: Prompt del usuario
        model: Modelo a usar
        url_base: URL base de Ollama
        images: Lista de imágenes codificadas en base64

    Returns:
        dict con la respuesta parseada o None si hubo error
    """
    user_msg = {
        "role": "user",
        "content": prompt
    }

    if images:
        user_msg["images"] = images

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un analista que EXTRAERÁ campos para crear una tarea. "
                    "Responde EXCLUSIVAMENTE con un objeto JSON válido (sin texto adicional). "
                    "Claves: title (string), desc (string), due (string YYYY-MM-DD o vacío), "
                    "color (string en español o clave), priority (boolean). "
                    "Si no encuentras un título claro, deja title como cadena vacía."
                )
            },
            user_msg
        ]
    }

    logger.debug(f"Ollama.request: model={model} url={url_base}")
    logger.debug(f"Ollama.prompt: {prompt}")

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url_base + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            logger.debug(f"Ollama.raw = {raw_text}")
            raw = json.loads(raw_text)
    except urllib.error.URLError as e:
        # Sin UI aquí: el servicio puede llamarse desde un hilo. El diálogo
        # decide cómo informar el error en el hilo principal.
        log_exception(logger, "Ollama.URLError", e)
        return None
    except Exception as e:
        log_exception(logger, "Ollama.Error", e)
        return None

    content = (raw.get("message") or {}).get("content", "")
    logger.debug(f"Ollama.content = {content}")

    # Intentar parsear JSON limpio
    try:
        obj = json.loads(content)
        logger.debug(f"Ollama.json = {obj}")
        return obj
    except Exception:
        obj = _extract_first_json_object(content)
        if obj is not None:
            logger.debug(f"Ollama.json.recovered = {obj}")
            return obj

    logger.warning("La respuesta de Ollama no fue un JSON válido")
    return None


def _extract_first_json_object(text: str) -> dict | None:
    """Extrae el primer objeto JSON bien balanceado encontrado en el texto"""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i+1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


def build_extraction_prompt(natural_text: str, images_present: bool = False) -> str:
    """Construye el prompt para extraer campos de una tarea desde lenguaje natural"""
    allowed_colors = ", ".join(sorted(set(list(VIBRANT_COLORS.keys()) + list(COLOR_LABELS.values()))))
    today = fmt_date(today_date())

    extra = (
        "\nSi se adjuntaron imágenes: léelas/transcribe su texto o interpreta lo que hay en la imagen, "
        "usa esa información como entrada y luego "
        "devuélveme SOLO un JSON con los campos abajo.\n"
        if images_present else ""
    )

    return (
        f"Hoy es {today}. {extra}"
        f"Devuélveme SOLO un JSON con:\n"
        f'  title   : string (si el usuario no da título, inventa uno breve y claro)\n'
        f'  desc    : string (puede estar vacío)\n'
        f'  due     : YYYY-MM-DD (si no hay fecha, vacío)\n'
        f'  color   : uno de [{allowed_colors}] (vacío si no menciona color)\n'
        f'  priority: boolean (true si detectas urgente / prioridad / importante)\n\n'
        f"Texto usuario:\n{natural_text}\n"
    )


def extract_task_with_ollama(
    natural_text: str,
    image_paths: list[str] | None = None
) -> dict | None:
    """
    Extrae campos de tarea desde lenguaje natural usando Ollama.

    Returns:
        dict con keys: title, desc, due, color, priority
    """
    images_b64 = [_encode_image_base64(p) for p in (image_paths or [])]

    # Si no hay texto pero hay imágenes, pedir descripción primero
    if not natural_text.strip() and images_b64:
        desc_prompt = "Describe con detalle el contenido de la(s) imagen(es)."
        desc_raw = ollama_chat_json(desc_prompt, images=images_b64)
        if isinstance(desc_raw, dict):
            natural_text = json.dumps(desc_raw, ensure_ascii=False)
        else:
            natural_text = str(desc_raw)

    # Construir prompt de extracción
    prompt = build_extraction_prompt(natural_text, images_present=bool(images_b64))
    obj = ollama_chat_json(prompt, images=images_b64)

    if obj is None:
        return None

    title = (obj.get("title") or "").strip()
    desc = (obj.get("desc") or "").strip()
    due = (obj.get("due") or "").strip()
    color_raw = (obj.get("color") or "").strip()
    priority = bool(obj.get("priority", False))

    if not title:
        title = (desc[:40] + "...") if desc else "(Sin título)"

    # Normalizar color
    color_key = normalize_color(color_raw)

    # Validar fecha
    if due and not valid_date(due):
        due = ""

    return {
        "title": title,
        "desc": desc,
        "due": due,
        "color": color_key,
        "priority": priority
    }


def _encode_image_base64(path: str) -> str:
    """Codifica una imagen en base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# --------------------------- Disponibilidad de Ollama ---------------------------

def is_ollama_running(url_base: str = OLLAMA_URL, timeout: float = 2.0) -> bool:
    """Devuelve True si el servidor de Ollama responde en ``url_base``."""
    try:
        with urllib.request.urlopen(url_base + "/api/tags", timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return 200 <= status < 300
    except Exception:
        return False


def start_ollama_server() -> bool:
    """
    Lanza ``ollama serve`` en segundo plano si el ejecutable está disponible.
    No espera a que esté listo (usar :func:`ensure_ollama_running`).
    """
    exe = shutil.which("ollama")
    if not exe:
        logger.warning("No se encontró el ejecutable 'ollama' en el PATH")
        return False
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        logger.info("Lanzado 'ollama serve' en segundo plano")
        return True
    except Exception as e:
        log_exception(logger, "Ollama.start", e)
        return False


def ensure_ollama_running(
    url_base: str = OLLAMA_URL,
    on_status=None,
    total_wait: float = 20.0,
) -> bool:
    """
    Garantiza que Ollama esté disponible: si no responde, intenta arrancarlo y
    espera hasta que conteste (o se agoten ``total_wait`` segundos).

    IMPORTANTE: hace ``time.sleep``; debe llamarse desde un hilo, NO desde la UI.

    Args:
        on_status: callback opcional ``on_status(str)`` para reportar progreso.
    """
    if is_ollama_running(url_base):
        return True

    if on_status:
        on_status("Iniciando Ollama…")

    if not start_ollama_server():
        return False

    deadline = time.monotonic() + total_wait
    while time.monotonic() < deadline:
        if is_ollama_running(url_base):
            return True
        time.sleep(0.5)

    return False

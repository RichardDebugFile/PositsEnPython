#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Posits Virtuales + IA local (Ollama)
- Lista de tareas (cards) con franja de color, prioridad, fecha y filtro por color
- Editor de notas: solo la caja de texto toma el color del posit
- Overlay "posit rápido" al doble clic del título (sin barra del SO)
- Botón "IA (Ollama)" para dictar/pegar una orden en lenguaje natural y crear la tarea
  * Si falta título -> notifica y abre diálogo "Nueva Tarea" con datos pre-rellenados (salvo título)
- Persistencia en ./data/notes.json
"""

import json
import os
import uuid
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime, timedelta, date
import urllib.request
import urllib.error
import logging, traceback
import base64
from tkinter import filedialog

# ---------------------------- Config de datos ----------------------------
APP_NAME = "✨ Posits Virtuales"
LOCAL_DATA_FILE = Path(__file__).with_name("tasks.json")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("STICKYTASKS_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "notes.json"

# ---------- Logging ----------
LOG_FILE = DATA_DIR / "app.log"
logger = logging.getLogger("posits")
logger.setLevel(logging.DEBUG)  # cámbialo a INFO si quieres menos verboso
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(fh)

ch = logging.StreamHandler()            # stdout
ch.setLevel(logging.DEBUG)              # mismo nivel que el archivo
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

def log_ex(prefix: str, e: Exception):
    logger.error("%s: %s\n%s", prefix, e, traceback.format_exc())




# Ollama (configurable por env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

# Paletas modernas
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

# ---------------------------- Utilidades ----------------------------
ISO_FMT = "%Y-%m-%d"

def today_date() -> date:
    return datetime.now().date()

def parse_date(s: str) -> date:
    return datetime.strptime(s, ISO_FMT).date()

def fmt_date(d: date) -> str:
    return d.strftime(ISO_FMT)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def mix_hex(c1: str, c2: str, t: float) -> str:
    c1 = c1.lstrip("#"); c2 = c2.lstrip("#")
    r1,g1,b1 = int(c1[0:2],16), int(c1[2:4],16), int(c1[4:6],16)
    r2,g2,b2 = int(c2[0:2],16), int(c2[2:4],16), int(c2[4:6],16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02X}{g:02X}{b:02X}"

def urgency_color(start: date, due: date, now: date) -> str:
    base = MODERN_COLORS["Text"]
    red  = MODERN_COLORS["Danger"]
    if due <= now:
        return red
    total_span = (due - start).days
    if total_span <= 0:
        return red
    remaining = (due - now).days
    t = 1.0 - (remaining / total_span)
    t = clamp(t, 0.0, 1.0)
    return mix_hex(base, red, t)

def safe_write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def valid_date(s: str) -> bool:
    try:
        parse_date(s.strip()); return True
    except Exception:
        return False

def clean_user_freeform(text: str) -> str:
    """Si el STT devolvió un JSON tipo {"text": "..."} extrae el 'text'; si no, devuelve el original."""
    if not text:
        return ""
    s = text.strip()
    if s.startswith("{") and '"text"' in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"]).strip()
        except Exception:
            pass
    return s


# ---------------------------- Botón pill (Canvas) ----------------------------
class PillButton(tk.Canvas):
    """Botón redondeado dibujado en Canvas (pill)."""
    def __init__(self, parent, text, command, color="Primary", size="normal", icon=""):
        self.base_color, self.hover_color = GRADIENTS.get(color, GRADIENTS["Primary"])
        self.text_str = f"{icon} {text}" if icon else text
        self.command = command

        if size == "small":
            self.font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
            self.padx, self.pady, self.radius = 10, 6, 12
        elif size == "large":
            self.font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
            self.padx, self.pady, self.radius = 18, 10, 16
        else:
            self.font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            self.padx, self.pady, self.radius = 14, 8, 14

        tw = self.font.measure(self.text_str)
        th = self.font.metrics("linespace")
        width = tw + 2 * self.padx
        height = th + 2 * self.pady

        super().__init__(parent, width=width, height=height, highlightthickness=0,
                         bd=0, bg=parent.cget("bg"), takefocus=1, cursor="hand2")
        self._draw(self.base_color)

        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.base_color))
        self.bind("<Button-1>", self._on_click)
        self.bind("<Return>", self._on_click)
        self.bind("<space>", self._on_click)

    def _on_click(self, event=None):
        if callable(self.command): self.command()

    def _draw_rounded(self, x1, y1, x2, y2, r, fill):
        self.create_rectangle(x1+r, y1, x2-r, y2, outline="", fill=fill)
        self.create_rectangle(x1, y1+r, x2, y2-r, outline="", fill=fill)
        self.create_oval(x1, y1, x1+2*r, y1+2*r, outline="", fill=fill)
        self.create_oval(x2-2*r, y1, x2, y1+2*r, outline="", fill=fill)
        self.create_oval(x1, y2-2*r, x1+2*r, y2, outline="", fill=fill)
        self.create_oval(x2-2*r, y2-2*r, x2, y2, outline="", fill=fill)

    def _draw(self, bg_color):
        self.delete("all")
        w = int(self.cget("width")); h = int(self.cget("height"))
        self._draw_rounded(0, 0, w, h, self.radius, bg_color)
        self.create_text(w//2, h//2, text=self.text_str, fill="white", font=self.font)

# ---------------------------- Helpers de layout ----------------------------
def create_centered_row(parent):
    row = tk.Frame(parent, bg=parent.cget("bg"))
    row.pack(fill="x")
    row.grid_columnconfigure(0, weight=1)
    row.grid_columnconfigure(1, weight=0)
    row.grid_columnconfigure(2, weight=1)
    tk.Frame(row, bg=parent.cget("bg")).grid(row=0, column=0, sticky="ew")
    center = tk.Frame(row, bg=parent.cget("bg")); center.grid(row=0, column=1)
    tk.Frame(row, bg=parent.cget("bg")).grid(row=0, column=2, sticky="ew")
    return center

# ---------------------------- UI estadísticas ----------------------------
def create_stat_card(parent, label, value, color="Primary", icon=""):
    card = tk.Frame(parent, bg=GRADIENTS[color][0], relief="flat", bd=0)
    card.pack(side="left", padx=4, pady=4)
    header = tk.Frame(card, bg=GRADIENTS[color][0]); header.pack(fill="x", padx=8, pady=(4,0))
    tk.Label(header, text=f"{icon} {label}", bg=GRADIENTS[color][0], fg="white",
             font=("Segoe UI", 9, "bold")).pack(anchor="w")
    value_frame = tk.Frame(card, bg=GRADIENTS[color][0]); value_frame.pack(fill="x", padx=8, pady=(0,4))
    value_label = tk.Label(value_frame, text=str(value), bg=GRADIENTS[color][0], fg="white",
                           font=("Segoe UI", 16, "bold"))
    value_label.pack(anchor="w")
    return card, value_label

def update_statistics(app):
    total_tasks = len(app.store.tasks)
    completed_tasks = sum(1 for t in app.store.tasks if t["done"])
    pending_tasks = total_tasks - completed_tasks
    if hasattr(app, 'total_label'): app.total_label.configure(text=str(total_tasks))
    if hasattr(app, 'completed_label'): app.completed_label.configure(text=str(completed_tasks))
    if hasattr(app, 'pending_label'): app.pending_label.configure(text=str(pending_tasks))
    if hasattr(app, 'footer_label'):
        app.footer_label.configure(text=f"✨ {total_tasks} tareas • {completed_tasks} completadas")

# ---------------------------- Persistencia ----------------------------
class TaskStore:
    """
    Tarea:
    {
      "id": str, "title": str, "desc": str,
      "start": "YYYY-MM-DD", "due": "YYYY-MM-DD",
      "priority": bool, "done": bool,
      "color": str, "pinned": bool, "alpha": float,
      "geometry": str, "open": bool
    }
    """
    def __init__(self, path: Path = DATA_FILE):
        self.path = path
        self.tasks = []
        self._after = None
        self._save_pending = False
        self.load()

    def set_after(self, after_callable): self._after = after_callable

    def save_throttled(self, delay_ms=400):
        if not self._after: return self.save()
        if self._save_pending: return
        self._save_pending = True
        def _flush():
            self._save_pending = False
            self.save()
        self._after(delay_ms, _flush)

    def _load_local_v2_if_any(self):
        try:
            if LOCAL_DATA_FILE.exists() and not self.path.exists():
                raw = LOCAL_DATA_FILE.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, list):
                    for t in data: self._apply_defaults(t)
                    safe_write_json(self.path, {"tasks": data})
                    return {"tasks": data}
        except Exception:
            pass
        return None

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                messagebox.showwarning(APP_NAME, "Archivo de notas dañado; se reinicia.")
                data = {"tasks": []}
        else:
            data = self._load_local_v2_if_any() or {"tasks": []}

        self.tasks = data.get("tasks", [])
        changed = False
        for t in self.tasks:
            changed |= self._apply_defaults(t)
        if changed: self.save()

    def _apply_defaults(self, t: dict) -> bool:
        changed = False
        if "id" not in t: t["id"] = uuid.uuid4().hex[:8]; changed = True
        if "title" not in t and "text" in t: t["title"] = t.pop("text"); changed = True
        if "desc" not in t: t["desc"] = ""; changed = True
        if "start" not in t: t["start"] = fmt_date(today_date()); changed = True
        if "due" not in t: t["due"] = fmt_date(today_date() + timedelta(days=3)); changed = True
        if "priority" not in t: t["priority"] = False; changed = True
        if "done" not in t: t["done"] = False; changed = True
        if "color" not in t: t["color"] = "Sunshine"; changed = True
        if "pinned" not in t: t["pinned"] = False; changed = True
        if "alpha" not in t: t["alpha"] = 0.98; changed = True
        if "geometry" not in t: t["geometry"] = "560x520+200+200"; changed = True
        if "open" not in t: t["open"] = False; changed = True
        return changed

    def save(self): safe_write_json(self.path, {"tasks": self.tasks})

    def add(self, title: str, desc: str, due: date, priority: bool, color: str | None = None):
        task = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "desc": desc,
            "start": fmt_date(today_date()),
            "due": fmt_date(due),
            "priority": priority,
            "done": False,
            "color": color or "Sunshine",
            "pinned": False,
            "alpha": 0.98,
            "geometry": "560x520+200+200",
            "open": False,
        }
        self.tasks.append(task)
        self.save()

    def toggle_done(self, index: int):
        self.tasks[index]["done"] = not self.tasks[index]["done"]; self.save()

    def toggle_priority(self, index: int):
        self.tasks[index]["priority"] = not self.tasks[index]["priority"]; self.save()

    def delete(self, index: int):
        self.tasks.pop(index); self.save()

    def index_by_id(self, tid: str) -> int:
        for i, t in enumerate(self.tasks):
            if t.get("id") == tid: return i
        return -1

# ---------------------------- Mapeo de colores aceptados ----------------------------
SPANISH_TO_KEY = {
    "amarillo": "Sunshine",
    "azul": "Ocean",
    "verde": "Nature",
    "naranja": "Sunset",
    "morado": "Lavender",
    "violeta": "Lavender",
    "lila": "Lavender",
    "coral": "Coral",
    "rojo": "Coral",  # aproximación
    "anaranjado": "Sunset",
}

def normalize_color(value: str | None) -> str | None:
    if not value: return None
    v = value.strip().lower()
    # acepta clave directa
    for k in VIBRANT_COLORS.keys():
        if v == k.lower(): return k
    # acepta etiqueta en español
    for k, lbl in COLOR_LABELS.items():
        if v == lbl.lower(): return k
    # sinonimia
    if v in SPANISH_TO_KEY: return SPANISH_TO_KEY[v]
    return None

# ---------------------------- Cliente Ollama ----------------------------



def ollama_chat_json(prompt: str, model: str = OLLAMA_MODEL, url_base: str = OLLAMA_URL, images: list[str] | None = None) -> dict | None:

    
    user_msg = {
        "role": "user",
        "content": prompt
    }
    # Si hay imágenes, las codificamos como data URIs
    if images:
        user_msg["images"] = images

    payload = {
        "model": model,
        "stream": False,
        #"format": "json",
        "messages": [
            {"role": "system", "content":
             "Eres un analista que EXTRAERÁ campos para crear una tarea. "
             "Responde EXCLUSIVAMENTE con un objeto JSON válido (sin texto adicional). "
             "Claves: title (string), desc (string), due (string YYYY-MM-DD o vacío), "
             "color (string en español o clave), priority (boolean). "
             "Si no encuentras un título claro, deja title como cadena vacía."},
            user_msg
        ]
    }
    logger.debug("Ollama.request: model=%s url=%s", model, url_base)
    logger.debug("Ollama.prompt: %s", prompt)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url_base + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            logger.debug("Ollama.raw = %s", raw_text)
            raw = json.loads(raw_text)
    except urllib.error.URLError as e:
        log_ex("Ollama.URLError", e)
        messagebox.showerror("Ollama", f"No se pudo conectar a {url_base}:\n{e}\nMira {LOG_FILE}")
        return None
    except Exception as e:
        log_ex("Ollama.Error", e)
        messagebox.showerror("Ollama", f"Error llamando a Ollama:\n{e}\nMira {LOG_FILE}")
        return None

    content = (raw.get("message") or {}).get("content", "")
    logger.debug("Ollama.content = %s", content)

    # Intentar parsear JSON limpio
    try:
        obj = json.loads(content)
        logger.debug("Ollama.json = %s", obj)
        return obj
    except Exception:
        obj = _extract_first_json_object(content)
        if obj is not None:
            logger.debug("Ollama.json.recovered = %s", obj)
            return obj

    logger.warning("La respuesta de Ollama no fue un JSON válido (ver log).")
    messagebox.showwarning("Ollama", "La respuesta de Ollama no fue un JSON válido. Revisa el archivo de log.")
    return None


def _extract_first_json_object(text: str):
    """Devuelve el primer objeto JSON bien balanceado encontrado en 'text', o None."""
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

# ---------------------------- Prompt de extracción de tareas ----------------------------


def build_extraction_prompt(natural_text: str, images_present:bool=False) -> str:
    """
    Prompt para extraer campos desde una frase en español.
    Se aceptan ejemplos del usuario como: 'crea una tarea título x desc y para el viernes color azul prioridad alta'.
    """
    allowed_colors = ", ".join(sorted(set(list(VIBRANT_COLORS.keys()) + list(COLOR_LABELS.values()))))
    today = fmt_date(today_date())
    extra = (
        "\nSi se adjuntaron imágenes: léelas/trascribe su texto o interpreta lo que hay en la imagen, "
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




def extract_task_with_ollama(natural_text: str,  image_paths: list[str] | None = None) -> dict | None:
    """
    Devuelve dict normalizado: {"title","desc","due","color","priority"}
    due puede venir vacío; color puede venir vacío; priority bool.
    """
    

    images_b64 = [_encode_image_base64(p) for p in (image_paths or [])]

    # --- 1) Si no hay texto del usuario, primero pedimos una descripción ---
    if not natural_text.strip() and images_b64:
        desc_prompt = "Describe con detalle el contenido de la(s) imagen(es)."
        desc_raw = ollama_chat_json(desc_prompt, images=images_b64)
        # Si responde con texto normal lo convertimos a string
        if isinstance(desc_raw, dict):                 # por si devolvió JSON
            natural_text = json.dumps(desc_raw, ensure_ascii=False)
        else:
            natural_text = str(desc_raw)

    # --- 2) Ahora sí construimos el prompt de extracción con 'natural_text' ---
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

    # Normalizar color a claves internas
    color_key = normalize_color(color_raw)

    # Normalizar fecha: si invalid, dejar vacío
    if due and not valid_date(due):
        due = ""

    return {
        "title": title,
        "desc": desc,
        "due": due,          # "" -> usaremos default
        "color": color_key,  # None -> default
        "priority": priority
    }


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------- Reconocimiento de voz (opcional) ----------------------------
# --- STT preferente: Faster-Whisper -> Vosk (ambos open-source y locales) ---
def stt_from_audio_bytes(raw_bytes: bytes, sample_rate: int = 16000) -> tuple[str | None, str | None]:
    try:
        import speech_recognition as sr
    except Exception:
        return None, "Instala SpeechRecognition: pip install SpeechRecognition"

    audio = sr.AudioData(raw_bytes, sample_rate, 2)
    r = sr.Recognizer()
    text = ""

    # 1) Faster-Whisper (si existe el método en tu SR)
    try:
        text = r.recognize_faster_whisper(audio, language="es")
        if text and text.strip():
            return text.strip(), None
    except Exception:
        pass

    # 2) Vosk
    try:
        text = r.recognize_vosk(audio)  # algunas versiones devuelven '{"text":"..."}'
        if text is not None:
            cleaned = clean_user_freeform(text).strip()
            if cleaned:
                return cleaned, None
    except Exception as e:
        return None, f"Voz: instala Vosk y su modelo en ./models (p. ej. vosk-model-small-es-0.42). Error: {e}"

    return None, "No fue posible transcribir con los motores locales (Faster-Whisper/Vosk)."


# --- Grabador push-to-talk con PyAudio ---
class PushToTalkRecorder:
    """Graba audio mientras presionas; devuelve bytes PCM al soltar."""
    def __init__(self, owner_widget, rate=16000, chunk=2048):
        self.owner = owner_widget
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.running = False
        self.stream = None
        self.pa = None

    def start(self):
        try:
            import pyaudio
        except Exception:
            messagebox.showwarning("Dictado", "Instala PyAudio (en Windows: pip install pipwin && pipwin install pyaudio)")
            return False

        try:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                                       input=True, frames_per_buffer=self.chunk)
        except Exception as e:
            messagebox.showwarning("Dictado", f"No se pudo abrir el micrófono.\n{e}")
            return False

        self.frames = []
        self.running = True
        self._loop()
        return True

    def _loop(self):
        if not self.running:
            return
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
        except Exception:
            pass
        # reprograma otro tirón
        self.owner.after(10, self._loop)

    def stop(self) -> bytes | None:
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pa:
                self.pa.terminate()
        except Exception:
            pass
        return b"".join(self.frames) if self.frames else None





# ---------------------------- Paleta y ventanas de notas (como antes) ----------------------------
class ColorPalette(tk.Toplevel):
    def __init__(self, master, on_pick):
        super().__init__(master)
        self.title("🎨 Paleta")
        self.resizable(False, False)
        self.on_pick = on_pick
        self.configure(bg=MODERN_COLORS["Background"])
        self.attributes("-topmost", True)
        row = 0; col = 0
        for name, hexv in VIBRANT_COLORS.items():
            sw = tk.Canvas(self, width=72, height=36, bg=hexv, highlightthickness=0, cursor="hand2")
            sw.grid(row=row, column=col, padx=6, pady=6)
            sw.bind("<Button-1>", lambda e, n=name: self._pick(n))
            tk.Label(self, text=COLOR_LABELS.get(name, name), bg=self.cget("bg"),
                     fg=MODERN_COLORS["Text"], font=("Segoe UI", 8)).grid(row=row+1, column=col)
            col += 1
            if col >= 3: col = 0; row += 2
    def _pick(self, name):
        try: self.on_pick(name)
        finally: self.destroy()

# --- Editor de nota: solo el Text toma el color ---
class ModernNoteWindow(tk.Toplevel):
    def __init__(self, app, tid: str, task: dict):
        super().__init__(app)
        self.app = app
        self.tid = tid
        self.palette_win = None
        self.dirty = False
        self._load_snapshot(task)

        geom = task.get("geometry", "560x520+200+200")
        try: self.geometry(geom)
        except Exception: self.geometry("560x520+200+200")
        self.minsize(520, 420)

        self.base_bg = MODERN_COLORS["Background"]
        self.configure(bg=self.base_bg)

        self.is_top = bool(task.get("pinned", False))
        self.attributes("-topmost", self.is_top)
        self.alpha = float(task.get("alpha", 0.98))
        try: self.attributes("-alpha", self.alpha)
        except Exception: pass

        self._update_title()
        self.protocol("WM_DELETE_WINDOW", self._on_request_close)

        header = tk.Frame(self, bg=self.base_bg, relief="flat", bd=0); header.pack(fill="x", padx=8, pady=(8,4))
        title_row = tk.Frame(header, bg=self.base_bg); title_row.pack(fill="x")
        self.title_var = tk.StringVar(value=self.pending_title)
        ent = tk.Entry(title_row, textvariable=self.title_var, bd=0, relief="flat",
                       font=("Segoe UI", 12, "bold"), bg=self.base_bg, fg=MODERN_COLORS["Text"])
        ent.pack(fill="x", padx=(6,4), pady=4)
        ent.bind("<KeyRelease>", lambda e: self._mark_dirty())

        controls = tk.Frame(header, bg=self.base_bg); controls.pack(fill="x", pady=(2,0))
        PillButton(controls, "Pin", self.toggle_pin, "Primary", "small", "📌").pack(side="left", padx=4)
        PillButton(controls, "Paleta", self.open_palette, "Secondary", "small", "🎨").pack(side="left", padx=4)
        self.btn_save = PillButton(controls, "Guardar", self._save_changes, "Success", "small", "💾")
        self.btn_save.pack(side="left", padx=4)
        PillButton(controls, "Cancelar", self._revert_changes, "Warning", "small", "↩️").pack(side="left", padx=4)

        op = tk.Frame(controls, bg=self.base_bg); op.pack(side="left", padx=8)
        tk.Label(op, text="🔍", bg=self.base_bg, font=("Segoe UI", 10)).pack(side="left")
        self.op_scale = ttk.Scale(op, from_=0.6, to=1.0, value=self.alpha, command=self.on_opacity, length=110)
        self.op_scale.pack(side="left", padx=4)

        self.text = tk.Text(self, wrap="word", undo=True, relief="flat", bd=0,
                            padx=12, pady=12, height=12,
                            font=("Segoe UI", 10), fg=MODERN_COLORS["Dark"])
        self.text.pack(fill="both", expand=True, padx=8, pady=4)
        self.text.insert("1.0", self.pending_desc)
        self.text.bind("<<Modified>>", self._on_text_modified)
        self._apply_color_to_text()

        footer = tk.Frame(self, bg=self.base_bg, relief="flat", bd=0); footer.pack(fill="x", padx=8, pady=8)

        date_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0); date_card.pack(side="left", padx=4)
        tk.Label(date_card, text="📅 Vence:", bg=GRADIENTS["Card"][0], font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4,0))
        self.due_var = tk.StringVar(value=self.pending_due)
        self.due_entry = tk.Entry(date_card, textvariable=self.due_var, width=12, font=("Segoe UI", 9), relief="flat", bd=0)
        self.due_entry.pack(padx=8, pady=(0,4))
        self.due_var.trace_add("write", lambda *_: (self._validate_due(), self._mark_dirty()))
        atajos = tk.Frame(date_card, bg=GRADIENTS["Card"][0]); atajos.pack(padx=6, pady=(0,6))
        for label, delta in [("Hoy", 0), ("+1d", 1), ("+3d", 3), ("+7d", 7), ("+30d", 30)]:
            PillButton(atajos, label, lambda d=delta: self._set_due_delta(d), "Primary", "small").pack(side="left", padx=2, pady=2)

        priority_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0); priority_card.pack(side="left", padx=4)
        tk.Label(priority_card, text="⭐ Prioridad:", bg=GRADIENTS["Card"][0], font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4,0))
        self.prio_var = tk.StringVar(value=("Alta" if self.pending_priority else "Baja"))
        prio = ttk.Combobox(priority_card, textvariable=self.prio_var, values=["Alta", "Baja"], width=6, state="readonly", font=("Segoe UI", 9))
        prio.pack(padx=8, pady=(0,4))
        prio.bind("<<ComboboxSelected>>", lambda e: self._mark_dirty())

        done_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0); done_card.pack(side="left", padx=4)
        self.done_var = tk.BooleanVar(value=self.pending_done)
        tk.Checkbutton(done_card, text="✅ Hecha", variable=self.done_var, command=self._mark_dirty,
                       bg=GRADIENTS["Card"][0], font=("Segoe UI", 9, "bold"), fg=MODERN_COLORS["Text"]).pack(padx=8, pady=8)

        actions_right = tk.Frame(footer, bg=self.base_bg); actions_right.pack(side="right")
        PillButton(actions_right, "Eliminar", self.delete_task, "Danger", "small", "🗑️").pack(side="left", padx=4)

        self.bind("<Control-s>", lambda e: self._save_changes())
        self.bind("<Escape>", lambda e: self._revert_changes())

        self.bind("<Configure>", self.on_configure)
        self.after(50, lambda: self.text.focus_set())
        self._validate_due(initial=True)
        self._set_save_state(False)

    # snapshot / utils
    def _load_snapshot(self, task):
        self.pending_title = task.get("title", "")
        self.pending_desc = task.get("desc", "")
        self.pending_due = task.get("due", fmt_date(today_date() + timedelta(days=3)))
        self.pending_priority = bool(task.get("priority", False))
        self.pending_done = bool(task.get("done", False))
        self.pending_color = task.get("color", "Sunshine")

    def _update_title(self):
        star = " • ✴" if self.dirty else ""
        self.title(f"✨ Nota • {self.pending_title or 'Sin título'}{star}")

    def _apply_color_to_text(self):
        color = VIBRANT_COLORS.get(self.pending_color, "#FFD54F")
        self.text.configure(bg=color)

    def _set_save_state(self, enabled: bool):
        self.dirty = enabled
        self._update_title()

    def _mark_dirty(self): self._set_save_state(True)

    def _validate_due(self, initial=False):
        s = self.due_var.get().strip()
        ok = valid_date(s)
        self.due_entry.configure(fg=MODERN_COLORS["Text"] if ok else MODERN_COLORS["Danger"])
        return ok

    def _set_due_delta(self, days: int):
        self.due_var.set(fmt_date(today_date() + timedelta(days=days)))
        self._mark_dirty()

    def _on_text_modified(self, event=None):
        if self.text.edit_modified():
            self.text.edit_modified(False)
            self._mark_dirty()

    def open_palette(self):
        if hasattr(self, "palette_win") and self.palette_win and self.palette_win.winfo_exists():
            self.palette_win.lift(); return
        self.palette_win = ColorPalette(self, self._on_pick_color)

    def _on_pick_color(self, name):
        self.pending_color = name
        self._apply_color_to_text()
        self._mark_dirty()

    def on_configure(self, event=None):
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0: return
        self.app.store.tasks[idx]["geometry"] = self.geometry()
        self.app.store.save_throttled()

    def on_opacity(self, val):
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0: return
        self.alpha = float(val)
        try: self.attributes("-alpha", self.alpha)
        except Exception: pass
        self.app.store.tasks[idx]["alpha"] = self.alpha
        self.app.store.save_throttled()

    def toggle_pin(self):
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0: return
        self.is_top = not self.is_top
        try: self.attributes("-topmost", self.is_top)
        except Exception: pass
        self.app.store.tasks[idx]["pinned"] = self.is_top
        self.app.store.save_throttled()

    def _save_changes(self):
        if not self._validate_due():
            messagebox.showwarning("Guardar", "Fecha fin inválida. Usa formato YYYY-MM-DD.")
            return
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0: return
        t = self.app.store.tasks[idx]
        self.pending_title = self.title_var.get().strip()
        self.pending_desc = self.text.get("1.0", "end-1c")
        self.pending_due = self.due_var.get().strip()
        self.pending_priority = (self.prio_var.get() == "Alta")
        self.pending_done = bool(self.done_var.get())
        t.update({
            "title": self.pending_title,
            "desc": self.pending_desc,
            "due": self.pending_due,
            "priority": self.pending_priority,
            "done": self.pending_done,
            "color": self.pending_color,
        })
        self.app.store.save()
        self.app.render_tasks(); update_statistics(self.app)
        self._set_save_state(False)

    def _revert_changes(self):
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0: return
        t = self.app.store.tasks[idx]
        self._load_snapshot(t)
        # re-aplicar a widgets
        self.title_var.set(self.pending_title)
        self.text.delete("1.0", "end"); self.text.insert("1.0", self.pending_desc)
        self.due_var.set(self.pending_due)
        self._apply_color_to_text()
        self._set_save_state(False)

    def delete_task(self):
        if messagebox.askyesno("Eliminar", "¿Eliminar esta tarea?"):
            idx = self.app.store.index_by_id(self.tid)
            if idx >= 0:
                self.app.delete_task_by_index(idx, self.tid)
            try: self.destroy()
            except Exception: pass

    def _on_request_close(self):
        if not self.dirty: return self._final_close()
        res = messagebox.askyesnocancel("Cambios sin guardar",
                                        "Tienes cambios sin guardar.\n¿Quieres guardarlos?",
                                        default=messagebox.YES, icon=messagebox.WARNING)
        if res is None: return
        if res: self._save_changes(); return self._final_close()
        return self._final_close()

    def _final_close(self):
        idx = self.app.store.index_by_id(self.tid)
        if idx >= 0:
            self.app.store.tasks[idx]["open"] = False
            self.app.store.save_throttled()
        self.app.note_windows.pop(self.tid, None)
        self.destroy()

    

# --- Overlay "posit rápido" (sin barra del SO) ---
class QuickStickyWindow(tk.Toplevel):
    def __init__(self, master, task: dict):
        super().__init__(master)
        self.task = task
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        color = VIBRANT_COLORS.get(task.get("color", "Sunshine"), "#FFD54F")
        fg = MODERN_COLORS["Dark"]
        self.configure(bg=color)
        pad = 12
        tk.Label(self, text=task.get("title") or "(Sin título)", bg=color, fg=fg,
                 font=("Segoe UI", 13, "bold"), anchor="w").pack(fill="x", padx=pad, pady=(pad, 4))
        tk.Label(self, text=task.get("desc", ""), bg=color, fg=fg, font=("Segoe UI", 10),
                 anchor="nw", justify="left", wraplength=380).pack(fill="both", expand=True, padx=pad, pady=(0, 6))
        tk.Label(self, text=f"📅 {task.get('due', '')}", bg=color, fg=fg,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=pad, pady=(0, 4))
        btn_frame = tk.Frame(self, bg=color); btn_frame.pack(fill="x", padx=pad, pady=(0, pad))
        PillButton(btn_frame, "Cerrar", self.close, "Danger", "small", "✖").pack(side="right")
        self.bind("<Button-1>", self._start_move)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<Escape>", lambda e: self.close())
        self.after(10, self._center_over_master)

    def _center_over_master(self):
        self.update_idletasks()
        mw = self.master.winfo_width(); mh = self.master.winfo_height()
        mx = self.master.winfo_rootx(); my = self.master.winfo_rooty()
        w = min(420, max(320, int(mw * 0.5)))
        h = self.winfo_height()
        if h < 200: h = 220
        self.geometry(f"{w}x{h}+{mx + (mw - w)//2}+{my + (mh - h)//2}")

    def _start_move(self, e):
        self._ox, self._oy = e.x, e.y

    def _on_move(self, e):
        x = self.winfo_x() + e.x - self._ox
        y = self.winfo_y() + e.y - self._oy
        self.geometry(f"+{x}+{y}")

    def close(self):
        try: self.destroy()
        except Exception: pass

# ---------------------------- Diálogo IA (Ollama) ----------------------------
class OllamaCaptureDialog(tk.Toplevel):
    """
    Diálogo para dictar o pegar texto, enviar a Ollama y crear la tarea.
    - Si falta título -> muestra aviso y abre diálogo 'Nueva Tarea' con datos pre-rellenados (menos título)
    """
    def __init__(self, master, on_create_task):
        super().__init__(master)
        self.title("🤖 IA (Ollama)")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.configure(bg=GRADIENTS["Card"][0])
        self.on_create_task = on_create_task


        header = tk.Frame(self, bg=GRADIENTS["Primary"][0]); header.pack(fill="x")
        tk.Label(header, text="Habla o pega tu instrucción", bg=GRADIENTS["Primary"][0], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(padx=16, pady=10)

        content = tk.Frame(self, bg=GRADIENTS["Card"][0]); content.pack(fill="both", expand=True, padx=16, pady=16)
        tk.Label(content, text="🗣️ Instrucción (ej.: \"crea tarea 'Enviar reporte' para el viernes, color azul, prioridad alta... \")",
                 bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"], font=("Segoe UI", 9, "bold"),
                 wraplength=480, justify="left").pack(anchor="w")
        self.txt_input = tk.Text(content, height=6, bg=MODERN_COLORS["White"], fg=MODERN_COLORS["Text"],
                                 relief="flat", bd=0, font=("Segoe UI", 10), wrap="word")
        self.txt_input.pack(fill="x", pady=(6, 10))
        self.image_paths = []
        self.img_info = tk.Label(content, text="Sin imágenes", bg=GRADIENTS["Card"][0],
                                fg=MODERN_COLORS["TextLight"], font=("Segoe UI", 9))
        self.img_info.pack(anchor="w", pady=(0,6))

        self.status_var = tk.StringVar(value="")
        self.status_lbl = tk.Label(content, textvariable=self.status_var,
                                bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["TextLight"],
                                font=("Segoe UI", 9, "italic"))
        self.status_lbl.pack(anchor="w", pady=(4,0))



        # Botones
        row = create_centered_row(content)

        # Botón "mantener para dictar"
        self.btn_ptt = PillButton(row, "🎙️ Mantener para dictar", lambda: None, "Secondary", "normal")
        self.btn_ptt.pack(side="left", padx=6, pady=4)
        self.btn_ptt.bind("<ButtonPress-1>", lambda e: self._ptt_start())
        self.btn_ptt.bind("<ButtonRelease-1>", lambda e: self._ptt_stop())

        PillButton(row, "▶ Analizar y crear", self._analyze_and_create, "Success", "normal", "✅").pack(side="left", padx=6, pady=4)
        PillButton(row, "🖼 Adjuntar imagen", self._pick_images, "Secondary", "normal").pack(side="left", padx=6, pady=4)
        PillButton(row, "Cerrar", self.destroy, "Danger", "normal", "✖").pack(side="left", padx=6, pady=4)


    


    def _ptt_start(self):
        # feedback visual
        try:
            self.btn_ptt._draw(GRADIENTS["Danger"][0])  # pone el botón en rojo mientras graba
        except Exception:
            pass
        self._rec = PushToTalkRecorder(self)
        ok = self._rec.start()
        if not ok:
            # vuelve al color normal si no logró iniciar
            try: self.btn_ptt._draw(GRADIENTS["Secondary"][0])
            except Exception: pass

    def _ptt_stop(self):
        if not hasattr(self, "_rec") or self._rec is None:
            return
        raw = self._rec.stop()
        logger.debug("PTT.bytes = %s", len(raw) if raw else 0)
        try:
            self.btn_ptt._draw(GRADIENTS["Secondary"][0])
        except Exception:
            pass
        self._rec = None
        if not raw:
            return
        text, err = stt_from_audio_bytes(raw, sample_rate=16000)
        if err:
            messagebox.showwarning("Dictado", err)
            return
        if text:
            prev = self.txt_input.get("1.0", "end").strip()
            self.txt_input.delete("1.0", "end")
            self.txt_input.insert("1.0", (prev + "\n" if prev else "") + text)


    def _pick_images(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona imagen(es)",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("Todos", "*.*")]
        )
        if not paths: 
            return
        self.image_paths = list(paths)
        self.img_info.configure(text=f"{len(self.image_paths)} imagen(es) adjunta(s)")

    def _analyze_and_create(self):
        raw = self.txt_input.get("1.0", "end").strip()
        if not raw and not self.image_paths:
            messagebox.showinfo("IA (Ollama)",
                                "Escribe/dicta una instrucción o adjunta una imagen.")
            return

        user_text = clean_user_freeform(raw)
        self._set_status("Consultando a Ollama…")
        try:
            logger.debug("IA.input = %s", user_text)
            logger.debug("IA.images = %s", len(self.image_paths))
            extracted = extract_task_with_ollama(
                user_text,
                image_paths=self.image_paths
            )
        except Exception as e:
            self._set_status("Error")
            log_ex("ollama_chat_json", e)
            messagebox.showerror("IA (Ollama)",
                                f"Ocurrió un error.\nRevisa {LOG_FILE}")
            return
        finally:
            self._set_status("")

        # --- VALIDAR SALIDA ---
        if not extracted:
            messagebox.showwarning("IA (Ollama)",
                                "No pude entender la orden. Revisa el log.")
            return

        # ---------------------- crear tarea ----------------------
        title = extracted["title"]
        desc  = extracted["desc"]
        due_s = extracted["due"] or fmt_date(today_date() + timedelta(days=3))
        if not valid_date(due_s):           # fallback seguro
            due_s = fmt_date(today_date() + timedelta(days=3))
        due = parse_date(due_s)
        prio  = bool(extracted["priority"])
        color = extracted["color"] or "Sunshine"

        logger.debug("IA.output = %s", extracted)
        self.on_create_task(title, desc, due, prio, color)  # ¡CREA LA TAREA!
        self.destroy()


    def _set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()



# ---------------------------- App principal ----------------------------
class ModernStickyApp(tk.Tk):
    BG = MODERN_COLORS["Background"]

    def __init__(self, store: TaskStore):
        super().__init__()
        self.store = store
        self.store.set_after(self.after)
        self.title(APP_NAME)
        self.configure(bg=self.BG)
        self.geometry("640x800")
        self.minsize(600, 680)
        self.resizable(True, True)

        self._create_header()
        self._create_toolbar()
        self._create_content_area()
        self._create_footer()

        self.note_windows: dict[str, ModernNoteWindow] = {}
        self.quick_windows: dict[str, QuickStickyWindow] = {}

        self.render_tasks()
        self.after(200, self.reopen_notes)

    def _create_header(self):
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0], relief="flat", bd=0); header.pack(fill="x")
        title_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0]); title_frame.pack(fill="x", padx=16, pady=12)
        tk.Label(title_frame, text=APP_NAME, bg=GRADIENTS["Primary"][0], fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        stats_frame = tk.Frame(header, bg=GRADIENTS["Primary"][0]); stats_frame.pack(fill="x", padx=16, pady=(0,12))
        _, self.total_label = create_stat_card(stats_frame, "Total", 0, "Primary", "📋")
        _, self.completed_label = create_stat_card(stats_frame, "Completadas", 0, "Success", "✅")
        _, self.pending_label = create_stat_card(stats_frame, "Pendientes", 0, "Warning", "⏳")
        update_statistics(self)

    def _create_toolbar(self):
        toolbar = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0); toolbar.pack(fill="x", padx=8, pady=8)
        center = create_centered_row(toolbar)

        PillButton(center, "Nueva Tarea", self.open_add_dialog, "Primary", "normal", "➕").pack(side="left", padx=6)

        self.var_only_pending = tk.BooleanVar(value=False)
        tk.Checkbutton(center, text="👁️ Solo Pendientes", variable=self.var_only_pending,
                       command=self.render_tasks, bg=center.cget("bg"),
                       font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)

        PillButton(center, "Abrir Notas", self.reopen_notes, "Secondary", "normal", "📝").pack(side="left", padx=6)

        # --- Botón IA (Ollama) ---
        PillButton(center, "IA (Ollama)", self.open_ollama_dialog, "Success", "normal", "🤖").pack(side="left", padx=6)

        # Filtro de color
        self.color_filter_frame = tk.Frame(center, bg=center.cget("bg"))
        tk.Label(self.color_filter_frame, text="Color:", bg=center.cget("bg"),
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8,4))
        self.var_color_filter = tk.StringVar(value="Todos")
        self.cb_color_filter = ttk.Combobox(self.color_filter_frame, textvariable=self.var_color_filter,
                                            state="readonly", width=12, values=["Todos"])
        self.cb_color_filter.pack(side="left")
        self.cb_color_filter.bind("<<ComboboxSelected>>", lambda e: self.render_tasks())

        self.var_topmost = tk.BooleanVar(value=True)
        def _toggle_topmost(): self.attributes("-topmost", self.var_topmost.get())
        tk.Checkbutton(center, text="Siempre arriba", variable=self.var_topmost,
                       command=_toggle_topmost, bg=center.cget("bg"),
                       font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)
        _toggle_topmost()

    def _create_content_area(self):
        content_frame = tk.Frame(self, bg=self.BG); content_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(content_frame, bg=self.BG, highlightthickness=0, relief="flat", bd=0)
        self.scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=self.canvas.yview)
        self.task_frame = tk.Frame(self.canvas, bg=self.BG)
        self.task_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.task_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel()

    def _bind_mousewheel(self):
        def _on_mousewheel(e):
            delta = e.delta
            if delta == 0: return
            self.canvas.yview_scroll(int(-1*(delta/120)), "units")
        def _on_mousewheel_linux_up(e): self.canvas.yview_scroll(-3, "units")
        def _on_mousewheel_linux_down(e): self.canvas.yview_scroll(3, "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)

    def _create_footer(self):
        footer = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0); footer.pack(fill="x", padx=8, pady=8)
        self.footer_label = tk.Label(footer, text="✨ 0 tareas • 0 completadas",
                                     bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["TextLight"], font=("Segoe UI", 9))
        self.footer_label.pack(pady=4)

    # ---------- Diálogo agregar ----------
    def open_add_dialog(self):
        ModernAddTaskDialog(self, self._on_add)

    def open_add_dialog_prefilled(self, title: str, desc: str, due: date, priority: bool, color: str):
        ModernAddTaskDialog(self, self._on_add, preset={"title": title, "desc": desc, "due": due, "priority": priority, "color": color})

    def _on_add(self, title: str, desc: str, due: date, priority: bool, color: str | None = None):
        self.store.add(title, desc, due, priority, color=color)
        # Asegúrate de que la nueva tarea sea visible aunque haya un filtro activo:
        if hasattr(self, "var_color_filter"):
            self.var_color_filter.set("Todos")
        self.render_tasks()
        update_statistics(self)


    # ---------- IA (Ollama) ----------
    def open_ollama_dialog(self):
        OllamaCaptureDialog(self, self._on_add)


    # ---------- Filtro de color dinámico ----------
    def _refresh_color_filter(self):
        colors_present = sorted({t.get("color", "Sunshine") for t in self.store.tasks})
        if len(self.store.tasks) == 0:
            self.color_filter_frame.pack_forget()
            self.var_color_filter.set("Todos")
            return
        labels = ["Todos"] + [COLOR_LABELS.get(k, k) for k in colors_present]
        current = self.var_color_filter.get()
        self.cb_color_filter["values"] = labels
        if not self.color_filter_frame.winfo_ismapped():
            self.color_filter_frame.pack(side="left", padx=8)
        if current not in labels:
            self.var_color_filter.set("Todos")

    # ---------- Render ----------
    def render_tasks(self):
        self._refresh_color_filter()
        logger.debug("Render: only_pending=%s color_filter=%s", 
                    getattr(self, "var_only_pending", tk.BooleanVar(value=False)).get(),
                    getattr(self, "var_color_filter", tk.StringVar(value="Todos")).get())
        for w in self.task_frame.winfo_children():
            w.destroy()

        now = today_date()
        selected_label = self.var_color_filter.get()
        selected_key = LABEL_TO_KEY.get(selected_label) if selected_label and selected_label != "Todos" else None

        for task in self.store.tasks:
            if self.var_only_pending.get() and task.get("done", False):
                continue
            if selected_key and task.get("color") != selected_key:
                continue
            self._render_modern_task_card(task, now)

        update_statistics(self)

    def _render_modern_task_card(self, task: dict, now: date):
        tid = task.get("id")
        border_color = self._get_task_border_color(task, now)

        outer = tk.Frame(self.task_frame, bg=border_color); outer.pack(fill="x", pady=4, padx=4)
        card = tk.Frame(outer, bg=GRADIENTS["Card"][0], relief="flat", bd=0); card.pack(fill="x", padx=2, pady=2)

        stripe_color = VIBRANT_COLORS.get(task.get("color", "Sunshine"), "#CCCCCC")
        stripe = tk.Frame(card, bg=stripe_color, width=10); stripe.pack(side="left", fill="y")

        body = tk.Frame(card, bg=GRADIENTS["Card"][0]); body.pack(side="left", fill="both", expand=True)

        header = tk.Frame(body, bg=GRADIENTS["Card"][0]); header.pack(fill="x", padx=12, pady=8)

        var_done = tk.BooleanVar(value=task["done"])
        tk.Checkbutton(header, variable=var_done,
                       command=lambda t=tid: self._toggle_done_with_update_by_id(t),
                       bg=GRADIENTS["Card"][0], font=("Segoe UI", 12)).pack(side="left")

        title_font = ("Segoe UI", 12, "bold")
        title_color = MODERN_COLORS["Text"] if not task["done"] else MODERN_COLORS["TextLight"]

        title_frame = tk.Frame(header, bg=GRADIENTS["Card"][0]); title_frame.pack(side="left", fill="x", expand=True, padx=8)
        lbl_title = tk.Label(title_frame, text=task["title"], bg=GRADIENTS["Card"][0], fg=title_color,
                             font=title_font, anchor="w", wraplength=460, justify="left", cursor="hand2")
        lbl_title.pack(fill="x")
        lbl_title.bind("<Double-Button-1>", lambda e, t=tid: self.open_quick_sticky_by_id(t))  # overlay
        # Botón extra para abrir editor completo sigue en acciones

        if task.get("priority"):
            tk.Label(header, text="⭐ PRIORIDAD", bg=GRADIENTS["Warning"][0], fg="white",
                     font=("Segoe UI", 8, "bold"), padx=6, pady=2).pack(side="right")

        content = tk.Frame(body, bg=GRADIENTS["Card"][0]); content.pack(fill="x", padx=12, pady=(0,8))
        if task.get("desc"):
            desc_font = ("Segoe UI", 10, "italic" if task["done"] else "normal")
            desc_fg = MODERN_COLORS["TextLight"] if task["done"] else MODERN_COLORS["Text"]
            tk.Label(content, text=task["desc"], bg=GRADIENTS["Card"][0], fg=desc_fg,
                     font=desc_font, anchor="w", wraplength=460, justify="left").pack(fill="x", pady=(0,8))

        footer = tk.Frame(body, bg=GRADIENTS["Card"][0]); footer.pack(fill="x", padx=12, pady=(0,8))
        try:
            start_d = parse_date(task["start"]); due_d = parse_date(task["due"])
        except Exception:
            start_d = due_d = now

        due_txt = f"📅 {task['due']}"
        overdue = (due_d <= now)
        due_fg = MODERN_COLORS["Danger"] if overdue and not task["done"] else MODERN_COLORS["TextLight"]
        tk.Label(footer, text=due_txt, bg=GRADIENTS["Card"][0], fg=due_fg, font=("Segoe UI", 9)).pack(side="left")

        actions = tk.Frame(footer, bg=GRADIENTS["Card"][0]); actions.pack(side="right")
        priority_icon = "⭐" if task.get("priority") else "☆"
        self._create_small_button(actions, priority_icon, lambda t=tid: self._toggle_priority_with_update_by_id(t), "Warning")
        self._create_small_button(actions, "📝", lambda t=tid: self.open_note_by_id(t), "Primary")
        self._create_small_button(actions, "🗑️", lambda t=tid: self._delete_task_with_update_by_id(t), "Danger")

    def _create_small_button(self, parent, icon, command, color):
        btn = tk.Label(parent, text=icon, bg=GRADIENTS[color][0], fg="white",
                       font=("Segoe UI", 10), padx=6, pady=2, relief="flat", bd=0, takefocus=1, cursor="hand2")
        btn.pack(side="left", padx=2)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Return>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(bg=GRADIENTS[color][1]))
        btn.bind("<Leave>", lambda e: btn.configure(bg=GRADIENTS[color][0]))

    def _toggle_done_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.store.toggle_done(idx)
        self.render_tasks(); update_statistics(self)

    def _toggle_priority_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.store.toggle_priority(idx)
        self.render_tasks(); update_statistics(self)

    def _delete_task_with_update_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        self.delete_task_by_index(idx, tid)
        update_statistics(self)

    def _get_task_border_color(self, task: dict, now: date) -> str:
        if task["done"]: return MODERN_COLORS["Success"]
        if task.get("priority"): return MODERN_COLORS["Warning"]
        try:
            start_d = parse_date(task["start"]); due_d = parse_date(task["due"])
            return urgency_color(start_d, due_d, now)
        except Exception:
            return MODERN_COLORS["TextLight"]

    def delete_task_by_index(self, idx: int, tid: str):
        win = self.note_windows.pop(tid, None)
        if win is not None and win.winfo_exists():
            try: win.destroy()
            except Exception: pass
        qwin = self.quick_windows.pop(tid, None)
        if qwin is not None and qwin.winfo_exists():
            try: qwin.destroy()
            except Exception: pass
        self.store.delete(idx)
        self.render_tasks()

    # ---------- Notas flotantes (editor) ----------
    def open_note_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        task = self.store.tasks[idx]
        win = self.note_windows.get(tid)
        if win and win.winfo_exists():
            win.deiconify(); win.lift(); return
        task["open"] = True
        self.store.save_throttled()
        win = ModernNoteWindow(self, tid, task)
        self.note_windows[tid] = win

    # ---------- Posit rápido (overlay) ----------
    def open_quick_sticky_by_id(self, tid: str):
        idx = self.store.index_by_id(tid)
        if idx < 0: return
        task = self.store.tasks[idx]
        qwin = self.quick_windows.get(tid)
        if qwin and qwin.winfo_exists():
            qwin.deiconify(); qwin.lift(); return
        qwin = QuickStickyWindow(self, task)
        self.quick_windows[tid] = qwin

    def reopen_notes(self):
        for t in self.store.tasks:
            if t.get("open") or t.get("pinned"):
                try: self.open_note_by_id(t.get("id"))
                except Exception: pass

# ---------------------------- Diálogo "Nueva Tarea" ----------------------------
class ModernAddTaskDialog(tk.Toplevel):
    def __init__(self, master, on_ok, preset: dict | None = None):
        super().__init__(master)
        self.title("➕ Nueva Tarea")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.configure(bg=GRADIENTS["Card"][0])
        self.on_ok = on_ok
        self.preset = preset or {}

        header = tk.Frame(self, bg=GRADIENTS["Primary"][0], relief="flat", bd=0); header.pack(fill="x")
        tk.Label(header, text="➕ Crear Nueva Tarea", bg=GRADIENTS["Primary"][0], fg="white",
                 font=("Segoe UI", 14, "bold")).pack(padx=16, pady=12)

        content = tk.Frame(self, bg=GRADIENTS["Card"][0]); content.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(content, text="📝 Título", bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self.ent_title = tk.Entry(content, font=("Segoe UI", 11), relief="flat", bd=0,
                                  bg=MODERN_COLORS["White"], fg=MODERN_COLORS["Text"])
        self.ent_title.pack(fill="x", pady=(0,12))

        tk.Label(content, text="📄 Descripción", bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self.txt_desc = tk.Text(content, font=("Segoe UI", 10), relief="flat", bd=0,
                                bg=MODERN_COLORS["White"], fg=MODERN_COLORS["Text"], height=4)
        self.txt_desc.pack(fill="x", pady=(0,12))

        tk.Label(content, text="📅 Fecha de Vencimiento", bg=GRADIENTS["Card"][0], fg=MODERN_COLORS["Text"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        date_frame = tk.Frame(content, bg=GRADIENTS["Card"][0]); date_frame.pack(fill="x", pady=(0,12))
        self.ent_due = tk.Entry(date_frame, font=("Segoe UI", 10), relief="flat", bd=0,
                                bg=MODERN_COLORS["White"], fg=MODERN_COLORS["Text"])
        default_due = self.preset.get("due", today_date() + timedelta(days=3))
        self.ent_due.insert(0, fmt_date(default_due))
        self.ent_due.pack(side="left", fill="x", expand=True)

        atajos_row = create_centered_row(content)
        for label, delta in [("🕐 Hoy", 0), ("📅 +1d", 1), ("📅 +3d", 3), ("📅 +7d", 7), ("📅 +30d", 30)]:
            PillButton(atajos_row, label, lambda d=delta: self._set_due_delta(d),
                       "Primary", "small").pack(side="left", padx=4, pady=2)

        priority_row = create_centered_row(content)
        self.var_priority = tk.BooleanVar(value=bool(self.preset.get("priority", False)))
        tk.Checkbutton(priority_row, text="⭐ Marcar como Prioridad", variable=self.var_priority,
                       bg=priority_row.cget("bg"), fg=MODERN_COLORS["Text"],
                       font=("Segoe UI", 10, "bold")).pack(pady=6)

        # Paleta de color preseleccionada si viene del preset
        self.preset_color = self.preset.get("color", "Sunshine")
        color_row = create_centered_row(content)
        tk.Label(color_row, text=f"🎨 Color: {COLOR_LABELS.get(self.preset_color, self.preset_color)}",
                 bg=color_row.cget("bg"), fg=MODERN_COLORS["Text"], font=("Segoe UI", 9, "bold")).pack(side="left", padx=6)
        PillButton(color_row, "Cambiar", self._pick_color, "Secondary", "small", "🎨").pack(side="left", padx=6)

        btn_row = create_centered_row(content)
        PillButton(btn_row, "Cancelar", self.destroy, "Danger", "normal", "✖").pack(side="left", padx=6, pady=4)
        PillButton(btn_row, "Crear Tarea", self._submit, "Success", "normal", "✅").pack(side="left", padx=6, pady=4)

        # Rellenar campos desde preset
        if self.preset:
            self.ent_title.insert(0, self.preset.get("title", ""))
            self.txt_desc.insert("1.0", self.preset.get("desc", ""))

        self.ent_title.focus()

    def _set_due_delta(self, d: int):
        self.ent_due.delete(0, "end")
        self.ent_due.insert(0, fmt_date(today_date() + timedelta(days=d)))

    def _pick_color(self):
        def _on_pick(name):
            self.preset_color = name
        ColorPalette(self, _on_pick)

    def _submit(self):
        title = self.ent_title.get().strip()
        desc = self.txt_desc.get("1.0", "end").strip()
        due_s = self.ent_due.get().strip()
        if not title:
            messagebox.showwarning("Nueva tarea", "Coloca un título."); return
        if not valid_date(due_s):
            messagebox.showwarning("Nueva tarea", "Fecha fin inválida. Usa formato YYYY-MM-DD."); return
        self.on_ok(title, desc, parse_date(due_s), self.var_priority.get(), color=self.preset_color)
        self.destroy()

# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    ModernStickyApp(TaskStore()).mainloop()

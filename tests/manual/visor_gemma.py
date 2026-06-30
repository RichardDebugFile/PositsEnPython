#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pequeña GUI Tkinter que envía imágenes al modelo gemma3:12b vía Ollama
y muestra la descripción devuelta.
"""

import base64
import requests
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from pathlib import Path
from PIL import Image   # solo para validar que es imagen

# --- Configuración ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"
PROMPT      = "Describe con detalle el contenido de la imagen."

def encode_image_b64(path: Path) -> str:
    """Lee la imagen y la devuelve en base64 para el endpoint /api/generate."""
    with path.open("rb") as fp:
        return base64.b64encode(fp.read()).decode()

def analyze_image(path: Path) -> str:
    """Envía la imagen al modelo y devuelve la respuesta en texto plano."""
    # 1) Validamos que sea realmente una imagen para evitar errores
    try:
        Image.open(path).verify()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"El archivo seleccionado no es una imagen válida: {e}") from e

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [encode_image_b64(path)],
        "stream": False          # respuesta completa, no fragmentada
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    # El campo principal se llama "response" en /api/generate
    return data.get("response", "")

# ---------------------- Interfaz Tkinter ---------------------- #
def choose_file():
    filetypes = [
        ("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
        ("Todos los archivos", "*.*")
    ]
    path_str = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
    if not path_str:
        return

    path = Path(path_str)
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, "⏳ Analizando la imagen…\n")
    root.update()

    try:
        description = analyze_image(path)
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, description)
    except Exception as exc:   # noqa: BLE001
        result_box.delete("1.0", tk.END)
        messagebox.showerror("Error", str(exc))

root = tk.Tk()
root.title("Analizador de fotos con Gemma 3 (12B)")

select_btn = tk.Button(root, text="📷 Elegir imagen y analizar", command=choose_file, width=30)
select_btn.pack(pady=10)

result_box = scrolledtext.ScrolledText(root, width=90, height=25, wrap=tk.WORD)
result_box.pack(padx=10, pady=10)

root.mainloop()

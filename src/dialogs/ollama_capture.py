#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diálogo para captura con IA (Ollama)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import timedelta

from ..config import MODERN_COLORS, GRADIENTS
from ..utils.dates import fmt_date, today_date, parse_date, valid_date
from ..utils.logger import logger
from ..services import extract_task_with_ollama
from ..services.stt import PushToTalkRecorder, stt_from_audio_bytes, clean_user_freeform
from ..ui.components import PillButton, create_centered_row

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


        header = tk.Frame(self, bg=GRADIENTS["Primary"][0])
        header.pack(fill="x")
        tk.Label(
            header,
            text="Habla o pega tu instrucción",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 13, "bold")
        ).pack(padx=16, pady=10)

        content = tk.Frame(self, bg=GRADIENTS["Card"][0])
        content.pack(fill="both", expand=True, padx=16, pady=16)
        tk.Label(
            content,
            text='🗣️ Instrucción (ej.: "crea tarea \'Enviar reporte\' '
            'para el viernes, color azul, prioridad alta... ")',
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9, "bold"),
            wraplength=480,
            justify="left"
        ).pack(anchor="w")
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

        PillButton(
            row,
            "▶ Analizar y crear",
            self._analyze_and_create,
            "Success",
            "normal",
            "✅"
        ).pack(side="left", padx=6, pady=4)
        PillButton(
            row,
            "🖼 Adjuntar imagen",
            self._pick_images,
            "Secondary",
            "normal"
        ).pack(side="left", padx=6, pady=4)
        PillButton(
            row, "Cerrar", self.destroy, "Danger", "normal", "✖"
        ).pack(side="left", padx=6, pady=4)



    def _ptt_start(self):
        # feedback visual
        try:
            # pone el botón en rojo mientras graba
            self.btn_ptt._draw(GRADIENTS["Danger"][0])
        except (AttributeError, KeyError):
            pass
        self._rec = PushToTalkRecorder(self)
        ok = self._rec.start()
        if not ok:
            # vuelve al color normal si no logró iniciar
            try:
                self.btn_ptt._draw(GRADIENTS["Secondary"][0])
            except (AttributeError, KeyError):
                pass

    def _ptt_stop(self):
        if not hasattr(self, "_rec") or self._rec is None:
            return
        raw = self._rec.stop()
        logger.debug("PTT.bytes = %s", len(raw) if raw else 0)
        try:
            self.btn_ptt._draw(GRADIENTS["Secondary"][0])
        except (AttributeError, KeyError):
            pass
        self._rec = None
        if not raw:
            return

        # Transcribir el audio a texto
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
            filetypes=[
                ("Imágenes", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
                ("Todos", "*.*")
            ]
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.img_info.configure(
            text=f"{len(self.image_paths)} imagen(es) adjunta(s)"
        )

    def _analyze_and_create(self):
        raw = self.txt_input.get("1.0", "end").strip()
        if not raw and not self.image_paths:
            messagebox.showinfo(
                "IA (Ollama)",
                "Escribe/dicta una instrucción o adjunta una imagen."
            )
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
        except (ConnectionError, TimeoutError, ValueError) as e:
            self._set_status("Error")
            logger.error("Error en Ollama: %s", e)
            messagebox.showerror(
                "IA (Ollama)",
                f"Ocurrió un error al conectar con Ollama.\n{e}"
            )
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


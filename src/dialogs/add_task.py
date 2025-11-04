#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diálogo para agregar nueva tarea
"""

import tkinter as tk
from tkinter import messagebox
from datetime import timedelta

from ..config import MODERN_COLORS, GRADIENTS, COLOR_LABELS
from ..utils.dates import fmt_date, today_date, valid_date, parse_date
from ..ui.components import PillButton, ColorPalette, create_centered_row

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

        header = tk.Frame(self, bg=GRADIENTS["Primary"][0], relief="flat", bd=0)
        header.pack(fill="x")
        tk.Label(
            header,
            text="➕ Crear Nueva Tarea",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(padx=16, pady=12)

        content = tk.Frame(self, bg=GRADIENTS["Card"][0])
        content.pack(fill="both", expand=True, padx=16, pady=16)

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

        tk.Label(
            content,
            text="📅 Fecha de Vencimiento",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))
        date_frame = tk.Frame(content, bg=GRADIENTS["Card"][0])
        date_frame.pack(fill="x", pady=(0, 12))
        self.ent_due = tk.Entry(date_frame, font=("Segoe UI", 10), relief="flat", bd=0,
                                bg=MODERN_COLORS["White"], fg=MODERN_COLORS["Text"])
        default_due = self.preset.get("due", today_date() + timedelta(days=3))
        self.ent_due.insert(0, fmt_date(default_due))
        self.ent_due.pack(side="left", fill="x", expand=True)

        atajos_row = create_centered_row(content)
        for label, delta in [
            ("🕐 Hoy", 0), ("📅 +1d", 1), ("📅 +3d", 3),
            ("📅 +7d", 7), ("📅 +30d", 30)
        ]:
            PillButton(
                atajos_row,
                label,
                lambda d=delta: self._set_due_delta(d),
                "Primary",
                "small"
            ).pack(side="left", padx=4, pady=2)

        priority_row = create_centered_row(content)
        self.var_priority = tk.BooleanVar(value=bool(self.preset.get("priority", False)))
        tk.Checkbutton(priority_row, text="⭐ Marcar como Prioridad", variable=self.var_priority,
                       bg=priority_row.cget("bg"), fg=MODERN_COLORS["Text"],
                       font=("Segoe UI", 10, "bold")).pack(pady=6)

        # Paleta de color preseleccionada si viene del preset
        self.preset_color = self.preset.get("color", "Sunshine")
        color_row = create_centered_row(content)
        tk.Label(
            color_row,
            text=f"🎨 Color: {COLOR_LABELS.get(self.preset_color, self.preset_color)}",
            bg=color_row.cget("bg"),
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=6)
        PillButton(
            color_row, "Cambiar", self._pick_color, "Secondary", "small", "🎨"
        ).pack(side="left", padx=6)

        btn_row = create_centered_row(content)
        PillButton(
            btn_row, "Cancelar", self.destroy, "Danger", "normal", "✖"
        ).pack(side="left", padx=6, pady=4)
        PillButton(
            btn_row, "Crear Tarea", self._submit, "Success", "normal", "✅"
        ).pack(side="left", padx=6, pady=4)

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
            messagebox.showwarning("Nueva tarea", "Coloca un título.")
            return
        if not valid_date(due_s):
            messagebox.showwarning(
                "Nueva tarea",
                "Fecha fin inválida. Usa formato YYYY-MM-DD."
            )
            return
        self.on_ok(
            title,
            desc,
            parse_date(due_s),
            self.var_priority.get(),
            color=self.preset_color
        )
        self.destroy()

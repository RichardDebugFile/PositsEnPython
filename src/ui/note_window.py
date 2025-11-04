#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ventana de edición de notas flotantes
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import timedelta
import os

from ..config import MODERN_COLORS, GRADIENTS, VIBRANT_COLORS
from ..utils.dates import fmt_date, today_date, valid_date
from .components import PillButton, ColorPalette
from ..services.attachments import AttachmentManager


class ModernNoteWindow(tk.Toplevel):
    """Ventana flotante para editar una nota/tarea completa"""

    def __init__(self, app, tid: str, task):
        super().__init__(app)
        self.app = app
        self.tid = tid
        self.palette_win = None
        self.dirty = False
        self._load_snapshot(task)

        geom = getattr(task, 'geometry', "560x520+200+200")
        try:
            self.geometry(geom)
        except Exception:
            self.geometry("560x520+200+200")
        self.minsize(520, 420)

        self.base_bg = MODERN_COLORS["Background"]
        self.configure(bg=self.base_bg)

        self.is_top = bool(getattr(task, "pinned", False))
        self.attributes("-topmost", self.is_top)
        self.alpha = float(getattr(task, "alpha", 0.98))
        try:
            self.attributes("-alpha", self.alpha)
        except Exception:
            pass

        self._update_title()
        self.protocol("WM_DELETE_WINDOW", self._on_request_close)

        # Header
        header = tk.Frame(self, bg=self.base_bg, relief="flat", bd=0)
        header.pack(fill="x", padx=8, pady=(8, 4))

        title_row = tk.Frame(header, bg=self.base_bg)
        title_row.pack(fill="x")
        self.title_var = tk.StringVar(value=self.pending_title)
        ent = tk.Entry(
            title_row,
            textvariable=self.title_var,
            bd=0,
            relief="flat",
            font=("Segoe UI", 12, "bold"),
            bg=self.base_bg,
            fg=MODERN_COLORS["Text"]
        )
        ent.pack(fill="x", padx=(6, 4), pady=4)
        ent.bind("<KeyRelease>", lambda e: self._mark_dirty())

        # Controls
        controls = tk.Frame(header, bg=self.base_bg)
        controls.pack(fill="x", pady=(2, 0))
        PillButton(controls, "Pin", self.toggle_pin, "Primary", "small", "📌").pack(side="left", padx=4)
        PillButton(controls, "Paleta", self.open_palette, "Secondary", "small", "🎨").pack(side="left", padx=4)
        PillButton(controls, "Adjuntar", self._attach_file, "Secondary", "small", "📎").pack(side="left", padx=4)
        self.btn_save = PillButton(controls, "Guardar", self._save_changes, "Success", "small", "💾")
        self.btn_save.pack(side="left", padx=4)
        PillButton(controls, "Cancelar", self._revert_changes, "Warning", "small", "↩️").pack(side="left", padx=4)

        # Opacity control
        op = tk.Frame(controls, bg=self.base_bg)
        op.pack(side="left", padx=8)
        tk.Label(op, text="🔍", bg=self.base_bg, font=("Segoe UI", 10)).pack(side="left")
        self.op_scale = ttk.Scale(op, from_=0.6, to=1.0, value=self.alpha, command=self.on_opacity, length=110)
        self.op_scale.pack(side="left", padx=4)

        # Text area
        self.text = tk.Text(
            self,
            wrap="word",
            undo=True,
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            height=12,
            font=("Segoe UI", 10),
            fg=MODERN_COLORS["Dark"]
        )
        self.text.pack(fill="both", expand=True, padx=8, pady=4)
        self.text.insert("1.0", self.pending_desc)
        self.text.bind("<<Modified>>", self._on_text_modified)
        self._apply_color_to_text()

        # Attachments section
        attach_frame = tk.Frame(self, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        attach_frame.pack(fill="x", padx=8, pady=8)
        tk.Label(
            attach_frame,
            text="📎 Archivos Adjuntos",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(padx=8, pady=(8, 4), anchor="w")

        self.attachments_listbox = tk.Listbox(
            attach_frame,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            height=3,
            relief="flat",
            bd=0,
            font=("Segoe UI", 9)
        )
        self.attachments_listbox.pack(fill="x", padx=8, pady=(0, 8))
        self.attachments_listbox.bind("<Double-Button-1>", lambda e: self._open_attachment())

        attach_buttons = tk.Frame(attach_frame, bg=GRADIENTS["Card"][0])
        attach_buttons.pack(fill="x", padx=8, pady=(0, 8))
        PillButton(attach_buttons, "Abrir", self._open_attachment, "Primary", "small", "📂").pack(side="left", padx=2)
        PillButton(attach_buttons, "Eliminar", self._remove_attachment, "Danger", "small", "❌").pack(side="left", padx=2)

        self._refresh_attachments()

        # Footer
        footer = tk.Frame(self, bg=self.base_bg, relief="flat", bd=0)
        footer.pack(fill="x", padx=8, pady=8)

        # Date card
        date_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        date_card.pack(side="left", padx=4)
        tk.Label(
            date_card,
            text="📅 Vence:",
            bg=GRADIENTS["Card"][0],
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=8, pady=(4, 0))

        self.due_var = tk.StringVar(value=self.pending_due)
        self.due_entry = tk.Entry(
            date_card,
            textvariable=self.due_var,
            width=12,
            font=("Segoe UI", 9),
            relief="flat",
            bd=0
        )
        self.due_entry.pack(padx=8, pady=(0, 4))
        self.due_var.trace_add("write", lambda *_: (self._validate_due(), self._mark_dirty()))

        atajos = tk.Frame(date_card, bg=GRADIENTS["Card"][0])
        atajos.pack(padx=6, pady=(0, 6))
        for label, delta in [("Hoy", 0), ("+1d", 1), ("+3d", 3), ("+7d", 7), ("+30d", 30)]:
            PillButton(atajos, label, lambda d=delta: self._set_due_delta(d), "Primary", "small").pack(side="left", padx=2, pady=2)

        # Priority card
        priority_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        priority_card.pack(side="left", padx=4)
        tk.Label(
            priority_card,
            text="⭐ Prioridad:",
            bg=GRADIENTS["Card"][0],
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=8, pady=(4, 0))

        self.prio_var = tk.StringVar(value=("Alta" if self.pending_priority else "Baja"))
        prio = ttk.Combobox(
            priority_card,
            textvariable=self.prio_var,
            values=["Alta", "Baja"],
            width=6,
            state="readonly",
            font=("Segoe UI", 9)
        )
        prio.pack(padx=8, pady=(0, 4))
        prio.bind("<<ComboboxSelected>>", lambda e: self._mark_dirty())

        # Done card
        done_card = tk.Frame(footer, bg=GRADIENTS["Card"][0], relief="flat", bd=0)
        done_card.pack(side="left", padx=4)
        self.done_var = tk.BooleanVar(value=self.pending_done)
        tk.Checkbutton(
            done_card,
            text="✅ Hecha",
            variable=self.done_var,
            command=self._mark_dirty,
            bg=GRADIENTS["Card"][0],
            font=("Segoe UI", 9, "bold"),
            fg=MODERN_COLORS["Text"]
        ).pack(padx=8, pady=8)

        # Actions
        actions_right = tk.Frame(footer, bg=self.base_bg)
        actions_right.pack(side="right")
        PillButton(actions_right, "Eliminar", self.delete_task, "Danger", "small", "🗑️").pack(side="left", padx=4)

        # Bindings
        self.bind("<Control-s>", lambda e: self._save_changes())
        self.bind("<Escape>", lambda e: self._revert_changes())
        self.bind("<Configure>", self.on_configure)

        self.after(50, lambda: self.text.focus_set())
        self._validate_due(initial=True)
        self._set_save_state(False)

    def _load_snapshot(self, task):
        """Carga el estado actual de la tarea"""
        from ..utils.dates import fmt_date
        self.pending_title = getattr(task, "title", "")
        self.pending_desc = getattr(task, "desc", "")
        # Manejar task.due que puede ser date o str
        due = getattr(task, "due", None)
        if due is None:
            self.pending_due = fmt_date(today_date() + timedelta(days=3))
        elif isinstance(due, str):
            self.pending_due = due
        else:
            self.pending_due = fmt_date(due)
        self.pending_priority = bool(getattr(task, "priority", False))
        self.pending_done = bool(getattr(task, "done", False))
        self.pending_color = getattr(task, "color", "Sunshine")

    def _update_title(self):
        """Actualiza el título de la ventana"""
        star = " • ✴" if self.dirty else ""
        self.title(f"✨ Nota • {self.pending_title or 'Sin título'}{star}")

    def _apply_color_to_text(self):
        """Aplica el color del posit al área de texto"""
        color = VIBRANT_COLORS.get(self.pending_color, "#FFD54F")
        self.text.configure(bg=color)

    def _set_save_state(self, enabled: bool):
        """Marca o desmarca el estado de cambios sin guardar"""
        self.dirty = enabled
        self._update_title()

    def _mark_dirty(self):
        """Marca como modificado"""
        self._set_save_state(True)

    def _validate_due(self, initial=False):
        """Valida la fecha de vencimiento"""
        s = self.due_var.get().strip()
        ok = valid_date(s)
        self.due_entry.configure(fg=MODERN_COLORS["Text"] if ok else MODERN_COLORS["Danger"])
        return ok

    def _set_due_delta(self, days: int):
        """Establece fecha de vencimiento relativa a hoy"""
        self.due_var.set(fmt_date(today_date() + timedelta(days=days)))
        self._mark_dirty()

    def _on_text_modified(self, event=None):
        """Callback cuando se modifica el texto"""
        if self.text.edit_modified():
            self.text.edit_modified(False)
            self._mark_dirty()

    def open_palette(self):
        """Abre la paleta de colores"""
        if hasattr(self, "palette_win") and self.palette_win and self.palette_win.winfo_exists():
            self.palette_win.lift()
            return
        self.palette_win = ColorPalette(self, self._on_pick_color)

    def _on_pick_color(self, name):
        """Callback al seleccionar un color"""
        self.pending_color = name
        self._apply_color_to_text()
        self._mark_dirty()

    def on_configure(self, event=None):
        """Guarda la geometría de la ventana"""
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0:
            return
        self.app.store.tasks[idx].geometry = self.geometry()
        self.app.store.save_throttled()

    def on_opacity(self, val):
        """Cambia la opacidad de la ventana"""
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0:
            return
        self.alpha = float(val)
        try:
            self.attributes("-alpha", self.alpha)
        except Exception:
            pass
        self.app.store.tasks[idx].alpha = self.alpha
        self.app.store.save_throttled()

    def toggle_pin(self):
        """Alterna el estado de siempre arriba"""
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0:
            return
        self.is_top = not self.is_top
        try:
            self.attributes("-topmost", self.is_top)
        except Exception:
            pass
        self.app.store.tasks[idx].pinned = self.is_top
        self.app.store.save_throttled()

    def _save_changes(self):
        """Guarda los cambios en la tarea"""
        if not self._validate_due():
            messagebox.showwarning("Guardar", "Fecha fin inválida. Usa formato YYYY-MM-DD.")
            return

        idx = self.app.store.index_by_id(self.tid)
        if idx < 0:
            return

        t = self.app.store.tasks[idx]
        self.pending_title = self.title_var.get().strip()
        self.pending_desc = self.text.get("1.0", "end-1c")
        self.pending_due = self.due_var.get().strip()
        self.pending_priority = (self.prio_var.get() == "Alta")
        self.pending_done = bool(self.done_var.get())

        # Actualizar atributos del objeto Task
        t.title = self.pending_title
        t.desc = self.pending_desc
        from ..utils.dates import parse_date
        t.due = parse_date(self.pending_due)
        t.priority = self.pending_priority
        t.done = self.pending_done
        t.color = self.pending_color

        self.app.store.save()
        self.app.render_tasks()
        self.app._update_statistics()
        self._set_save_state(False)

    def _revert_changes(self):
        """Revierte los cambios no guardados"""
        idx = self.app.store.index_by_id(self.tid)
        if idx < 0:
            return

        t = self.app.store.tasks[idx]
        self._load_snapshot(t)

        # Re-aplicar a widgets
        self.title_var.set(self.pending_title)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.pending_desc)
        self.due_var.set(self.pending_due)
        self._apply_color_to_text()
        self._set_save_state(False)

    def _attach_file(self):
        """Adjunta un archivo a la tarea"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo para adjuntar",
            filetypes=[
                ("Todos los archivos", "*.*"),
                ("PDF", "*.pdf"),
                ("Imágenes", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Excel", "*.xlsx *.xls"),
                ("Word", "*.docx *.doc"),
                ("Texto", "*.txt *.md")
            ]
        )
        if not file_path:
            return

        try:
            dest_path = AttachmentManager.copy_to_data(file_path)
            idx = self.app.store.index_by_id(self.tid)
            if idx >= 0:
                self.app.store.tasks[idx].attachments.append(dest_path)
                self.app.store.save()
                self._refresh_attachments()
                self._mark_dirty()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo adjuntar el archivo:\n{e}")

    def _refresh_attachments(self):
        """Actualiza la lista de archivos adjuntos"""
        self.attachments_listbox.delete(0, tk.END)
        idx = self.app.store.index_by_id(self.tid)
        if idx >= 0:
            task = self.app.store.tasks[idx]
            for attachment in task.attachments:
                filename = os.path.basename(attachment)
                self.attachments_listbox.insert(tk.END, filename)

    def _open_attachment(self):
        """Abre el archivo adjunto seleccionado"""
        selection = self.attachments_listbox.curselection()
        if not selection:
            messagebox.showinfo("Abrir", "Selecciona un archivo primero")
            return

        idx = self.app.store.index_by_id(self.tid)
        if idx >= 0:
            task = self.app.store.tasks[idx]
            attach_idx = selection[0]
            if attach_idx < len(task.attachments):
                file_path = task.attachments[attach_idx]
                try:
                    AttachmentManager.open_file(file_path)
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo abrir el archivo:\n{e}")

    def _remove_attachment(self):
        """Elimina el archivo adjunto seleccionado"""
        selection = self.attachments_listbox.curselection()
        if not selection:
            messagebox.showinfo("Eliminar", "Selecciona un archivo primero")
            return

        idx = self.app.store.index_by_id(self.tid)
        if idx >= 0:
            task = self.app.store.tasks[idx]
            attach_idx = selection[0]
            if attach_idx < len(task.attachments):
                if messagebox.askyesno("Eliminar", "¿Eliminar este archivo adjunto?"):
                    file_path = task.attachments.pop(attach_idx)
                    try:
                        AttachmentManager.delete_file(file_path)
                    except Exception:
                        pass  # Archivo ya no existe
                    self.app.store.save()
                    self._refresh_attachments()
                    self._mark_dirty()

    def delete_task(self):
        """Elimina la tarea"""
        if messagebox.askyesno("Eliminar", "¿Eliminar esta tarea?"):
            idx = self.app.store.index_by_id(self.tid)
            if idx >= 0:
                self.app.delete_task_by_index(idx, self.tid)
            try:
                self.destroy()
            except Exception:
                pass

    def _on_request_close(self):
        """Maneja el cierre de la ventana"""
        if not self.dirty:
            return self._final_close()

        res = messagebox.askyesnocancel(
            "Cambios sin guardar",
            "Tienes cambios sin guardar.\n¿Quieres guardarlos?",
            default=messagebox.YES,
            icon=messagebox.WARNING
        )

        if res is None:
            return
        if res:
            self._save_changes()
            return self._final_close()
        return self._final_close()

    def _final_close(self):
        """Cierre final de la ventana"""
        idx = self.app.store.index_by_id(self.tid)
        if idx >= 0:
            self.app.store.tasks[idx].open = False
            self.app.store.save_throttled()
        self.app.note_windows.pop(self.tid, None)
        self.destroy()

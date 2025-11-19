#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diálogo para descargar música desde YouTube
"""

import tkinter as tk
from tkinter import messagebox, ttk

from ..config import MODERN_COLORS, GRADIENTS
from ..services import YouTubeDownloader
from ..ui.components import PillButton, create_centered_row
from ..utils.logger import logger


class MusicDownloaderDialog(tk.Toplevel):
    """
    Diálogo para descargar música desde YouTube
    Integrado con el estilo visual de Posits Virtuales
    """

    def __init__(self, master, music_player=None):
        super().__init__(master)
        self.title("🎵 Descargar Música")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.configure(bg=GRADIENTS["Card"][0])

        self.music_player = music_player
        self.downloader = YouTubeDownloader()

        # Variables
        self.url_var = tk.StringVar()
        self.quality_var = tk.StringVar(value="192")
        self.status_var = tk.StringVar(value="Listo para descargar")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()

    def _build_ui(self):
        """Construye la interfaz del diálogo"""

        # Header
        header = tk.Frame(self, bg=GRADIENTS["Primary"][0])
        header.pack(fill="x")
        tk.Label(
            header,
            text="Descargar Música desde YouTube",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 13, "bold")
        ).pack(padx=16, pady=10)

        # Content
        content = tk.Frame(self, bg=GRADIENTS["Card"][0])
        content.pack(fill="both", expand=True, padx=16, pady=16)

        # URL del video
        tk.Label(
            content,
            text="🔗 URL del video de YouTube:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))

        url_frame = tk.Frame(content, bg=GRADIENTS["Card"][0])
        url_frame.pack(fill="x", pady=(0, 12))

        self.url_entry = tk.Entry(
            url_frame,
            textvariable=self.url_var,
            bg=MODERN_COLORS["White"],
            fg=MODERN_COLORS["Text"],
            relief="flat",
            bd=0,
            font=("Segoe UI", 10),
            width=50
        )
        self.url_entry.pack(side="left", fill="x", expand=True, ipady=6, ipadx=8)
        self.url_entry.focus()

        # Botón para obtener info
        PillButton(
            url_frame,
            "ℹ️ Info",
            self._get_video_info,
            "Secondary",
            "normal"
        ).pack(side="left", padx=(8, 0))

        # Info del video (oculto inicialmente)
        self.info_frame = tk.Frame(content, bg=GRADIENTS["Card"][1])
        self.info_label = tk.Label(
            self.info_frame,
            text="",
            bg=GRADIENTS["Card"][1],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9),
            justify="left",
            wraplength=480
        )
        self.info_label.pack(padx=12, pady=8)

        # Calidad de audio
        tk.Label(
            content,
            text="🎚️ Calidad de audio:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))

        quality_frame = tk.Frame(content, bg=GRADIENTS["Card"][0])
        quality_frame.pack(fill="x", pady=(0, 12))

        qualities = [
            ("128 kbps (Baja)", "128"),
            ("192 kbps (Media)", "192"),
            ("256 kbps (Alta)", "256"),
            ("320 kbps (Máxima)", "320")
        ]

        for label, value in qualities:
            tk.Radiobutton(
                quality_frame,
                text=label,
                variable=self.quality_var,
                value=value,
                bg=GRADIENTS["Card"][0],
                fg=MODERN_COLORS["Text"],
                selectcolor=GRADIENTS["Card"][1],
                font=("Segoe UI", 9)
            ).pack(side="left", padx=8)

        # Barra de progreso
        progress_frame = tk.Frame(content, bg=GRADIENTS["Card"][0])
        progress_frame.pack(fill="x", pady=(0, 8))

        self.status_label = tk.Label(
            progress_frame,
            textvariable=self.status_var,
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9, "italic")
        )
        self.status_label.pack(anchor="w", pady=(0, 4))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=480,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x")

        # Botones
        row = create_centered_row(content)

        self.btn_download = PillButton(
            row,
            "⬇️ Descargar",
            self._start_download,
            "Success",
            "normal"
        )
        self.btn_download.pack(side="left", padx=6, pady=4)

        self.btn_cancel = PillButton(
            row,
            "✖ Cancelar Descarga",
            self._cancel_download,
            "Danger",
            "disabled"
        )
        self.btn_cancel.pack(side="left", padx=6, pady=4)

        PillButton(
            row,
            "Cerrar",
            self.destroy,
            "Secondary",
            "normal"
        ).pack(side="left", padx=6, pady=4)

    def _get_video_info(self):
        """Obtiene y muestra información del video"""
        url = self.url_var.get().strip()

        if not url:
            messagebox.showwarning(
                "URL requerida",
                "Por favor ingresa la URL del video de YouTube"
            )
            return

        self.status_var.set("Obteniendo información del video...")
        self.update_idletasks()

        try:
            info = self.downloader.get_video_info(url)

            if info:
                duration_min = info.get("duration", 0) // 60 if info.get("duration") else 0
                duration_sec = info.get("duration", 0) % 60 if info.get("duration") else 0

                info_text = (
                    f"📺 {info.get('title', 'Sin título')}\n"
                    f"👤 {info.get('uploader', 'Desconocido')}\n"
                    f"⏱️ Duración: {duration_min}:{duration_sec:02d}\n"
                    f"👁️ Vistas: {info.get('view_count', 0):,}"
                )

                self.info_label.config(text=info_text)
                self.info_frame.pack(fill="x", pady=(0, 12))
                self.status_var.set("Información obtenida correctamente")
            else:
                messagebox.showerror(
                    "Error",
                    "No se pudo obtener información del video"
                )
                self.status_var.set("Error al obtener información")

        except Exception as e:
            logger.error(f"Error obteniendo info: {e}")
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.status_var.set("Error")

    def _start_download(self):
        """Inicia la descarga del video"""
        url = self.url_var.get().strip()

        if not url:
            messagebox.showwarning(
                "URL requerida",
                "Por favor ingresa la URL del video de YouTube"
            )
            return

        if self.downloader.is_downloading:
            messagebox.showinfo(
                "Descarga en progreso",
                "Ya hay una descarga en progreso. Espera a que termine."
            )
            return

        # Deshabilitar botones
        self.btn_download.configure(state="disabled")
        self.btn_cancel.configure(state="normal")

        # Resetear progreso
        self.progress_var.set(0)
        self.status_var.set("Iniciando descarga...")

        # Iniciar descarga
        quality = self.quality_var.get()
        self.downloader.download_audio(
            url,
            quality=quality,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error
        )

    def _on_progress(self, info: dict):
        """Callback de progreso"""
        status = info.get("status")

        if status == "downloading":
            percentage = info.get("percentage", 0)
            speed = info.get("speed", 0)
            eta = info.get("eta", 0)

            self.progress_var.set(percentage)

            # Formatear velocidad
            speed_mb = speed / (1024 * 1024) if speed else 0
            self.status_var.set(
                f"Descargando... {percentage:.1f}% - "
                f"{speed_mb:.2f} MB/s - "
                f"ETA: {eta}s"
            )

        elif status == "processing":
            self.progress_var.set(99)
            self.status_var.set("Procesando y convirtiendo a MP3...")

        self.update_idletasks()

    def _on_complete(self, info: dict):
        """Callback de descarga completada"""
        self.progress_var.set(100)
        self.status_var.set("✅ Descarga completada")

        # Habilitar botones
        self.btn_download.configure(state="normal")
        self.btn_cancel.configure(state="disabled")

        # Mostrar mensaje
        filename = info.get("filename", "Desconocido")
        size_mb = info.get("size", 0) / (1024 * 1024)

        messagebox.showinfo(
            "Descarga completada",
            f"✅ Descarga completada exitosamente!\n\n"
            f"Archivo: {filename}\n"
            f"Tamaño: {size_mb:.2f} MB\n"
            f"Ubicación: data/music/\n\n"
            f"La canción ha sido agregada a tu biblioteca de música."
        )

        # Si hay un reproductor de música, recargar la playlist
        if self.music_player:
            try:
                self.music_player.scan_music_folder()
                logger.info("Playlist actualizada con nueva canción")
            except Exception as e:
                logger.error(f"Error actualizando playlist: {e}")

        # Limpiar
        self.url_var.set("")
        self.progress_var.set(0)
        self.info_frame.pack_forget()

    def _on_error(self, error: str):
        """Callback de error"""
        self.status_var.set("❌ Error en descarga")

        # Habilitar botones
        self.btn_download.configure(state="normal")
        self.btn_cancel.configure(state="disabled")

        # Mostrar error
        messagebox.showerror(
            "Error en descarga",
            f"Ocurrió un error durante la descarga:\n\n{error}"
        )

        # Resetear progreso
        self.progress_var.set(0)

    def _cancel_download(self):
        """Cancela la descarga actual"""
        if self.downloader.is_downloading:
            self.downloader.cancel()
            self.status_var.set("Cancelando...")
            self.btn_cancel.configure(state="disabled")

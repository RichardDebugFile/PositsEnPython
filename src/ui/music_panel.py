#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel de control de música de fondo con playlist y canción principal
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from ..config import MODERN_COLORS, GRADIENTS
from .components import PillButton


class MusicPanel(tk.Frame):
    """Panel de control de música con playlist, canción principal y drag & drop"""

    def __init__(self, parent, music_player, **kwargs):
        super().__init__(parent, **kwargs)
        self.music_player = music_player
        self.configure(bg=GRADIENTS["Card"][0])

        # Configurar callback para cuando cambie la canción
        self.music_player.on_track_change = self._on_track_change

        self._create_ui()
        self._update_display()

        # Iniciar comprobación periódica de finalización de canciones
        self._check_music_end()

    def _check_music_end(self):
        """Comprueba periódicamente si la canción terminó"""
        self.music_player.check_and_play_next()
        # Llamar cada 1 segundo
        self.after(1000, self._check_music_end)

    def _create_ui(self):
        """Crea la interfaz del panel"""

        # Header
        header = tk.Frame(self, bg=GRADIENTS["Card"][0])
        header.pack(fill="x", padx=12, pady=(12, 8))

        tk.Label(
            header,
            text="🎵 Música de Fondo",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 11, "bold")
        ).pack(side="left")

        # Indicador de modo
        self.mode_label = tk.Label(
            header,
            text="[Principal]",
            bg=GRADIENTS["Primary"][0],
            fg="white",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2
        )
        self.mode_label.pack(side="right")

        # Archivo actual
        self.file_label = tk.Label(
            self,
            text="Sin musica seleccionada",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 9),
            wraplength=300,
            anchor="w",
            justify="left",
            padx=8,
            pady=4
        )
        self.file_label.pack(fill="x", padx=12, pady=(0, 8))

        # Zona de drop (drag & drop)
        drop_zone = tk.Frame(
            self,
            bg=MODERN_COLORS["Light"],
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground=MODERN_COLORS["TextLight"]
        )
        drop_zone.pack(fill="x", padx=12, pady=(0, 8))

        drop_label = tk.Label(
            drop_zone,
            text="🎼 Arrastra aqui tu archivo de musica\n(MP3, WAV, OGG, FLAC)",
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["TextLight"],
            font=("Segoe UI", 8),
            pady=15
        )
        drop_label.pack(fill="both", expand=True)

        # Configurar drag & drop (opcional, requiere tkinterdnd2)
        try:
            from tkinterdnd2 import DND_FILES
            drop_zone.drop_target_register("DND_Files")
            drop_label.drop_target_register("DND_Files")
            drop_zone.dnd_bind('<<Drop>>', self._on_drop)
            drop_label.dnd_bind('<<Drop>>', self._on_drop)
        except:
            # Si no está disponible tkinterdnd2, solo usar botón examinar
            pass

        # Botón agregar canción
        btn_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        btn_frame.pack(fill="x", padx=12, pady=(0, 8))

        PillButton(
            btn_frame,
            "📁 Agregar Cancion",
            self._browse_file,
            color="Primary",
            size="small"
        ).pack(side="left")

        # Lista de canciones (Listbox con scrollbar)
        playlist_label = tk.Label(
            self,
            text="🎶 Lista de Reproduccion:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 9, "bold")
        )
        playlist_label.pack(fill="x", padx=12, pady=(8, 4))

        # Frame para la lista con scrollbar
        list_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox
        self.playlist_listbox = tk.Listbox(
            list_frame,
            bg=MODERN_COLORS["Light"],
            fg=MODERN_COLORS["Text"],
            selectbackground=GRADIENTS["Primary"][0],
            selectforeground="white",
            font=("Segoe UI", 8),
            height=5,
            yscrollcommand=scrollbar.set,
            activestyle="none"
        )
        self.playlist_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.playlist_listbox.yview)

        # Bind para doble click (reproducir)
        self.playlist_listbox.bind("<Double-Button-1>", self._on_playlist_doubleclick)
        # Bind para click derecho (establecer como principal)
        self.playlist_listbox.bind("<Button-3>", self._on_right_click)

        # Botones de gestión de playlist
        btn_manage_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        btn_manage_frame.pack(fill="x", padx=12, pady=(0, 8))

        PillButton(
            btn_manage_frame,
            "🗑️ Eliminar",
            self._remove_selected,
            color="Danger",
            size="small"
        ).pack(side="left", padx=(0, 4))

        PillButton(
            btn_manage_frame,
            "⭐ Marcar Principal",
            self._set_as_main,
            color="Warning",
            size="small"
        ).pack(side="left")

        # Selector de modo
        mode_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        mode_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            mode_frame,
            text="Modo:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 8, "bold")
        ).pack(side="left", padx=(0, 8))

        PillButton(
            mode_frame,
            "🔁 Principal",
            self._switch_to_main,
            color="Warning",
            size="small"
        ).pack(side="left", padx=(0, 4))

        PillButton(
            mode_frame,
            "📋 Playlist",
            self._switch_to_playlist,
            color="Secondary",
            size="small"
        ).pack(side="left")

        # Controles de reproducción
        controls = tk.Frame(self, bg=GRADIENTS["Card"][0])
        controls.pack(fill="x", padx=12, pady=(0, 8))

        # Botón anterior
        PillButton(
            controls,
            "⏮",
            self._previous,
            color="Secondary",
            size="small"
        ).pack(side="left", padx=(0, 4))

        # Botón Play/Pause
        self.play_pause_btn = PillButton(
            controls,
            "▶ Play",
            self._toggle_play_pause,
            color="Success",
            size="normal"
        )
        self.play_pause_btn.pack(side="left", padx=(0, 4))

        # Botón Stop
        PillButton(
            controls,
            "⏹ Stop",
            self._stop,
            color="Danger",
            size="small"
        ).pack(side="left", padx=(0, 4))

        # Botón siguiente
        PillButton(
            controls,
            "⏭",
            self._next,
            color="Secondary",
            size="small"
        ).pack(side="left")

        # Control de volumen
        volume_frame = tk.Frame(self, bg=GRADIENTS["Card"][0])
        volume_frame.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(
            volume_frame,
            text="🔊 Volumen:",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 8)
        ).pack(side="left", padx=(0, 8))

        self.volume_var = tk.IntVar(value=int(self.music_player.get_volume() * 100))
        self.volume_slider = ttk.Scale(
            volume_frame,
            from_=0,
            to=100,
            variable=self.volume_var,
            command=self._on_volume_change,
            orient="horizontal"
        )
        self.volume_slider.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.volume_label = tk.Label(
            volume_frame,
            text=f"{self.volume_var.get()}%",
            bg=GRADIENTS["Card"][0],
            fg=MODERN_COLORS["Text"],
            font=("Segoe UI", 8),
            width=4
        )
        self.volume_label.pack(side="left")

    def _browse_file(self):
        """Abre diálogo para seleccionar archivo de música"""
        filetypes = [
            ("Archivos de Audio", "*.mp3 *.wav *.ogg *.flac"),
            ("MP3", "*.mp3"),
            ("WAV", "*.wav"),
            ("OGG", "*.ogg"),
            ("FLAC", "*.flac"),
            ("Todos los archivos", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de musica",
            filetypes=filetypes
        )

        if file_path:
            self._add_track(file_path)

    def _on_drop(self, event):
        """Maneja el evento de arrastrar y soltar"""
        try:
            # Obtener ruta del archivo
            file_path = event.data.strip('{}')  # Remover llaves si existen
            if file_path.startswith('{') and file_path.endswith('}'):
                file_path = file_path[1:-1]

            # Validar extensión
            valid_extensions = ['.mp3', '.wav', '.ogg', '.flac']
            if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                self._add_track(file_path)
            else:
                messagebox.showwarning("Formato no soportado", f"El archivo debe ser MP3, WAV, OGG o FLAC")

        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar archivo: {e}")

    def _add_track(self, file_path: str):
        """Agrega una canción a la playlist"""
        if self.music_player.add_to_playlist(file_path):
            self._update_playlist_display()
            # Si era la primera canción y no hay canción principal, establecer como principal
            if len(self.music_player.get_playlist()) == 1 and not self.music_player.get_main_track():
                self.music_player.set_main_track(file_path)
                self.music_player.play()
                self._update_display()

    def _remove_selected(self):
        """Elimina la canción seleccionada"""
        selection = self.playlist_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_player.remove_from_playlist(index)
            self._update_playlist_display()
            self._update_display()

    def _set_as_main(self):
        """Establece la canción seleccionada como canción principal"""
        selection = self.playlist_listbox.curselection()
        if selection:
            index = selection[0]
            playlist = self.music_player.get_playlist()
            if index < len(playlist):
                file_path = playlist[index]
                self.music_player.set_main_track(file_path)
                messagebox.showinfo("Cancion Principal", f"Cancion principal establecida:\n{Path(file_path).name}")
                self._update_display()
        else:
            messagebox.showwarning("Sin seleccion", "Por favor selecciona una cancion de la lista")

    def _on_right_click(self, event):
        """Menú contextual con click derecho"""
        # Seleccionar el item bajo el cursor
        index = self.playlist_listbox.nearest(event.y)
        self.playlist_listbox.selection_clear(0, tk.END)
        self.playlist_listbox.selection_set(index)
        self.playlist_listbox.activate(index)

        # Mostrar menú
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="⭐ Establecer como Principal", command=self._set_as_main)
        menu.add_command(label="▶ Reproducir", command=lambda: self._on_playlist_doubleclick(None))
        menu.add_separator()
        menu.add_command(label="🗑️ Eliminar", command=self._remove_selected)

        menu.tk_popup(event.x_root, event.y_root)

    def _on_playlist_doubleclick(self, event):
        """Reproduce la canción seleccionada al hacer doble click"""
        selection = self.playlist_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_player.play_track_at(index)
            self._update_display()

    def _switch_to_main(self):
        """Cambia al modo canción principal"""
        self.music_player.play_main_track()
        self._update_display()

    def _switch_to_playlist(self):
        """Cambia al modo playlist"""
        self.music_player.play_playlist()
        self._update_display()

    def _toggle_play_pause(self):
        """Alterna entre play y pause"""
        if self.music_player.is_music_playing():
            self.music_player.pause()
        elif self.music_player.is_paused:
            self.music_player.resume()
        else:
            self.music_player.play()
        self._update_display()

    def _stop(self):
        """Detiene la reproducción"""
        self.music_player.stop()
        self._update_display()

    def _previous(self):
        """Canción anterior"""
        if self.music_player.get_mode() == "playlist":
            self.music_player.previous_track()
            self._update_display()

    def _next(self):
        """Siguiente canción"""
        if self.music_player.get_mode() == "playlist":
            self.music_player.next_track()
            self._update_display()

    def _on_volume_change(self, value):
        """Callback cuando cambia el volumen"""
        volume = float(value) / 100.0
        self.music_player.set_volume(volume)
        self.volume_label.configure(text=f"{int(float(value))}%")

    def _on_track_change(self):
        """Callback cuando cambia la canción (automáticamente)"""
        self._update_display()

    def _update_display(self):
        """Actualiza la visualización del panel"""
        mode = self.music_player.get_mode()

        # Actualizar indicador de modo
        if mode == "main":
            self.mode_label.configure(text="[Principal]", bg=GRADIENTS["Warning"][0])
            main_track = self.music_player.get_main_track_filename()
            self.file_label.configure(text=f"🔁 {main_track}")
        else:
            self.mode_label.configure(text="[Playlist]", bg=GRADIENTS["Secondary"][0])
            filename = self.music_player.get_current_filename()
            current_index = self.music_player.get_current_index()
            playlist_len = len(self.music_player.get_playlist())

            if playlist_len > 0:
                self.file_label.configure(text=f"🎵 ({current_index + 1}/{playlist_len}) {filename}")
            else:
                self.file_label.configure(text="Sin musica en la playlist")

        # Actualizar botón play/pause
        if self.music_player.is_music_playing():
            self.play_pause_btn.set_text("⏸ Pause")
        else:
            self.play_pause_btn.set_text("▶ Play")

        # Resaltar canción actual en la lista
        self._update_playlist_display()

    def _update_playlist_display(self):
        """Actualiza la visualización de la playlist"""
        self.playlist_listbox.delete(0, tk.END)

        filenames = self.music_player.get_playlist_filenames()
        current_index = self.music_player.get_current_index()
        mode = self.music_player.get_mode()
        main_track = self.music_player.get_main_track()
        playlist = self.music_player.get_playlist()

        for i, name in enumerate(filenames):
            # Marcar canción actual en modo playlist
            if mode == "playlist" and i == current_index:
                prefix = "▶ "
            # Marcar canción principal
            elif main_track and i < len(playlist) and playlist[i] == main_track:
                prefix = "⭐ "
            else:
                prefix = "   "

            self.playlist_listbox.insert(tk.END, f"{prefix}{name}")

            # Resaltar la canción actual en modo playlist
            if mode == "playlist" and i == current_index:
                self.playlist_listbox.itemconfig(i, bg=GRADIENTS["Primary"][0], fg="white")
            # Resaltar canción principal
            elif main_track and i < len(playlist) and playlist[i] == main_track:
                self.playlist_listbox.itemconfig(i, bg=GRADIENTS["Warning"][0], fg="white")

    def refresh(self):
        """Refresca el estado del panel"""
        self._update_display()

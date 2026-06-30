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
from ..utils.audio_detection import get_browser_playing_audio
from ..utils.cdp_helper import CDPHelper


class MusicPanel(tk.Frame):
    """Panel de control de música con playlist, canción principal y drag & drop"""

    def __init__(self, parent, music_player, app=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.music_player = music_player
        self.app = app  # Referencia a la app principal
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
        ).pack(side="left", padx=(0, 4))

        # Botón para capturar música en reproducción de Windows
        PillButton(
            btn_frame,
            "🎧 Capturar",
            self._capture_playing_music,
            color="Success",
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

    def _capture_playing_music(self):
        """Captura la música en reproducción de Windows y abre el descargador"""
        try:
            # PASO 1: Verificar si hay URL de YouTube ya copiada en el portapapeles
            print("🔍 Verificando portapapeles...")
            clipboard_url = None
            try:
                clipboard_text = self.clipboard_get()
                if clipboard_text and isinstance(clipboard_text, str):
                    clipboard_text = clipboard_text.strip()
                    if ("youtube.com/watch" in clipboard_text or "youtu.be/" in clipboard_text) and \
                       (clipboard_text.startswith("http://") or clipboard_text.startswith("https://")):
                        clipboard_url = clipboard_text
            except Exception:
                pass

            if clipboard_url:
                print(f"✅ URL encontrada en portapapeles: {clipboard_url}")
                # Preguntar al usuario si quiere usar esta URL o capturar una nueva
                response = messagebox.askyesno(
                    "URL en portapapeles",
                    f"Se encontró una URL de YouTube en el portapapeles:\n\n"
                    f"{clipboard_url[:60]}...\n\n"
                    "¿Usar esta URL?\n\n"
                    "(NO = Capturar URL del navegador actual)",
                    icon='question'
                )

                if response:
                    # Usuario quiere usar la URL del portapapeles
                    if self.app and hasattr(self.app, 'open_music_downloader'):
                        self.app.open_music_downloader(initial_query=clipboard_url)
                    return
                else:
                    # Usuario quiere capturar nueva URL - limpiar portapapeles primero
                    print("🗑️ Usuario eligió capturar nueva URL, limpiando portapapeles...")
                    try:
                        self.clipboard_clear()
                        self.update()
                    except Exception:
                        pass

            # PASO 2: Obtener información de la música en reproducción
            media_info = self._get_windows_media_info()

            if not media_info or (not media_info.get("title") and not media_info.get("artist")):
                messagebox.showwarning(
                    "Sin música",
                    "No se detectó música en reproducción.\n\n"
                    "Asegúrate de tener un video de YouTube reproduciéndose."
                )
                return

            title = media_info.get("title", "")
            artist = media_info.get("artist", "")
            cleaned_title = self._clean_music_title(title)
            cleaned_artist = self._clean_music_title(artist)
            song_name = f"{cleaned_artist} - {cleaned_title}" if cleaned_artist and cleaned_title else (cleaned_artist or cleaned_title)

            # PASO 3: Preguntar al usuario cómo quiere capturar
            response = messagebox.askyesno(
                "Capturar URL de YouTube",
                f"🎵 Música detectada:\n{song_name}\n\n"
                "Para descargar el video exacto:\n\n"
                "1. Haz clic en SÍ\n"
                "2. Cambia a la ventana del navegador con YouTube\n"
                "3. El sistema copiará la URL automáticamente\n\n"
                "¿Intentar capturar la URL automáticamente?\n\n"
                "(NO = buscar por título en YouTube)",
                icon='question'
            )

            if not response:
                # Usuario eligió NO - usar búsqueda por título
                print("ℹ️ Usuario eligió buscar por título")
                if self.app and hasattr(self.app, 'open_music_downloader'):
                    self.app.open_music_downloader(initial_query=song_name)
                return

            # PASO 4: Dar tiempo al usuario para cambiar a la ventana del navegador
            continue_capture = messagebox.askokcancel(
                "Preparar navegador",
                "El sistema buscará y activará automáticamente\n"
                "la ventana del navegador con YouTube.\n\n"
                "Proceso:\n"
                "1. Se detectará la ventana del navegador\n"
                "2. Se activará automáticamente\n"
                "3. Se copiará la URL del video\n\n"
                "¿Continuar con la captura automática?",
                icon='question'
            )

            if not continue_capture:
                print("❌ Usuario canceló la captura")
                return

            # PASO 5: Intentar capturar URL (primero con CDP, luego con teclado)
            captured_url = None

            # Intento 1: Usar CDP (Chrome DevTools Protocol)
            print("🔍 Intentando capturar con CDP...")
            try:
                cdp_helper = CDPHelper()
                if cdp_helper.is_available():
                    print("✓ CDP disponible, buscando pestaña reproduciendo...")
                    captured_url = cdp_helper.get_youtube_playing_url()

                    if captured_url:
                        print(f"✅ URL capturada vía CDP: {captured_url}")
                    else:
                        print("⚠️ CDP disponible pero no se encontró pestaña reproduciendo")
                else:
                    print("ℹ️ CDP no disponible, usando método alternativo...")
            except Exception as e:
                print(f"⚠️ Error con CDP: {e}")

            # Intento 2: Si CDP falló, usar método de teclado
            if not captured_url:
                print("🔍 Intentando capturar URL con teclado (espera 3 seg)...")
                captured_url = self._capture_url_with_countdown()

            if captured_url and ("youtube.com/watch" in captured_url or "youtu.be/" in captured_url):
                print(f"✅ URL capturada: {captured_url}")

                # Abrir descargador con la URL
                if self.app and hasattr(self.app, 'open_music_downloader'):
                    self.app.open_music_downloader(initial_query=captured_url)

                # Limpiar portapapeles para evitar ciclo infinito en próxima captura
                print("🗑️ Limpiando portapapeles para evitar reutilización...")
                try:
                    self.clipboard_clear()
                    self.update()
                except Exception:
                    pass

                return

            # PASO 6: Si falló, ofrecer usar búsqueda por título
            print("⚠️ No se pudo capturar URL")
            fallback_response = messagebox.askyesno(
                "URL no capturada",
                f"No se pudo capturar la URL automáticamente.\n\n"
                f"Música detectada: {song_name}\n\n"
                "¿Buscar por título en YouTube?\n\n"
                "(Alternativamente, puedes copiar la URL manualmente\n"
                "y volver a hacer clic en Capturar)",
                icon='warning'
            )

            if fallback_response:
                # Usuario quiere buscar por título
                if self.app and hasattr(self.app, 'open_music_downloader'):
                    self.app.open_music_downloader(initial_query=song_name)

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error al capturar música:\n{str(e)}"
            )

    def _get_browser_url_uiautomation(self):
        """
        Captura automáticamente la URL del navegador activo usando UI Automation.
        Lee directamente de la barra de direcciones sin simular eventos de teclado.
        Soporta: Chrome, Edge, Firefox, Brave, Opera.
        """
        try:
            import subprocess

            # Script de PowerShell con UI Automation para leer URL de la barra de direcciones
            powershell_script = r"""
# Configurar salida UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

# Definir Win32 API
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Win32API {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
}
"@

try {
    # Obtener ventana activa
    $hwnd = [Win32API]::GetForegroundWindow()

    if ($hwnd -eq [IntPtr]::Zero) {
        exit 1
    }

    # Obtener título de ventana
    $title = New-Object System.Text.StringBuilder 512
    [Win32API]::GetWindowText($hwnd, $title, $title.Capacity) | Out-Null
    $windowTitle = $title.ToString()

    # Verificar si es un navegador
    $browsers = @("Chrome", "Firefox", "Edge", "Brave", "Opera", "Chromium", "YouTube")
    $isBrowser = $false
    foreach ($browser in $browsers) {
        if ($windowTitle -like "*$browser*") {
            $isBrowser = $true
            break
        }
    }

    if (-not $isBrowser) {
        exit 1
    }

    # Obtener elemento de UI Automation desde el handle
    $windowElement = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)

    if ($null -eq $windowElement) {
        exit 1
    }

    # Buscar elementos Edit (barra de direcciones)
    $editCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Edit
    )

    $elements = $windowElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        $editCondition
    )

    # Buscar la URL en los elementos Edit
    foreach ($element in $elements) {
        try {
            $valuePattern = $element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
            if ($null -ne $valuePattern) {
                $value = $valuePattern.Current.Value

                if ($value -and ($value.StartsWith("http://") -or $value.StartsWith("https://"))) {
                    Write-Output $value.Trim()
                    exit 0
                }
            }
        }
        catch {
            # Continuar
        }
    }

    exit 1
}
catch {
    exit 1
}
"""

            # Ejecutar PowerShell
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", powershell_script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0 and result.stdout.strip():
                url = result.stdout.strip()
                if url.startswith("http://") or url.startswith("https://"):
                    return url

            # Debug: mostrar stderr si hay errores
            if result.stderr:
                print(f"PowerShell stderr: {result.stderr}")

            return None

        except subprocess.TimeoutExpired:
            print("⏱️ Timeout al obtener URL del navegador")
            return None
        except Exception as e:
            print(f"❌ Error obteniendo URL del navegador: {e}")
            return None

    def _get_browser_url_keyboard(self):
        """
        Captura la URL del navegador usando automatización de teclado mejorada.
        Envía Ctrl+L y Ctrl+C para copiar la URL de la barra de direcciones.
        Preserva el clipboard original.
        """
        try:
            import subprocess

            # PowerShell script que simula las teclas y captura la URL
            powershell_script = r"""
# Script para capturar URL usando SendKeys
Add-Type -AssemblyName System.Windows.Forms

# Guardar clipboard actual
$oldClipboard = $null
try {
    $oldClipboard = [System.Windows.Forms.Clipboard]::GetText()
} catch {}

# Limpiar clipboard
[System.Windows.Forms.Clipboard]::Clear()

# Esperar un poco
Start-Sleep -Milliseconds 100

# Enviar Ctrl+L (seleccionar barra de direcciones)
[System.Windows.Forms.SendKeys]::SendWait("^l")
Start-Sleep -Milliseconds 200

# Enviar Ctrl+C (copiar)
[System.Windows.Forms.SendKeys]::SendWait("^c")
Start-Sleep -Milliseconds 300

# Obtener contenido del clipboard
$url = $null
try {
    $url = [System.Windows.Forms.Clipboard]::GetText()
} catch {}

# Restaurar clipboard original
if ($oldClipboard) {
    try {
        [System.Windows.Forms.Clipboard]::SetText($oldClipboard)
    } catch {}
}

# Devolver URL si es válida
if ($url -and ($url.StartsWith("http://") -or $url.StartsWith("https://"))) {
    Write-Output $url.Trim()
    exit 0
}

exit 1
"""

            # Ejecutar PowerShell
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", powershell_script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0 and result.stdout.strip():
                url = result.stdout.strip()
                if url.startswith("http://") or url.startswith("https://"):
                    return url

            return None

        except subprocess.TimeoutExpired:
            print("⏱️ Timeout en captura con teclado")
            return None
        except Exception as e:
            print(f"❌ Error en captura con teclado: {e}")
            return None

    def _capture_url_with_countdown(self):
        """
        Detecta el navegador reproduciendo audio y captura la URL de YouTube.
        Usa Core Audio API para detectar el proceso exacto reproduciendo audio.
        """
        try:
            import ctypes
            import time
            from ctypes import wintypes

            # Definir constantes de Win32
            VK_CONTROL = 0x11
            VK_L = 0x4C
            VK_C = 0x43
            KEYEVENTF_KEYUP = 0x0002
            CF_UNICODETEXT = 13

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # Funciones de Win32
            EnumWindows = user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            GetWindowTextW = user32.GetWindowTextW
            GetWindowTextLengthW = user32.GetWindowTextLengthW
            GetWindowThreadProcessId = user32.GetWindowThreadProcessId
            IsWindowVisible = user32.IsWindowVisible
            SetForegroundWindow = user32.SetForegroundWindow

            # Funciones de Clipboard
            OpenClipboard = user32.OpenClipboard
            OpenClipboard.argtypes = [wintypes.HWND]
            OpenClipboard.restype = wintypes.BOOL

            CloseClipboard = user32.CloseClipboard
            CloseClipboard.restype = wintypes.BOOL

            GetClipboardData = user32.GetClipboardData
            GetClipboardData.argtypes = [wintypes.UINT]
            GetClipboardData.restype = wintypes.HANDLE

            GlobalLock = kernel32.GlobalLock
            GlobalLock.argtypes = [wintypes.HGLOBAL]
            GlobalLock.restype = wintypes.LPVOID

            GlobalUnlock = kernel32.GlobalUnlock
            GlobalUnlock.argtypes = [wintypes.HGLOBAL]
            GlobalUnlock.restype = wintypes.BOOL

            # PASO 1: Detectar qué navegador está reproduciendo audio
            print("🎵 Detectando navegador reproduciendo audio...")
            browser_info = get_browser_playing_audio()

            if not browser_info:
                print("❌ No se detectó ningún navegador reproduciendo audio")
                messagebox.showwarning(
                    "Sin audio detectado",
                    "No se detectó ningún navegador reproduciendo audio.\n\n"
                    "Asegúrate de que YouTube esté reproduciendo música."
                )
                return None

            target_pid = browser_info['pid']
            browser_name = browser_info['name']
            browser_process_name = browser_name.lower().replace('.exe', '')
            print(f"✓ Detectado: {browser_name} (PID: {target_pid})")

            # PASO 2: Buscar ventanas con el NOMBRE del navegador que contengan "YouTube"
            # Nota: En navegadores multi-proceso (Chrome/Brave), el proceso que reproduce audio
            # es un renderer child que NO tiene ventanas. Las ventanas están en el proceso padre.
            print(f"🔍 Buscando ventanas de YouTube en procesos {browser_name}...")
            browser_hwnd = None
            youtube_windows = []

            # Importar psutil para obtener info de procesos
            import psutil

            def enum_windows_callback(hwnd, lParam):
                nonlocal browser_hwnd
                if IsWindowVisible(hwnd):
                    # Obtener PID de esta ventana
                    window_pid = ctypes.c_ulong()
                    GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))

                    # Verificar si el proceso de esta ventana es el mismo navegador
                    try:
                        process = psutil.Process(window_pid.value)
                        process_name = process.name().lower().replace('.exe', '')

                        # Solo procesar ventanas del mismo navegador (no importa el PID exacto)
                        if process_name == browser_process_name:
                            length = GetWindowTextLengthW(hwnd)
                            if length > 0:
                                buffer = ctypes.create_unicode_buffer(length + 1)
                                GetWindowTextW(hwnd, buffer, length + 1)
                                title = buffer.value

                                # Buscar "YouTube" en el título
                                if "youtube" in title.lower():
                                    print(f"✓ YouTube encontrado: {title[:60]}...")
                                    youtube_windows.append((hwnd, title))
                                    if not browser_hwnd:
                                        browser_hwnd = hwnd
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                return True  # Continuar enumeración

            # Enumerar todas las ventanas (primer intento)
            EnumWindows(EnumWindowsProc(enum_windows_callback), 0)

            # Si no se encontró YouTube, pedir al usuario que active la pestaña
            if not browser_hwnd:
                print("⚠️ No se encontró 'YouTube' en el título de la ventana activa")

                response = messagebox.askyesno(
                    "Activar pestaña de YouTube",
                    f"Se detectó audio en {browser_name}, pero la pestaña con YouTube\n"
                    "no está activa.\n\n"
                    "INSTRUCCIONES:\n"
                    "1. Ve a tu navegador\n"
                    "2. Haz click en la PESTAÑA de YouTube\n"
                    "3. Vuelve aquí y presiona 'Sí'\n\n"
                    "¿Activaste la pestaña de YouTube?",
                    icon='question'
                )

                if not response:
                    print("⏹️ Usuario canceló la captura")
                    return None

                # Reintentar la búsqueda
                print("🔄 Reintentando búsqueda de YouTube...")
                browser_hwnd = None
                youtube_windows = []

                time.sleep(1)  # Dar tiempo al usuario
                EnumWindows(EnumWindowsProc(enum_windows_callback), 0)

                if not browser_hwnd:
                    print("❌ Aún no se detectó YouTube en la ventana activa")
                    messagebox.showerror(
                        "YouTube no detectado",
                        "No se pudo detectar YouTube en la ventana del navegador.\n\n"
                        "Verifica que:\n"
                        "• La pestaña de YouTube esté ACTIVA (visible)\n"
                        "• El título de la pestaña contenga la palabra 'YouTube'"
                    )
                    return None

            print(f"✅ Encontradas {len(youtube_windows)} ventana(s) de YouTube")

            # Dar foco a la ventana del navegador
            print("🎯 Activando ventana del navegador...")
            SetForegroundWindow(browser_hwnd)
            time.sleep(1)

            print("⌨️ Simulando Ctrl+L...")
            # Simular Ctrl+L (seleccionar barra de direcciones)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)  # Ctrl down
            time.sleep(0.05)
            user32.keybd_event(VK_L, 0, 0, 0)  # L down
            time.sleep(0.05)
            user32.keybd_event(VK_L, 0, KEYEVENTF_KEYUP, 0)  # L up
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)  # Ctrl up

            time.sleep(0.3)

            print("⌨️ Simulando Ctrl+C...")
            # Simular Ctrl+C (copiar)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)  # Ctrl down
            time.sleep(0.05)
            user32.keybd_event(VK_C, 0, 0, 0)  # C down
            time.sleep(0.05)
            user32.keybd_event(VK_C, 0, KEYEVENTF_KEYUP, 0)  # C up
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)  # Ctrl up

            time.sleep(0.5)

            # Leer URL del portapapeles usando Win32 API
            print("📋 Leyendo portapapeles con Win32...")
            url = None
            try:
                if OpenClipboard(None):
                    try:
                        handle = GetClipboardData(CF_UNICODETEXT)
                        if handle:
                            pdata = GlobalLock(handle)
                            if pdata:
                                try:
                                    url = ctypes.wstring_at(pdata)
                                    print(f"✓ Clipboard: {url[:80]}...")
                                finally:
                                    GlobalUnlock(handle)
                        else:
                            print("⚠️ Clipboard vacío")
                    finally:
                        CloseClipboard()
                else:
                    print("❌ No se pudo abrir el portapapeles")
            except Exception as e:
                print(f"❌ Error leyendo portapapeles Win32: {e}")

            # Validar URL
            if url and isinstance(url, str):
                url = url.strip()
                if url.startswith("http://") or url.startswith("https://"):
                    print(f"✅ URL válida capturada!")
                    return url
                else:
                    print(f"⚠️ No es una URL: {url[:50]}...")

            return None

        except Exception as e:
            print(f"❌ Error en captura con countdown: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _clean_music_title(self, text: str) -> str:
        """
        Limpia el título de música para mejorar la búsqueda en YouTube.
        Elimina nombres de canales, símbolos extra y texto innecesario.
        """
        if not text:
            return ""

        import re

        # Eliminar texto después de símbolos comunes de separación
        separators = ["|", "//", "【", "】", "[", "]", "(Official", "(official"]
        for sep in separators:
            if sep in text:
                text = text.split(sep)[0]

        # Si el título empieza con "Nombre - ", eliminar el nombre del canal
        # Ejemplo: "Sheer's HD Video & Music - ZERO" -> "ZERO"
        if " - " in text:
            parts = text.split(" - ", 1)
            # Si la primera parte parece un nombre de canal (más de 15 caracteres o contiene palabras clave)
            channel_keywords = ["video", "music", "official", "records", "entertainment", "media", "hd", "vevo"]
            if len(parts[0]) > 20 or any(keyword in parts[0].lower() for keyword in channel_keywords):
                text = parts[1]

        # Eliminar sufijos comunes
        suffixes = [
            "- Topic", "- Official", "VEVO", "(Audio)", "(Video)", "(Lyrics)",
            "(Official Music Video)", "(Official Video)", "(Official Audio)"
        ]
        for suffix in suffixes:
            text = re.sub(re.escape(suffix), "", text, flags=re.IGNORECASE)

        # Limpiar espacios múltiples y caracteres extra
        text = re.sub(r'\s+', ' ', text)
        text = text.strip(" -/|#")

        return text

    def _get_windows_media_info(self):
        """Obtiene información de reproducción de Windows Media Session usando PowerShell"""
        try:
            import subprocess
            import json

            # Script de PowerShell para obtener información de reproducción con UTF-8
            powershell_script = """
            # Configurar salida UTF-8
            [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
            $OutputEncoding = [System.Text.Encoding]::UTF8

            Add-Type -AssemblyName System.Runtime.WindowsRuntime
            $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

            Function Await($WinRtTask, $ResultType) {
                $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
                $netTask = $asTask.Invoke($null, @($WinRtTask))
                $netTask.Wait(-1) | Out-Null
                $netTask.Result
            }

            [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager,Windows.Media.Control,ContentType=WindowsRuntime] | Out-Null
            $sessionManager = Await ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])

            $session = $sessionManager.GetCurrentSession()
            if ($session) {
                $mediaProperties = Await ($session.TryGetMediaPropertiesAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties])

                $result = @{
                    title = $mediaProperties.Title
                    artist = $mediaProperties.Artist
                    album = $mediaProperties.AlbumTitle
                }

                $result | ConvertTo-Json -Compress
            }
            """

            # Ejecutar PowerShell con encoding UTF-8
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", powershell_script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON result
                media_info = json.loads(result.stdout.strip())

                # Verificar que tenga datos válidos
                if media_info.get("title") or media_info.get("artist"):
                    return media_info
                else:
                    return None
            else:
                return None

        except subprocess.TimeoutExpired:
            print("Timeout al obtener información de media")
            return None
        except json.JSONDecodeError:
            print("Error al parsear respuesta de PowerShell")
            return None
        except FileNotFoundError:
            messagebox.showwarning(
                "PowerShell no encontrado",
                "No se pudo encontrar PowerShell en tu sistema."
            )
            return None
        except Exception as e:
            print(f"Error obteniendo media info: {e}")
            return None

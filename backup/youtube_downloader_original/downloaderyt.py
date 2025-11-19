#!/usr/bin/env python3
import os, shutil, glob, threading, hashlib, requests, tkinter as tk, time, logging, subprocess
from tkinter import ttk, messagebox, filedialog
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadCancelled

# ===== Logging =====
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mini-dl")
fh = logging.FileHandler("downloader.log", encoding="utf-8")
fh.setLevel(LOG_LEVEL)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(fh)

# ---- FFmpeg (pon la carpeta si no está en PATH) ----
FFMPEG_DIR = r""  # ej: r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin"

# ===== Utils =====
def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()

def fmt_bytes(n):
    if n is None: return "?"
    units = ["B","KiB","MiB","GiB","TiB"]; i = 0; n = float(n)
    while n >= 1024 and i < len(units)-1: n /= 1024; i += 1
    return f"{n:.2f}{units[i]}"

def fmt_rate(bps): return f"{fmt_bytes(bps)}/s" if bps else ""
def fmt_eta(seconds):
    if not seconds: return ""
    m, s = divmod(int(seconds), 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def build_format_for_quality(q: str) -> str:
    """Intenta altura exacta y luego <=altura. Permisivo en contenedor/códec."""
    limits = {"Mejor": None, "1080p": 1080, "720p": 720, "480p": 480, "360p": 360}
    h = limits.get(q, None)
    if h is None:
        return "bv*+ba/b"  # mejor disponible (cualquier códec/contenedor)
    return (
        f"bv*[height={h}]+ba/"      # exacto
        f"bv*[height<={h}]+ba/"     # <=H
        f"b[height<={h}]"           # single-file <=H
    )

def cleanup_residuals(final_or_base_path: str):
    deleted = []
    base, _ = os.path.splitext(final_or_base_path)
    for p in glob.glob(base + ".*"):
        ext = os.path.splitext(p)[1].lower()
        if ext in (".webm", ".m4a", ".part", ".temp", ".ytdl"):
            try: os.remove(p); deleted.append(p)
            except Exception as e: log.debug(f"No se pudo borrar {p}: {e}")
    return deleted

def ffmpeg_path():
    if FFMPEG_DIR and os.path.exists(FFMPEG_DIR):
        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        return os.path.join(FFMPEG_DIR, exe)
    return shutil.which("ffmpeg")

def next_output_path(base_no_ext: str, ext: str, overwrite: bool) -> str:
    out_path = f"{base_no_ext}{ext}"
    if overwrite:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return out_path
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base_no_ext} ({i}){ext}"
        i += 1
    return out_path

def list_formats(url):
    opts = {"quiet": True, "no_warnings": True, "logger": GuiLogger()}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        fmts = info.get("formats") or []
        log.info(f"=== {info.get('title')} · {len(fmts)} formatos ===")
        for f in fmts:
            log.info(f"itag={f.get('format_id')} "
                     f"{f.get('ext')} {f.get('vcodec')}+{f.get('acodec')} "
                     f"{f.get('height')}p proto={f.get('protocol')} "
                     f"br={f.get('tbr')} fps={f.get('fps')} drm={f.get('has_drm')}")

# ===== yt-dlp logger =====
class GuiLogger:
    def debug(self, msg): log.debug(msg)
    def warning(self, msg): log.warning(msg)
    def error(self, msg):
        s = str(msg)
        if "Got error: HTTP Error 403" in s:
            log.debug("Fragmento 403 (ignorado)"); return
        log.error(s)

# ===== Conversión a MP3 con cancelación =====
def convert_file_to_mp3(src_path, status_var, btn_convert, btn_cancel, cancel_ev: threading.Event,
                        overwrite: bool, conversion_active: threading.Event):
    try:
        if not src_path or not os.path.exists(src_path):
            messagebox.showwarning("Archivo", "Selecciona un archivo válido."); return
        if src_path.lower().endswith(".mp3"):
            messagebox.showinfo("Conversión", "El archivo ya es MP3."); return

        ff = ffmpeg_path()
        if not ff:
            messagebox.showerror("FFmpeg", "FFmpeg no encontrado. Configura FFMPEG_DIR o PATH."); return

        base, _ = os.path.splitext(src_path)
        out_path = next_output_path(base, ".mp3", overwrite)

        status_var.set("Convirtiendo a MP3…")
        log.info(f"FFmpeg convert: '{src_path}' → '{out_path}'")

        cmd = [ff, "-hide_banner", "-nostdin", "-y", "-i", src_path, "-vn",
               "-acodec", "libmp3lame", "-b:a", "192k", out_path]
        btn_convert["state"] = "disabled"; btn_cancel["state"] = "normal"

        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )

        while True:
            if cancel_ev.is_set():
                status_var.set("Cancelando…")
                try:
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill()
                except Exception: pass
                if os.path.exists(out_path):
                    try: os.remove(out_path); log.debug(f"Borrado parcial: {out_path}")
                    except: pass
                messagebox.showinfo("Cancelado", "Conversión cancelada.")
                log.info("Conversión cancelada.")
                return

            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                continue

            s = line.decode("utf-8", errors="replace")
            if "time=" in s:
                try:
                    t = s.split("time=")[1].split()[0]
                    status_var.set(f"Convirtiendo… (t={t})")
                except Exception:
                    pass

        rc = proc.returncode
        if rc == 0:
            status_var.set("✔ Conversión completa")
            messagebox.showinfo("Conversión",
                                f"MP3 creado:\n{out_path}\nTamaño: {fmt_bytes(os.path.getsize(out_path))}")
            log.info(f"MP3 final: {out_path} · {fmt_bytes(os.path.getsize(out_path))}")
        else:
            err_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-600:]
            status_var.set("Error en conversión")
            messagebox.showerror("FFmpeg", f"Falló conversión (rc={rc}).\n{err_tail}")
            log.error(f"FFmpeg rc={rc}\n{err_tail}")
    finally:
        btn_convert["state"] = "normal"; btn_cancel["state"] = "disabled"
        cancel_ev.clear(); conversion_active.clear()

# ===== Descarga con cancelación =====
def download(url, audio_only, quality, str_var, pct_var, btn_download, btn_cancel, cancel_ev: threading.Event):
    btn_download["state"] = "disabled"; btn_cancel["state"] = "normal"

    ffmpeg_ok = bool(ffmpeg_path())
    log.info(f"URL: {url}")
    log.info(f"Modo: {'MP3' if audio_only else 'Video'} · Calidad: {quality}")
    log.info(f"FFmpeg: {'OK' if ffmpeg_ok else 'NO DETECTADO'}")

    last_log = 0.0
    last_tmpfile = None

    def progress_hook(d):
        nonlocal last_log, last_tmpfile
        if cancel_ev.is_set():
            raise DownloadCancelled("Cancelado por el usuario")
        status = d.get("status")
        if status == "downloading":
            last_tmpfile = d.get("tmpfilename") or d.get("filename") or last_tmpfile
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            speed = d.get("speed"); eta = d.get("eta"); now = time.time()

            if total:
                pct = downloaded * 100.0 / total
                str_var.set(f"{pct:.1f}% · {fmt_rate(speed)} · ETA {fmt_eta(eta)}")
                pct_var.set(pct)
            else:
                str_var.set(f"{fmt_bytes(downloaded)} · {fmt_rate(speed)}")

            if now - last_log >= 1.0:
                if total:
                    log.info(f"Progreso: {pct:.1f}% · {fmt_rate(speed)} · ETA {fmt_eta(eta)}")
                else:
                    log.info(f"Descargado: {fmt_bytes(downloaded)} · {fmt_rate(speed)}")
                last_log = now
        elif status == "finished":
            str_var.set("Procesando…"); log.info("Descarga finalizada, procesando/merging…")

    # Construcción de opciones (sin impersonate)
    ydl_opts = {
        "outtmpl": "%(title)s.%(ext)s",
        "progress_hooks": [progress_hook],
        "quiet": True, "no_warnings": True,
        "prefer_ffmpeg": True, "keepvideo": False, "keep_fragments": False,
        "retries": 10, "fragment_retries": 10,
        "logger": GuiLogger(),
        # 1) web_safari (HLS) -> web
        "extractor_args": {"youtube": {"player_client": ["web_safari", "web"]}},
        # 2) Prioridad de protocolos y calidad
        "format_sort": [
            "proto:m3u8,m3u8_native,https",
            "res",
            "br",
            "codec:h264:vp9:av01",
            "ext:mp4:webm:m4a"
        ],
        "format_sort_force": True,
        "check_formats": "selected",
        # No forzamos merge_output_format; dejamos que yt-dlp decida (evita líos con Opus)
    }

    # PO Token opcional (si lo defines, se usará mweb primero)
    po_token = os.getenv("YT_PO_TOKEN")
    if po_token:
        ydl_opts["extractor_args"]["youtube"]["player_client"] = ["mweb", "web_safari", "web"]
        ydl_opts["extractor_args"]["youtube"]["po_token"] = po_token

    if ffmpeg_ok and FFMPEG_DIR:
        ydl_opts["ffmpeg_location"] = FFMPEG_DIR

    # Selección de formato
    if audio_only:
        ydl_opts |= {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }
    else:
        fmt = build_format_for_quality(quality)
        ydl_opts |= {"format": fmt}
        log.info(f"Selector de formato: {fmt}")

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if audio_only:
                local = f"{info['title']}.mp3"
            else:
                local = ydl.prepare_filename(info)
                if not os.path.exists(local):
                    base, _ = os.path.splitext(local)
                    for ext in (".mp4", ".mkv", ".webm"):
                        cand = base + ext
                        if os.path.exists(cand):
                            local = cand; break

        deleted = cleanup_residuals(local)
        for p in deleted: log.debug(f"Borrado residuo: {p}")

        size = os.path.getsize(local); sha = sha256sum(local)
        log.info(f"Archivo final: {local}")
        log.info(f"Tamaño final: {fmt_bytes(size)} · SHA-256: {sha}")

        str_var.set("✔ Descarga completa"); pct_var.set(100)
        messagebox.showinfo("Completado", f"Archivo: {local}\nSHA-256: {sha}\nTamaño: {fmt_bytes(size)}")
    except DownloadCancelled:
        str_var.set("Cancelado"); pct_var.set(0); log.info("Descarga cancelada por el usuario.")
        if last_tmpfile and os.path.exists(last_tmpfile):
            try: os.remove(last_tmpfile); log.debug(f"Borrado tmp: {last_tmpfile}")
            except: pass
        if last_tmpfile: cleanup_residuals(last_tmpfile)
        messagebox.showinfo("Cancelado", "Descarga cancelada y archivos temporales eliminados.")
    except Exception as e:
        # Fallback: best single-file
        if "Requested format is not available" in str(e):
            try:
                log.warning("Reintentando con selector genérico 'b'…")
                ydl_opts_fallback = dict(ydl_opts)
                ydl_opts_fallback.pop("format_sort", None)
                ydl_opts_fallback.pop("format_sort_force", None)
                ydl_opts_fallback["format"] = "b"
                with YoutubeDL(ydl_opts_fallback) as ydl2:
                    info = ydl2.extract_info(url, download=True)
                    local = ydl2.prepare_filename(info)
                    if not os.path.exists(local):
                        base, _ = os.path.splitext(local)
                        for ext in (".mp4", ".mkv", ".webm"):
                            cand = base + ext
                            if os.path.exists(cand):
                                local = cand; break
                deleted = cleanup_residuals(local)
                for p in deleted: log.debug(f"Borrado residuo: {p}")
                size = os.path.getsize(local); sha = sha256sum(local)
                log.info(f"Archivo final: {local}")
                log.info(f"Tamaño final: {fmt_bytes(size)} · SHA-256: {sha}")
                str_var.set("✔ Descarga completa (fallback)"); pct_var.set(100)
                messagebox.showinfo("Completado", f"(Fallback)\nArchivo: {local}\nSHA-256: {sha}\nTamaño: {fmt_bytes(size)}")
                return
            except Exception:
                log.exception("Fallback también falló")
        log.exception("Fallo en descarga")
        messagebox.showerror("Error", str(e)); str_var.set("Error")
    finally:
        btn_download["state"] = "normal"; btn_cancel["state"] = "disabled"; cancel_ev.clear()

# ===== GUI =====
def main():
    root = tk.Tk()
    root.title("Mini-descargador YouTube")
    root.geometry("520x420"); root.resizable(False, False)

    # ---- Descarga desde URL ----
    tk.Label(root, text="URL del video:").pack(anchor="w", padx=10, pady=(12, 0))
    url_entry = tk.Entry(root, width=66); url_entry.pack(padx=10, pady=4); url_entry.focus()

    opts_frame = tk.Frame(root); opts_frame.pack(fill="x", padx=10, pady=2)
    audio_var = tk.BooleanVar(value=False)
    tk.Checkbutton(opts_frame, text="Sólo audio (MP3)", variable=audio_var).pack(side="left")

    tk.Label(root, text="Calidad de video:").pack(anchor="w", padx=10, pady=(6, 0))
    quality_var = tk.StringVar(value="Mejor")
    ttk.Combobox(root, textvariable=quality_var, state="readonly",
                 values=["Mejor", "1080p", "720p", "480p", "360p"]).pack(padx=10, pady=2, fill="x")

    status_var = tk.StringVar(value="—"); pct_var = tk.DoubleVar(value=0.0)
    ttk.Label(root, textvariable=status_var).pack(padx=10, pady=(6, 2))
    ttk.Progressbar(root, length=480, variable=pct_var, maximum=100).pack(padx=10)

    # Botones Descargar / Cancelar
    btns = tk.Frame(root); btns.pack(pady=8)
    cancel_event = threading.Event()
    def on_click_download():
        url = url_entry.get().strip()
        if not url:
            messagebox.showwarning("Falta URL", "Pega primero el enlace del video."); return
        status_var.set("Preparando…"); pct_var.set(0)
        threading.Thread(target=download,
                         args=(url, audio_var.get(), quality_var.get(), status_var, pct_var,
                               btn_download, btn_cancel, cancel_event),
                         daemon=True).start()
    def on_click_cancel():
        cancel_event.set(); btn_cancel["state"] = "disabled"

    btn_download = ttk.Button(btns, text="Descargar", command=on_click_download, width=20)
    btn_cancel   = ttk.Button(btns, text="Cancelar",  command=on_click_cancel, state="disabled", width=20)
    btn_download.grid(row=0, column=0, padx=6); btn_cancel.grid(row=0, column=1, padx=6)

    # ---- Conversión local a MP3 ----
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=8)
    tk.Label(root, text="Convertir archivo local a MP3").pack(anchor="w", padx=10)

    conv_frame = tk.Frame(root); conv_frame.pack(fill="x", padx=10, pady=4)
    pick_var = tk.StringVar(value="")
    tk.Entry(conv_frame, textvariable=pick_var, width=66, state="readonly").pack(side="left", padx=(0,6))
    def pick_file():
        path = filedialog.askopenfilename(
            title="Selecciona un archivo de video/audio",
            filetypes=[("Video/Audio", "*.mp4;*.mkv;*.webm;*.mov;*.m4a;*.wav;*.flac;*.aac;*.opus;*.wma;*.avi"),
                       ("Todos", "*.*")]
        )
        if path: pick_var.set(path)
    ttk.Button(conv_frame, text="Elegir…", command=pick_file).pack(side="left")

    overwrite_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Sobrescribir si existe", variable=overwrite_var)\
        .pack(anchor="w", padx=10, pady=(4, 2))

    conv_status = tk.StringVar(value="—")
    ttk.Label(root, textvariable=conv_status).pack(anchor="w", padx=10, pady=(2,2))

    conv_btns = tk.Frame(root); conv_btns.pack(pady=4)
    conv_cancel_ev   = threading.Event()
    conversion_active = threading.Event()
    def on_convert():
        if conversion_active.is_set():
            messagebox.showinfo("Conversión", "Ya hay una conversión en curso."); return
        path = pick_var.get().strip()
        conversion_active.set()
        btn_convert["state"] = "disabled"
        threading.Thread(target=convert_file_to_mp3,
                         args=(path, conv_status, btn_convert, btn_cancel_conv,
                               conv_cancel_ev, overwrite_var.get(), conversion_active),
                         daemon=True).start()
    def on_cancel_convert():
        conv_cancel_ev.set(); btn_cancel_conv["state"] = "disabled"

    btn_convert     = ttk.Button(conv_btns, text="Convertir a MP3", command=on_convert, width=20)
    btn_cancel_conv = ttk.Button(conv_btns, text="Cancelar conversión",
                                 command=on_cancel_convert, state="disabled", width=20)
    btn_convert.grid(row=0, column=0, padx=6); btn_cancel_conv.grid(row=0, column=1, padx=6)

    root.mainloop()

if __name__ == "__main__":
    main()

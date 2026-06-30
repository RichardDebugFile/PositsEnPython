# Roadmap / Cosas por hacer — Posits Virtuales

Backlog de mejoras y nuevas funciones. Cada ítem indica objetivo, enfoque y,
cuando aplica, el procedimiento técnico para implementarlo.

> Convención de ramas: GitFlow (ver [CONTRIBUTING.md](CONTRIBUTING.md)). Cada
> ítem se desarrolla en una rama `feature/<descripcion>` desde `develop`.

---

## 🟢 Próxima función — Monitoreo de tiempo de uso por aplicación

**Objetivo:** que Posits se conecte con el sistema para medir cuánto tiempo pasa
el usuario en cada aplicación (p. ej. navegador, editor, juego) y mostrar
estadísticas diarias/semanales. Útil para productividad y para alimentar la
gamificación (XP por foco, alertas de distracción).

### Enfoque técnico (Windows)

La app ya es de escritorio Windows y usa `psutil`, `pywin32` y `ctypes`
(ver `src/ui/music_panel.py` y `src/utils/audio_detection.py`), así que no hacen
falta dependencias nuevas. La idea: **sondear periódicamente la ventana en primer
plano**, resolver su proceso y acumular segundos por aplicación, descartando el
tiempo en que el usuario está inactivo.

Piezas clave de la API de Windows:
- `user32.GetForegroundWindow()` → handle de la ventana activa.
- `user32.GetWindowThreadProcessId(hwnd)` → PID de esa ventana.
- `psutil.Process(pid).name()` → nombre del ejecutable (p. ej. `chrome.exe`).
- `user32.GetLastInputInfo()` → milisegundos desde la última entrada (teclado/
  ratón) para detectar inactividad (idle).

### Procedimiento

1. **Servicio de seguimiento** — crear `src/services/usage_tracker.py` con una
   clase `UsageTracker` modelada como `NotificationService`
   (`src/services/notifications.py`): un hilo *daemon* con `start()`/`stop()`.
   - Cada ~5 s: obtener proceso en primer plano y, si el usuario NO está idle
     (umbral configurable, p. ej. 60 s sin input), sumar el intervalo al
     contador del proceso actual.
   - Mapear nombre de proceso → nombre legible (tabla opcional:
     `chrome.exe → Chrome`).

2. **Persistencia** — guardar en `data/usage_stats.json` con forma
   `{"YYYY-MM-DD": {"chrome.exe": 3600, "Code.exe": 5400, ...}}` (segundos por
   app por día). Reutilizar el patrón de escritura con throttling de
   `TaskStore.save_throttled` para no escribir en cada tick.

3. **Modelo/consultas** — añadir helpers para leer agregados: total por día,
   top-N apps, rango de fechas (para el calendario/heatmap, igual que
   `GamificationManager.get_xp_for_date_range`).

4. **UI** — nueva pestaña o panel "Tiempo de uso":
   - Lista/!barra horizontal con el top de apps del día y su tiempo.
   - Selector de día/semana; opción de exportar.
   - Integrar en el `Notebook` de `src/app.py` (ya hay tabs Tareas/Calendario).

5. **Arranque/cierre** — iniciar `UsageTracker` en `ModernStickyApp.__init__`
   (junto a `NotificationService`) y detenerlo en `_on_closing` (ya existe el
   handler de cierre limpio).

6. **Configuración** — intervalo de sondeo, umbral de idle, lista de apps a
   ignorar; persistir en `data/` como el resto de configs.

### Consideraciones
- **Privacidad:** todo es local; no se envía nada fuera del equipo. Documentarlo.
- **Rendimiento:** sondeo cada 5 s es despreciable; evitar trabajo pesado en el
  hilo. No registrar títulos de ventana por defecto (solo proceso) para no
  guardar datos sensibles.
- **Idle/bloqueo:** no contar tiempo con la sesión bloqueada o sin input.
- **Multiplataforma (futuro):** en Linux/macOS la API difiere; encapsular la
  parte específica de Windows tras una interfaz para poder portarla.

### Criterios de aceptación
- [ ] El tracker acumula tiempo correcto por app durante una sesión real.
- [ ] No cuenta tiempo en estado idle.
- [ ] Los datos persisten y sobreviven a reinicios.
- [ ] La UI muestra el top de apps del día.
- [ ] Tests unitarios de la agregación (sin depender de la API de Windows:
      inyectar el "proceso activo" y el "idle" para poder testear la lógica).

---

## 🟡 Pendientes menores (detectados durante el trabajo reciente)

- **Tema claro más vibrante (Ola 4):** subir contraste, sombras/tarjetas y
  acentos en la paleta (`src/config.py`: `MODERN_COLORS`/`GRADIENTS`). Iterativo;
  requiere validación visual.
- **Posits fuera de pantalla:** posiciones guardadas con coordenadas de un
  monitor desconectado (x negativos) quedan invisibles. Evaluar un *clamp*
  seguro al área visible (cuidado con multi-monitor) o reforzar el botón de
  "reset de posiciones".
- **Barras de carga restantes:** escaneo inicial de música
  (`MusicPlayer.scan_music_folder`) y arranque de la app (splash) aún sin
  feedback visual; ya existe `LoadingOverlay` reutilizable.
- **Smoke-test de descarga en CI:** opcional, un test marcado que verifique
  `get_video_info` contra YouTube (red), separado de la suite por defecto.

---

## ✅ Hecho recientemente
- Capas/arranque: ventana principal sin always-on-top, sin "flash negro",
  cierre limpio.
- Suite de tests (pytest) + CI en GitHub Actions; GitFlow.
- Ollama de un clic (auto-arranque + barra de carga); posits redimensionables.
- Pomodoro con anillo de progreso por fase; toolbar agrupada con tooltips.
- Autostart robusto (relanza con el Python del venv).

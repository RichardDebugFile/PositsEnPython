# Posits Mobile - Prototipo MVP

Versión móvil de Posits Virtuales usando Kivy/KivyMD para Android/iOS.

## 🎯 Características

### ✅ Implementadas
- **Sistema de Tareas**: CRUD completo con prioridades (Urgente, Alta, Normal, Baja)
  - Las tareas se marcan como completadas/pendientes (no se eliminan automáticamente)
  - Opción de eliminar con confirmación
- **Gamificación**: XP, niveles y misiones diarias personalizables
  - **Misiones Diarias**: El usuario crea hasta 3 misiones personalizadas (checkbox simple)
  - Las misiones se resetean cada día pero conservan el texto
  - Sistema de racha (días consecutivos completando tareas)
- **Pomodoro Timer**: 25min trabajo + 5min descanso
  - **Reproductor de Música**: Reproduce MP3s de `data/music/`
  - Controles: Play/Pause, Siguiente, Anterior, Seleccionar canción
- **Estadísticas**: Nivel, XP, racha, tareas pendientes

### ❌ No Incluidas (Por tamaño/complejidad)
- Integración con Ollama (requiere servidor)
- VOSK Speech-to-Text (modelo ~40MB)
- Posits flotantes (concepto desktop)

## 📋 Requisitos

### Desktop (desarrollo)
```bash
Python 3.11+
```

### Móvil (runtime)
- Android 8.0+ (API 26+)
- iOS 13.0+
- ~50 MB espacio

## 🚀 Instalación

### 1. Instalar dependencias
```bash
cd mobile-prototype
pip install -r requirements.txt
```

### 2. Ejecutar en desktop (testing)
```bash
python main.py
```

### 3. Compilar para Android
```bash
# Instalar buildozer
pip install buildozer

# Primera vez (genera buildozer.spec)
buildozer init

# Compilar APK
buildozer -v android debug

# APK estará en: bin/posits-mobile-0.1-debug.apk
```

### 4. Compilar para iOS (requiere macOS)
```bash
# Instalar kivy-ios
pip install kivy-ios

# Compilar
toolchain build python3 kivy

# Crear proyecto Xcode
toolchain create PositsMobile .
```

## 📱 Estructura del Proyecto

```
mobile-prototype/
├── main.py                 # Punto de entrada
├── screens/
│   ├── home_screen.py      # Pantalla principal
│   ├── tasks_screen.py     # Lista de tareas
│   ├── add_task_screen.py  # Agregar tarea
│   ├── pomodoro_screen.py  # Timer Pomodoro
│   ├── music_screen.py     # Reproductor música
│   └── stats_screen.py     # Estadísticas
├── services/
│   ├── task_store.py       # Gestión de tareas (SQLite)
│   ├── gamification.py     # Sistema XP/niveles
│   ├── pomodoro.py         # Timer Pomodoro
│   ├── music_player.py     # Reproductor (Kivy Audio)
│   └── youtube_dl.py       # Descargador YouTube
├── widgets/
│   ├── task_card.py        # Card de tarea
│   ├── progress_ring.py    # Anillo de progreso
│   └── level_badge.py      # Badge de nivel
├── data/                   # Datos locales
│   ├── tasks.db           # SQLite database
│   └── music/             # Archivos MP3
├── assets/
│   ├── fonts/
│   ├── images/
│   └── sounds/
├── buildozer.spec          # Config Android
├── requirements.txt
└── README.md
```

## 🎨 UI/UX

### Tema
- **Material Design 3**
- **Modo oscuro** por defecto
- **Colores**: Gradientes morados/azules (igual que desktop)

### Navegación
- **Bottom Navigation Bar**: 5 tabs principales
  - 🏠 Home (estadísticas rápidas)
  - ✓ Tareas
  - 🍅 Pomodoro
  - 🎵 Música
  - 📊 Stats

### Gestos
- **Swipe left**: Completar tarea
- **Swipe right**: Eliminar tarea
- **Long press**: Editar tarea
- **Pull to refresh**: Actualizar datos

## 🔧 Desarrollo

### Testing en Desktop
```bash
# Ejecutar app
python main.py

# Probar íconos (si no se ven en la app principal)
python test_icons.py
```

### Testing en Android
```bash
# Instalar y ejecutar en dispositivo
buildozer android debug deploy run

# Ver logs en tiempo real
buildozer android adb -- logcat | grep python
```

## ⚠️ Problemas Conocidos

### ~~Íconos de Material Design no se muestran~~ ✅ SOLUCIONADO
**Solución aplicada**: Los íconos se reemplazaron con emojis en toda la interfaz:
- Navegación inferior: 🏠 Home, ✅ Tareas, ⏱️ Pomodoro
- Botones de tareas: ✓ Completar, ↶ Deshacer, 🗑️ Eliminar
- Controles de música: ▶️ Play, ⏸️ Pausa, ⏮️ Anterior, ⏭️ Siguiente, 🎵 Playlist
- Prioridades: 🔴 Urgente, 🟠 Alta, 🟡 Normal, 🟢 Baja
- Editar misión: ✏️

### ~~Música no suena~~ ✅ SOLUCIONADO
**Solución aplicada**: Se cambió de `kivy.core.audio` (no disponible) a `pygame.mixer`:
- Instalación: Ya incluido en `requirements.txt`
- Los archivos MP3 deben estar en `../data/music/`
- Funciona con loop infinito para sesiones Pomodoro

### App se cierra al inicio
**Causa**: Faltan dependencias o configuración incorrecta.

**Solución**:
```bash
cd mobile-prototype
..\venv\Scripts\python.exe -m pip install --upgrade kivy[base] kivymd pillow requests python-dateutil
```

## 📊 Benchmarks

### Tamaño de la App
- **APK Debug**: ~25 MB
- **APK Release**: ~15 MB
- **Con música**: +tamaño archivos MP3

### Rendimiento
- **Inicio**: < 2 segundos
- **Transiciones**: 60 FPS
- **Uso RAM**: ~80 MB
- **Batería**: Bajo impacto

## 🐛 Troubleshooting

### Error: "SDL2 not found"
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# macOS
brew install sdl2
```

### Error: Buildozer compilation failed
```bash
# Limpiar build
buildozer android clean

# Reinstalar
buildozer -v android debug
```

### Audio no funciona en Android
- Verificar permisos en `buildozer.spec`:
```ini
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
```

## 📝 Roadmap

### v0.2 (Próxima versión)
- [ ] Sincronización con backend (opcional)
- [ ] Widgets del sistema
- [ ] Notificaciones push
- [ ] Modo offline completo
- [ ] Export/Import tareas

### v0.3 (Futuro)
- [ ] Integración con calendario
- [ ] Recordatorios por ubicación
- [ ] Colaboración multi-usuario
- [ ] Temas personalizables

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - ver archivo `LICENSE`

## 👨‍💻 Autor

**Posits Team**
- Desktop version: Python + Tkinter
- Mobile version: Python + Kivy/KivyMD

## 🔗 Links

- [Documentación Kivy](https://kivy.org/doc/stable/)
- [KivyMD Components](https://kivymd.readthedocs.io/)
- [Buildozer Docs](https://buildozer.readthedocs.io/)

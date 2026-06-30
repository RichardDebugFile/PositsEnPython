# ✨ Posits Virtuales v2.6

Aplicación multiplataforma de gestión de tareas con sistema de gamificación integrado, modo Pomodoro y versión móvil.

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)
![Kivy](https://img.shields.io/badge/Kivy-Mobile-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Características

### Versión de Escritorio
- 🎮 **Sistema de Gamificación Progresivo**: Gana XP, sube de nivel y completa misiones diarias personalizables
- 📌 **Posits Flotantes Automáticos**: Misiones diarias se abren como posits siempre visibles en tu escritorio
- 📐 **Posits Redimensionables**: Arrastra el grip de la esquina para cambiar el tamaño; recuerda posición **y** tamaño
- 💾 **Memoria de Posiciones**: Los posits recuerdan dónde los colocaste
- 🔄 **Reset de Posiciones**: Botón para reiniciar posiciones si quedan en lugares incómodos
- 🍅 **Modo Pomodoro Flotante**: Temporizador con **anillo de progreso** y color por fase (trabajo/descanso), cola de tareas y música integrada
- 🎵 **Reproductor de Música**: Control de música para sesiones de trabajo productivas
- 🎧 **Capturar Música en Reproducción**: Detecta el video de YouTube que suena (vía CDP) y abre el descargador con la URL lista
- 📎 **Archivos Adjuntos**: Adjunta PDFs, imágenes, documentos a tus tareas
- 🔔 **Notificaciones Automáticas**: Alertas para tareas próximas a vencer
- 🎨 **8 Colores Vibrantes**: Organiza visualmente tus tareas
- 🤖 **IA de un Clic (Ollama)**: Crea tareas desde lenguaje natural; **detecta/arranca Ollama** solo y muestra barra de carga (sin congelar la app)
- 🎙️ **Dictado por Voz**: Transcripción de audio a texto con barra de carga
- 📊 **Ordenamiento Avanzado**: Por fecha, prioridad, título o color
- 🧭 **Barra de Herramientas Agrupada**: Acciones por categoría con *tooltips*
- 🚀 **Inicio Automático con Windows**: Arranca con tu PC, usando siempre el entorno virtual del proyecto

### Versión Móvil (Nueva!)
- 📱 **Interfaz Móvil con KivyMD**: Aplicación completa para dispositivos móviles
- 🎯 **Gestión de Tareas**: Crea, edita, completa y elimina tareas desde tu móvil
- ✨ **Misiones Diarias Editables**: 3 slots para misiones personalizadas con recompensas XP
- 🔥 **Sistema de Rachas**: Mantén tu productividad día a día
- 🍅 **Pomodoro con Cola de Tareas**: Agrega tareas al Pomodoro y gestiónalas durante tus sesiones
- 🎶 **Reproductor de Música Integrado**: Control de música con pygame durante Pomodoro
- 📊 **Estadísticas en Tiempo Real**: Nivel, XP, tareas pendientes y racha actual
- 🎨 **Prioridades con Colores**: Sistema visual de prioridades (Urgente, Alta, Normal, Baja)
- 🌙 **Tema Oscuro**: Interfaz optimizada para la vista

## 🚀 Instalación

### Requisitos
- Python 3.13 o superior
- pip

### Instalación rápida - Versión de Escritorio

```bash
# Clonar repositorio
git clone https://github.com/RichardDebugFile/PositsEnPython.git
cd PositsEnPython

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # En Windows
source venv/bin/activate  # En Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python main.py
```

### Instalación - Versión Móvil

```bash
# Navegar a la carpeta móvil
cd mobile-prototype

# Instalar dependencias móviles
pip install -r requirements.txt

# Ejecutar aplicación móvil
python main.py
```

### Dependencias opcionales

Para funcionalidades avanzadas:

```bash
# Dictado por voz (Escritorio)
pip install -r requirements_voice.txt

# Integración con IA (Ollama)
pip install ollama

# Inicio automático en Windows
pip install pywin32 winshell

# Descarga de música de YouTube (Opcional para móvil)
pip install yt-dlp
```

## 🚀 Configurar Inicio Automático

Para que **Posits Virtuales** se inicie automáticamente con Windows:

```bash
# Opción 1: doble clic
scripts\activar_inicio_automatico.bat

# Opción 2: por consola (usa el venv)
venv\Scripts\python scripts\install_startup.py
```

Selecciona la opción **1** (Activar inicio automático). Para desactivarlo, usa
`scripts\desactivar_inicio_automatico.bat`.

> El launcher (`scripts/posits_launcher.pyw`) se relanza con el Python del
> entorno virtual, de modo que el arranque automático siempre tiene todas las
> dependencias disponibles.

## 📖 Uso

### Versión de Escritorio

#### Inicio Rápido
1. Ejecuta `python main.py`
2. Haz clic en "➕ Nueva Tarea" para crear tareas
3. Completa tareas para ganar XP y subir de nivel
4. Crea y personaliza misiones diarias desde el panel de gamificación

#### Panel de Gamificación
- **Nivel y XP**: Sube de nivel completando tareas (sistema progresivo)
- **Misiones Diarias**: 3 slots para misiones personalizadas
- **Sistema de Rachas**: Mantén la racha completando al menos una tarea diaria
- **Edición de Misiones**: Crea, edita y elimina misiones con recompensas XP personalizadas

#### Modo Pomodoro
1. Haz clic en "🍅 Pomodoro" en la ventana principal
2. Agrega tareas a la cola desde la lista de tareas (drag & drop)
3. Gestiona tu música de fondo
4. Inicia el timer y trabaja en bloques de 25 minutos
5. Las tareas en cola te ayudan a mantener el foco

#### Reproductor de Música
- Carga música desde `data/music/`
- Control completo: play, pause, siguiente, anterior
- Selección de canciones desde playlist
- Volumen ajustable

#### Adjuntar Archivos
1. Abre una tarea (botón 📝)
2. Haz clic en "📎 Adjuntar"
3. Selecciona el archivo
4. Doble clic para abrir archivos adjuntos

#### Ordenamiento
Usa el dropdown "Ordenar" para organizar tareas por:
- 📅 Fecha
- ⭐ Prioridad
- 🔤 Título
- 🎨 Color

### Versión Móvil

#### Navegación
La app móvil tiene 3 pantallas principales:
- **Inicio**: Dashboard con estadísticas, misiones y racha
- **Tareas**: Lista completa de tareas con filtros por estado
- **Pomodoro**: Timer con cola de tareas y reproductor de música

#### Gestión de Tareas
1. Ve a la pantalla "Tareas"
2. Presiona "+ Agregar" para crear una nueva tarea
3. Selecciona la prioridad (Urgente, Alta, Normal, Baja)
4. Presiona el botón **"P"** para agregar la tarea al Pomodoro
5. Presiona **"OK"** para completar una tarea
6. Presiona **"X"** para eliminar

#### Misiones Diarias Móvil
1. En la pantalla "Inicio", verás 3 slots de misiones
2. Presiona **"Editar"** para crear o modificar una misión
3. Define el título y la recompensa XP (default: 50 XP)
4. Toca la misión para marcarla como completada y ganar XP
5. Las misiones se resetean automáticamente cada día

#### Pomodoro Móvil
1. Agrega tareas a la cola desde la pantalla "Tareas" (botón "P")
2. Ve a la pantalla "Pomodoro"
3. Verás tus tareas en cola en el card "Tareas en Cola"
4. Controla la música de fondo (archivos .mp3 en `data/music/`)
5. Presiona "Iniciar" para comenzar el timer
6. Toca una tarea en cola para removerla
7. Usa "Limpiar" para vaciar toda la cola

## 🏗️ Arquitectura

### Versión de Escritorio
```
PositsEnPython/
├── main.py                      # Punto de entrada
├── requirements.txt
├── CONTRIBUTING.md              # Flujo de ramas (GitFlow)
├── ROADMAP.md                   # Backlog y próximas funciones
├── pytest.ini
├── .github/workflows/ci.yml     # CI: compileall + pylint + bandit + pytest
├── src/
│   ├── app.py                   # Aplicación principal Tkinter
│   ├── config.py                # Configuración centralizada
│   ├── models/                  # task.py, store.py, gamification.py, pomodoro.py
│   ├── services/                # ollama.py, stt.py, attachments.py, notifications.py,
│   │                            #   music_player.py, youtube_downloader.py
│   ├── ui/                      # paneles, pomodoro_window.py, loading.py,
│   │                            #   components.py (PillButton, Tooltip, ...)
│   ├── dialogs/                 # add_task.py, ollama_capture.py, music_downloader.py
│   └── utils/                   # dates, colors, logger, cdp_helper,
│                                #   audio_detection, media_capture
├── scripts/
│   ├── posits_launcher.pyw      # Launcher de autostart (relanza con el venv)
│   ├── install_startup.py       # Registrar/quitar inicio automático
│   └── cdp/                     # Scripts y guías para CDP (captura de YouTube)
├── tests/                       # pytest (unitarios + integración)
│   └── manual/                  # diagnósticos manuales (red/audio/Vosk/CDP)
├── data/
│   ├── notes.json               # Base de datos de tareas
│   ├── gamification.json        # Progreso de gamificación
│   ├── posit_positions.json     # Posición y tamaño de los posits
│   ├── attachments/             # Archivos adjuntos
│   └── music/                   # Archivos MP3 para Pomodoro
└── mobile-prototype/            # Versión móvil (KivyMD)
```

### Versión Móvil
```
mobile-prototype/
├── main.py                  # Punto de entrada móvil
├── services/
│   ├── task_store.py       # Persistencia compartida
│   ├── gamification.py     # Sistema gamificación móvil
│   ├── pomodoro.py         # Timer Pomodoro con cola
│   ├── simple_music_player.py  # Reproductor pygame
│   └── youtube_downloader.py   # Descarga música (opcional)
├── data/
│   ├── tasks.json          # Base de datos tareas
│   ├── gamification.json   # Progreso y misiones
│   └── music/             # Archivos MP3
├── requirements.txt        # Dependencias móviles
└── TECHNICAL.md           # Documentación técnica
```

## 🎯 Sistema de Puntos

### Sistema de XP y Niveles

| Acción | XP Ganado | Notas |
|--------|-----------|-------|
| Tarea Baja prioridad | 5 XP | Verde |
| Tarea Normal | 10 XP | Azul |
| Tarea Alta prioridad | 20 XP | Naranja |
| Tarea Urgente | 30 XP | Rojo |
| Sesión Pomodoro | 30 XP | 25 min trabajo |
| Misión diaria | **50 XP** | Personalizable |
| **Con racha activa** | **+50% XP** | Bonus diario |

### Sistema de Niveles Progresivo
- **Nivel 1-5**: 100 XP por nivel
- **Nivel 6-10**: 150 XP por nivel
- **Nivel 11+**: 200 XP por nivel
- Cada nivel es más desafiante que el anterior

### Sistema de Rachas
- Completa al menos **1 tarea** cada día para mantener la racha
- La racha otorga **+50% XP** en todas las acciones
- Se resetea si pasas un día sin completar tareas
- Registra tu **racha más larga** como récord personal

## 📱 Características Avanzadas

### Modo Pomodoro (Escritorio y Móvil)
- **Timer 25/5**: 25 minutos trabajo, 5 minutos descanso
- **Cola de Tareas**: Organiza las tareas que trabajarás durante la sesión
- **Reproductor de Música Integrado**:
  - Carga archivos MP3 desde `data/music/`
  - Control completo: play, pause, next, previous
  - Selección desde playlist
  - Loop infinito durante trabajo
- **Estadísticas**: Sesiones completadas, tiempo total trabajado
- **Ventana Flotante** (Escritorio): Siempre visible mientras trabajas

### Misiones Diarias Personalizables
- **3 Slots de Misiones**: Crea tus propias misiones diarias
- **Recompensas XP Ajustables**: Define cuánto XP otorga cada misión
- **Reset Automático**: Las misiones se resetean cada día (mantienen el título)
- **Editor Integrado**: Crea, edita y elimina misiones desde la app
- **Posits Flotantes** (Escritorio): Las misiones se muestran como ventanas flotantes

### Notificaciones (Escritorio)
- Alertas automáticas para tareas próximas a vencer
- Revisa cada 60 segundos en segundo plano
- Compatible con Windows, macOS y Linux

### IA con Ollama (Escritorio)
- Extrae tareas desde lenguaje natural
- Analiza imágenes para crear tareas
- Requiere Ollama instalado localmente

### Dictado por Voz (Escritorio)
- Push-to-talk en editor de notas
- Transcripción automática a texto
- Soporta Faster-Whisper y Vosk

### Descarga de Música de YouTube (Opcional)
- Descarga audio de YouTube como MP3
- Sanitiza nombres de archivos automáticamente
- Guarda en `data/music/` para usar en Pomodoro
- Requiere `yt-dlp` instalado

## 🛠️ Configuración

Edita `src/config.py` para personalizar:
- Colores y temas
- Puntos XP por acción
- Número de misiones diarias
- Multiplicador de racha
- Intervalo de notificaciones

## 📚 Documentación

### General
- [Guía de contribución / GitFlow](CONTRIBUTING.md)
- [Roadmap y backlog](ROADMAP.md)
- [Instalación Detallada](docs/INSTALACION_RAPIDA.md)
- [Documentación de Migración](docs/migration/)
- [Arquitectura del Sistema](docs/REFACTORIZACION_V2.md)
- [Guía de Desarrollo](docs/development/)

### Versión Móvil
- [Documentación Técnica Móvil](mobile-prototype/TECHNICAL.md)
- [Fix de Íconos KivyMD](mobile-prototype/ICONO_FIX.md)
- [API Móvil (Propuesta)](docs/api/MOBILE_INTEGRATION.md)

### Características Específicas
- [Inicio Automático en Windows](INICIO_AUTOMATICO.md)
- Sistema de Gamificación (ver `src/models/gamification.py`)
- Modo Pomodoro (ver `src/models/pomodoro.py` y `mobile-prototype/services/pomodoro.py`)

## 🧪 Desarrollo

### Tests

Suite con **pytest** (unitarios + integración/flujo), aislada en archivos
temporales (no toca `data/`):

```bash
venv\Scripts\python -m pytest
```

Los scripts que requieren red, audio, modelos Vosk o un navegador con CDP están
en `tests/manual/` y se ejecutan a mano (no entran en la suite automática).

### Integración continua (CI)

`.github/workflows/ci.yml` corre en cada `push`/`pull request` a `main` y
`develop` sobre **windows-latest**: `compileall` + `pylint` + `bandit` + `pytest`.

### Flujo de ramas (GitFlow)

- `main`: estable/publicado · `develop`: integración · `feature/*`,
  `release/*`, `hotfix/*`: temporales.
- Detalles y comandos en **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## 🤝 Contribuir

Las contribuciones son bienvenidas:

1. Haz fork del proyecto.
2. Crea tu rama desde `develop`: `git checkout -b feature/mi-mejora`.
3. Asegúrate de que `pytest` pase.
4. Abre un Pull Request hacia `develop` (ver [CONTRIBUTING.md](CONTRIBUTING.md)).

## 📝 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 👥 Autor

**Richard** - [GitHub](https://github.com/RichardDebugFile)

## 🙏 Agradecimientos

- Comunidad de Python
- Proyecto Ollama
- Tkinter documentation
- Todos los contribuidores

---

## 🆕 Novedades en v2.6

### Estabilidad y arranque
- ✅ La ventana principal ya **no** se queda encima de todas las apps (solo los posits)
- ✅ Arranque sin "flash negro" (la ventana se muestra ya construida y pintada)
- ✅ Cierre limpio (detiene servicios en segundo plano y guarda estado)
- ✅ Inicio automático robusto: el launcher se relanza con el Python del venv

### Experiencia de uso
- ✅ Pomodoro con **anillo de progreso** y color por fase (trabajo/descanso)
- ✅ Posits **redimensionables** (grip de esquina + memoria de tamaño)
- ✅ IA (Ollama) de **un clic**: auto-arranque + barra de carga, sin congelar la UI
- ✅ Barras de carga en IA y dictado por voz (STT)
- ✅ Barra de herramientas **agrupada** con *tooltips*
- ✅ **🎧 Capturar** la música de YouTube en reproducción (vía CDP)

### Calidad e infraestructura
- ✅ Suite de **tests (pytest)** + **CI** (GitHub Actions, windows-latest)
- ✅ Flujo de ramas **GitFlow** ([CONTRIBUTING.md](CONTRIBUTING.md))
- ✅ Refactor: lógica de captura extraída a `utils/media_capture` (testeable)

## 🆕 Novedades en v2.5

### Versión Móvil Completa
- ✅ Aplicación móvil funcional con KivyMD
- ✅ Sistema de gamificación completo
- ✅ Misiones diarias editables
- ✅ Pomodoro con cola de tareas
- ✅ Reproductor de música integrado
- ✅ Sincronización de datos con versión escritorio

### Mejoras en Escritorio
- ✅ Sistema de misiones personalizable
- ✅ Ventana flotante de Pomodoro mejorada
- ✅ Reproductor de música para productividad
- ✅ Descargador de música de YouTube
- ✅ Sistema de rachas mejorado

### Próximas Características (Roadmap)
Ver **[ROADMAP.md](ROADMAP.md)** para el backlog completo y los procedimientos. Destacados:
- 🖥️ **Monitoreo de tiempo de uso por aplicación** (medir cuánto tiempo pasas en cada app)
- 🎨 Tema claro más vibrante (paleta y jerarquía visual)
- 🔄 Sincronización en tiempo real entre escritorio y móvil
- 📊 Gráficas de productividad
- 📱 Compilación APK para Android
- 🍎 Soporte para iOS

---

**Versión 2.6** - Estabilidad, UX (Pomodoro/posits/IA) y CI/CD • 2026

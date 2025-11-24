# ✨ Posits Virtuales v2.5

Aplicación multiplataforma de gestión de tareas con sistema de gamificación integrado, modo Pomodoro y versión móvil.

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)
![Kivy](https://img.shields.io/badge/Kivy-Mobile-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Características

### Versión de Escritorio
- 🎮 **Sistema de Gamificación Progresivo**: Gana XP, sube de nivel y completa misiones diarias personalizables
- 📌 **Posits Flotantes Automáticos**: Misiones diarias se abren como posits siempre visibles en tu escritorio
- 💾 **Memoria de Posiciones**: Los posits recuerdan dónde los colocaste
- 🔄 **Reset de Posiciones**: Botón para reiniciar posiciones si quedan en lugares incómodos
- 🍅 **Modo Pomodoro Flotante**: Ventana de Pomodoro con cola de tareas y reproductor de música integrado
- 🎵 **Reproductor de Música**: Control de música para sesiones de trabajo productivas
- 📎 **Archivos Adjuntos**: Adjunta PDFs, imágenes, documentos a tus tareas
- 🔔 **Notificaciones Automáticas**: Alertas para tareas próximas a vencer
- 🎨 **8 Colores Vibrantes**: Organiza visualmente tus tareas
- 🤖 **Integración con IA (Ollama)**: Crea tareas desde lenguaje natural
- 🎙️ **Dictado por Voz**: Transcripción de audio a texto
- 📊 **Ordenamiento Avanzado**: Por fecha, prioridad, título o color
- 🚀 **Inicio Automático con Windows**: Configura la app para que se abra al arrancar tu PC

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
python configurar_inicio_automatico.py
```

Selecciona la opción **1** (Activar inicio automático).

📖 **Guía completa**: Ver [INICIO_AUTOMATICO.md](INICIO_AUTOMATICO.md)

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
├── main.py                    # Punto de entrada
├── src/
│   ├── app.py                # Aplicación principal Tkinter
│   ├── config.py             # Configuración centralizada
│   ├── models/
│   │   ├── task.py          # Modelo de tarea
│   │   ├── task_store.py    # Persistencia de tareas
│   │   ├── gamification.py  # Sistema de gamificación
│   │   └── pomodoro.py      # Gestor de Pomodoro
│   ├── services/
│   │   ├── ollama_service.py      # Integración IA
│   │   ├── stt_service.py         # Speech-to-text
│   │   ├── attachment_manager.py  # Gestión de adjuntos
│   │   ├── notification_service.py # Notificaciones
│   │   ├── music_player.py        # Reproductor música
│   │   └── youtube_downloader.py  # Descarga música YouTube
│   ├── ui/
│   │   ├── components/      # Componentes reutilizables
│   │   ├── pomodoro_window.py # Ventana flotante Pomodoro
│   │   └── ...
│   ├── dialogs/             # Diálogos modales
│   └── utils/               # Utilidades
├── data/
│   ├── tasks.json           # Base de datos tareas
│   ├── gamification.json    # Progreso gamificación
│   ├── attachments/         # Archivos adjuntos
│   └── music/              # Archivos MP3 para Pomodoro
└── docs/                    # Documentación
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

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

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
- 🔄 Sincronización en tiempo real entre escritorio y móvil
- 📊 Gráficas de productividad
- 🎨 Temas personalizables
- 🌐 Exportar/importar datos
- 📱 Compilación APK para Android
- 🍎 Soporte para iOS

---

**Versión 2.5** - Versión Móvil + Pomodoro Mejorado • Noviembre 2025

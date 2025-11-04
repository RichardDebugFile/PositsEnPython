# ✨ Posits Virtuales v2.3

Aplicación de gestión de tareas con sistema de gamificación integrado y posits flotantes automáticos.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Características

- 🎮 **Sistema de Gamificación Progresivo**: Gana XP, sube de nivel (cada nivel más difícil) y completa misiones diarias
- 📌 **Posits Flotantes Automáticos**: Misiones diarias se abren como posits siempre visibles en tu escritorio
- 💾 **Memoria de Posiciones**: Los posits recuerdan dónde los colocaste
- 🔄 **Reset de Posiciones**: Botón para reiniciar posiciones si quedan en lugares incómodos
- 📎 **Archivos Adjuntos**: Adjunta PDFs, imágenes, documentos a tus tareas
- 🔔 **Notificaciones Automáticas**: Alertas para tareas próximas a vencer
- 🎨 **8 Colores Vibrantes**: Organiza visualmente tus tareas
- 🤖 **Integración con IA (Ollama)**: Crea tareas desde lenguaje natural
- 🎙️ **Dictado por Voz**: Transcripción de audio a texto
- 📊 **Ordenamiento Avanzado**: Por fecha, prioridad, título o color
- 🚀 **Inicio Automático con Windows**: Configura la app para que se abra al arrancar tu PC

## 🚀 Instalación

### Requisitos
- Python 3.8 o superior
- pip

### Instalación rápida

```bash
# Clonar repositorio
git clone https://github.com/RichardDebugFile/PositsEnPython.git
cd PositsEnPython

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python main.py
```

### Dependencias opcionales

Para funcionalidades avanzadas:

```bash
# Dictado por voz
pip install -r requirements_voice.txt

# Integración con IA (Ollama)
pip install ollama

# Inicio automático en Windows
pip install pywin32 winshell
```

## 🚀 Configurar Inicio Automático

Para que **Posits Virtuales** se inicie automáticamente con Windows:

```bash
python configurar_inicio_automatico.py
```

Selecciona la opción **1** (Activar inicio automático).

📖 **Guía completa**: Ver [INICIO_AUTOMATICO.md](INICIO_AUTOMATICO.md)

## 📖 Uso

### Inicio Rápido

1. Ejecuta `python main.py`
2. Haz clic en "➕ Nueva Tarea" para crear tareas
3. Completa tareas para ganar XP y subir de nivel
4. Haz clic en el botón ➕ del panel "Productividad" para crear misiones diarias

### Panel de Gamificación

- **Nivel y XP**: Sube de nivel completando tareas
- **Misiones Diarias**: 3 misiones que se resetean cada día
- **Sistema de Rachas**: Mantén la racha completando misiones
- **Botón ➕**: Crea las misiones diarias como tareas reales

### Adjuntar Archivos

1. Abre una tarea (botón 📝)
2. Haz clic en "📎 Adjuntar"
3. Selecciona el archivo
4. Doble clic para abrir archivos adjuntos

### Ordenamiento

Usa el dropdown "Ordenar" para organizar tareas por:
- 📅 Fecha
- ⭐ Prioridad
- 🔤 Título
- 🎨 Color

## 🏗️ Arquitectura

```
PositsEnPython/
├── main.py                 # Punto de entrada (16 líneas)
├── src/
│   ├── app.py             # Aplicación principal
│   ├── config.py          # Configuración centralizada
│   ├── models/            # Task, TaskStore, GamificationManager
│   ├── services/          # Ollama, STT, Attachments, Notifications
│   ├── ui/                # Componentes de interfaz
│   ├── dialogs/           # Diálogos modales
│   └── utils/             # Utilidades (fechas, colores, logger)
├── data/                  # Datos persistentes
│   ├── tasks.json
│   ├── gamification.json
│   └── attachments/
└── docs/                  # Documentación
```

## 🎯 Sistema de Puntos

| Acción | XP Ganado |
|--------|-----------|
| Completar tarea normal | 10 XP |
| Completar tarea prioritaria | 25 XP |
| Completar misión diaria | 50 XP |
| **Con racha activa** | +50% XP |

Cada 100 XP subes 1 nivel.

## 📱 Características Avanzadas

### Notificaciones
- Alertas automáticas para tareas próximas a vencer
- Revisa cada 60 segundos en segundo plano
- Compatible con Windows, macOS y Linux

### IA con Ollama
- Extrae tareas desde lenguaje natural
- Analiza imágenes para crear tareas
- Requiere Ollama instalado localmente

### Dictado por Voz
- Push-to-talk en editor de notas
- Transcripción automática a texto
- Soporta Faster-Whisper y Vosk

## 🛠️ Configuración

Edita `src/config.py` para personalizar:
- Colores y temas
- Puntos XP por acción
- Número de misiones diarias
- Multiplicador de racha
- Intervalo de notificaciones

## 📚 Documentación

- [Instalación Detallada](docs/INSTALACION_RAPIDA.md)
- [Documentación de Migración](docs/migration/)
- [Arquitectura del Sistema](docs/REFACTORIZACION_V2.md)
- [Guía de Desarrollo](docs/development/)
- [API Móvil (Propuesta)](docs/api/MOBILE_INTEGRATION.md)

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

**Versión 2.0** - Migración completada • Noviembre 2025

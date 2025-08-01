# 🚀 Instalación Rápida - Posits Virtuales

## ⚡ Instalación en 3 Pasos

### 1. Verificar Python
```bash
python --version
# Debe mostrar Python 3.8 o superior
```

### 2. Ejecutar la Aplicación
```bash
python main.py
```

¡Listo! La aplicación debería abrirse con todas las funcionalidades básicas.

---

## 🔧 Instalación Completa (Opcional)

### Para Reconocimiento de Voz
```bash
# Windows
pip install pipwin
pipwin install pyaudio
pip install SpeechRecognition

# Linux/macOS
pip install pyaudio SpeechRecognition
```

### Para IA Local (Ollama)
1. **Instalar Ollama**: [https://ollama.ai](https://ollama.ai)
2. **Descargar modelo**:
   ```bash
   ollama pull gemma3
   ```

### Para Reconocimiento de Voz Local
```bash
pip install faster-whisper
```

---

## 🎯 Funcionalidades Disponibles

### ✅ Sin Instalación Adicional
- ✅ Gestión de tareas con colores
- ✅ Editor de notas flotantes
- ✅ Filtros y estadísticas
- ✅ Posits rápidos (overlay)
- ✅ Persistencia de datos

### 🔧 Con Dependencias Opcionales
- 🎙️ **Dictado por voz** (con pyaudio + SpeechRecognition)
- 🤖 **IA local** (con Ollama)
- 🖼️ **Análisis de imágenes** (con Ollama)
- 🚀 **Reconocimiento de voz local** (con faster-whisper)

---

## 🐛 Problemas Comunes

### Error: "No module named 'tkinter'"
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk

# Windows: Reinstalar Python con tkinter incluido
```

### Error de micrófono en Windows
```bash
pip install pipwin
pipwin install pyaudio
```

### Ollama no responde
```bash
# Verificar que esté corriendo
curl http://localhost:11434/api/tags
```

---

## 📁 Estructura de Archivos

```
PositsEnPython/
├── main.py                 # Aplicación principal
├── README.md              # Documentación completa
├── requirements.txt       # Dependencias opcionales
├── config.example.env     # Configuración de ejemplo
├── INSTALACION_RAPIDA.md # Esta guía
├── test_tasks.py         # Tests unitarios
└── data/                 # Datos de la aplicación (se crea automáticamente)
    ├── notes.json        # Tareas guardadas
    └── app.log          # Logs de la aplicación
```

---

**✨ ¡Disfruta organizando tu vida con Posits Virtuales!** 
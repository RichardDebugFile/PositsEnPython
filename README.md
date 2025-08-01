# ✨ Posits Virtuales

Una aplicación de escritorio moderna para gestionar tareas y notas con inteligencia artificial local (Ollama). Combina la simplicidad de los post-its virtuales con la potencia de la IA para crear tareas desde lenguaje natural.

## 🚀 Características Principales

### 📋 Gestión de Tareas
- **Cards modernas** con franjas de color, prioridad y fechas de vencimiento
- **Filtros inteligentes** por color y estado (pendientes/completadas)
- **Estadísticas en tiempo real** (total, completadas, pendientes)
- **Colores vibrantes** para categorizar tareas (Amarillo, Azul, Verde, Naranja, Morado, Coral)

### 🎨 Editor de Notas Avanzado
- **Ventanas flotantes** con transparencia ajustable
- **Solo el área de texto** toma el color del posit
- **Persistencia de geometría** y estado de ventanas
- **Modo "siempre arriba"** para notas importantes
- **Atajos de fecha** (Hoy, +1d, +3d, +7d, +30d)

### 🤖 Inteligencia Artificial Local
- **Integración con Ollama** para crear tareas desde lenguaje natural
- **Dictado por voz** con reconocimiento de voz local
- **Análisis de imágenes** para extraer texto y crear tareas
- **Prompt inteligente** que extrae título, descripción, fecha, color y prioridad

### ⚡ Funcionalidades Rápidas
- **Overlay "posit rápido"** al doble clic (sin barra del sistema)
- **Ventanas arrastrables** y redimensionables
- **Atajos de teclado** (Ctrl+S para guardar, Escape para cancelar)
- **Auto-guardado** con throttling para mejor rendimiento

## 📋 Requisitos

### Dependencias Principales
```bash
# Python 3.8+ (incluido tkinter)
python --version
```

### Dependencias Opcionales (para funcionalidades avanzadas)

#### Para Reconocimiento de Voz
```bash
# Windows
pip install pipwin
pipwin install pyaudio
pip install SpeechRecognition

# Linux/macOS
pip install pyaudio
pip install SpeechRecognition
```

#### Para Ollama (IA Local)
1. **Instalar Ollama**: [https://ollama.ai](https://ollama.ai)
2. **Descargar modelo**: `ollama pull gemma3` (o el modelo que prefieras)

#### Para Reconocimiento de Voz Local (Opcional)
- **Faster-Whisper**: `pip install faster-whisper`
- **Vosk**: Descargar modelo español en `./model/vosk-model-small-es-0.42/`

## 🛠️ Instalación

### 1. Clonar o Descargar
```bash
git clone <tu-repositorio>
cd PositsEnPython
```

### 2. Verificar Python
```bash
python --version  # Debe ser 3.8+
```

### 3. Instalar Dependencias (Opcional)
```bash
# Para funcionalidad completa
pip install pyaudio SpeechRecognition

# Para reconocimiento de voz local
pip install faster-whisper
```

### 4. Configurar Ollama (Opcional)
```bash
# Instalar Ollama desde https://ollama.ai
# Luego descargar un modelo
ollama pull gemma3
```

## 🚀 Uso

### Ejecutar la Aplicación
```bash
python main.py
```

### Interfaz Principal

#### 📋 Barra de Herramientas
- **➕ Nueva Tarea**: Crear tarea manualmente
- **👁️ Solo Pendientes**: Filtro para ver solo tareas no completadas
- **📝 Abrir Notas**: Reabrir todas las notas cerradas
- **🤖 IA (Ollama)**: Crear tareas con inteligencia artificial
- **🎨 Filtro por Color**: Filtrar tareas por color
- **📌 Siempre Arriba**: Mantener ventana principal visible

#### 📊 Estadísticas
- **Total**: Número total de tareas
- **Completadas**: Tareas marcadas como hechas
- **Pendientes**: Tareas por completar

### 🎯 Gestión de Tareas

#### Crear Tarea Manualmente
1. Clic en **"➕ Nueva Tarea"**
2. Completar:
   - **Título** (obligatorio)
   - **Descripción** (opcional)
   - **Fecha de vencimiento** (formato YYYY-MM-DD)
   - **Prioridad** (marcar checkbox)
   - **Color** (seleccionar de la paleta)

#### Crear Tarea con IA
1. Clic en **"🤖 IA (Ollama)"**
2. **Dictar** o **escribir** instrucción natural:
   ```
   "crea tarea 'Enviar reporte mensual' para el viernes, 
   color azul, prioridad alta"
   ```
3. **Adjuntar imagen** (opcional) para análisis
4. Clic en **"▶ Analizar y crear"**

#### Gestionar Tareas Existentes
- **✅ Checkbox**: Marcar como completada
- **⭐ Estrella**: Cambiar prioridad
- **📝 Nota**: Abrir editor completo
- **🗑️ Eliminar**: Borrar tarea

### 🎨 Editor de Notas

#### Características
- **Ventana flotante** con transparencia ajustable
- **Color del texto** según el posit
- **Persistencia** de posición y tamaño
- **Modo pin** para mantener arriba

#### Controles
- **📌 Pin**: Mantener ventana siempre visible
- **🎨 Paleta**: Cambiar color del posit
- **💾 Guardar**: Guardar cambios (Ctrl+S)
- **↩️ Cancelar**: Descartar cambios (Escape)
- **🔍 Opacidad**: Ajustar transparencia (60%-100%)

#### Campos Editables
- **Título**: Nombre de la tarea
- **Descripción**: Texto libre con formato
- **📅 Vence**: Fecha de vencimiento
- **⭐ Prioridad**: Alta/Baja
- **✅ Hecha**: Estado de completado

### ⚡ Posit Rápido (Overlay)
- **Doble clic** en el título de cualquier tarea
- **Ventana sin bordes** del sistema
- **Arrastrable** con el mouse
- **Cerrar** con Escape o botón ✖

### 🎙️ Dictado por Voz
1. En el diálogo IA, **mantener presionado** "🎙️ Mantener para dictar"
2. **Hablar** la instrucción
3. **Soltar** para procesar
4. El texto aparecerá en el campo de entrada

## ⚙️ Configuración

### Variables de Entorno
```bash
# Directorio de datos (por defecto: ./data/)
export STICKYTASKS_DATA_DIR="/ruta/personalizada"

# Configuración de Ollama
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="gemma3"
```

### Archivos de Datos
- **`./data/notes.json`**: Tareas y configuraciones
- **`./data/app.log`**: Logs de la aplicación
- **`./data/`**: Directorio de datos por defecto

## 🔧 Funcionalidades Avanzadas

### Reconocimiento de Voz Local
La aplicación intenta usar estos motores en orden:
1. **Faster-Whisper** (más rápido)
2. **Vosk** (más preciso)

### Análisis de Imágenes con IA
- **Soporte para**: PNG, JPG, JPEG, WebP, BMP
- **Análisis de texto** en imágenes
- **Interpretación visual** para crear tareas

### Persistencia Inteligente
- **Auto-guardado** cada 400ms después de cambios
- **Migración automática** desde archivos antiguos
- **Backup temporal** durante escritura

## 🎨 Paleta de Colores

| Color | Código | Uso |
|-------|--------|-----|
| **Amarillo** | `#FFD54F` | Tareas generales |
| **Azul** | `#64B5F6` | Trabajo/Proyectos |
| **Verde** | `#81C784` | Personal/Salud |
| **Naranja** | `#FF8A65` | Urgente/Importante |
| **Morado** | `#BA68C8` | Creativo/Inspiración |
| **Coral** | `#FF7043` | Crítico/Deadline |

## ⌨️ Atajos de Teclado

### Editor de Notas
- **Ctrl+S**: Guardar cambios
- **Escape**: Cancelar/Cerrar
- **Enter**: Confirmar en campos

### Aplicación Principal
- **Mouse Wheel**: Scroll en lista de tareas
- **Doble Clic**: Abrir posit rápido

## 🐛 Solución de Problemas

### Ollama no responde
```bash
# Verificar que Ollama esté corriendo
curl http://localhost:11434/api/tags

# Verificar modelo instalado
ollama list
```

### Error de micrófono
```bash
# Windows: Instalar pyaudio correctamente
pip install pipwin
pipwin install pyaudio

# Linux: Instalar dependencias del sistema
sudo apt-get install portaudio19-dev
```

### Ventanas no se abren
- Verificar que no haya errores en `./data/app.log`
- Reiniciar la aplicación
- Verificar permisos de escritura en `./data/`

## 📝 Ejemplos de Uso

### Crear Tarea Simple
```
"Recordar llamar al cliente mañana"
```

### Crear Tarea Compleja
```
"crea tarea 'Preparar presentación Q4' para el 15 de diciembre, 
color azul, prioridad alta, descripción: incluir gráficos de ventas"
```

### Análisis de Imagen
1. Adjuntar captura de pantalla de email
2. Escribir: "crear tarea desde esta imagen"
3. La IA extraerá información y creará la tarea


## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Ollama** por proporcionar IA local
- **Tkinter** por la interfaz gráfica
- **Vosk** y **Faster-Whisper** por reconocimiento de voz
- **SpeechRecognition** por la abstracción de STT

---

**✨ Posits Virtuales** - Organiza tu vida con inteligencia artificial local 
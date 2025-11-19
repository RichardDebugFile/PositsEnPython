# Descargador de YouTube

Una aplicación de escritorio simple y eficiente para descargar videos de YouTube con interfaz gráfica desarrollada en Python.

## Características

- ✅ **Interfaz gráfica intuitiva** con Tkinter
- ✅ **Descarga de video y audio** en múltiples calidades
- ✅ **Conversión automática a MP3** para descargas de audio
- ✅ **Barra de progreso** con información detallada
- ✅ **Múltiples calidades** disponibles (Mejor, 1080p, 720p, 480p, 360p)
- ✅ **Verificación de integridad** con hash SHA-256
- ✅ **Limpieza automática** de archivos temporales
- ✅ **Reintentos automáticos** en caso de errores

## Requisitos Previos

### Software Necesario

1. **Python 3.8 o superior**
   - Descarga desde [python.org](https://www.python.org/downloads/)
   - Asegúrate de marcar "Add Python to PATH" durante la instalación

2. **FFmpeg (Opcional pero recomendado)**
   - Necesario para conversión de audio y fusión de video+audio
   - Descarga desde [ffmpeg.org](https://ffmpeg.org/download.html)
   - O instala con Chocolatey: `choco install ffmpeg`

## Instalación

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Si tienes Git instalado
git clone <url-del-repositorio>
cd DescargadorYT

# O simplemente descarga y extrae el archivo ZIP
```

### Paso 2: Crear Entorno Virtual

```powershell
# Crear entorno virtual
py -3 -m venv venv

# Activar el entorno virtual
.\venv\Scripts\Activate.ps1
```

### Paso 3: Instalar Dependencias

```powershell
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install yt-dlp requests

# Generar archivo de requisitos
pip freeze > requirements.txt
```

### Paso 4: Configurar FFmpeg (Opcional)

Si tienes FFmpeg instalado en una ubicación personalizada, edita la línea 7 en `downloaderyt.py`:

```python
FFMPEG_DIR = r"C:\ruta\a\tu\ffmpeg\bin"
```

## Uso

### Ejecutar la Aplicación

```powershell
# Asegúrate de que el entorno virtual esté activado
.\venv\Scripts\Activate.ps1

# Ejecutar la aplicación
python downloaderyt.py
```

### Cómo Usar la Interfaz

1. **Pega la URL** del video de YouTube en el campo de texto
2. **Selecciona las opciones**:
   - ✅ **Sólo audio (MP3)**: Para descargar solo el audio en formato MP3
   - **Calidad de video**: Elige entre Mejor, 1080p, 720p, 480p, 360p
3. **Haz clic en "Descargar"**
4. **Espera** a que se complete la descarga

### Información de Progreso

Durante la descarga verás:
- **Porcentaje de progreso**
- **Velocidad de descarga**
- **Tiempo estimado restante**
- **Estado del proceso**

## Características Técnicas

### Formatos Soportados

- **Video**: MP4, MKV, WebM
- **Audio**: MP3, M4A, WebM
- **Calidades**: Hasta 1080p (dependiendo del video original)

### Configuración Avanzada

El script incluye configuraciones optimizadas:
- **Reintentos**: 10 intentos para descargas fallidas
- **Fragmentos**: 10 reintentos para fragmentos de video
- **Limpieza**: Eliminación automática de archivos temporales
- **Logging**: Silenciamiento de mensajes innecesarios

## Solución de Problemas

### Error: "FFmpeg no detectado"
- **Solución**: Instala FFmpeg y agrégalo al PATH del sistema
- **Alternativa**: La aplicación funcionará sin FFmpeg, pero no podrá convertir audio

### Error: "HTTP Error 403"
- **Causa**: Restricciones temporales de YouTube
- **Solución**: Espera unos minutos y vuelve a intentar

### Error: "Video no disponible"
- **Causa**: Video privado, eliminado o con restricciones geográficas
- **Solución**: Verifica que el video sea público y accesible

### Error de Dependencias
```powershell
# Reinstalar dependencias
pip uninstall yt-dlp requests
pip install yt-dlp requests
```

## Estructura del Proyecto

```
DescargadorYT/
├── downloaderyt.py      # Aplicación principal
├── requirements.txt     # Dependencias del proyecto
├── venv/               # Entorno virtual (generado)
└── README.md          # Este archivo
```

## Dependencias

- **yt-dlp**: Descargador de videos de YouTube
- **requests**: Cliente HTTP para Python
- **tkinter**: Interfaz gráfica (incluido con Python)

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz un fork del proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Notas Importantes

- **Uso Responsable**: Respeta los términos de servicio de YouTube
- **Derechos de Autor**: Solo descarga contenido que tengas permiso para usar
- **Ancho de Banda**: Considera tu conexión a internet al descargar videos grandes
- **Almacenamiento**: Los videos pueden ocupar mucho espacio en disco

---

**Desarrollado con ❤️ en Python** 
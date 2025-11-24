# Guía de Instalación - Posits Mobile

## 📋 Prerrequisitos

### Windows (Desarrollo)
```bash
Python 3.11 o superior
Git
```

### Linux (Desarrollo + Compilación Android)
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev git \
    build-essential libssl-dev libffi-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
    zlib1g-dev
```

### macOS (Desarrollo + Compilación iOS)
```bash
# Instalar Homebrew primero
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar dependencias
brew install python3 git sdl2 sdl2_image sdl2_ttf sdl2_mixer
```

## 🚀 Instalación Rápida (Desktop Testing)

### 1. Clonar el repositorio
```bash
cd mobile-prototype
```

### 2. Crear entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar la app
```bash
python main.py
```

## 📱 Compilación para Android

### Requisitos Adicionales
- Ubuntu 20.04+ o WSL2 (en Windows)
- 8 GB RAM mínimo
- 20 GB espacio en disco
- Java JDK 11

### 1. Instalar Buildozer
```bash
pip install buildozer
pip install cython
```

### 2. Instalar Android SDK/NDK (automático)
```bash
# Buildozer descargará automáticamente al compilar por primera vez
# Esto puede tardar 1-2 horas en la primera compilación
```

### 3. Inicializar Buildozer (si no existe buildozer.spec)
```bash
buildozer init
```

### 4. Compilar APK Debug
```bash
# Modo verbose para ver progreso
buildozer -v android debug

# El APK estará en: bin/positsmobile-0.1-debug.apk
```

### 5. Compilar APK Release (para publicar)
```bash
buildozer -v android release

# Firmar el APK (necesitas crear un keystore)
# Ver: https://developer.android.com/studio/publish/app-signing
```

### 6. Instalar en dispositivo Android
```bash
# Conectar dispositivo por USB con depuración USB activada
buildozer android deploy

# O instalar y ejecutar directamente
buildozer android debug deploy run
```

## 🍎 Compilación para iOS

### Requisitos
- macOS 11.0+
- Xcode 13.0+
- Cuenta de desarrollador de Apple ($99/año para publicar)

### 1. Instalar kivy-ios
```bash
pip install kivy-ios
```

### 2. Compilar dependencias
```bash
toolchain build python3 kivy
```

### 3. Crear proyecto Xcode
```bash
toolchain create PositsMobile .
```

### 4. Abrir en Xcode
```bash
open PositsMobile-ios/PositsMobile.xcodeproj
```

### 5. Configurar firma y compilar
1. Seleccionar tu equipo de desarrollo en "Signing & Capabilities"
2. Conectar iPhone/iPad
3. Build and Run (⌘R)

## 🐛 Solución de Problemas

### Error: "SDL2 not found"
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

# macOS
brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer

# Windows
# Kivy incluye SDL2, solo asegúrate de instalar correctamente
pip install --upgrade kivy
```

### Error: "Permission denied" en Linux
```bash
# Dar permisos a buildozer
chmod +x ~/.buildozer/android/platform/android-sdk/tools/bin/*
```

### Error: Compilación muy lenta
```bash
# Limpiar caché y reintentar
buildozer android clean

# Usar menos CPUs si tienes poca RAM
export BUILDOZER_WARN_ON_ROOT=0
```

### Error: "Java not found"
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Configurar JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### App crashea al iniciar en Android
```bash
# Ver logs en tiempo real
adb logcat | grep python

# O con buildozer
buildozer android adb -- logcat | grep python
```

## 📊 Tiempos de Compilación Estimados

### Primera vez (descarga todo)
- Android: 1-2 horas
- iOS: 30-60 minutos

### Compilaciones posteriores
- Android Debug: 5-10 minutos
- Android Release: 10-15 minutos
- iOS: 3-5 minutos

## 💾 Tamaños de Archivos

- **APK Debug**: ~25 MB
- **APK Release**: ~15 MB
- **IPA**: ~20 MB

## 🔧 Configuración Avanzada

### Reducir tamaño del APK
```ini
# En buildozer.spec
android.archs = arm64-v8a  # Solo para dispositivos modernos
android.add_src = False    # No incluir código fuente
```

### Agregar ícono personalizado
```ini
# En buildozer.spec
icon.filename = %(source.dir)s/assets/icon.png
# Ícono debe ser 512x512 PNG
```

### Agregar splash screen
```ini
# En buildozer.spec
presplash.filename = %(source.dir)s/assets/splash.png
# Splash debe ser 1280x720 PNG
```

### Cambiar permisos
```ini
# En buildozer.spec
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,CAMERA,VIBRATE
```

## 📱 Testing en Emulador Android

### Usando Android Studio
1. Instalar Android Studio
2. Crear AVD (Android Virtual Device)
3. Iniciar emulador
4. `adb install bin/positsmobile-0.1-debug.apk`

### Usando comandos
```bash
# Listar dispositivos
adb devices

# Instalar APK
adb install -r bin/positsmobile-0.1-debug.apk

# Ver logs
adb logcat | grep python
```

## 🔄 Actualizar la App

### Para desarrollo
```bash
# Recompilar solo código Python (más rápido)
buildozer android debug

# Reinstalar en dispositivo
adb install -r bin/positsmobile-0.1-debug.apk
```

### Para producción
```bash
# Incrementar versión en buildozer.spec
version = 0.2

# Recompilar release
buildozer android release
```

## 📚 Recursos Adicionales

- [Documentación Kivy](https://kivy.org/doc/stable/)
- [Documentación KivyMD](https://kivymd.readthedocs.io/)
- [Buildozer Docs](https://buildozer.readthedocs.io/)
- [Guía Android](https://kivy.org/doc/stable/guide/packaging-android.html)
- [Guía iOS](https://kivy.org/doc/stable/guide/packaging-ios.html)

## 🆘 Soporte

Si encuentras problemas:
1. Revisa los logs: `buildozer -v android debug`
2. Limpia el build: `buildozer android clean`
3. Actualiza buildozer: `pip install --upgrade buildozer`
4. Busca en [Stack Overflow](https://stackoverflow.com/questions/tagged/kivy)
5. Pregunta en [Kivy Discord](https://chat.kivy.org/)

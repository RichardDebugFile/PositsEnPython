# 🚀 Configurar Inicio Automático de Posits Virtuales

Este documento explica cómo hacer que **Posits Virtuales** se inicie automáticamente cuando arranca Windows.

---

## 📋 Requisitos Previos

Antes de configurar el inicio automático, asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install pywin32 winshell
```

O instala todas las dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🎯 Método 1: Script Automático (Recomendado)

### Paso 1: Ejecutar el configurador

Desde la carpeta del proyecto, ejecuta:

```bash
python configurar_inicio_automatico.py
```

### Paso 2: Seleccionar opción

El script te mostrará un menú:

```
====================================================================
   CONFIGURADOR DE INICIO AUTOMÁTICO - POSITS VIRTUALES
====================================================================

¿Qué deseas hacer?
1. Activar inicio automático
2. Desactivar inicio automático
3. Salir

Selecciona una opción (1-3):
```

Selecciona **1** para activar el inicio automático.

### Paso 3: ¡Listo!

El script creará un atajo en tu carpeta de Inicio de Windows. La próxima vez que reinicies tu PC, **Posits Virtuales** se abrirá automáticamente.

---

## 🔧 Método 2: Manual

Si prefieres hacerlo manualmente:

### Paso 1: Abrir carpeta de Inicio

1. Presiona `Win + R`
2. Escribe: `shell:startup`
3. Presiona Enter

### Paso 2: Crear atajo

1. Haz clic derecho en la carpeta que se abrió
2. Selecciona **Nuevo → Acceso directo**
3. En "Ubicación", escribe la ruta completa a `iniciar_posits.bat`:
   ```
   "G:\Documentos G\Ing. Sotware\ExperimentosPy\PositsEnPython\iniciar_posits.bat"
   ```
   *(Ajusta la ruta según donde tengas el proyecto)*

4. Haz clic en **Siguiente**
5. Ponle un nombre: **Posits Virtuales**
6. Haz clic en **Finalizar**

### Paso 3: Probar

Reinicia tu PC y verifica que la aplicación se abra automáticamente.

---

## ❌ Desactivar Inicio Automático

### Método Automático:

```bash
python configurar_inicio_automatico.py
```

Selecciona la opción **2** (Desactivar inicio automático).

### Método Manual:

1. Presiona `Win + R`
2. Escribe: `shell:startup`
3. Presiona Enter
4. Elimina el atajo "Posits Virtuales"

---

## 🐛 Solución de Problemas

### La app no se abre al iniciar Windows

**Causa posible:** Python no está en el PATH del sistema.

**Solución:**
1. Abre `iniciar_posits.bat` con un editor de texto
2. Reemplaza `pythonw` por la ruta completa a Python:
   ```bat
   start "" "C:\Python311\pythonw.exe" main.py
   ```

### La app se abre pero se cierra inmediatamente

**Causa posible:** Error en el código o dependencias faltantes.

**Solución:**
1. Edita `iniciar_posits.bat`
2. Cambia `pythonw` por `python` para ver los errores:
   ```bat
   start "" python main.py
   ```
3. Revisa los errores en la consola que se abre

### Quiero ver la consola de debug al iniciar

Edita `iniciar_posits.bat` y usa:

```bat
start "" python main.py
```

En lugar de:

```bat
start "" pythonw main.py
```

---

## 📚 Notas Adicionales

- **pythonw.exe**: Ejecuta sin mostrar ventana de consola (silencioso)
- **python.exe**: Ejecuta mostrando ventana de consola (para debug)
- La app se ejecuta en segundo plano con los posits flotantes visibles
- Los posits recuerdan su posición de la sesión anterior

---

## ✅ Verificación

Para verificar que el inicio automático está configurado:

1. Presiona `Win + R`
2. Escribe: `shell:startup`
3. Busca el atajo "Posits Virtuales"
4. Si está ahí, ¡está configurado correctamente!

---

© 2025 Posits Virtuales v2.3
Sistema de Inicio Automático

# Scripts para Iniciar Navegadores con CDP

## 📁 Archivos Incluidos

- **`start_chrome_cdp.bat`** - Inicia Google Chrome con CDP
- **`start_brave_cdp.bat`** - Inicia Brave Browser con CDP

## 🚀 Uso Rápido

### Para Chrome:

1. **Doble click** en `start_chrome_cdp.bat`
2. Sigue las instrucciones en pantalla
3. ✅ Chrome se abrirá con CDP habilitado

### Para Brave:

1. **Doble click** en `start_brave_cdp.bat`
2. Sigue las instrucciones en pantalla
3. ✅ Brave se abrirá con CDP habilitado

---

## 🎯 Características

### ✨ Búsqueda Automática
Los scripts buscan automáticamente la instalación del navegador en:
- `C:\Program Files\[Navegador]\`
- `C:\Program Files (x86)\[Navegador]\`
- `%LOCALAPPDATA%\[Navegador]\`
- `%USERPROFILE%\AppData\Local\[Navegador]\`

**No necesitas editar nada**, funciona automáticamente.

### 🔄 Gestión de Procesos
Los scripts detectan si el navegador ya está abierto y te dan opciones:

```
Opciones:
[1] Cerrar automáticamente (recomendado)
[2] Yo cerraré manualmente
[3] Cancelar
```

### ✅ Verificación CDP
Al finalizar, verifica que CDP esté respondiendo correctamente:

```
✓ CDP respondiendo correctamente en puerto 9222
✓ Pestañas activas: 3
```

---

## 📋 Flujo Detallado

### Ejemplo con Chrome:

```
═══════════════════════════════════════════════════════════════════
         INICIAR GOOGLE CHROME CON CDP (Puerto 9222)
═══════════════════════════════════════════════════════════════════

[1] Buscando instalación de Google Chrome...

   ✓ Encontrado en: C:\Program Files\Google\Chrome\Application\

[2] Verificando procesos de Chrome en ejecución...

   ⚠ ATENCIÓN: Hay procesos de Chrome en ejecución

   Procesos activos:
   chrome.exe      12345
   chrome.exe      12346
   chrome.exe      12347

   ┌────────────────────────────────────────────────────────┐
   │  Para usar CDP, Chrome debe reiniciarse con flags     │
   │  especiales. Necesitas cerrar Chrome completamente.   │
   └────────────────────────────────────────────────────────┘

   Opciones:
   [1] Cerrar Chrome automáticamente (recomendado)
   [2] Yo cerraré Chrome manualmente
   [3] Cancelar

   Elige una opción (1, 2 o 3): 1

   [2.1] Cerrando Chrome automáticamente...

   ✓ Chrome cerrado correctamente

[3] Iniciando Chrome con CDP habilitado...

   Configuración:
   - Puerto CDP: 9222
   - Host: 127.0.0.1 (solo local)
   - Ejecutable: C:\Program Files\Google\Chrome\Application\chrome.exe

   ✓ Chrome iniciado correctamente con CDP

[4] Verificando conexión CDP...

   ✓ CDP respondiendo correctamente en puerto 9222
   ✓ Pestañas activas: 1

═══════════════════════════════════════════════════════════════════
                   ✓ CHROME CON CDP LISTO
═══════════════════════════════════════════════════════════════════

   Ahora puedes:
   1. Abrir YouTube en Chrome
   2. Reproducir un video
   3. Usar la función "Capturar" en la aplicación

   Para verificar que funciona, ejecuta: python test_cdp.py

═══════════════════════════════════════════════════════════════════
```

---

## 🔧 Solución de Problemas

### Error: "No se encontró [Navegador]"

**Causa:** El navegador no está instalado o está en una ubicación no estándar.

**Solución:**
1. Instala el navegador desde:
   - Chrome: https://www.google.com/chrome/
   - Brave: https://brave.com/download/

2. O edita el script `.bat` y agrega la ruta manualmente:
   ```batch
   :: Línea ~40 en el script
   set "CHROME_PATH=C:\Tu\Ruta\Personalizada\chrome.exe"
   goto :found
   ```

---

### Error: "Algunos procesos no se pudieron cerrar"

**Causa:** Hay procesos del navegador bloqueados o con permisos elevados.

**Solución:**
1. Abre el Administrador de Tareas (Ctrl+Shift+Esc)
2. Busca todos los procesos `chrome.exe` o `brave.exe`
3. Selecciónalos todos
4. Click derecho → "Finalizar tarea"
5. Ejecuta el script de nuevo

---

### Error: "CDP iniciado pero aún no responde"

**Causa:** El navegador acaba de iniciar y CDP aún no está listo.

**Solución:**
- **Espera 5-10 segundos** y ejecuta `python test_cdp.py`
- Si persiste, reinicia el script

---

## 🎓 Uso Avanzado

### Cambiar el Puerto CDP

Por defecto usa el puerto `9222`. Para cambiarlo:

1. Abre el script `.bat` en un editor de texto
2. Busca la línea:
   ```batch
   start "" "!BRAVE_PATH!" --remote-debugging-port=9222
   ```
3. Cambia `9222` por el puerto que desees (ej: `9223`)
4. Guarda el archivo

**IMPORTANTE:** También debes actualizar `src/utils/cdp_helper.py`:
```python
DEFAULT_PORT = 9223  # Cambiar aquí también
```

---

### Agregar Más Flags

Puedes agregar más flags de Chromium al inicio:

```batch
start "" "!BRAVE_PATH!" --remote-debugging-port=9222 --disable-extensions --incognito
```

Flags útiles:
- `--disable-extensions` - Deshabilita extensiones
- `--incognito` - Modo incógnito
- `--start-maximized` - Inicia maximizado
- `--new-window` - Nueva ventana

---

## 📊 Comparación de Métodos

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Script .BAT** | ✅ Automático<br>✅ Busca instalación<br>✅ Gestiona procesos<br>✅ Verifica CDP | ⚠️ Requiere cerrar navegador |
| **Acceso Directo** | ✅ Permanente<br>✅ Un solo click | ❌ Ruta fija<br>❌ Manual |
| **Línea de Comandos** | ✅ Control total | ❌ Temporal<br>❌ Repetitivo |

---

## 🔐 Seguridad

### ¿Es seguro usar CDP?

**Sí, completamente seguro:**

- ✅ CDP solo escucha en `127.0.0.1` (localhost)
- ✅ **NO** está expuesto a internet
- ✅ Solo aplicaciones locales pueden conectarse
- ✅ Es el mismo protocolo que usan las DevTools (F12)

### ¿Qué permisos necesita el script?

- ❌ **NO necesita permisos de administrador**
- ✅ Solo necesita poder cerrar procesos del usuario actual
- ✅ No modifica archivos del sistema
- ✅ No instala nada

---

## 📝 Notas Importantes

### Puerto 9222 en Uso

Si obtienes un error que el puerto está en uso:

1. **Verifica si ya hay un navegador con CDP abierto:**
   ```bash
   python test_cdp.py
   ```

2. **Si funciona**, ya tienes CDP habilitado, no necesitas el script

3. **Si no funciona**, otro programa está usando el puerto 9222:
   - Cierra todos los navegadores
   - Usa otro puerto (ver "Uso Avanzado")

### Usar Siempre con CDP

Para usar siempre el navegador con CDP:

**Opción 1: Pin el script a la barra de tareas**
1. Click derecho en `start_brave_cdp.bat`
2. "Anclar a la barra de tareas"
3. Usa ese acceso siempre

**Opción 2: Crear acceso directo en el escritorio**
1. Click derecho en `start_brave_cdp.bat`
2. "Enviar a" → "Escritorio (crear acceso directo)"
3. Renombra a "Brave (CDP)"

---

## 🎯 Verificar que Funciona

Después de ejecutar el script:

### Test 1: Script de Python

```bash
cd "G:\Documentos G\Ing. Sotware\ExperimentosPy\PositsEnPython"
venv\Scripts\python test_cdp.py
```

Deberías ver:
```
[OK] CDP esta disponible
[OK] Conectado exitosamente
[OK] Se encontraron X pestana(s)
```

### Test 2: En el Navegador

1. Abre: http://127.0.0.1:9222/json
2. Deberías ver JSON con información de las pestañas

### Test 3: En la Aplicación

1. Abre YouTube en el navegador
2. Reproduce un video
3. Haz click en "🎧 Capturar"
4. Debería capturar automáticamente la URL

---

## 💡 Consejos

### Primer Uso
1. **Lee todos los mensajes** del script
2. **No cierres la ventana** hasta que termine
3. **Prueba con `test_cdp.py`** antes de usar la app

### Uso Diario
- Usa el script/acceso directo en lugar del navegador normal
- Si olvidas y abres el navegador normal, ciérralo y usa el script
- CDP solo funciona cuando se inicia con el flag

### Alternancia
Si necesitas usar el navegador **sin** CDP:
- Usa el acceso directo normal del navegador
- La app seguirá funcionando con el método básico

---

## 📞 Soporte

Si tienes problemas:

1. **Lee la sección de "Solución de Problemas"**
2. **Ejecuta `python test_cdp.py`** y revisa los mensajes
3. **Verifica que el navegador se cerró completamente**
4. **Intenta con el otro navegador** (Chrome vs Brave)

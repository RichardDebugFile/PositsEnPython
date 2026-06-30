# Instrucciones para Habilitar CDP (Chrome DevTools Protocol)

## ¿Qué es CDP y por qué lo necesito?

CDP (Chrome DevTools Protocol) permite a la aplicación **detectar automáticamente qué pestaña de YouTube está reproduciendo**, incluso si tienes múltiples pestañas de YouTube abiertas o la pestaña activa es otra (WhatsApp, GitHub, etc.).

### Sin CDP:
- ❌ Solo detecta la pestaña activa visible
- ❌ Si YouTube está en background, no lo encuentra
- ⚠️ Puede capturar la URL incorrecta si hay múltiples YouTubes

### Con CDP:
- ✅ Ve TODAS las pestañas abiertas
- ✅ Detecta automáticamente cuál está reproduciendo
- ✅ Funciona con pestañas en background
- ✅ 100% preciso con múltiples pestañas de YouTube

---

## Cómo Habilitar CDP

### Opción 1: Crear Acceso Directo (Recomendado)

#### Para Brave:

1. **Cierra Brave completamente** (Ctrl+Shift+Q o desde el Administrador de Tareas)

2. Haz **click derecho en el escritorio** → Nuevo → Acceso directo

3. En "Ubicación", pega esto:
   ```
   "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --remote-debugging-port=9222
   ```

4. Dale un nombre: **"Brave (con CDP)"**

5. **Usa este acceso directo** para abrir Brave de ahora en adelante

#### Para Chrome:

1. **Cierra Chrome completamente**

2. Haz **click derecho en el escritorio** → Nuevo → Acceso directo

3. En "Ubicación", pega esto:
   ```
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
   ```

4. Dale un nombre: **"Chrome (con CDP)"**

5. **Usa este acceso directo** para abrir Chrome de ahora en adelante

---

### Opción 2: Desde la Terminal (Temporal)

#### Para Brave:

Abre PowerShell o CMD y ejecuta:
```powershell
& "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --remote-debugging-port=9222
```

#### Para Chrome:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

⚠️ **Nota:** Este método solo funciona mientras la terminal está abierta.

---

## Verificar que CDP está Funcionando

1. **Abre el navegador con CDP habilitado** (usando el acceso directo)

2. **Ejecuta el script de test:**
   ```bash
   python scripts\cdp\test_cdp.py
   ```

3. Deberías ver:
   ```
   [OK] CDP esta disponible en http://127.0.0.1:9222
   [OK] Conectado exitosamente
   [OK] Se encontraron X pestana(s)
   ```

---

## Solución de Problemas

### Error: "CDP NO esta disponible"

**Causa:** El navegador no está ejecutándose con el flag `--remote-debugging-port=9222`

**Solución:**
1. Cierra COMPLETAMENTE el navegador (incluyendo procesos en segundo plano)
2. Abre el Administrador de Tareas (Ctrl+Shift+Esc)
3. Busca "brave.exe" o "chrome.exe" y finaliza TODOS los procesos
4. Abre el navegador usando el acceso directo con CDP

---

### Error: "No se encontraron pestanas de YouTube"

**Causa:** No hay pestañas de YouTube abiertas o no contienen "/watch" en la URL

**Solución:**
1. Abre YouTube.com
2. **Reproduce un video** (no solo la página principal)
3. Verifica que la URL contenga `/watch?v=`

---

### Error: "No se encontro pestana reproduciendo"

**Causa:** El video está pausado o no se está reproduciendo

**Solución:**
1. Asegúrate de que el video esté **REPRODUCIENDO** (no pausado)
2. Espera unos segundos después de dar play
3. Ejecuta el test de nuevo

---

## ¿Es Seguro?

**Sí, es completamente seguro.**

- El puerto 9222 **solo acepta conexiones locales** (127.0.0.1)
- **NO** está expuesto a internet
- Es el mismo protocolo que usan las DevTools del navegador (F12)
- Usado por millones de desarrolladores diariamente

---

## Desactivar CDP

Si quieres volver a usar el navegador sin CDP:

1. Usa el **acceso directo normal** de Brave/Chrome (sin el flag)
2. O simplemente abre el navegador desde el menú inicio

La aplicación seguirá funcionando, pero usará el método de detección básico (solo pestaña activa).

---

## Preguntas Frecuentes

### ¿Tengo que hacer esto cada vez que abra el navegador?

**No.** Si usas el acceso directo con CDP, solo tienes que usarlo en lugar del acceso directo normal. No necesitas configurar nada más.

### ¿Puedo tener múltiples instancias del navegador?

**No recomendado.** Solo una instancia puede usar el puerto 9222. Si intentas abrir otra, obtendrás un error.

### ¿Afecta el rendimiento?

**No.** CDP usa recursos mínimos y no afecta el rendimiento del navegador.

### ¿Funciona con otros navegadores?

Sí, funciona con cualquier navegador basado en Chromium:
- ✅ Brave
- ✅ Google Chrome
- ✅ Microsoft Edge
- ✅ Opera
- ❌ Firefox (usa protocolo diferente)

---

## Soporte

Si tienes problemas:

1. Ejecuta `python scripts\cdp\test_cdp.py` y revisa los mensajes de error
2. Verifica que el navegador esté corriendo con el flag correcto
3. Revisa que no haya otro navegador usando el puerto 9222

# Tests manuales / diagnósticos

Scripts de verificación que **no** se ejecutan en la suite automatizada de
`pytest` (ni en CI) porque dependen de recursos externos o hardware:

| Script | Requiere |
|--------|----------|
| `test_youtube_downloader.py` | Conexión a internet (consulta a YouTube) |
| `test_music_scan.py` | `pygame` + archivos reales en `data/music/` |
| `test_vosk.py` | Modelo Vosk descargado y `SpeechRecognition`/`PyAudio` |
| `test_task_compatibility.py` | Reemplaza `sys.stdout` (diagnóstico de migración) |
| `visor_gemma.py` | Ollama/Gemma en ejecución |

Se ejecutan a mano desde la raíz del repo, por ejemplo:

```bash
python tests/manual/test_youtube_downloader.py
```

La suite automatizada (unitarios + integración) está en `tests/` y se corre con:

```bash
pytest
```

# Documentación Técnica - Posits Mobile

## 🏗️ Arquitectura

### Patrón de Diseño
- **MVP (Model-View-Presenter)** adaptado para Kivy
- **Servicios independientes** reutilizables
- **Separación de concerns** clara

```
┌─────────────────────────────────────┐
│         main.py (App)               │
│  ┌───────────────────────────────┐  │
│  │  MDBottomNavigation           │  │
│  │  ├── HomeScreen                │  │
│  │  ├── TasksScreen               │  │
│  │  └── PomodoroScreen            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
          ↓           ↓           ↓
┌─────────────────────────────────────┐
│         Services Layer              │
│  ├── TaskStore (SQLite)             │
│  ├── GamificationSystem (JSON)     │
│  └── PomodoroTimer (In-memory)     │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│         Data Layer                  │
│  ├── data/tasks.db                  │
│  ├── data/gamification.json         │
│  └── data/music/                    │
└─────────────────────────────────────┘
```

## 📦 Estructura de Módulos

### `services/task_store.py`
**Responsabilidad**: Gestión de tareas con SQLite

**Métodos principales**:
- `add_task(title, description, priority, ...)` → int
- `get_tasks(filter_completed=None)` → List[Dict]
- `complete_task(task_id)` → bool
- `delete_task(task_id)` → bool
- `get_statistics()` → Dict

**Esquema de BD**:
```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT DEFAULT 'normal',
    color TEXT DEFAULT 'blue',
    completed BOOLEAN DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    deadline TEXT,
    tags TEXT  -- JSON array
)
```

**Ventajas vs JSON (versión desktop)**:
- ✅ Consultas más rápidas
- ✅ Filtrado eficiente
- ✅ Soporte para índices
- ✅ Transacciones atómicas
- ✅ Menor uso de memoria

### `services/gamification.py`
**Responsabilidad**: Sistema XP, niveles y misiones

**Métodos principales**:
- `add_xp(amount, reason)` → Dict
- `on_task_completed(priority)` → Dict
- `on_pomodoro_completed()` → Dict
- `get_status()` → Dict
- `get_missions()` → List[Dict]

**Fórmula de nivel**:
```python
nivel = floor(sqrt(xp / 100))
xp_para_nivel_n = n² × 100

Ejemplos:
Nivel 1: 0-99 XP
Nivel 2: 100-399 XP
Nivel 3: 400-899 XP
Nivel 4: 900-1599 XP
```

**Misiones diarias**:
1. Completar 3 tareas → 50 XP
2. Completar tarea de prioridad alta → 75 XP
3. Completar 2 Pomodoros → 60 XP

### `services/pomodoro.py`
**Responsabilidad**: Timer Pomodoro con callbacks

**Configuración**:
- Trabajo: 25 minutos
- Descanso corto: 5 minutos
- Descanso largo: 15 minutos
- Sesiones hasta descanso largo: 4

**Estados**:
```python
{
    "is_running": bool,
    "is_paused": bool,
    "is_work_time": bool,
    "sessions_completed": int,
    "time_remaining": float,  # segundos
    "progress": float  # 0-100%
}
```

**Callbacks disponibles**:
- `on_tick(status)` - Cada segundo
- `on_work_complete(sessions)` - Al completar trabajo
- `on_break_complete()` - Al completar descanso
- `on_session_complete(sessions)` - Al completar ciclo completo

## 🎨 UI Components

### KivyMD Widgets Usados

**Layout**:
- `MDBoxLayout` - Layout flexible
- `MDCard` - Cards con elevación
- `MDBottomNavigation` - Navegación inferior

**Inputs**:
- `MDTextField` - Input de texto Material
- `MDCheckbox` - Checkbox Material
- `MDSwitch` - Switch Material

**Buttons**:
- `MDRaisedButton` - Botón elevado
- `MDIconButton` - Botón de ícono
- `MDFlatButton` - Botón plano

**Lists**:
- `MDList` - Lista de items
- `OneLineAvatarIconListItem` - Item con avatar e ícono

**Dialogs**:
- `MDDialog` - Diálogo modal
- `MDFileManager` - Explorador de archivos

### Tema y Colores

**Paleta**:
```python
primary_palette = "DeepPurple"  # #5E35B1
theme_style = "Dark"
```

**Colores personalizados**:
```python
# Cards de nivel
level_card_bg = (0.3, 0.2, 0.5, 1)  # Morado oscuro

# Prioridades de tareas
priority_colors = {
    "low": (0.5, 0.5, 0.5, 1),      # Gris
    "normal": (0.3, 0.6, 1, 1),     # Azul
    "high": (1, 0.6, 0.2, 1),       # Naranja
    "urgent": (1, 0.2, 0.2, 1)      # Rojo
}
```

## 🔄 Flujo de Datos

### Agregar Tarea
```
Usuario → TasksScreen.show_add_dialog()
         ↓
      MDDialog con MDTextField
         ↓
      TasksScreen.add_task()
         ↓
      TaskStore.add_task() → INSERT INTO tasks
         ↓
      TasksScreen.refresh_tasks() → UI update
```

### Completar Tarea
```
Usuario → TaskItem.complete_task()
         ↓
      TaskStore.complete_task() → UPDATE tasks
         ↓
      GamificationSystem.on_task_completed()
         ↓
      ├── Calcular XP según prioridad
      ├── Actualizar misiones diarias
      └── Verificar level up
         ↓
      TasksScreen.refresh_tasks() → UI update
```

### Ciclo Pomodoro
```
Usuario → PomodoroScreen.toggle_timer()
         ↓
      PomodoroTimer.start()
         ↓
      Clock.schedule_interval(update_timer, 1)
         ↓
      Cada segundo:
         ├── PomodoroTimer.update()
         ├── Actualizar UI (timer_label)
         └── Si completó fase:
                ↓
             GamificationSystem.on_pomodoro_completed()
                ↓
             Notificar usuario (vibración/sonido)
```

## 📊 Optimizaciones para Móvil

### 1. Base de Datos SQLite
**Por qué**: Más eficiente que JSON para queries frecuentes

**Ventajas**:
- Índices automáticos en PRIMARY KEY
- Queries optimizadas por SQLite engine
- Menor uso de memoria (no cargar todo en RAM)

### 2. Lazy Loading
**Implementación**:
```python
def refresh_tasks(self):
    # Solo cargar tareas visibles
    tasks = self.task_store.get_tasks(filter_completed=False)
    # NO cargar tareas completadas automáticamente
```

### 3. Event Scheduling
**Clock.schedule_interval**:
```python
# Actualizar timer cada segundo, NO usar threading
Clock.schedule_interval(self.update_timer, 1)
```

**Ventaja**: Kivy maneja el ciclo de vida automáticamente

### 4. Widget Recycling
**Para listas largas (futuro)**:
```python
from kivy.uix.recycleview import RecycleView

class TaskRecycleView(RecycleView):
    # Recicla widgets fuera de pantalla
    # Reduce uso de memoria
```

## 🔒 Seguridad y Privacidad

### Datos Locales
- **Ubicación**: `data/` en almacenamiento privado de la app
- **Acceso**: Solo la app puede leer/escribir
- **Backup**: Android Auto Backup (opcional)

### Permisos Mínimos
```ini
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,VIBRATE
```

**Justificación**:
- `INTERNET`: Descargar de YouTube
- `WRITE_EXTERNAL_STORAGE`: Guardar MP3s
- `VIBRATE`: Notificaciones Pomodoro

### Sin Tracking
- ❌ No analytics
- ❌ No ads
- ❌ No servidor remoto (todo local)

## 🧪 Testing

### Unit Tests (futuro)
```python
# tests/test_task_store.py
import unittest
from services.task_store import TaskStore

class TestTaskStore(unittest.TestCase):
    def setUp(self):
        self.store = TaskStore(db_path="test.db")

    def test_add_task(self):
        task_id = self.store.add_task("Test")
        self.assertGreater(task_id, 0)

    def tearDown(self):
        os.remove("test.db")
```

### Integration Tests
```bash
# Ejecutar en emulador
pytest tests/ --android

# Con coverage
pytest tests/ --cov=services
```

## 📈 Métricas de Rendimiento

### Objetivos
- **Inicio**: < 2 segundos
- **Transiciones**: 60 FPS (16ms/frame)
- **Uso RAM**: < 100 MB
- **Batería**: < 5% por hora de uso activo

### Profiling
```python
from kivy.core.profiling import Profiler

profiler = Profiler()
profiler.start()
# ... código a medir ...
profiler.stop()
print(profiler.results())
```

## 🔌 Extensibilidad

### Agregar Nueva Pantalla
```python
# 1. Crear screen
class MusicScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... build UI ...

# 2. Agregar a app
class PositsMobileApp(MDApp):
    def build(self):
        # ...
        music_item = MDBottomNavigationItem(
            name='music',
            text='Música',
            icon='music'
        )
        bottom_nav.add_widget(music_item)
```

### Agregar Servicio
```python
# services/new_service.py
class NewService:
    def __init__(self, data_file="data/new.json"):
        # ...

# main.py
self.new_service = NewService()
```

## 🚀 Roadmap Técnico

### v0.2
- [ ] RecycleView para listas
- [ ] Async database operations
- [ ] Animaciones suaves
- [ ] Sound effects

### v0.3
- [ ] Backend sync (FastAPI)
- [ ] Push notifications
- [ ] Widget del sistema
- [ ] Share extension

### v1.0
- [ ] Multi-usuario
- [ ] Cloud backup
- [ ] Temas personalizables
- [ ] Plugins system

## 📚 Referencias

- [Kivy Architecture](https://kivy.org/doc/stable/guide/architecture.html)
- [KivyMD Components](https://kivymd.readthedocs.io/en/latest/components/)
- [SQLite Best Practices](https://www.sqlite.org/bestpractice.html)
- [Android Performance](https://developer.android.com/topic/performance)

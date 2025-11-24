"""
Posits Mobile - Prototipo MVP (Versión Corregida)
App de productividad con Kivy/KivyMD
"""

from kivy.config import Config
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '640')

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.list import MDList, ThreeLineListItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.metrics import dp

from services.task_store import TaskStore
from services.gamification import GamificationSystem
from services.pomodoro import PomodoroTimer
from services.simple_music_player import SimpleMusicPlayer


class HomeScreen(MDScreen):
    """Pantalla principal con resumen"""

    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app = app_instance
        self.mission_dialog = None
        self.current_mission_index = None
        self.build_ui()

    def build_ui(self):
        """Construye la interfaz completa"""
        # Usar ScrollView para que todo el contenido sea scrollable
        scroll = ScrollView()
        layout = MDBoxLayout(orientation='vertical', padding=dp(15), spacing=dp(12), size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        # Título
        title = MDLabel(
            text="Posits Mobile",
            font_style="H5",
            halign="center",
            size_hint_y=None,
            height=dp(40)
        )
        layout.add_widget(title)

        # Card de nivel y XP
        level_card = MDCard(
            size_hint_y=None,
            height=dp(120),
            padding=dp(15),
            md_bg_color=(0.3, 0.2, 0.5, 1),
            elevation=3
        )
        level_box = MDBoxLayout(orientation='vertical', spacing=dp(5))

        self.level_label = MDLabel(
            text="Nivel 1",
            font_style="H4",
            halign="center",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1)
        )
        self.xp_label = MDLabel(
            text="0 XP",
            font_style="H6",
            halign="center",
            theme_text_color="Custom",
            text_color=(0.9, 0.9, 0.9, 1)
        )
        self.xp_progress_label = MDLabel(
            text="Progreso: 0%",
            font_style="Caption",
            halign="center",
            theme_text_color="Custom",
            text_color=(0.8, 0.8, 0.8, 1)
        )

        level_box.add_widget(self.level_label)
        level_box.add_widget(self.xp_label)
        level_box.add_widget(self.xp_progress_label)
        level_card.add_widget(level_box)
        layout.add_widget(level_card)

        # Card de misiones diarias
        missions_card = MDCard(
            size_hint_y=None,
            height=dp(180),
            padding=dp(15),
            elevation=2
        )
        missions_box = MDBoxLayout(orientation='vertical', spacing=dp(8))

        missions_title = MDLabel(
            text="Misiones Diarias",
            font_style="H6",
            size_hint_y=None,
            height=dp(30)
        )
        missions_box.add_widget(missions_title)

        self.mission_labels = []
        self.mission_buttons = []
        self.mission_edit_buttons = []
        for i in range(3):
            # Contenedor horizontal para checkbox y botón editar
            mission_row = MDBoxLayout(orientation='horizontal', size_hint_y=None, height=dp(35), spacing=dp(5))

            # Botón checkbox (completa/desmarca)
            mission_btn = MDFlatButton(
                text="",
                size_hint_x=0.8,
                on_release=lambda x, idx=i: self.toggle_mission(idx)
            )
            self.mission_labels.append(mission_btn)

            # Botón editar (texto claro)
            edit_btn = MDRaisedButton(
                text="Editar",
                size_hint_x=0.2,
                on_release=lambda x, idx=i: self.show_mission_dialog(idx)
            )
            self.mission_edit_buttons.append(edit_btn)

            mission_row.add_widget(mission_btn)
            mission_row.add_widget(edit_btn)
            self.mission_buttons.append(mission_row)
            missions_box.add_widget(mission_row)

        missions_card.add_widget(missions_box)
        layout.add_widget(missions_card)

        # Card de racha y estadísticas
        stats_card = MDCard(
            size_hint_y=None,
            height=dp(100),
            padding=dp(15),
            elevation=2,
            md_bg_color=(0.2, 0.5, 0.3, 1)
        )
        stats_box = MDBoxLayout(orientation='vertical', spacing=dp(5))

        self.streak_label = MDLabel(
            text="Racha: 0 dias",
            font_style="H6",
            halign="center",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1)
        )
        self.pending_label = MDLabel(
            text="0 tareas pendientes",
            halign="center",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=(0.9, 0.9, 0.9, 1)
        )

        stats_box.add_widget(self.streak_label)
        stats_box.add_widget(self.pending_label)
        stats_card.add_widget(stats_box)
        layout.add_widget(stats_card)

        # Botón acción rápida
        quick_btn = MDRaisedButton(
            text="+ Nueva Tarea",
            pos_hint={'center_x': 0.5},
            size_hint_x=0.8,
            size_hint_y=None,
            height=dp(50),
            md_bg_color=(0.4, 0.3, 0.6, 1),
            on_release=self.go_to_tasks
        )
        layout.add_widget(quick_btn)

        scroll.add_widget(layout)
        self.add_widget(scroll)
        self.refresh_stats()

    def go_to_tasks(self, *args):
        """Cambia a la pantalla de tareas y abre diálogo de agregar"""
        if self.parent and self.parent.parent:
            bottom_nav = self.parent.parent
            # Cambiar al tab de tareas
            for item in bottom_nav.children:
                if hasattr(item, 'name') and item.name == 'tasks':
                    bottom_nav.switch_to(item)
                    # Esperar un frame para que la pantalla esté activa
                    Clock.schedule_once(lambda dt: self.app.tasks_screen.show_add_dialog(), 0.1)
                    break

    def refresh_stats(self):
        """Actualiza estadísticas"""
        if not self.app:
            return

        stats = self.app.task_store.get_statistics()
        gam_status = self.app.gamification.get_status()
        streak_info = self.app.gamification.get_streak()

        # Actualizar nivel y XP
        self.level_label.text = f"Nivel {gam_status['level']}"
        self.xp_label.text = f"{gam_status['xp']} XP"

        # Calcular progreso al siguiente nivel
        progress_pct = gam_status.get('progress_percentage', 0)
        self.xp_progress_label.text = f"Progreso: {int(progress_pct)}% al siguiente nivel"

        # Actualizar misiones diarias
        missions = gam_status.get('missions', [])

        # Llenar los labels de misiones
        for i in range(3):
            if i < len(self.mission_labels):
                if i < len(missions):
                    mission = missions[i]
                    completed_icon = "[X]" if mission.get('completed') else "[ ]"
                    title = mission.get('title', '')
                    self.mission_labels[i].text = f"{completed_icon} {title}"
                else:
                    # Slot vacío - mostrar hint para crear misión
                    self.mission_labels[i].text = "+ Tap para crear mision"

        # Actualizar racha y tareas pendientes
        current_streak = streak_info.get('current_streak', 0)
        self.streak_label.text = f"Racha: {current_streak} dias"
        self.pending_label.text = f"{stats['pending']} tareas pendientes"

    def toggle_mission(self, index):
        """Marca/desmarca una misión como completada"""
        missions = self.app.gamification.get_missions()
        if index < len(missions):
            mission = missions[index]
            result = self.app.gamification.toggle_mission(mission['id'])
            self.refresh_stats()

            # Mostrar mensaje si ganó XP
            if result.get('mission_completed'):
                xp_gained = result.get('xp_gained', 0)
                # Opcionalmente mostrar toast/snackbar
                pass

    def show_mission_dialog(self, index):
        """Muestra diálogo para crear/editar misión"""
        missions = self.app.gamification.get_missions()
        self.current_mission_index = index

        # Verificar si existe una misión en este índice
        is_edit = index < len(missions)
        mission = missions[index] if is_edit else None

        content = MDBoxLayout(
            orientation='vertical',
            spacing=dp(10),
            size_hint_y=None,
            height=dp(120),
            padding=dp(10)
        )

        self.mission_title_field = MDTextField(
            hint_text="Título de la misión",
            text=mission['title'] if mission else "",
            mode="rectangle"
        )
        self.mission_xp_field = MDTextField(
            hint_text="Recompensa XP",
            text=str(mission.get('xp_reward', 50)) if mission else "50",
            mode="rectangle",
            input_filter="int"
        )

        content.add_widget(self.mission_title_field)
        content.add_widget(self.mission_xp_field)

        buttons = [
            MDFlatButton(
                text="CANCELAR",
                on_release=lambda x: self.mission_dialog.dismiss()
            ),
            MDRaisedButton(
                text="GUARDAR",
                md_bg_color=(0.2, 0.6, 0.3, 1),
                on_release=lambda x: self.save_mission(is_edit, mission['id'] if mission else None)
            )
        ]

        # Si está editando, agregar botón eliminar
        if is_edit:
            delete_btn = MDRaisedButton(
                text="ELIMINAR",
                md_bg_color=(0.9, 0.3, 0.3, 1),
                on_release=lambda x: self.delete_mission(mission['id'])
            )
            buttons.insert(1, delete_btn)

        self.mission_dialog = MDDialog(
            title="Editar Misión" if is_edit else "Nueva Misión",
            type="custom",
            content_cls=content,
            buttons=buttons
        )
        self.mission_dialog.open()

    def save_mission(self, is_edit, mission_id):
        """Guarda una misión (crear o actualizar)"""
        title = self.mission_title_field.text.strip()
        if not title:
            return

        try:
            xp_reward = int(self.mission_xp_field.text or "50")
        except ValueError:
            xp_reward = 50

        if is_edit:
            self.app.gamification.update_mission(mission_id, title, xp_reward)
        else:
            self.app.gamification.add_mission(title, xp_reward)

        self.mission_dialog.dismiss()
        self.refresh_stats()

    def delete_mission(self, mission_id):
        """Elimina una misión"""
        self.app.gamification.delete_mission(mission_id)
        self.mission_dialog.dismiss()
        self.refresh_stats()


class TasksScreen(MDScreen):
    """Pantalla de tareas"""

    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app = app_instance
        self.dialog = None
        self.build_ui()

    def build_ui(self):
        """Construye la interfaz"""
        layout = MDBoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))

        # Header con botón
        header = MDBoxLayout(size_hint_y=None, height=dp(60), padding=dp(5))

        title = MDLabel(
            text="Mis Tareas",
            font_style="H6",
            size_hint_x=0.6
        )

        add_btn = MDRaisedButton(
            text="+ Agregar",
            size_hint_x=0.4,
            md_bg_color=(0.2, 0.6, 0.3, 1),
            on_release=self.show_add_dialog
        )

        header.add_widget(title)
        header.add_widget(add_btn)
        layout.add_widget(header)

        # ScrollView con lista de tareas
        scroll = ScrollView()
        self.task_list = MDList()
        scroll.add_widget(self.task_list)
        layout.add_widget(scroll)

        self.add_widget(layout)
        self.refresh_tasks()

    def refresh_tasks(self):
        """Actualiza lista de tareas"""
        if not self.app:
            return

        self.task_list.clear_widgets()

        # Obtener todas las tareas (completadas y no completadas)
        all_tasks = self.app.task_store.get_tasks(filter_completed=None)

        # Separar en pendientes y completadas
        pending_tasks = [t for t in all_tasks if not t['completed']]
        completed_tasks = [t for t in all_tasks if t['completed']]

        if not all_tasks:
            no_tasks = MDLabel(
                text="No hay tareas\nCrea tu primera tarea!",
                halign="center",
                font_style="Body1",
                size_hint_y=None,
                height=dp(100)
            )
            self.task_list.add_widget(no_tasks)
        else:
            # Primero mostrar tareas pendientes
            if pending_tasks:
                pending_label = MDLabel(
                    text="PENDIENTES",
                    font_style="Subtitle1",
                    size_hint_y=None,
                    height=dp(35),
                    padding=(dp(10), dp(5))
                )
                self.task_list.add_widget(pending_label)

                for task in pending_tasks:
                    item = TaskItem(task=task, screen=self, app=self.app)
                    self.task_list.add_widget(item)

            # Luego mostrar tareas completadas
            if completed_tasks:
                completed_label = MDLabel(
                    text="COMPLETADAS",
                    font_style="Subtitle1",
                    size_hint_y=None,
                    height=dp(35),
                    padding=(dp(10), dp(5)),
                    theme_text_color="Custom",
                    text_color=(0.6, 0.6, 0.6, 1)
                )
                self.task_list.add_widget(completed_label)

                for task in completed_tasks:
                    item = TaskItem(task=task, screen=self, app=self.app)
                    self.task_list.add_widget(item)

    def show_add_dialog(self, *args):
        """Muestra diálogo para agregar tarea"""
        if not self.dialog:
            content = MDBoxLayout(
                orientation='vertical',
                spacing=dp(10),
                size_hint_y=None,
                height=dp(220),
                padding=dp(10)
            )

            self.title_field = MDTextField(
                hint_text="Título de la tarea",
                mode="rectangle"
            )
            self.desc_field = MDTextField(
                hint_text="Descripción (opcional)",
                mode="rectangle"
            )

            content.add_widget(self.title_field)
            content.add_widget(self.desc_field)

            # Label de prioridad
            priority_label = MDLabel(
                text="Prioridad:",
                size_hint_y=None,
                height=dp(20),
                font_style="Caption"
            )
            content.add_widget(priority_label)

            # Botones de prioridad
            priority_box = MDBoxLayout(
                size_hint_y=None,
                height=dp(50),
                spacing=dp(5)
            )

            self.selected_priority = "normal"
            self.priority_buttons = {}

            priorities = [
                ("urgent", "Urgente", (1, 0.27, 0.27, 1)),
                ("high", "Alta", (1, 0.55, 0, 1)),
                ("normal", "Normal", (1, 0.84, 0, 1)),
                ("low", "Baja", (0.3, 0.69, 0.31, 1))
            ]

            for priority, emoji, color in priorities:
                btn = MDRaisedButton(
                    text=emoji,
                    md_bg_color=color if priority == "normal" else (0.5, 0.5, 0.5, 1),
                    on_release=lambda x, p=priority: self.select_priority(p)
                )
                self.priority_buttons[priority] = btn
                priority_box.add_widget(btn)

            content.add_widget(priority_box)

            self.dialog = MDDialog(
                title="Nueva Tarea",
                type="custom",
                content_cls=content,
                buttons=[
                    MDFlatButton(
                        text="CANCELAR",
                        on_release=lambda x: self.dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="AGREGAR",
                        md_bg_color=(0.2, 0.6, 0.3, 1),
                        on_release=self.add_task
                    ),
                ],
            )

        self.dialog.open()

    def select_priority(self, priority):
        """Selecciona la prioridad de la tarea"""
        self.selected_priority = priority

        # Colores para cada prioridad
        colors = {
            "urgent": (1, 0.27, 0.27, 1),
            "high": (1, 0.55, 0, 1),
            "normal": (1, 0.84, 0, 1),
            "low": (0.3, 0.69, 0.31, 1)
        }

        # Actualizar colores de botones
        for p, btn in self.priority_buttons.items():
            if p == priority:
                btn.md_bg_color = colors[p]
            else:
                btn.md_bg_color = (0.5, 0.5, 0.5, 1)

    def add_task(self, *args):
        """Agrega una tarea"""
        title = self.title_field.text.strip()
        if not title:
            return

        desc = self.desc_field.text.strip()
        priority = getattr(self, 'selected_priority', 'normal')

        self.app.task_store.add_task(
            title=title,
            description=desc,
            priority=priority
        )

        # Limpiar y cerrar
        self.title_field.text = ""
        self.desc_field.text = ""
        self.selected_priority = "normal"  # Reset priority
        self.dialog.dismiss()

        # Refrescar
        self.refresh_tasks()

        # Actualizar home screen si existe
        if hasattr(self.app, 'home_screen'):
            self.app.home_screen.refresh_stats()


class TaskItem(MDBoxLayout):
    """Item de tarea mejorado con soporte para completar/descompletar y eliminar"""

    def __init__(self, task, screen, app, **kwargs):
        super().__init__(orientation='horizontal', size_hint_y=None, height=dp(80), spacing=dp(5), **kwargs)
        self.task = task
        self.screen = screen
        self.app = app
        self.delete_dialog = None

        is_completed = task['completed']

        # Información de la tarea
        info_box = MDBoxLayout(orientation='vertical', size_hint_x=0.55, padding=(dp(10), dp(5)))

        # Textos - aplicar estilo tachado si está completada
        title_label = MDLabel(
            text=f"[s]{task['title']}[/s]" if is_completed else task['title'],
            font_style="Subtitle1",
            markup=True,
            theme_text_color="Custom",
            text_color=(0.6, 0.6, 0.6, 1) if is_completed else (1, 1, 1, 1),
            size_hint_y=None,
            height=dp(25)
        )
        info_box.add_widget(title_label)

        desc_label = MDLabel(
            text=task['description'] if task['description'] else "Sin descripción",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=(0.7, 0.7, 0.7, 1),
            size_hint_y=None,
            height=dp(20)
        )
        info_box.add_widget(desc_label)

        # Prioridad con color
        priority_text = {
            "low": "Baja",
            "normal": "Normal",
            "high": "Alta",
            "urgent": "Urgente"
        }
        priority_colors = {
            "low": (0.3, 0.69, 0.31, 1),     # Verde
            "normal": (0.3, 0.6, 1, 1),       # Azul
            "high": (1, 0.55, 0, 1),          # Naranja
            "urgent": (1, 0.27, 0.27, 1)      # Rojo
        }
        priority_label = MDLabel(
            text=f"[{priority_text.get(task['priority'], 'Normal')}]",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=priority_colors.get(task['priority'], (1, 1, 1, 1)),
            size_hint_y=None,
            height=dp(20)
        )
        info_box.add_widget(priority_label)

        self.add_widget(info_box)

        # Botones en el lado derecho
        buttons_box = MDBoxLayout(orientation='horizontal', size_hint_x=0.45, spacing=dp(2))

        # Botón Pomodoro (solo para tareas no completadas)
        if not is_completed:
            # Verificar si ya está en la cola
            task_in_queue = task['id'] in app.pomodoro.get_tasks_in_queue()
            pomodoro_btn = MDRaisedButton(
                text="P" if not task_in_queue else "P-",
                size_hint_x=0.33,
                md_bg_color=(0.5, 0.3, 0.8, 1) if not task_in_queue else (0.3, 0.3, 0.3, 1),
                on_release=lambda x: self.toggle_pomodoro()
            )
            buttons_box.add_widget(pomodoro_btn)
            self.pomodoro_btn = pomodoro_btn

        # Si está completada, mostrar botón de desmarcar
        if is_completed:
            uncomplete_btn = MDRaisedButton(
                text="Deshacer",
                size_hint_x=0.5,
                md_bg_color=(0.8, 0.6, 0.2, 1),
                on_release=lambda x: self.uncomplete_task()
            )
            buttons_box.add_widget(uncomplete_btn)
        else:
            # Botón completar
            complete_btn = MDRaisedButton(
                text="OK",
                size_hint_x=0.33,
                md_bg_color=(0.2, 0.8, 0.3, 1),
                on_release=lambda x: self.toggle_complete()
            )
            buttons_box.add_widget(complete_btn)

        # Botón eliminar (siempre visible)
        delete_btn = MDRaisedButton(
            text="X",
            size_hint_x=0.33,
            md_bg_color=(0.9, 0.3, 0.3, 1),
            on_release=lambda x: self.show_delete_dialog()
        )
        buttons_box.add_widget(delete_btn)

        self.add_widget(buttons_box)

    def toggle_complete(self):
        """Marca tarea como completada"""
        if not self.task['completed']:
            self.app.task_store.complete_task(self.task['id'])
            result = self.app.gamification.on_task_completed(self.task['priority'])

            # Mostrar XP ganado
            if result and result.get('level_up'):
                # TODO: Mostrar diálogo de level up
                pass

            self.screen.refresh_tasks()

            # Actualizar home
            if hasattr(self.app, 'home_screen'):
                self.app.home_screen.refresh_stats()

    def uncomplete_task(self):
        """Desmarca tarea como no completada"""
        self.app.task_store.uncomplete_task(self.task['id'])
        self.screen.refresh_tasks()

        # Actualizar home
        if hasattr(self.app, 'home_screen'):
            self.app.home_screen.refresh_stats()

    def show_delete_dialog(self):
        """Muestra diálogo de confirmación para eliminar"""
        if not self.delete_dialog:
            self.delete_dialog = MDDialog(
                title="Eliminar Tarea",
                text=f"¿Estás seguro de eliminar la tarea '{self.task['title']}'?",
                buttons=[
                    MDFlatButton(
                        text="CANCELAR",
                        on_release=lambda x: self.delete_dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="ELIMINAR",
                        md_bg_color=(0.9, 0.3, 0.3, 1),
                        on_release=lambda x: self.delete_task()
                    ),
                ],
            )
        self.delete_dialog.open()

    def delete_task(self):
        """Elimina la tarea permanentemente"""
        self.app.task_store.delete_task(self.task['id'])
        self.delete_dialog.dismiss()
        self.screen.refresh_tasks()

        # Actualizar home
        if hasattr(self.app, 'home_screen'):
            self.app.home_screen.refresh_stats()

    def toggle_pomodoro(self):
        """Agrega o remueve la tarea de la cola del Pomodoro"""
        task_id = self.task['id']
        task_in_queue = task_id in self.app.pomodoro.get_tasks_in_queue()

        if task_in_queue:
            # Remover de la cola
            self.app.pomodoro.remove_task_from_queue(task_id)
            if hasattr(self, 'pomodoro_btn'):
                self.pomodoro_btn.text = "P"
                self.pomodoro_btn.md_bg_color = (0.5, 0.3, 0.8, 1)
        else:
            # Agregar a la cola
            self.app.pomodoro.add_task_to_queue(task_id)
            if hasattr(self, 'pomodoro_btn'):
                self.pomodoro_btn.text = "P-"
                self.pomodoro_btn.md_bg_color = (0.3, 0.3, 0.3, 1)

        # Actualizar pantalla de Pomodoro si existe
        if hasattr(self.app, 'pomodoro_screen'):
            self.app.pomodoro_screen.refresh_task_queue()


class PomodoroScreen(MDScreen):
    """Pantalla de Pomodoro con reproductor de música"""

    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app = app_instance
        self.timer_event = None
        self.music_dialog = None
        self.build_ui()

    def build_ui(self):
        """Construye la UI de Pomodoro"""
        layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(20),
            spacing=dp(20)
        )

        # Título
        title = MDLabel(
            text="Pomodoro Timer",
            font_style="H5",
            halign="center",
            size_hint_y=None,
            height=dp(40)
        )
        layout.add_widget(title)

        # Timer display grande
        self.timer_label = MDLabel(
            text="25:00",
            font_style="H2",
            halign="center",
            size_hint_y=None,
            height=dp(120),
            theme_text_color="Custom",
            text_color=(0.3, 0.6, 1, 1)
        )
        layout.add_widget(self.timer_label)

        # Fase actual
        self.phase_label = MDLabel(
            text="Tiempo de Trabajo",
            font_style="H6",
            halign="center",
            size_hint_y=None,
            height=dp(40)
        )
        layout.add_widget(self.phase_label)

        # Sesiones completadas
        self.sessions_label = MDLabel(
            text="0 sesiones completadas",
            halign="center",
            size_hint_y=None,
            height=dp(30),
            font_style="Caption"
        )
        layout.add_widget(self.sessions_label)

        # Card de música
        music_card = MDCard(
            size_hint_y=None,
            height=dp(120),
            padding=dp(10),
            elevation=2
        )
        music_box = MDBoxLayout(orientation='vertical', spacing=dp(5))

        music_title = MDLabel(
            text="Musica de Fondo",
            font_style="Caption",
            halign="center",
            size_hint_y=None,
            height=dp(20)
        )
        music_box.add_widget(music_title)

        self.current_track_label = MDLabel(
            text="Sin cancion",
            font_style="Body2",
            halign="center",
            size_hint_y=None,
            height=dp(30)
        )
        music_box.add_widget(self.current_track_label)

        # Controles de música (texto claro)
        music_controls = MDBoxLayout(
            spacing=dp(5),
            size_hint_y=None,
            height=dp(50)
        )

        prev_btn = MDRaisedButton(
            text="<",
            on_release=self.previous_track
        )
        music_controls.add_widget(prev_btn)

        self.play_music_btn = MDRaisedButton(
            text="Play",
            md_bg_color=(0.3, 0.6, 0.3, 1),
            on_release=self.toggle_music
        )
        music_controls.add_widget(self.play_music_btn)

        next_btn = MDRaisedButton(
            text=">",
            on_release=self.next_track
        )
        music_controls.add_widget(next_btn)

        select_btn = MDRaisedButton(
            text="Lista",
            on_release=self.show_playlist
        )
        music_controls.add_widget(select_btn)

        music_box.add_widget(music_controls)
        music_card.add_widget(music_box)
        layout.add_widget(music_card)

        # Card de tareas en cola
        tasks_card = MDCard(
            size_hint_y=None,
            height=dp(200),
            padding=dp(10),
            elevation=2
        )
        tasks_box = MDBoxLayout(orientation='vertical', spacing=dp(5))

        tasks_header = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(30),
            spacing=dp(5)
        )

        tasks_title = MDLabel(
            text="Tareas en Cola",
            font_style="Caption",
            size_hint_x=0.7
        )
        tasks_header.add_widget(tasks_title)

        clear_queue_btn = MDRaisedButton(
            text="Limpiar",
            size_hint_x=0.3,
            md_bg_color=(0.7, 0.3, 0.3, 1),
            on_release=self.clear_task_queue
        )
        tasks_header.add_widget(clear_queue_btn)
        tasks_box.add_widget(tasks_header)

        # ScrollView para la lista de tareas
        self.tasks_scroll = ScrollView(size_hint=(1, 1))
        self.tasks_queue_list = MDList()
        self.tasks_scroll.add_widget(self.tasks_queue_list)
        tasks_box.add_widget(self.tasks_scroll)

        tasks_card.add_widget(tasks_box)
        layout.add_widget(tasks_card)

        # Espacio
        layout.add_widget(MDLabel(size_hint_y=0.05))

        # Botones del timer
        btn_box = MDBoxLayout(
            spacing=dp(10),
            size_hint_y=None,
            height=dp(60),
            padding=dp(5)
        )

        self.start_btn = MDRaisedButton(
            text="Iniciar",
            pos_hint={'center_x': 0.5},
            md_bg_color=(0.2, 0.7, 0.3, 1),
            on_release=self.toggle_timer
        )
        btn_box.add_widget(self.start_btn)

        self.reset_btn = MDRaisedButton(
            text="Reset",
            pos_hint={'center_x': 0.5},
            md_bg_color=(0.7, 0.3, 0.3, 1),
            on_release=self.reset_timer
        )
        btn_box.add_widget(self.reset_btn)

        layout.add_widget(btn_box)
        self.add_widget(layout)

        # Actualizar etiqueta de canción actual
        self.update_music_display()

        # Actualizar lista de tareas en cola
        self.refresh_task_queue()

    def refresh_task_queue(self):
        """Actualiza la lista de tareas en la cola del Pomodoro"""
        from kivymd.uix.list import TwoLineListItem

        self.tasks_queue_list.clear_widgets()

        task_ids = self.app.pomodoro.get_tasks_in_queue()
        if not task_ids:
            # Mostrar mensaje si no hay tareas
            empty_label = MDLabel(
                text="No hay tareas en cola\nAgrega tareas desde la pantalla Tareas",
                halign="center",
                font_style="Caption",
                theme_text_color="Secondary"
            )
            self.tasks_queue_list.add_widget(empty_label)
            return

        # Obtener todas las tareas
        all_tasks = self.app.task_store.get_tasks(filter_completed=None)
        tasks_dict = {task['id']: task for task in all_tasks}

        # Mostrar tareas en cola
        for task_id in task_ids:
            if task_id in tasks_dict:
                task = tasks_dict[task_id]
                item = TwoLineListItem(
                    text=task['title'],
                    secondary_text=task['description'] if task['description'] else "Sin descripcion",
                    on_release=lambda x, tid=task_id: self.remove_task_from_queue(tid)
                )
                self.tasks_queue_list.add_widget(item)

    def remove_task_from_queue(self, task_id: str):
        """Remueve una tarea de la cola del Pomodoro"""
        self.app.pomodoro.remove_task_from_queue(task_id)
        self.refresh_task_queue()

    def clear_task_queue(self, *args):
        """Limpia toda la cola de tareas"""
        self.app.pomodoro.clear_queue()
        self.refresh_task_queue()

    def toggle_timer(self, *args):
        """Inicia/pausa el timer"""
        if not self.app.pomodoro.is_running:
            self.app.pomodoro.start()
            self.start_btn.text = "Pausar"
            self.timer_event = Clock.schedule_interval(self.update_timer, 1)
        elif self.app.pomodoro.is_paused:
            self.app.pomodoro.resume()
            self.start_btn.text = "Pausar"
        else:
            self.app.pomodoro.pause()
            self.start_btn.text = "Reanudar"

    def reset_timer(self, *args):
        """Resetea el timer"""
        self.app.pomodoro.reset()
        self.start_btn.text = "Iniciar"
        if self.timer_event:
            self.timer_event.cancel()
        self.update_display()

    def update_timer(self, dt):
        """Actualiza el timer cada segundo"""
        changed = self.app.pomodoro.update()
        self.update_display()

        if changed and not self.app.pomodoro.is_work_time:
            # Completó trabajo, obtuvo XP
            self.app.gamification.on_pomodoro_completed()

            # Actualizar home
            if hasattr(self.app, 'home_screen'):
                self.app.home_screen.refresh_stats()

    def update_display(self):
        """Actualiza la visualización"""
        status = self.app.pomodoro.get_status()

        self.timer_label.text = f"{status['minutes']:02d}:{status['seconds']:02d}"

        if status['is_work_time']:
            self.phase_label.text = "Tiempo de Trabajo"
            self.timer_label.text_color = (0.3, 0.6, 1, 1)
        else:
            self.phase_label.text = "Descanso"
            self.timer_label.text_color = (0.3, 0.8, 0.4, 1)

        self.sessions_label.text = f"{status['sessions_completed']} sesiones completadas"

    def toggle_music(self, *args):
        """Reproduce/pausa la música"""
        if self.app.music_player.is_playing:
            self.app.music_player.pause()
            self.play_music_btn.text = "Play"
        else:
            self.app.music_player.play()
            self.play_music_btn.text = "Pausa"
        self.update_music_display()

    def next_track(self, *args):
        """Siguiente canción"""
        self.app.music_player.next_track()
        self.update_music_display()

    def previous_track(self, *args):
        """Canción anterior"""
        self.app.music_player.previous_track()
        self.update_music_display()

    def update_music_display(self):
        """Actualiza el nombre de la canción actual"""
        track_name = self.app.music_player.get_current_track_name()
        # Truncar si es muy largo
        if len(track_name) > 35:
            track_name = track_name[:32] + "..."
        self.current_track_label.text = track_name

    def show_playlist(self, *args):
        """Muestra diálogo con la lista de canciones"""
        from kivymd.uix.list import OneLineListItem

        songs = self.app.music_player.get_playlist_names()
        if not songs:
            return

        # Crear contenedor con altura fija
        content = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=dp(10),
            spacing=dp(5)
        )

        # Calcular altura basada en número de canciones (máximo 400dp)
        item_height = dp(48)
        max_height = min(len(songs) * item_height, dp(400))
        content.height = max_height

        # Crear lista de canciones
        song_list = MDList()
        for i, song_name in enumerate(songs):
            item = OneLineListItem(
                text=song_name,
                on_release=lambda x, idx=i: self.select_song(idx)
            )
            song_list.add_widget(item)

        # Scroll view con altura fija
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(song_list)
        content.add_widget(scroll)

        self.music_dialog = MDDialog(
            title="Seleccionar Cancion",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(
                    text="CERRAR",
                    on_release=lambda x: self.music_dialog.dismiss()
                )
            ]
        )
        self.music_dialog.open()

    def select_song(self, index):
        """Selecciona una canción de la playlist"""
        self.app.music_player.play(index)
        self.play_music_btn.text = "Pausa"
        self.update_music_display()
        if self.music_dialog:
            self.music_dialog.dismiss()


class PositsMobileApp(MDApp):
    """App principal"""

    def build(self):
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.theme_style = "Dark"

        # Inicializar servicios
        self.task_store = TaskStore()
        self.gamification = GamificationSystem()
        self.pomodoro = PomodoroTimer()
        self.music_player = SimpleMusicPlayer()

        # Crear pantallas pasando la instancia de la app
        self.home_screen = HomeScreen(self, name='home')
        self.tasks_screen = TasksScreen(self, name='tasks')
        self.pomodoro_screen = PomodoroScreen(self, name='pomodoro')

        # Bottom navigation
        self.bottom_nav = MDBottomNavigation()

        # Items del bottom nav (solo texto, sin íconos)
        home_item = MDBottomNavigationItem(
            name='home',
            text='Inicio',
            icon='home'  # Requerido por KivyMD pero no se mostrará
        )
        home_item.add_widget(self.home_screen)

        tasks_item = MDBottomNavigationItem(
            name='tasks',
            text='Tareas',
            icon='checkbox-marked-outline'
        )
        tasks_item.add_widget(self.tasks_screen)

        pomodoro_item = MDBottomNavigationItem(
            name='pomodoro',
            text='Pomodoro',
            icon='timer'
        )
        pomodoro_item.add_widget(self.pomodoro_screen)

        self.bottom_nav.add_widget(home_item)
        self.bottom_nav.add_widget(tasks_item)
        self.bottom_nav.add_widget(pomodoro_item)

        return self.bottom_nav


if __name__ == '__main__':
    PositsMobileApp().run()

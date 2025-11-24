"""
Prueba simple de íconos de KivyMD
"""

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDIconButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel


class IconTestApp(MDApp):
    def build(self):
        screen = MDScreen()
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)

        # Título
        layout.add_widget(MDLabel(
            text="Test de Íconos",
            halign="center",
            font_style="H5"
        ))

        # Probar varios íconos
        icons = [
            "home",
            "pencil",
            "check-circle",
            "delete",
            "play",
            "pause",
            "skip-next",
            "skip-previous",
            "playlist-music",
            "timer"
        ]

        for icon_name in icons:
            btn_layout = MDBoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=60,
                spacing=10
            )

            btn = MDIconButton(
                icon=icon_name,
                md_bg_color=(0.3, 0.6, 0.8, 1)
            )

            label = MDLabel(
                text=icon_name,
                size_hint_x=0.7
            )

            btn_layout.add_widget(btn)
            btn_layout.add_widget(label)
            layout.add_widget(btn_layout)

        screen.add_widget(layout)
        return screen


if __name__ == '__main__':
    IconTestApp().run()

"""
Módulo de servicios
"""

from .ollama import extract_task_with_ollama, ollama_chat_json
from .stt import stt_from_audio_bytes, PushToTalkRecorder
from .attachments import AttachmentManager
from .notifications import NotificationService
from .youtube_downloader import YouTubeDownloader

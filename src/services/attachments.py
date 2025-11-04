#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servicio de gestión de archivos adjuntos
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from ..config import ATTACHMENTS_DIR, ALLOWED_ATTACHMENT_EXTENSIONS, MAX_ATTACHMENT_SIZE_MB
from ..utils.logger import logger


class AttachmentManager:
    """
    Gestiona archivos adjuntos a las tareas.
    Copia archivos a la carpeta de attachments y mantiene referencias.
    """

    def __init__(self, attachments_dir: Path = ATTACHMENTS_DIR):
        self.attachments_dir = attachments_dir
        self.attachments_dir.mkdir(parents=True, exist_ok=True)

    def add_attachment(self, source_path: str | Path, task_id: str) -> Optional[str]:
        """
        Copia un archivo a la carpeta de attachments.

        Args:
            source_path: Ruta del archivo original
            task_id: ID de la tarea a la que se adjunta

        Returns:
            Nombre del archivo guardado (relativo a attachments_dir) o None si falla
        """
        source_path = Path(source_path)

        # Validaciones
        if not source_path.exists():
            logger.error(f"Archivo no existe: {source_path}")
            return None

        if source_path.suffix.lower() not in ALLOWED_ATTACHMENT_EXTENSIONS:
            logger.warning(f"Extensión no permitida: {source_path.suffix}")
            return None

        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_ATTACHMENT_SIZE_MB:
            logger.warning(f"Archivo muy grande: {file_size_mb:.2f} MB")
            return None

        # Generar nombre único
        unique_name = f"{task_id}_{uuid.uuid4().hex[:8]}{source_path.suffix}"
        dest_path = self.attachments_dir / unique_name

        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Adjunto guardado: {unique_name}")
            return unique_name
        except Exception as e:
            logger.error(f"Error copiando adjunto: {e}")
            return None

    def get_attachment_path(self, attachment_name: str) -> Path:
        """Retorna la ruta completa de un adjunto"""
        return self.attachments_dir / attachment_name

    def delete_attachment(self, attachment_name: str) -> bool:
        """Elimina un archivo adjunto"""
        try:
            file_path = self.get_attachment_path(attachment_name)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Adjunto eliminado: {attachment_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando adjunto: {e}")
            return False

    def delete_task_attachments(self, task_id: str):
        """Elimina todos los adjuntos de una tarea"""
        for file_path in self.attachments_dir.glob(f"{task_id}_*"):
            try:
                file_path.unlink()
                logger.info(f"Adjunto eliminado: {file_path.name}")
            except Exception as e:
                logger.error(f"Error eliminando adjunto: {e}")

    def get_attachment_size(self, attachment_name: str) -> Optional[int]:
        """Retorna el tamaño de un adjunto en bytes"""
        try:
            file_path = self.get_attachment_path(attachment_name)
            if file_path.exists():
                return file_path.stat().st_size
            return None
        except Exception:
            return None

    def format_size(self, size_bytes: int) -> str:
        """Formatea un tamaño en bytes a formato legible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

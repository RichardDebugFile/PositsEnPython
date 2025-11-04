#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración del sistema de logging
"""

import logging
import traceback
from ..config import LOG_FILE


def setup_logger(name: str = "posits", level: int = logging.DEBUG) -> logging.Logger:
    """
    Configura y retorna un logger con handlers de archivo y consola
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers si ya está configurado
    if logger.handlers:
        return logger

    # Handler de archivo
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Handler de consola
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


def log_exception(logger: logging.Logger, prefix: str, e: Exception):
    """
    Registra una excepción con su traceback completo
    """
    logger.error("%s: %s\n%s", prefix, e, traceback.format_exc())


# Logger global de la aplicación
logger = setup_logger()

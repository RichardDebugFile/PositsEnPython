#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para manejo de fechas
"""

from datetime import datetime, date
from ..config import ISO_FMT


def today_date() -> date:
    """Retorna la fecha actual"""
    return datetime.now().date()


def parse_date(s: str) -> date:
    """Convierte string YYYY-MM-DD a objeto date"""
    return datetime.strptime(s, ISO_FMT).date()


def fmt_date(d: date) -> str:
    """Convierte objeto date a string YYYY-MM-DD"""
    return d.strftime(ISO_FMT)


def valid_date(s: str) -> bool:
    """Valida si un string es una fecha válida en formato YYYY-MM-DD"""
    try:
        parse_date(s.strip())
        return True
    except Exception:
        return False


def days_until(target_date: date) -> int:
    """Retorna cuántos días faltan hasta una fecha (negativo si ya pasó)"""
    return (target_date - today_date()).days


def is_overdue(target_date: date) -> bool:
    """Retorna True si la fecha ya pasó"""
    return target_date < today_date()


def is_today(target_date: date) -> bool:
    """Retorna True si la fecha es hoy"""
    return target_date == today_date()


def is_this_week(target_date: date) -> bool:
    """Retorna True si la fecha está en esta semana"""
    delta = days_until(target_date)
    return 0 <= delta <= 7

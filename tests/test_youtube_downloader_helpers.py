#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitarios de los helpers puros de YouTubeDownloader (sin red ni descargas).
"""

import pytest

from src.services.youtube_downloader import YouTubeDownloader


@pytest.fixture
def downloader(tmp_path):
    # output_dir temporal: NO toca data/music real
    return YouTubeDownloader(output_dir=str(tmp_path / "music"))


class TestInicializacion:
    def test_crea_directorio_de_salida(self, downloader, tmp_path):
        assert (tmp_path / "music").exists()

    def test_estado_inicial(self, downloader):
        assert downloader.is_downloading is False


class TestFormatBytes:
    @pytest.mark.parametrize("value,unidad", [
        (500, "B"),
        (1024, "KB"),
        (1048576, "MB"),
        (1073741824, "GB"),
    ])
    def test_unidades(self, downloader, value, unidad):
        assert downloader._format_bytes(value).endswith(unidad)

    def test_none_es_interrogante(self, downloader):
        assert downloader._format_bytes(None) == "?"


class TestFormatEta:
    @pytest.mark.parametrize("segundos,esperado", [
        (30, "0:30"),
        (90, "1:30"),
        (3665, "1:01:05"),
        (0, ""),
        (None, ""),
    ])
    def test_eta(self, downloader, segundos, esperado):
        assert downloader._format_eta(segundos) == esperado


class TestSanitizeFilename:
    def test_elimina_caracteres_prohibidos(self, downloader):
        sucio = 'a<b>c:d"e/f\\g|h?i*j'
        limpio = downloader._sanitize_filename(sucio)
        assert not any(c in limpio for c in '<>:"/\\|?*')

    def test_colapsa_espacios(self, downloader):
        assert "  " not in downloader._sanitize_filename("hola    mundo")

    def test_limita_longitud(self, downloader):
        largo = "x" * 300 + ".mp3"
        assert len(downloader._sanitize_filename(largo)) <= 204

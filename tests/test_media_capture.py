#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitarios de src/utils/media_capture.py (funciones puras, sin UI ni red).
"""

import pytest

from src.utils.media_capture import sanitize_youtube_url, clean_music_title


class TestSanitizeYoutubeUrl:
    def test_quita_parametros_de_playlist(self):
        url = "https://www.youtube.com/watch?v=H2kUfHhAL3M&list=RDMM&index=2"
        assert sanitize_youtube_url(url) == "https://www.youtube.com/watch?v=H2kUfHhAL3M"

    def test_formato_corto_youtu_be(self):
        url = "https://youtu.be/H2kUfHhAL3M?si=xyz123"
        assert sanitize_youtube_url(url) == "https://www.youtube.com/watch?v=H2kUfHhAL3M"

    def test_formato_embed(self):
        url = "https://www.youtube.com/embed/H2kUfHhAL3M"
        assert sanitize_youtube_url(url) == "https://www.youtube.com/watch?v=H2kUfHhAL3M"

    def test_url_ya_limpia_se_mantiene(self):
        url = "https://www.youtube.com/watch?v=abcdefghijk"
        assert sanitize_youtube_url(url) == url

    def test_url_no_youtube_pasa_igual(self):
        assert sanitize_youtube_url("https://example.com/x?y=1") == "https://example.com/x?y=1"

    def test_texto_plano_pasa_igual(self):
        assert sanitize_youtube_url("buscar esta cancion") == "buscar esta cancion"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_valores_vacios_o_none(self, value):
        # None y "" se devuelven tal cual; "   " no es URL de YouTube -> se devuelve trim
        result = sanitize_youtube_url(value)
        assert result == value or result == (value or "").strip()


class TestCleanMusicTitle:
    def test_quita_prefijo_de_canal(self):
        assert clean_music_title("Sheer's HD Video & Music - ZERO") == "ZERO"

    def test_quita_sufijo_official(self):
        assert clean_music_title("Song Name (Official Video)") == "Song Name"

    def test_corta_en_pipe(self):
        assert clean_music_title("Artist - Track | Lyrics") == "Artist - Track"

    def test_colapsa_espacios(self):
        assert clean_music_title("a      b") == "a b"

    @pytest.mark.parametrize("value", ["", None])
    def test_vacio_devuelve_cadena_vacia(self, value):
        assert clean_music_title(value) == ""

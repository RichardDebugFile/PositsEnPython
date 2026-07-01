#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests del helper de Spotify (normalizacion de URL/URI, sin abrir la app).
"""

import pytest

from src.services.spotify import normalize_spotify_uri


class TestNormalizeSpotifyUri:
    def test_uri_passthrough(self):
        uri = "spotify:playlist:37i9dQZF1DWZeKCadgRdKQ"
        assert normalize_spotify_uri(uri) == uri

    def test_url_playlist_con_query(self):
        url = "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ?si=abc123"
        assert normalize_spotify_uri(url) == "spotify:playlist:37i9dQZF1DWZeKCadgRdKQ"

    def test_url_track(self):
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        assert normalize_spotify_uri(url) == "spotify:track:4cOdK2wGLETKBW3PvgPWqT"

    @pytest.mark.parametrize("value", ["", None, "hola mundo", "https://youtube.com/x"])
    def test_no_reconocido_devuelve_none(self, value):
        assert normalize_spotify_uri(value) is None

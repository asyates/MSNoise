"""Unit tests for msnoise.core.fdsn pure-logic helpers.

Covers functions that require no network, no ObsPy client, and no DB:
  - parse_datasource_scheme
  - is_remote_source
  - get_auth
"""
import os
import pytest

from ..core.fdsn import get_auth, is_remote_source, parse_datasource_scheme


# ---------------------------------------------------------------------------
# parse_datasource_scheme
# ---------------------------------------------------------------------------

class TestParseDatasourceScheme:
    def test_empty_string_is_local(self):
        assert parse_datasource_scheme("") == "local"

    def test_none_is_local(self):
        assert parse_datasource_scheme(None) == "local"

    def test_sds_scheme(self):
        assert parse_datasource_scheme("sds://") == "sds"

    def test_fdsn_scheme(self):
        assert parse_datasource_scheme("fdsn://http://service.iris.edu") == "fdsn"

    def test_eida_scheme(self):
        assert parse_datasource_scheme("edsn://eida-routing") == "local"  # unknown → local
        assert parse_datasource_scheme("eida://eida-routing") == "eida"

    def test_unknown_scheme_is_local(self):
        assert parse_datasource_scheme("sftp://server/path") == "local"

    def test_plain_path_is_local(self):
        assert parse_datasource_scheme("/data/archive") == "local"


# ---------------------------------------------------------------------------
# is_remote_source
# ---------------------------------------------------------------------------

class TestIsRemoteSource:
    def test_fdsn_is_remote(self):
        assert is_remote_source("fdsn://http://service.iris.edu") is True

    def test_eida_is_remote(self):
        assert is_remote_source("eida://eida-routing") is True

    def test_local_is_not_remote(self):
        assert is_remote_source("") is False
        assert is_remote_source(None) is False
        assert is_remote_source("/data/archive") is False

    def test_sds_is_not_remote(self):
        assert is_remote_source("sds://") is False


# ---------------------------------------------------------------------------
# get_auth
# ---------------------------------------------------------------------------

class TestGetAuth:
    def test_returns_dict_with_three_keys(self):
        result = get_auth("MSNOISE")
        assert set(result.keys()) == {"user", "password", "token"}

    def test_missing_env_vars_return_none(self, monkeypatch):
        monkeypatch.delenv("TESTPREFIX_FDSN_USER", raising=False)
        monkeypatch.delenv("TESTPREFIX_FDSN_PASSWORD", raising=False)
        monkeypatch.delenv("TESTPREFIX_FDSN_TOKEN", raising=False)
        result = get_auth("TESTPREFIX")
        assert result["user"] is None
        assert result["password"] is None
        assert result["token"] is None

    def test_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("MYPREFIX_FDSN_USER", "alice")
        monkeypatch.setenv("MYPREFIX_FDSN_PASSWORD", "secret")
        monkeypatch.setenv("MYPREFIX_FDSN_TOKEN", "/path/to/token")
        result = get_auth("MYPREFIX")
        assert result["user"] == "alice"
        assert result["password"] == "secret"
        assert result["token"] == "/path/to/token"

    def test_prefix_is_uppercased(self, monkeypatch):
        monkeypatch.setenv("LOWER_FDSN_USER", "bob")
        result = get_auth("lower")
        assert result["user"] == "bob"

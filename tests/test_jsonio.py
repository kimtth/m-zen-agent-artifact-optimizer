import pytest

from zen.domain.core import JsonResponseError, parse_json


def test_parse_json_accepts_fence() -> None:
    assert parse_json('```json\n{"언어":"한국어"}\n```') == {"언어": "한국어"}


def test_parse_json_rejects_trailing_prose() -> None:
    with pytest.raises(JsonResponseError):
        parse_json('{"ok": true} trailing')

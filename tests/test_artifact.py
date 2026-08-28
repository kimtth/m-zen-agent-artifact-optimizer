from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from zen.domain.core import ArtifactError, load_artifact


def test_frontmatter_is_frozen_and_encoding_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "한국어.instructions.md"
    source = "---\r\napplyTo: '**/*.py'\r\n---\r\n결과를 먼저 쓰세요.\r\n"
    path.write_bytes(codecs.BOM_UTF8 + source.encode())

    artifact = load_artifact(path)
    candidate = artifact.candidate_bytes("결과를 쓰세요.\n")

    assert candidate.startswith(codecs.BOM_UTF8)
    decoded = candidate[len(codecs.BOM_UTF8) :].decode()
    assert decoded.startswith("---\r\napplyTo: '**/*.py'\r\n---\r\n")
    assert decoded.endswith("결과를 쓰세요.\r\n")


def test_malformed_frontmatter_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.prompt.md"
    path.write_text("---\nname: bad\nbody", encoding="utf-8")

    with pytest.raises(ArtifactError, match="closing"):
        load_artifact(path)


def test_invalid_yaml_frontmatter_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.agent.md"
    path.write_text("---\ntools: [read_file\n---\nbody", encoding="utf-8")

    with pytest.raises(ArtifactError, match="invalid YAML"):
        load_artifact(path)


def test_unrecognized_markdown_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Instructions", encoding="utf-8")

    with pytest.raises(ArtifactError, match="unsupported"):
        load_artifact(path)

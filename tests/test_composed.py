# Copyright 2026 Query Farm LLC - https://query.farm

"""Placement guards for the composed example queries (vgi_audio.composed).

Each composed example must appear in the schema-level ``vgi.example_queries`` list
AND within the first five examples of the object that drives it (the only lists
the vgi-lint agent simulator shows). Execution against a live worker is covered
by ``test/sql/composed.test``.
"""

from __future__ import annotations

import base64
import json

import pytest

from vgi_audio import composed, scalars, tables, worker

_ALL = (composed.BEAT_GRID_VS_TEMPO, composed.LOUD_CLIPS_PROFILE, composed.MFCC_TIMBRE_DISTANCE)


def _pairs(tag_value: str) -> list[tuple[str, str]]:
    return [(q["description"], q["sql"]) for q in json.loads(tag_value)]


@pytest.mark.parametrize("example", _ALL)
def test_composed_example_in_schema_list(example: tuple[str, str]) -> None:
    """Every composed example is published at schema level."""
    assert example in _pairs(worker._SCHEMA_EXAMPLE_QUERIES)


@pytest.mark.parametrize(
    ("example", "tag_value", "native_lists"),
    [
        (
            composed.BEAT_GRID_VS_TEMPO,
            tables._TAGS_BEATS["vgi.example_queries"],
            [tables.BeatsPathFunction.Meta.examples, tables.BeatsBytesFunction.Meta.examples],
        ),
        (composed.LOUD_CLIPS_PROFILE, tables.CLIPS_TABLE.tags["vgi.example_queries"], []),
        (
            composed.MFCC_TIMBRE_DISTANCE,
            scalars._TAGS_MFCC["vgi.example_queries"],
            [scalars.MfccPathFunction.Meta.examples, scalars.MfccBytesFunction.Meta.examples],
        ),
    ],
)
def test_composed_example_on_driving_object(
    example: tuple[str, str], tag_value: str, native_lists: list[list[object]]
) -> None:
    """The driving object shows the composed example within its first five."""
    assert example in _pairs(tag_value)[:5]
    for examples in native_lists:
        assert example[1] in [e.sql for e in examples[:5]]  # type: ignore[attr-defined]


@pytest.mark.parametrize("tag_value", [worker._SCHEMA_EXAMPLE_QUERIES, tables._TAGS_BEATS["vgi.example_queries"]])
def test_example_lists_fit_vgi417_bound(tag_value: str) -> None:
    """Example lists carrying the ~3 KB click blob stay under the 8,000-char bound."""
    assert len(tag_value) <= 8000


def test_example_clips_audio_doc_matches_formats() -> None:
    """The click clip is FLAC, so the audio column must not be documented as WAV-only."""
    formats = {base64.b64decode(b64)[:4] for _, _, b64 in tables._EXAMPLE_CLIPS}
    assert formats == {b"RIFF", b"fLaC"}
    for doc in (
        tables.CLIPS_TABLE.column_comments["audio"],
        tables._CLIPS_SCHEMA.field("audio").metadata[b"comment"].decode(),
        tables._CLIPS_DOC_MD,
    ):
        assert "FLAC" in doc

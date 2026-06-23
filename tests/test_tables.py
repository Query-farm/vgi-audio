"""Integration tests for the audio table functions.

Drives ``beats`` and ``audio_info`` through the real bind -> init -> process
lifecycle in-process (no worker subprocess), for both the path and bytes
overloads.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

from vgi_audio.tables import (
    AudioInfoBytesFunction,
    AudioInfoPathFunction,
    BeatsBytesFunction,
    BeatsPathFunction,
)

from .harness import invoke_table_function
from .synth import click_track_bytes, sine_wav_bytes


@pytest.fixture(scope="module")
def tone_bytes() -> bytes:
    return sine_wav_bytes(freq=440.0, sr=22050, seconds=2.0)


@pytest.fixture(scope="module")
def tone_path(tone_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(tone_bytes)
    return path


@pytest.fixture(scope="module")
def click_bytes() -> bytes:
    return click_track_bytes(bpm=120.0, sr=22050, seconds=6.0)


class TestBeats:
    def test_columns_and_rows_bytes(self, click_bytes: bytes) -> None:
        table = invoke_table_function(BeatsBytesFunction, positional=(pa.scalar(click_bytes, type=pa.binary()),))
        assert table.column_names == ["seq", "time"]
        assert table.num_rows > 3
        times = table.column("time").to_pylist()
        assert times == sorted(times)
        assert table.column("seq").to_pylist() == list(range(table.num_rows))

    def test_path_overload(self, tone_path: str) -> None:
        table = invoke_table_function(BeatsPathFunction, positional=(pa.scalar(tone_path, type=pa.string()),))
        assert table.column_names == ["seq", "time"]

    def test_garbage_no_rows(self) -> None:
        table = invoke_table_function(BeatsBytesFunction, positional=(pa.scalar(b"not audio", type=pa.binary()),))
        assert table.num_rows == 0


class TestAudioInfo:
    def test_single_row_bytes(self, tone_bytes: bytes) -> None:
        table = invoke_table_function(AudioInfoBytesFunction, positional=(pa.scalar(tone_bytes, type=pa.binary()),))
        assert table.column_names == ["duration", "sample_rate", "channels"]
        assert table.num_rows == 1
        assert table.column("sample_rate").to_pylist() == [22050]
        assert table.column("channels").to_pylist() == [1]
        assert abs(table.column("duration").to_pylist()[0] - 2.0) < 1e-3

    def test_path_overload(self, tone_path: str) -> None:
        table = invoke_table_function(AudioInfoPathFunction, positional=(pa.scalar(tone_path, type=pa.string()),))
        assert table.num_rows == 1
        assert table.column("sample_rate").to_pylist() == [22050]

    def test_garbage_no_rows(self) -> None:
        table = invoke_table_function(AudioInfoBytesFunction, positional=(pa.scalar(b"junk", type=pa.binary()),))
        assert table.num_rows == 0

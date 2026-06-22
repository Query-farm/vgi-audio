"""End-to-end tests for the per-row scalar audio functions.

Spawns ``audio_worker.py`` as a subprocess via ``vgi.client.Client`` and calls
each scalar exactly as DuckDB would after ``ATTACH``, exercising both the path
(VARCHAR) and bytes (BLOB) overloads, plus the ``mfcc`` arity overload. The
audio column travels in the input batch; the constant ``n`` argument goes in
``positional``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

from .synth import click_track_bytes, sine_wav_bytes

_WORKER = str(Path(__file__).resolve().parent.parent / "audio_worker.py")


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


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


def _scalar_bytes(client: Client, name: str, blobs: list, *, positional: list | None = None) -> list:
    batch = pa.RecordBatch.from_pydict({"b": pa.array(blobs, type=pa.binary())})
    results = list(
        client.scalar_function(
            function_name=name,
            input=iter([batch]),
            arguments=Arguments(positional=positional or []),
        )
    )
    return results[0]["result"].to_pylist()


def _scalar_path(client: Client, name: str, paths: list, *, positional: list | None = None) -> list:
    batch = pa.RecordBatch.from_pydict({"p": pa.array(paths, type=pa.string())})
    results = list(
        client.scalar_function(
            function_name=name,
            input=iter([batch]),
            arguments=Arguments(positional=positional or []),
        )
    )
    return results[0]["result"].to_pylist()


class TestMetadataScalars:
    def test_sample_rate_bytes(self, client: Client, tone_bytes: bytes) -> None:
        assert _scalar_bytes(client, "sample_rate", [tone_bytes]) == [22050]

    def test_sample_rate_path(self, client: Client, tone_path: str) -> None:
        assert _scalar_path(client, "sample_rate", [tone_path]) == [22050]

    def test_duration_bytes(self, client: Client, tone_bytes: bytes) -> None:
        d = _scalar_bytes(client, "duration", [tone_bytes])[0]
        assert d is not None and abs(d - 2.0) < 1e-3

    def test_channels_bytes(self, client: Client, tone_bytes: bytes) -> None:
        assert _scalar_bytes(client, "channels", [tone_bytes]) == [1]


class TestFeatureScalars:
    def test_spectral_centroid_near_tone(self, client: Client, tone_bytes: bytes) -> None:
        sc = _scalar_bytes(client, "spectral_centroid", [tone_bytes])[0]
        assert sc is not None and 350.0 < sc < 900.0

    def test_rms_and_zcr(self, client: Client, tone_bytes: bytes) -> None:
        assert _scalar_bytes(client, "rms_energy", [tone_bytes])[0] > 0.0
        zcr = _scalar_bytes(client, "zero_crossing_rate", [tone_bytes])[0]
        assert 0.0 <= zcr <= 1.0

    def test_spectral_bandwidth(self, client: Client, tone_bytes: bytes) -> None:
        assert _scalar_bytes(client, "spectral_bandwidth", [tone_bytes])[0] > 0.0

    def test_tempo_near_120(self, client: Client, click_bytes: bytes) -> None:
        bpm = _scalar_bytes(client, "tempo", [click_bytes])[0]
        assert bpm is not None
        assert 100.0 < bpm < 140.0 or 55.0 < bpm < 70.0 or 230.0 < bpm < 250.0

    def test_estimated_key_format(self, client: Client, tone_bytes: bytes) -> None:
        key = _scalar_bytes(client, "estimated_key", [tone_bytes])[0]
        assert key is None or key.split()[-1] in {"major", "minor"}


class TestMfccOverloads:
    def test_default_13(self, client: Client, tone_bytes: bytes) -> None:
        coeffs = _scalar_bytes(client, "mfcc", [tone_bytes])[0]
        assert coeffs is not None and len(coeffs) == 13

    def test_n_overload(self, client: Client, tone_bytes: bytes) -> None:
        coeffs = _scalar_bytes(client, "mfcc", [tone_bytes], positional=[pa.scalar(20, type=pa.int64())])[0]
        assert coeffs is not None and len(coeffs) == 20

    def test_mfcc_path(self, client: Client, tone_path: str) -> None:
        coeffs = _scalar_path(client, "mfcc", [tone_path])[0]
        assert coeffs is not None and len(coeffs) == 13


class TestHostileInputE2E:
    def test_garbage_bytes_null(self, client: Client) -> None:
        garbage = b"\x00not-audio\xff" * 50
        assert _scalar_bytes(client, "duration", [garbage]) == [None]
        assert _scalar_bytes(client, "tempo", [garbage]) == [None]
        assert _scalar_bytes(client, "sample_rate", [garbage]) == [None]
        assert _scalar_bytes(client, "mfcc", [garbage]) == [None]

    def test_empty_bytes_null(self, client: Client) -> None:
        assert _scalar_bytes(client, "duration", [b""]) == [None]

    def test_null_input(self, client: Client) -> None:
        assert _scalar_bytes(client, "duration", [None]) == [None]
        assert _scalar_path(client, "duration", [None]) == [None]

    def test_nonexistent_path_null(self, client: Client) -> None:
        assert _scalar_path(client, "duration", ["/no/such/file.wav"]) == [None]

    def test_mixed_good_and_bad(self, client: Client, tone_bytes: bytes) -> None:
        # A bad blob next to a good one must not poison the good result.
        out = _scalar_bytes(client, "sample_rate", [tone_bytes, b"junk", None])
        assert out == [22050, None, None]

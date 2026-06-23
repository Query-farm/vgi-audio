# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python>=0.8.3",
#     "librosa>=0.10",
#     "soundfile>=0.12",
#     "numpy",
# ]
# ///
"""VGI worker exposing librosa audio-feature extraction to SQL.

Assembles the audio functions in ``vgi_audio`` into a single ``audio`` catalog
and runs the worker over stdio (DuckDB subprocess) or HTTP. Each feature accepts
either a ``VARCHAR`` filesystem path (the worker opens the file) or a ``BLOB`` of
raw audio bytes.

Native decoding (WAV/FLAC/OGG) needs no external tools (libsndfile via
soundfile). Compressed formats (mp3, m4a, ...) require ``ffmpeg`` / ``audioread``
at runtime; if absent, those decodes return NULL rather than crashing.

Usage:
    uv run audio_worker.py              # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'audio' (TYPE vgi, LOCATION 'uv run audio_worker.py');

    SELECT audio.duration('/tmp/tone.wav');           -- seconds
    SELECT audio.sample_rate('/tmp/tone.wav');         -- Hz
    SELECT audio.tempo('/tmp/click.wav');              -- estimated BPM
    SELECT audio.spectral_centroid('/tmp/tone.wav');   -- Hz
    SELECT audio.mfcc('/tmp/tone.wav', 20);            -- DOUBLE[20]
    SELECT audio.estimated_key('/tmp/tone.wav');       -- 'C major' (heuristic)
    SELECT * FROM audio.beats('/tmp/click.wav');       -- (seq, time)
    SELECT * FROM audio.audio_info('/tmp/tone.wav');   -- (duration, sample_rate, channels)
"""

from __future__ import annotations

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_audio import features
from vgi_audio.scalars import SCALAR_FUNCTIONS
from vgi_audio.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [
    *SCALAR_FUNCTIONS,
    *TABLE_FUNCTIONS,
]

_AUDIO_CATALOG = Catalog(
    name="audio",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for SQL",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class AudioWorker(Worker):
    """Worker process hosting the ``audio`` catalog."""

    catalog = _AUDIO_CATALOG


def main() -> None:
    """Run the audio worker process (stdio or, via flags, HTTP)."""
    # Pay librosa's slow first-import cost once, up front, before serving.
    features.warm_up()
    AudioWorker.main()


if __name__ == "__main__":
    main()

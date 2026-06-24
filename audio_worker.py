# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
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

_CATALOG_DESCRIPTION_LLM = (
    "Extract acoustic and musical features from audio files or raw audio bytes directly in SQL. "
    "Scalars return per-file metadata (duration, sample_rate, channels) and signal/spectral/rhythmic "
    "features (tempo in BPM, rms_energy, zero_crossing_rate, spectral_centroid, spectral_bandwidth, "
    "mfcc as DOUBLE[], and a heuristic estimated_key like 'C major'). Table functions return beat onset "
    "times (beats) and combined metadata (audio_info). Every function accepts either a VARCHAR filesystem "
    "path the worker opens, or a BLOB of raw audio bytes. Use it to index, filter, or cluster audio "
    "collections by their acoustic properties without leaving SQL."
)

_CATALOG_DESCRIPTION_MD = (
    "# audio\n\n"
    "librosa-powered audio-feature extraction for DuckDB over Apache Arrow.\n\n"
    "Every function takes either a `VARCHAR` filesystem path or a `BLOB` of raw audio bytes "
    "(WAV/FLAC/OGG decode natively; compressed formats need `ffmpeg`). Undecodable or hostile "
    "input yields `NULL` / no rows rather than crashing.\n\n"
    "**Scalars:** `duration`, `sample_rate`, `channels`, `tempo`, `rms_energy`, "
    "`zero_crossing_rate`, `spectral_centroid`, `spectral_bandwidth`, `mfcc`, `estimated_key`.\n\n"
    "**Table functions:** `beats` (beat onset times), `audio_info` (duration / sample_rate / channels)."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Audio-feature functions: per-file metadata (duration, sample_rate, channels), signal/spectral/"
    "rhythmic scalars (tempo, rms_energy, zero_crossing_rate, spectral_centroid, spectral_bandwidth, "
    "mfcc, estimated_key), and table functions for beat onset times (beats) and combined metadata "
    "(audio_info). Inputs are a VARCHAR path or a BLOB of audio bytes."
)

_SCHEMA_DESCRIPTION_MD = (
    "librosa audio-feature extraction functions (duration, tempo, MFCC, key, spectral features, "
    "beats, ...) over Apache Arrow."
)

_AUDIO_CATALOG = Catalog(
    name="audio",
    default_schema="main",
    comment="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for SQL.",
    source_url="https://github.com/Query-farm/vgi-audio",
    tags={
        "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.description_md": _CATALOG_DESCRIPTION_MD,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-audio/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-audio/blob/main/README.md",
    },
    schemas=[
        Schema(
            name="main",
            comment="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for SQL",
            tags={
                "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
            },
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

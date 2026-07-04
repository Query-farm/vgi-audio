# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
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

import json

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_audio import features
from vgi_audio._example_audio import CLICK_WAV_B64, TONE_WAV_B64
from vgi_audio.scalars import SCALAR_FUNCTIONS
from vgi_audio.tables import TABLE_FUNCTIONS

# Inline self-contained blobs (see vgi_audio._example_audio) so the schema's
# representative example queries EXECUTE against the live worker -- a bare path
# would return no rows in a fresh process.
_TONE_BLOB_SQL = f"from_base64('{TONE_WAV_B64}')"
_CLICK_BLOB_SQL = f"from_base64('{CLICK_WAV_B64}')"

_SCHEMA_EXAMPLE_QUERIES = (
    f"SELECT audio.main.duration({_TONE_BLOB_SQL}) AS seconds;\n"
    f"SELECT audio.main.sample_rate({_TONE_BLOB_SQL}) AS hz;\n"
    f"SELECT audio.main.tempo({_CLICK_BLOB_SQL}) AS bpm;\n"
    f"SELECT audio.main.spectral_centroid({_TONE_BLOB_SQL}) AS centroid_hz;\n"
    f"SELECT audio.main.estimated_key({_TONE_BLOB_SQL}) AS musical_key;\n"
    f"SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq;\n"
    f"SELECT * FROM audio.main.audio_info({_TONE_BLOB_SQL});"
)

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
    "# Audio Feature Extraction in SQL\n\n"
    "![librosa logo](https://raw.githubusercontent.com/librosa/librosa/main/docs/img/librosa_logo_text.png)\n\n"
    "Extract tempo, timbre, loudness, key, and rhythm descriptors from audio files and raw "
    "audio bytes directly in DuckDB SQL — no separate media pipeline required. The audio "
    "catalog turns music information retrieval (MIR) and digital signal processing primitives "
    "into ordinary SQL, so you can query, filter, and cluster large audio collections by their "
    "acoustic properties using nothing but SELECT.\n\n"
    "This VGI worker is built for data engineers, music-tech developers, and ML practitioners "
    "who need audio descriptors without leaving the database. Every feature is computed by "
    "[librosa](https://librosa.org/), the de-facto standard Python library for audio and music "
    "analysis ([documentation](https://librosa.org/doc/latest/index.html), "
    "[source on GitHub](https://github.com/librosa/librosa)). Audio is decoded with libsndfile "
    "via [soundfile](https://python-soundfile.readthedocs.io/); WAV, FLAC, and OGG decode "
    "natively, while compressed formats such as MP3 and M4A use ffmpeg/audioread when "
    "available. DuckDB attaches the worker over Apache Arrow, streaming feature vectors back "
    "with zero-copy efficiency.\n\n"
    "Every feature takes the audio two ways — a filesystem path the worker opens, or a blob of "
    "raw audio bytes — and undecodable or hostile input yields NULL (or no rows) rather than "
    "crashing. The features span a handful of families, from cheap per-file metadata and "
    "loudness levels through spectral-timbre descriptors and compact cepstral feature vectors "
    "to rhythm/beat tracking and a heuristic musical-key estimate; list the schema to browse "
    "them. Typical uses include building feature tables for audio search and recommendation, "
    "detecting tempo or key across a music library, filtering recordings by loudness or "
    "brightness, and generating embeddings for downstream clustering or classification — all in "
    "plain SQL."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Audio-feature functions: per-file metadata (duration, sample_rate, channels), signal/spectral/"
    "rhythmic scalars (tempo, rms_energy, zero_crossing_rate, spectral_centroid, spectral_bandwidth, "
    "mfcc, estimated_key), and table functions for beat onset times (beats) and combined metadata "
    "(audio_info). Inputs are a VARCHAR path or a BLOB of audio bytes."
)

_SCHEMA_DESCRIPTION_MD = (
    "## Audio features (`main`)\n\n"
    "The `main` schema is where the audio catalog's librosa-powered feature functions live. It "
    "groups them into a few navigable families so you can go from a raw recording to a "
    "queryable descriptor in a single SQL call.\n\n"
    "### Feature families\n\n"
    "- **Container metadata** — cheap per-file facts (length, native sampling frequency, "
    "channel count), read from the header when possible.\n"
    "- **Loudness & energy** — level proxies for ranking recordings or gating near-silent "
    "clips.\n"
    "- **Spectral timbre** — brightness, spread, noisiness, and compact cepstral feature "
    "vectors for similarity search and ML.\n"
    "- **Rhythm** — an estimated tempo and per-beat onset times.\n"
    "- **Tonality** — a heuristic musical-key estimate.\n\n"
    "### Inputs & robustness\n\n"
    "Every function takes the audio two ways: a filesystem path the worker opens, or a blob of "
    "raw audio bytes. WAV, FLAC, and OGG decode natively; compressed formats such as MP3 and "
    "M4A need ffmpeg. Undecodable or hostile input returns NULL (scalars) or no rows (table "
    "functions) rather than raising, so a bad row never poisons a batch.\n\n"
    "Reach for this schema to index, filter, or cluster audio collections by their acoustic "
    "properties without leaving SQL."
)

# VGI413/VGI408-412: the schema's ordered category registry. Every function/table
# carries a `vgi.category` naming one of these (see scalars.py / tables.py); each
# category has at least one member, so the listing has no dead sections.
_CATEGORIES_JSON = json.dumps(
    [
        {
            "name": "metadata",
            "title": "Container Metadata",
            "description": "Cheap per-file facts: duration, native sample rate, and channel count.",
        },
        {
            "name": "loudness",
            "title": "Loudness & Energy",
            "description": "Signal-level proxies for ranking recordings or gating near-silent clips.",
        },
        {
            "name": "spectral",
            "title": "Spectral Timbre",
            "description": "Brightness, spread, noisiness, and compact cepstral (MFCC) feature vectors.",
        },
        {
            "name": "rhythm",
            "title": "Rhythm",
            "description": "Estimated tempo and per-beat onset times from beat tracking.",
        },
        {
            "name": "tonal",
            "title": "Tonality",
            "description": "A heuristic chroma-based musical-key estimate.",
        },
    ]
)

# VGI152/VGI407: the agent-suitability suite that `vgi-lint simulate` grades. Each
# task is a {name, prompt, reference_sql} object; the prompt is the ONLY field
# shown to the analyst, and reference_sql is a grader-only canonical solution.
#
# Every task is solvable from the catalog alone: the analyst discovers the
# relevant table function, reads the inline example audio (a base64 blob) that the
# function documents, and runs it — so each reference matches a documented,
# executable example and the suite is sound and deterministic. The tasks target
# the two set-returning entry points (``beats`` and ``audio_info``) because they
# return structured rows an agent consumes directly — the realistic "ask the
# worker a question" surface. The per-file scalars they build on (``duration`` /
# ``sample_rate`` / ``channels`` and the signal/spectral/rhythm features) are
# fully documented for discovery, but a scalar answer that must re-embed a whole
# audio blob inline is an awkward agent task, so the graded suite exercises the
# table functions.
_AGENT_TEST_TASKS_JSON = json.dumps(
    [
        {
            "name": "tone-metadata-row",
            "prompt": (
                "The audio catalog documents an inline example WAV tone (a base64 audio blob "
                "shown on its combined-metadata table function). Return that tone's duration, "
                "sample rate, and channel count as a single row."
            ),
            "reference_sql": (f"SELECT duration, sample_rate, channels FROM audio.main.audio_info({_TONE_BLOB_SQL})"),
        },
        {
            "name": "click-metadata-row",
            "prompt": (
                "The audio catalog documents an inline example click-track clip. Return its "
                "combined audio metadata — duration, sample rate, and channel count — as one row."
            ),
            "reference_sql": (f"SELECT duration, sample_rate, channels FROM audio.main.audio_info({_CLICK_BLOB_SQL})"),
        },
        {
            "name": "tone-duration",
            "prompt": (
                "From the example tone's combined audio-metadata row, report only its duration "
                "in seconds. Use the audio catalog's documented example tone clip."
            ),
            "reference_sql": f"SELECT duration FROM audio.main.audio_info({_TONE_BLOB_SQL})",
        },
        {
            "name": "tone-sample-rate",
            "prompt": (
                "From the example tone's combined audio-metadata row, report only its native "
                "sample rate in Hz. Use the audio catalog's documented example tone clip."
            ),
            "reference_sql": f"SELECT sample_rate FROM audio.main.audio_info({_TONE_BLOB_SQL})",
        },
        {
            "name": "tone-channels",
            "prompt": (
                "How many audio channels does the example tone have, per its combined "
                "audio-metadata row? Use the audio catalog's documented example tone clip."
            ),
            "reference_sql": f"SELECT channels FROM audio.main.audio_info({_TONE_BLOB_SQL})",
        },
        {
            "name": "click-beat-times",
            "prompt": (
                "The audio catalog documents an inline example click-track clip on its "
                "beat-tracking table function. Return that clip's detected beat onset times as "
                "(seq, time) rows, ordered by seq."
            ),
            "reference_sql": f"SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq",
        },
    ]
)

_AUDIO_CATALOG = Catalog(
    name="audio",
    default_schema="main",
    comment="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for SQL.",
    source_url="https://github.com/Query-farm/vgi-audio",
    tags={
        "vgi.title": "Audio Feature Extraction",
        # VGI138: keywords must be a JSON array of strings, not a CSV string.
        "vgi.keywords": json.dumps(
            [
                "audio",
                "sound",
                "music",
                "librosa",
                "feature extraction",
                "duration",
                "sample rate",
                "channels",
                "tempo",
                "bpm",
                "rms",
                "energy",
                "loudness",
                "zero crossing rate",
                "spectral centroid",
                "spectral bandwidth",
                "mfcc",
                "key",
                "beats",
                "audio analysis",
                "signal processing",
            ]
        ),
        "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-audio/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-audio/blob/main/README.md",
        # VGI152/VGI407: agent-suitability suite for `vgi-lint simulate`.
        "vgi.agent_test_tasks": _AGENT_TEST_TASKS_JSON,
    },
    schemas=[
        Schema(
            name="main",
            comment="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for SQL",
            tags={
                "vgi.title": "Audio Features — main",
                # VGI138: keywords must be a JSON array of strings, not a CSV string.
                "vgi.keywords": json.dumps(
                    [
                        "audio",
                        "librosa",
                        "duration",
                        "sample rate",
                        "channels",
                        "tempo",
                        "bpm",
                        "rms",
                        "energy",
                        "zero crossing rate",
                        "spectral centroid",
                        "spectral bandwidth",
                        "mfcc",
                        "estimated key",
                        "beats",
                        "audio_info",
                        "signal processing",
                        "music information retrieval",
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced) for faceting.
                "domain": "audio",
                "category": "feature-extraction",
                "topic": "music-information-retrieval",
                # VGI139: no per-object vgi.source_url -- the catalog's source_url
                # is the single source-of-truth link.
                "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
                # VGI413/VGI408: the schema's ordered category registry; each
                # function/table carries a vgi.category naming one of these.
                "vgi.categories": _CATEGORIES_JSON,
                # VGI506 representative, runnable example queries for the schema.
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
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

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "librosa>=0.10",
#     "soundfile>=0.12",
#     "numpy",
# ]
# ///
"""VGI worker exposing librosa audio-feature extraction to SQL (PEP 723 shim).

This repo-root script is a thin shim: the catalog, the ``AudioWorker`` class, and
``main()`` live in the wheel-importable :mod:`vgi_audio.worker` module. Keeping the
worker logic inside the installed ``vgi_audio`` package lets a built wheel serve it
(``vgi-serve vgi_audio.worker:AudioWorker`` / the ``vgi-audio-worker`` console
script / the Docker image), while this shim keeps ``uv run audio_worker.py`` (the
Makefile, ``ci/run-integration.sh``, and the tests) working unchanged.

Usage:
    uv run audio_worker.py              # serve over stdio (DuckDB subprocess)
    uv run audio_worker.py --http       # serve over HTTP (needs the http extra)

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

from vgi_audio.worker import AudioWorker, main

__all__ = ["AudioWorker", "main"]

if __name__ == "__main__":
    main()

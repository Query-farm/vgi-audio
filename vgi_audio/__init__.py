"""Extract audio features from audio bytes/paths as DuckDB SQL functions.

A [VGI](https://query.farm) worker that runs `librosa` feature extraction inside
SQL. The implementation is split so each concern stays focused:

- ``features``  -- pure feature logic over ``librosa`` / ``soundfile`` / ``numpy``;
  no Arrow or VGI dependency, directly unit-testable, **total** (never raises on
  hostile input; returns ``None``).
- ``scalars``   -- per-row VGI scalar functions. Each accepts a ``VARCHAR`` path
  *or* a ``BLOB`` of audio bytes, exposed as two same-name overloads resolved by
  argument type.
- ``tables``    -- set-returning table functions (``beats``, ``audio_info``),
  likewise in path/bytes overload pairs.

``audio_worker.py`` at the repo root assembles these into the ``audio`` catalog
and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"

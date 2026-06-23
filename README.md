<p align="center">
  <img src="docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-audio

[![CI](https://github.com/Query-farm/vgi-audio/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-audio/actions/workflows/ci.yml)

A [VGI](https://query.farm) worker that brings **audio feature extraction** into
DuckDB/SQL. It runs [`librosa`](https://librosa.org/) over audio you reference by
**file path** *or* pass as **raw bytes (BLOB)** — pulling out duration, sample
rate, tempo, MFCCs, spectral features, a heuristic musical key, and beat onset
times — as plain SQL functions.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'audio' (TYPE vgi, LOCATION 'uv run audio_worker.py');

SELECT audio.duration('/tmp/tone.wav');            -- 2.0   (seconds)
SELECT audio.sample_rate('/tmp/tone.wav');          -- 22050 (Hz)
SELECT audio.channels('/tmp/tone.wav');             -- 1
SELECT audio.tempo('/tmp/click.wav');               -- ~120.0 (estimated BPM)
SELECT audio.rms_energy('/tmp/tone.wav');           -- mean RMS energy
SELECT audio.zero_crossing_rate('/tmp/tone.wav');   -- mean ZCR
SELECT audio.spectral_centroid('/tmp/tone.wav');    -- Hz (spectrum centre of mass)
SELECT audio.spectral_bandwidth('/tmp/tone.wav');   -- Hz
SELECT audio.mfcc('/tmp/tone.wav');                 -- DOUBLE[13]
SELECT audio.mfcc('/tmp/tone.wav', 20);             -- DOUBLE[20]
SELECT audio.estimated_key('/tmp/tone.wav');        -- 'C major' (heuristic!)

SELECT * FROM audio.beats('/tmp/click.wav');        -- (seq, time) beat onsets
SELECT * FROM audio.audio_info('/tmp/tone.wav');    -- (duration, sample_rate, channels)
```

## Path *or* bytes — the polymorphic audio argument

Every function accepts the audio two ways:

* a **`VARCHAR` path** — the worker opens the file from its own filesystem; or
* a **`BLOB`** — raw audio bytes that travel over Arrow (e.g. from
  `read_blob(...)`), decoded in-memory.

```sql
-- bytes straight from DuckDB's read_blob:
SELECT file, audio.duration(content)
FROM read_blob('recordings/*.wav');
```

Under the hood each feature is registered as **two same-name overloads** — one
whose first argument is `VARCHAR`, one `BLOB` — and DuckDB resolves which to call
by the column type. (`mfcc` additionally has an optional coefficient-count
argument, so it has path/bytes × default/`n` overloads.)

## Scalars (per-row) vs. table functions

* **Scalars** take **positional** arguments only and resolve overloads by
  type/arity (DuckDB's `name := value` syntax is a table-function feature, not a
  scalar one). Every per-row answer is a scalar, so it works inline in any
  projection or predicate: `duration`, `sample_rate`, `channels`, `tempo`,
  `rms_energy`, `zero_crossing_rate`, `spectral_centroid`, `spectral_bandwidth`,
  `mfcc`, `estimated_key`.

* **Table functions** return rows: `beats(audio)` → `(seq BIGINT, time DOUBLE)`
  beat onset times; `audio_info(audio)` → a single
  `(duration DOUBLE, sample_rate INT, channels INT)` row.

## Robustness — audio bytes are untrusted

A SQL worker is fed whatever is in the column. Malformed, truncated, absurdly
large, or maliciously crafted audio **must never crash the worker process**, so:

* every decode/analysis is wrapped per row; failures return **NULL** (scalars) or
  **no rows** (table functions) instead of raising;
* `NULL` input → `NULL` output / no rows;
* decoded audio is **bounded** — at most 30 minutes (`MAX_DURATION_SECONDS`) is
  materialised, so a corrupt header claiming billions of frames can't exhaust
  memory; longer inputs are analysed on a truncated prefix;
* analysis runs at a fixed 22.05 kHz mono internally, bounding per-frame work.

A garbage blob next to a valid one yields `NULL` for the bad row and the correct
value for the good row — the process keeps serving.

## Supported formats & native dependencies

* **WAV / FLAC / OGG** decode natively via `soundfile`
  ([libsndfile](http://libsndfile.github.io/libsndfile/), bundled in the wheel) —
  **no external tools required**.
* **Compressed formats (mp3, m4a, ...)** require **`ffmpeg`** (or `audioread`)
  installed on the worker host. If neither is present, those decodes return
  `NULL` rather than crashing.

Python dependencies and their licenses:

| Library      | Purpose                  | License |
| ------------ | ------------------------ | ------- |
| `librosa`    | feature extraction       | ISC     |
| `soundfile`  | WAV/FLAC/OGG decode      | BSD-3-Clause |
| `numpy`      | array math               | BSD-3-Clause |

This worker itself is **MIT** licensed (see `LICENSE`).

`estimated_key` is a **heuristic** chroma-based (Krumhansl-Schmuckler) estimate —
useful as a hint, not ground truth.

## Layout

```
audio_worker.py        repo-root stdio entry point; PEP 723 inline deps; main()
vgi_audio/
  features.py          pure feature logic over librosa/soundfile/numpy; no Arrow/VGI; total
  scalars.py           per-row scalars (path/bytes overloads, mfcc n overload)
  tables.py            table functions: beats, audio_info (path/bytes overloads)
  schema_utils.py      pa.Field comment / column-doc helper
tests/                 pytest: test_features (pure), test_tables (in-proc), test_scalars (Client RPC)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / lint
```

## Development

```sh
uv sync --extra dev
uv run pytest -q                # unit + integration
make test-sql                   # end-to-end SQL via haybarn-unittest
uv run ruff check . && uv run mypy vgi_audio/
```

Test audio is generated deterministically in-process (numpy sine tones / click
tracks written to in-memory WAV via `soundfile`); a couple of tiny committed WAV
fixtures under `test/sql/data/` drive the SQL E2E.

---

## Authorship & License

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm


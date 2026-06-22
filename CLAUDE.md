# CLAUDE.md — vgi-audio

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker that extracts audio features with `librosa`
as DuckDB scalar + table functions. Every feature accepts the audio two ways:
a `VARCHAR` filesystem **path** the worker opens, or a `BLOB` of raw audio
**bytes**. `audio_worker.py` assembles every function into one `audio` catalog
(single `main` schema) over stdio. Sibling style/tooling to `vgi-conform` /
`vgi-calendar`.

## Layout

```
audio_worker.py        repo-root stdio entry point; PEP 723 inline deps; main() warms librosa then serves
vgi_audio/
  features.py          pure feature logic over librosa/soundfile/numpy; no Arrow/VGI; TOTAL (never raises)
  scalars.py           per-row scalars; path/bytes overloads (+ mfcc n overload)
  tables.py            table functions: beats, audio_info; path/bytes overloads
  schema_utils.py      pa.Field comment / column-doc helper
tests/
  synth.py             deterministic numpy -> in-memory WAV generators (sine tone, click track)
  test_features.py     pure logic + hostile-input contract (no subprocess)
  test_tables.py       table functions in-proc via harness.py (bind->init->process)
  test_scalars.py      scalars over a real worker subprocess via vgi.client.Client
  harness.py           in-process table-function lifecycle driver
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
test/sql/data/*.wav    tiny committed deterministic WAV fixtures
Makefile               test / test-unit / test-sql / lint
```

To add a feature: implement the logic in `features.py` (pure, **total** — catch
everything, return `None` on any failure), then add path + bytes scalar (or
table) overloads in the matching module and register them in `audio_worker.py`'s
`_FUNCTIONS`.

## Path OR bytes — the polymorphic argument (THE core convention)

A scalar's input Arrow type comes from its `compute()` annotation:
`pa.StringArray` → `VARCHAR`, `pa.BinaryArray` → `BLOB`. To accept *both* a path
and raw bytes, every feature is registered as **two same-name overloads** — one
`StringArray` (path), one `BinaryArray` (bytes) — and DuckDB resolves which by
the argument's column type. This is the same "share a `Meta.name` across
overloads" idiom `vgi-conform` uses for optional arguments.

* **Scalars are positional-only.** `name := value` is rejected for scalars; it's
  a table-function feature. `mfcc`'s optional `n` is therefore an extra *arity*
  overload (so `mfcc` has four classes: path/bytes × default/`n`).
* **LIST returns require an explicit `Returns(arrow_type=...)`.** `mfcc` returns
  `DOUBLE[]`, so its `compute()` is annotated
  `Returns(arrow_type=pa.list_(pa.float64()))` — the SDK raises without it.
* **Table-function args** go through `Arg(0, arrow_type=...)`; `beats` /
  `audio_info` likewise have a path overload (`pa.string()`) and a bytes overload
  (`pa.binary()`).

## Robustness — audio bytes are UNTRUSTED (production bar)

The single choke point is `features._load_mono` / `_probe`: **all** decoding and
analysis is wrapped in `try/except Exception` and returns `None` on any failure.
Consequences enforced by tests:

* malformed / truncated / garbage / empty bytes → `None` (scalar → NULL; table →
  no rows). A bad row beside a good one does **not** poison the good result.
* `NULL` input → `NULL` / no rows (the map helpers and `AudioInput.from_*` pass
  `None` straight through).
* decode is **bounded**: `librosa.load(..., duration=MAX_DURATION_SECONDS)` caps
  materialised audio at 30 min so a corrupt header claiming billions of frames
  can't exhaust RAM; analysis runs at fixed 22.05 kHz mono.
* non-finite samples are scrubbed (`nan_to_num`); non-finite feature values
  collapse to `None`.

Never let an exception escape a `features.*` function — that is the contract that
keeps the worker process alive under hostile input.

## librosa specifics

* **Imported once**, lazily + cached (`features._librosa`); `main()` calls
  `features.warm_up()` so the slow first import is paid before serving.
* **Tempo API moved between releases.** `_tempo_fn` probes, in order,
  `librosa.feature.rhythm.tempo` → `librosa.feature.tempo` → `librosa.beat.tempo`
  (0.11 lacks the `rhythm` submodule). If you bump librosa, re-check this.
* `estimated_key` is a heuristic Krumhansl-Schmuckler chroma correlation — a hint,
  not ground truth. Document it as such.

## Tests

* `make test-unit` (pytest): pure logic asserts a 22050 Hz sine tone exactly
  (sample rate / duration) and spectral centroid *near* the tone; a 120-BPM click
  track recovers tempo within a band; the hostile-input suite proves NULL-not-
  crash. `test_scalars.py` drives a real worker subprocess.
* `make test-sql` (haybarn-unittest): glob is `test/sql/*` (NOT `*.test` — the
  bare-name glob is what matches). Files use `statement ok` + `LOAD vgi;` then
  `ATTACH 'audio' ... LOCATION '${VGI_AUDIO_WORKER}'`; scalars are
  catalog-qualified (`audio.duration(...)`). Numeric features are asserted with
  ranges/tolerance, never exact floats. The hostile test proves the worker
  survives garbage and is still alive afterwards.
* Test audio is generated deterministically in `tests/synth.py`; the only
  committed blobs are the two tiny WAV fixtures under `test/sql/data/`.
* If `make test-sql` flakes once, re-run — only a *consistent* failure is real.

## Verify

```sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
uv run --no-sync pytest -q
make test-sql
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_audio/
```

## Licensing

Worker is MIT. Deps: `librosa` (ISC), `soundfile` (BSD-3-Clause), `numpy`
(BSD-3-Clause). `ffmpeg` (optional, for compressed formats) is not bundled.

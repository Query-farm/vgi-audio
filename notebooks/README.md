# Audio Feature Extraction examples

| Notebook | Question | One-parameter experiment |
|---|---|---|
| [01 · Louder or brighter?](01_louder_or_brighter.ipynb) | Can equal designed RMS coexist with different spectral balance? | Harmonic amplitude ratio 0.5 → 1.0 |
| [02 · A tempo estimate is not a detected beat list](02_a_tempo_estimate_is_not_a_beat_list.ipynb) | How do a whole-clip tempo and a rate calculated from selected event gaps differ? | Excluded window endpoint 2.0 → 1.25 seconds |

Each notebook has an **Open in Colab** link. Choose **Runtime → Run all**. Its
short bootstrap installs only the shared base package and retrieves an immutable,
SHA256-verified helper. For local execution install [uv](https://docs.astral.sh/uv/)
and Git, then run from `notebooks/`:

```sh
uv sync --locked
uv run --locked vgi-notebooks
uv run --locked vgi-notebooks 01_louder_or_brighter.ipynb --parameters '{"HARMONIC_RATIO":1.0}' --output-dir artifacts/equal-harmonics
uv run --locked vgi-notebooks 02_a_tempo_estimate_is_not_a_beat_list.ipynb --parameters '{"WINDOW_END_S":1.25}' --output-dir artifacts/shorter-window
```

For Jupyter select `.venv/bin/python`. The locked `[runner]` extra is local tooling;
Colab installs the base package only. Overrides require a separate destination,
unknown parameters fail and failed execution cannot replace a successful default.
If an older setup replaced kernel libraries, open this version in a fresh runtime.

## Portable release and shared presentation

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'audio' (TYPE vgi,
  LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-audio@v0.4.0 vgi-audio-worker');
```

The actual public v0.4.0 tag owns its worker dependencies. This short launch was
verified through Haybarn 2.0.0rc11 without older notebook transitive overrides. It
resolved vgi-python 0.51.1, vgi-rpc 0.53.1, PyArrow 26.0.1, librosa 1.0.1,
soundfile 0.14.0, NumPy 2.5.4 and Numba 0.68.1 in a separate Python 3.13 worker
environment. The notebook lock does not lock that worker environment.

Initial Git/package/extension/helper retrieval needs internet and compatible
numerical/audio wheels. Cold librosa/Numba warm-up can take tens of seconds.
No local worker checkout, root environment, credentials, external audio service
or recording download is required. The WAV/FLAC inputs need no ffmpeg. macOS arm64
was tested; other platforms were not executed. Shared connection setup isolates
persistent SQL secrets.

The [shared v0.3.25 standard](https://github.com/Query-farm/vgi-example-notebooks/blob/v0.3.25/docs/standard.md)
provides the catalog-derived title/subtitle, branding, isolated SQL connection,
SVG presentation and protected runner. Keep `charts/` beside saved notebooks for
GitHub image links; interactive runs retain native SVG. Generated audio, review
PNGs, logs and experiment copies remain ignored. Hosted Colab was not manually
operated; one fresh compatible kernel preloads eight libraries before executing
the second notebook’s visible bootstrap. Both books use the same helper/launch path.

## Coverage and current source

Complete actual catalogs contain **13 public named surfaces**: ten scalar functions,
two table functions and the `example_clips` table. Path/BLOB and MFCC arity
overloads count once; all 26 function rows and full table schemas are retained.
The two focused stories meaningfully cover **9/13**, exceeding the required seven:
`duration`, `sample_rate`, `channels`, `rms_energy`, `spectral_centroid`,
`example_clips`, `audio_info`, `tempo` and `beats`.

`zero_crossing_rate`, `spectral_bandwidth`, `mfcc` and `estimated_key` remain
uncovered. Catalog exports, reviewed mappings and hashed execution receipts live
in `coverage.json` and `coverage/`. The notebook runtime is v0.4.0, source
`6b1327f4fe9232262060bbe4c3219ad6a5048fd8`. Current public source
`83d5252f418b4c9a9bc9112599236ab1f399cf3f` has identical interfaces, unchanged
feature logic and unchanged root dependencies; differences add composed examples
and clarify literal arguments and WAV/FLAC metadata. Compatible current coverage
does not claim a separately executed current-main notebook.

Existing green [source CI](https://github.com/Query-farm/vgi-audio/actions/runs/43558931194)
is reused. This update changes only notebooks; no worker rebuild, repeated full
suite, container publication or new worker release is needed. The original
checkout’s modified root README and older source state are preserved.

## Controlled level and brightness

The first notebook authors three-second mono 22,050-Hz PCM-32 WAV signals. A 440-Hz
reference, half-amplitude copy and normalized 440/1320-Hz mixture form the main
comparison. Five additional clips sweep the harmonic ratio. Python defines inputs;
the actual worker supplies measured features and SQL computes comparisons.

Halving amplitude should change relative RMS by approximately −6.02 dB while
preserving spectral centroid. Normalization keeps designed whole-clip RMS fixed
as the harmonic ratio changes. The worker averages framed RMS/centroid values, so
window padding and spectral leakage cause small deviations from ideal formulas.
The comparison is an uncalibrated digital level ratio, not dB SPL, perceived
loudness, a mastering standard or listening quality. Centroid is neither pitch nor
a complete timbre description. Invalid bytes remain unavailable rather than zero.

The ratio experiment changes 0.5 to 1.0 while retaining the reference and
half-amplitude controls. Eight SQL checks use explicit tolerances for these
controlled signals. They do not claim a universal audio identity.

## Tempo, event lists and a half-open window

The second notebook reads the actual built-in sample table once: a 2-second,
6,000-Hz mono FLAC click clip and a 0.5-second, 200-Hz mono WAV tone. Their byte
strings are synthetic embedded fixtures, not external recordings. The table’s
`beat_count` is itself calculated by the worker’s detector, so its agreement with
`beats()` is consistency, not independent ground truth.

Scalar functions accept columns of bytes. The table functions require literal
arguments and reject lateral column/subquery inputs. The notebook copies the two
catalog byte strings unchanged into generated `.bin` files, verifies SQL SHA256,
and passes visible literal paths to `audio_info` and `beats`. Their real headers
determine decoding. This is input materialization, never a substitute fixture or
feature calculation. Invalid non-NULL bytes emit no table rows; SQL NULL arguments
are rejected at binding and are not used.

Native metadata differs from analysis rate: features resample to 22,050 Hz mono.
Beat `seq` is a zero-based output index; `time` is seconds from clip start. The
click list has events near 0.534, 1.045 and 1.533 seconds. The steady tone has no
detected events but a numeric tempo estimate near 136 BPM. That output is not
evidence of a reliable musical pulse. Invalid bytes retain unavailable metadata
and estimates, separately from a valid zero-event clip.

Selection uses **[0, WINDOW_END_S)** over the complete materialized event list.
Changing 2.0 to 1.25 seconds retains two of three click events and one rather than
two gaps. The resulting `60 / median(gap)` rate changes from about 120.19 to
117.45 BPM, while the whole-clip tempo remains about 117.45 BPM. No audio is cropped
or reanalyzed for the parameter change. One or two intervals cannot establish
tempo accuracy or confidence, and numerical agreement between estimators does not
validate either. Eleven SQL checks verify bytes, metadata, population, bounds and
interval arithmetic while retaining the zero-event tone and invalid input.

# Audio worker notebooks

[01 · Louder or brighter?](01_louder_or_brighter.ipynb) separates digital RMS level
from spectral balance. A reference tone, an attenuated copy, and a normalized
two-tone mix are measured by the real worker. SQL compares their relative dB and
spectral centroids, then tests a harmonic sweep against an approximate model.
The notebook contains successful outputs and two figures.

## Install and run

Install [uv](https://docs.astral.sh/uv/) and Git. This directory can be copied
away from the worker checkout; no root environment or local worker build is used.
From this directory:

```sh
uv sync --locked
uv run --locked python execute.py --timeout 300
```

Haybarn **2.0.0rc11**, JupySQL, plotting tools and the execution environment are
locked separately here. The visible literal ATTACH launches Python 3.13 through
uvx and the remotely available **v0.4.0** Git tag. Its command pins
**vgi-python 0.51.0**, **vgi-rpc 0.53.1**, **librosa 1.0.0**, **soundfile 0.14.0**,
and **NumPy 2.5.4**. Initial extension/package/Python downloads require internet.
The worker warms librosa/Numba at startup; a cold cache can take tens of seconds.
The generated PCM WAVs do not need ffmpeg.

The runner uses an isolated temporary kernel with the current interpreter and
notebook-directory cwd. It atomically saves only successful executions. Parameter
overrides require a separate destination. For an existing Jupyter frontend:

```sh
uv run --locked python -m ipykernel install --user \
  --name vgi-audio-notebooks --display-name "Python (vgi-audio notebooks)"
```

Select that kernel, then **Restart Kernel → Run All** after editing parameters.
Run the frontend from this directory. The runner does not register a global kernel.

## One-parameter experiment

```sh
uv run --locked python execute.py --timeout 300 \
  --parameters '{"HARMONIC_RATIO": 1.0}' --output-dir artifacts/equal-harmonics
```

The default ratio is **0.5**, and the accepted range is **0–1**. Predict a higher
spectral centroid near 880 Hz at equal component amplitudes, while the mix's RMS
level stays near 0 dB relative to the reference. Compare the first figure's two
panels and the sweep endpoint. The half-amplitude control should still measure
about −6.02 dB. Reset the ratio to **0.5** for the saved default.

Experiment copies include their branding assets, guide, helper, runner and lock.
Generated inputs, extracted charts, experiments and validation logs live under
ignored `artifacts/`; they are regenerated without another repository.

## Signal design and limits

All inputs are synthetic three-second, 22,050 Hz, mono, signed 32-bit PCM WAVs.
`notebook_support.py` generates them with NumPy and Python's standard `wave`
module; it imports no audio analysis library. The reference is a 440 Hz sine of
peak amplitude 0.4. A copy uses half that amplitude. The harmonic mix is

```text
0.4 / sqrt(1 + r²) × (sin(2π × 440 × t) + r × sin(2π × 1320 × t))
```

Five additional clips span `r = 0` through the chosen ratio. Distinct components
complete integer cycle counts over the clip, so the cross term averages to zero;
the mix's designed whole-clip RMS is 0.4/√2. The peak stays below full scale for
the accepted ratios. SQL reads WAV bytes into BLOBs and sends them through VGI.
Every measured duration, channel count, sample rate, RMS and centroid comes from
the remote worker. Materialized SQL tables supply the charts.

The worker averages RMS and spectral centroid over frames at 22.05 kHz mono.
Boundary padding, window leakage and finite precision mean those results need
not exactly equal whole-clip RMS or an ideal two-line spectrum. SQL uses stated
tolerances. RMS is digital amplitude, not calibrated sound pressure or perceived
loudness; a centroid is not pitch or a complete description of timbre. No playback
is needed. Invalid bytes return NULL and stay distinct from a measured zero.

The v0.4.0 release supplies wheel/source assets, so the example uses a pinned Git
launch instead of assuming a native release binary exists. To upgrade, change the
literal release/dependency pins, rerun both parameter sets, and inspect the figures
and SQL checks before replacing saved outputs. `INSTALL vgi FROM community`
downloads the community extension independently of this Python lock.

The unchanged local Strata Sun and VGI assets link to Query.Farm and VGI.

## Validation

The source gates pass: 55 pytest tests; 61/69/61 SQL assertions over
stdio/HTTP/Unix sockets; Ruff, mypy (nine files), and CI-prescribed isolated
`uvx pydoclint`. Current vgi-lint 2.5.1 against a warmed persistent HTTP worker reports
100/100, L2 behavioral assurance, and zero findings. Its bare stdio runs hit
10/30-second probe/example timeouts while repeatedly starting workers; those
timeouts are recorded separately and are not counted as passing metadata checks. The notebook's default and experiment checks are separate
executions against its pinned remote worker launch.

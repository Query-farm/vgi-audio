"""Per-row scalar audio-feature functions.

Every function here is a true DuckDB **scalar** -- one value (per row) in, one
value out -- so it works inline in any projection or predicate:

    SELECT audio.duration(path)          FROM tracks;       -- path column (VARCHAR)
    SELECT audio.tempo(blob)             FROM recordings;   -- bytes column (BLOB)
    SELECT id, audio.spectral_centroid(path) FROM tracks;

Path OR bytes -- the polymorphic first argument
----------------------------------------------
Every feature accepts either a ``VARCHAR`` filesystem path (the worker opens the
file) or a ``BLOB`` of raw audio bytes (read in-memory). VGI / DuckDB *scalar*
functions take **positional** arguments and resolve overloads by *type/arity*
(the ``name := value`` named-argument syntax is a table-function feature, not a
scalar one). So each feature is exposed as **two same-name overloads** -- one
whose first argument is annotated ``pa.StringArray`` (path) and one
``pa.BinaryArray`` (bytes) -- and DuckDB picks the right one by the column type.
This is the same "share a name across overloads" idiom the sibling
``vgi-conform`` worker uses for its optional-argument scalars.

``mfcc`` additionally takes an optional ``n`` (number of coefficients), which --
being a *scalar* optional arg -- is itself an extra arity overload, giving four
classes: ``mfcc(path)`` / ``mfcc(path, n)`` / ``mfcc(blob)`` / ``mfcc(blob, n)``.

NULL semantics: NULL input -> NULL output. Malformed / corrupt / undecodable
audio -> NULL (the worker never crashes on hostile bytes).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import composed, features, meta
from ._example_audio import TONE_WAV_B64
from .features import AudioInput

# ---------------------------------------------------------------------------
# Runnable example inputs.
#
# The strict linter EXECUTES every example and expects rows back, so examples
# must be self-contained -- a bare ``'/tmp/tone.wav'`` path either errors or
# returns no rows in a fresh process. We therefore decode a tiny embedded WAV
# tone inline via ``from_base64(...)`` (a ``BLOB``), which the ``BLOB`` overload
# of each scalar then analyses and returns a real value for.
# ---------------------------------------------------------------------------

_TONE_BLOB_SQL = f"from_base64('{TONE_WAV_B64}')"


def _tone_example(fn: str, description: str) -> FunctionExample:
    """A runnable ``SELECT audio.main.<fn>(<tone blob>)`` example (returns a row).

    The reference is fully catalog-qualified (``catalog.schema.function``) so it
    both executes when the worker is attached and counts toward per-object example
    coverage (VGI505/VGI511); a two-part ``audio.<fn>`` reference does neither.
    """
    return FunctionExample(sql=f"SELECT audio.main.{fn}({_TONE_BLOB_SQL})", description=description)


# ---------------------------------------------------------------------------
# Mapping helpers: apply a pure ``AudioInput -> X`` function across an input
# array, passing NULL straight through. One pair (path / bytes) per feature.
# ---------------------------------------------------------------------------


# ``array_cls`` is the concrete Arrow array class ``arrow_type`` builds; it types
# the result so each ``compute`` keeps its concrete return annotation (which
# ``Returns()`` infers the SQL type from).


def _map_path[T, A: pa.Array[Any]](
    arr: pa.StringArray,
    fn: Callable[[AudioInput | None], T],
    arrow_type: pa.DataType,
    array_cls: type[A],
) -> A:
    out = [fn(AudioInput.from_path(x)) for x in arr.to_pylist()]
    result = pa.array(out, type=arrow_type)
    assert isinstance(result, array_cls)
    return result


def _map_bytes[T, A: pa.Array[Any]](
    arr: pa.BinaryArray,
    fn: Callable[[AudioInput | None], T],
    arrow_type: pa.DataType,
    array_cls: type[A],
) -> A:
    out = [fn(AudioInput.from_bytes(x)) for x in arr.to_pylist()]
    result = pa.array(out, type=arrow_type)
    assert isinstance(result, array_cls)
    return result


_PATH_DOC = "Filesystem path to an audio file the worker will open."
_BLOB_DOC = "Raw audio bytes (e.g. the contents of a WAV/FLAC file)."

# ---------------------------------------------------------------------------
# Per-object discovery/description tags (VGI112/113/124/126/128), one set per
# logical feature. Both the path and bytes overloads of a feature carry the same
# set, since the linter presents them as one named function. Descriptions are
# Markdown narratives written for an LLM/agent (``_llm``) and for human docs
# (``_md``).
# ---------------------------------------------------------------------------

_TAGS_DURATION = meta.object_tags(
    title="Audio Duration (Seconds)",
    doc_llm=(
        "## audio.duration\n\n"
        "Return the **length of an audio recording in seconds** as a `DOUBLE`. "
        "Accepts either a `VARCHAR` filesystem path (the worker opens the file) "
        "or a `BLOB` of raw audio bytes.\n\n"
        "**When to use.** Filter or sort a table of recordings by length, compute "
        "total library duration, or skip clips that are too short/long before "
        "running heavier analysis.\n\n"
        "**Behavior.** Uses the container's native frame count when it can be "
        "probed (exact and cheap); otherwise decodes the stream to measure it. "
        "Returns `NULL` for `NULL` input and for anything that cannot be decoded "
        "(missing file, corrupt or unsupported bytes) -- it never raises."
    ),
    doc_md=(
        "# duration\n\n"
        "Audio duration in **seconds** (`DOUBLE`).\n\n"
        "```sql\n"
        "SELECT audio.main.duration('/music/track.flac');   -- e.g. 213.4\n"
        "```\n\n"
        "Reads the native frame count when available, else decodes to measure; "
        "works equally over a file path or a column of raw audio bytes. `NULL` for "
        "undecodable or `NULL` input."
    ),
    keywords=["duration", "length", "seconds", "runtime", "playtime", "audio length", "clip length", "time"],
    category="metadata",
    example_queries=[
        ("Duration in seconds of an inline WAV tone.", f"SELECT audio.main.duration({_TONE_BLOB_SQL})"),
    ],
)

_TAGS_SAMPLE_RATE = meta.object_tags(
    title="Native Sample Rate (Hz)",
    doc_llm=(
        "## audio.sample_rate\n\n"
        "Return the recording's **native sample rate in Hz** (e.g. 44100, 48000) "
        "as an `INTEGER`. Accepts a `VARCHAR` path or a `BLOB` of audio bytes.\n\n"
        "**When to use.** Group or validate a collection by sample rate, detect "
        "downsampled material, or decide resampling before further processing.\n\n"
        "**Behavior.** Read straight from the container header when possible; for "
        "formats that can't be probed, a brief decode recovers the source rate. "
        "This is the recording's **native/original** rate -- note the feature "
        "scalars (tempo, spectral, mfcc, ...) internally resample to 22.05 kHz for "
        "analysis, so their results are independent of this value. `NULL` on "
        "`NULL` or undecodable input."
    ),
    doc_md=(
        "# sample_rate\n\n"
        "Native **sample rate in Hz** (`INTEGER`), e.g. `44100`.\n\n"
        "```sql\n"
        "SELECT audio.main.sample_rate('/music/track.wav');\n"
        "```\n\n"
        "The recording's original rate, taken from the file header where available "
        "(the analysis features resample to 22.05 kHz regardless). Also accepts a "
        "`BLOB`. `NULL` for undecodable input."
    ),
    keywords=["sample rate", "samplerate", "hz", "frequency", "sampling", "44100", "48000", "khz"],
    category="metadata",
    example_queries=[
        ("Native sample rate (Hz) of an inline WAV tone.", f"SELECT audio.main.sample_rate({_TONE_BLOB_SQL})"),
    ],
)

_TAGS_CHANNELS = meta.object_tags(
    title="Audio Channel Count",
    doc_llm=(
        "## audio.channels\n\n"
        "Return the **number of audio channels** in a recording as an `INTEGER` "
        "(`1` = mono, `2` = stereo, higher for surround). Accepts either a "
        "`VARCHAR` filesystem path or a `BLOB` of raw audio bytes.\n\n"
        "**When to use.** Separate mono from stereo material, flag unexpected "
        "channel layouts before mixing, or pick a down-mix strategy.\n\n"
        "**Behavior.** The count is read from the container header, which is exact "
        "for WAV/FLAC/OGG. Only when the header cannot be read does the worker fall "
        "back to a decode, and because that analysis path is mono it then reports "
        "`1` -- so a `1` from a header-less source means 'decodable as mono', not a "
        "guaranteed true channel count. Returns `NULL` for `NULL` input and for "
        "anything undecodable; it never raises."
    ),
    doc_md=(
        "# channels\n\n"
        "Number of **audio channels** (`INTEGER`): `1` = mono, `2` = stereo, more "
        "for surround.\n\n"
        "```sql\n"
        "-- Path input:\n"
        "SELECT audio.main.channels('/music/track.wav');\n"
        "-- Bytes input (a BLOB column or literal):\n"
        "SELECT audio.main.channels(audio_bytes);\n"
        "```\n\n"
        "Read from the file header (exact for WAV/FLAC/OGG). If only a decode is "
        "possible the worker's mono analysis path reports `1`. `NULL` for `NULL` or "
        "undecodable input."
    ),
    keywords=["channels", "mono", "stereo", "surround", "channel count", "layout", "multichannel"],
    category="metadata",
    example_queries=[
        ("Channel count of an inline WAV tone (1 = mono).", f"SELECT audio.main.channels({_TONE_BLOB_SQL})"),
    ],
)

_TAGS_TEMPO = meta.object_tags(
    title="Estimated Tempo (BPM)",
    doc_llm=(
        "## audio.tempo\n\n"
        "Estimate the **tempo of a recording in beats per minute** as a `DOUBLE`, "
        "using librosa's onset/beat analysis. Accepts a `VARCHAR` path or a "
        "`BLOB` of bytes.\n\n"
        "**When to use.** Sort a music library by speed, find tracks within a BPM "
        "range for a playlist or DJ set, or cluster rhythmically similar audio.\n\n"
        "**Behavior.** This is a **heuristic estimate** -- it can land on a "
        "half/double-time multiple of the perceived tempo, and is unreliable on "
        "material with no steady pulse. Computed on a bounded mono signal at a "
        "fixed analysis rate. `NULL` on `NULL` or undecodable input."
    ),
    doc_md=(
        "# tempo\n\n"
        "Estimated **tempo in BPM** (`DOUBLE`).\n\n"
        "```sql\n"
        "SELECT audio.main.tempo('/music/track.mp3');   -- e.g. 120.0\n"
        "```\n\n"
        "> Heuristic: may report a half/double-time multiple, and is unreliable "
        "on audio without a steady beat. `NULL` for undecodable input."
    ),
    keywords=["tempo", "bpm", "beats per minute", "speed", "pace", "rhythm", "beat"],
    category="rhythm",
    example_queries=[
        ("Estimated tempo (BPM) of an inline WAV clip.", f"SELECT audio.main.tempo({_TONE_BLOB_SQL})"),
    ],
)

_TAGS_RMS = meta.object_tags(
    title="Mean RMS Energy Level",
    doc_llm=(
        "## audio.rms_energy\n\n"
        "Return the **mean root-mean-square (RMS) energy** of the waveform as a "
        "`DOUBLE` -- a loudness/level proxy. Accepts a `VARCHAR` path or a `BLOB` "
        "of bytes.\n\n"
        "**When to use.** Rank recordings by overall loudness, detect near-silent "
        "or clipped/hot clips, or normalise levels across a collection.\n\n"
        "**Behavior.** The RMS amplitude is computed per short analysis frame over "
        "a bounded mono signal and then averaged. Values are a unitless linear "
        "amplitude (not decibels), roughly `0`..`1` for normalised audio -- take "
        "`20*log10(x)` for a dBFS-like figure. It reflects signal level, not "
        "perceived loudness (no A/K-weighting). `NULL` on `NULL` or undecodable "
        "input."
    ),
    doc_md=(
        "# rms_energy\n\n"
        "Mean **RMS energy** (`DOUBLE`) -- a loudness proxy.\n\n"
        "```sql\n"
        "SELECT audio.main.rms_energy('/music/track.wav');\n"
        "```\n\n"
        "Higher means louder; near `0` means near-silent. Linear amplitude "
        "(~`0`..`1`), not dB -- use `20*log10(x)` for dBFS. Also accepts a `BLOB`. "
        "`NULL` for undecodable input."
    ),
    keywords=["rms", "energy", "loudness", "level", "amplitude", "volume", "power", "silence"],
    category="loudness",
    example_queries=[
        ("Mean RMS energy (loudness proxy) of an inline WAV tone.", f"SELECT audio.main.rms_energy({_TONE_BLOB_SQL})"),
    ],
)

_TAGS_ZCR = meta.object_tags(
    title="Mean Zero-Crossing Rate",
    doc_llm=(
        "## audio.zero_crossing_rate\n\n"
        "Return the **mean zero-crossing rate** of the signal as a `DOUBLE` -- the "
        "average fraction of adjacent samples where the waveform changes sign. "
        "Accepts a `VARCHAR` path or a `BLOB` of bytes.\n\n"
        "**When to use.** A cheap timbre/noisiness cue: percussive, noisy, or "
        "unvoiced content has a high ZCR; smooth tonal content a low one. Useful "
        "for rough speech-vs-music or voiced-vs-unvoiced splits.\n\n"
        "**Behavior.** Computed per analysis frame over a bounded mono signal and "
        "averaged; the value is a fraction in `0`..`1` (the share of sample pairs "
        "that cross zero). It tracks high-frequency/noise content but is not a "
        "pitch or frequency estimate. `NULL` on `NULL` or undecodable input."
    ),
    doc_md=(
        "# zero_crossing_rate\n\n"
        "Mean **zero-crossing rate** (`DOUBLE`, `0`..`1`).\n\n"
        "```sql\n"
        "SELECT audio.main.zero_crossing_rate('/music/track.wav');\n"
        "```\n\n"
        "High for noisy/percussive audio, low for smooth tones; a fraction of sign "
        "changes, not a frequency. Also accepts a `BLOB`. `NULL` for undecodable "
        "input."
    ),
    keywords=["zero crossing rate", "zcr", "noisiness", "timbre", "sign changes", "percussive", "voiced"],
    category="spectral",
    example_queries=[
        (
            "Mean zero-crossing rate of an inline WAV tone.",
            f"SELECT audio.main.zero_crossing_rate({_TONE_BLOB_SQL})",
        ),
    ],
)

_TAGS_CENTROID = meta.object_tags(
    title="Mean Spectral Centroid (Hz)",
    doc_llm=(
        "## audio.spectral_centroid\n\n"
        "Return the **mean spectral centroid in Hz** as a `DOUBLE` -- the "
        "amplitude-weighted center of mass of the magnitude spectrum, the standard "
        "measure of timbral brightness. Accepts a `VARCHAR` path or a `BLOB` of "
        "bytes.\n\n"
        "**When to use.** Quantify brightness to rank or cluster sounds (bright/"
        "sharp vs dark/dull), gate on tone colour, or feed a similarity model.\n\n"
        "**Behavior.** Computed per analysis frame and averaged over a bounded mono "
        "signal resampled to 22.05 kHz, so the value ranges from `0` up to the "
        "Nyquist limit of ~11 kHz; brighter sounds sit higher. It is energy-"
        "weighted, so near-silent frames contribute little. `NULL` on `NULL` or "
        "undecodable input."
    ),
    doc_md=(
        "# spectral_centroid\n\n"
        "Mean **spectral centroid in Hz** (`DOUBLE`) -- spectral brightness.\n\n"
        "```sql\n"
        "SELECT audio.main.spectral_centroid('/music/track.wav');\n"
        "```\n\n"
        "Higher = brighter/sharper timbre. Analysis runs at 22.05 kHz, so values "
        "top out near the ~11 kHz Nyquist limit. Also accepts a `BLOB` of raw "
        "bytes. `NULL` for undecodable input."
    ),
    keywords=["spectral centroid", "brightness", "timbre", "center of mass", "spectrum", "sharpness"],
    category="spectral",
    example_queries=[
        (
            "Mean spectral centroid (Hz), a brightness measure, of an inline WAV tone.",
            f"SELECT audio.main.spectral_centroid({_TONE_BLOB_SQL})",
        ),
    ],
)

_TAGS_BANDWIDTH = meta.object_tags(
    title="Mean Spectral Bandwidth (Hz)",
    doc_llm=(
        "## audio.spectral_bandwidth\n\n"
        "Return the **mean spectral bandwidth in Hz** as a `DOUBLE` -- the energy-"
        "weighted spread (a p=2 standard deviation) of the spectrum around its "
        "centroid. Accepts a `VARCHAR` path or a `BLOB` of bytes.\n\n"
        "**When to use.** Separate narrow-band tonal sounds (small bandwidth) from "
        "broadband or noisy sounds (large bandwidth); a complementary timbre "
        "feature alongside `spectral_centroid`.\n\n"
        "**Behavior.** Computed per analysis frame and averaged over a bounded mono "
        "signal resampled to 22.05 kHz, so it is bounded by the ~11 kHz Nyquist "
        "limit. `NULL` on `NULL` or undecodable input."
    ),
    doc_md=(
        "# spectral_bandwidth\n\n"
        "Mean **spectral bandwidth in Hz** (`DOUBLE`) -- spectral spread.\n\n"
        "```sql\n"
        "SELECT audio.main.spectral_bandwidth('/music/track.wav');\n"
        "```\n\n"
        "Small = tonal/narrow-band; large = broadband/noisy. Bounded by the "
        "~11 kHz Nyquist limit of the 22.05 kHz analysis. Also accepts a `BLOB`. "
        "`NULL` for undecodable input."
    ),
    keywords=["spectral bandwidth", "spread", "spectrum width", "timbre", "broadband", "narrowband"],
    category="spectral",
    example_queries=[
        (
            "Mean spectral bandwidth (Hz), a spectral-spread measure, of an inline WAV tone.",
            f"SELECT audio.main.spectral_bandwidth({_TONE_BLOB_SQL})",
        ),
    ],
)

_TAGS_MFCC = meta.object_tags(
    title="MFCC Coefficient Means",
    doc_llm=(
        "## audio.mfcc\n\n"
        "Return the **mean of each Mel-frequency cepstral coefficient (MFCC)** over "
        "the whole clip as a `DOUBLE[]` -- a compact timbral fingerprint.\n\n"
        "**Signature.** `mfcc(audio)` or `mfcc(audio, n)`. `audio` is a `VARCHAR` "
        "path or a `BLOB` of raw bytes; the optional `n` sets how many coefficients "
        "to keep (default `13`, clamped to `[1, 128]`), and the result is a "
        "length-`n` list of per-coefficient means.\n\n"
        "**When to use.** A ready-made feature vector for audio similarity search, "
        "clustering, or classification -- store the list per recording and compare "
        "with a distance metric.\n\n"
        "**Caveats for ML.** The first coefficient (`MFCC-0`) tracks overall "
        "log-energy/loudness rather than timbre and can dominate a raw Euclidean "
        "distance -- drop it or weight it down for timbral similarity. The means are "
        "unnormalized and the coefficients live on different scales, so standardize "
        "(z-score) them across your corpus before computing distances. Averaging "
        "over the whole clip also discards how timbre evolves over time. Returns "
        "`NULL` on `NULL` or undecodable input."
    ),
    doc_md=(
        "# mfcc\n\n"
        "Per-coefficient **MFCC means** over the clip as a `DOUBLE[]` (a timbral "
        "fingerprint).\n\n"
        "```sql\n"
        "SELECT audio.main.mfcc('/music/track.wav');       -- 13 coefficient means\n"
        "SELECT audio.main.mfcc('/music/track.wav', 20);   -- keep 20 instead\n"
        "```\n\n"
        "Call it as `mfcc(audio)` or `mfcc(audio, n)`: the optional `n` (default "
        "13, clamped to 1..128) sets the vector length, and `audio` is a path or a "
        "`BLOB`.\n\n"
        "**Using it as an ML feature.** `MFCC-0` (the first entry) mostly reflects "
        "overall loudness, not timbre -- drop or down-weight it for similarity. The "
        "means are unnormalized and per-coefficient scales differ, so z-score them "
        "across your data before Euclidean distance. `NULL` for undecodable input."
    ),
    keywords=["mfcc", "cepstral", "timbre", "feature vector", "fingerprint", "embedding", "similarity", "ml"],
    category="spectral",
    example_queries=[
        ("Default 13 MFCC coefficient means of an inline WAV tone.", f"SELECT audio.main.mfcc({_TONE_BLOB_SQL})"),
        ("Keep 20 MFCC coefficients instead of the default 13.", f"SELECT audio.main.mfcc({_TONE_BLOB_SQL}, 20)"),
        composed.MFCC_TIMBRE_DISTANCE,
    ],
)

# Composed example (mfcc on both sides of an example_clips self-join, compared
# with list_distance) carried by the default-arity mfcc overloads.
_MFCC_DISTANCE_EXAMPLE = FunctionExample(
    sql=composed.MFCC_TIMBRE_DISTANCE[1], description=composed.MFCC_TIMBRE_DISTANCE[0]
)

_TAGS_KEY = meta.object_tags(
    title="Estimated Musical Key",
    doc_llm=(
        "## audio.estimated_key\n\n"
        "Estimate the **musical key** of a recording, e.g. `'C major'` or "
        "`'A minor'`, as a `VARCHAR`. Accepts a `VARCHAR` path or a `BLOB` of "
        "bytes.\n\n"
        "**When to use.** Tag a music library by key for harmonic mixing "
        "(DJ-style), find compatible tracks, or filter by tonality.\n\n"
        "**Behavior.** A **heuristic** Krumhansl-Schmuckler correlation of the "
        "chroma profile against the 24 major/minor key templates -- treat it as "
        "a hint, not ground truth (it can confuse relative major/minor and "
        "struggles on atonal/percussive material). `NULL` on `NULL` or "
        "undecodable input."
    ),
    doc_md=(
        "# estimated_key\n\n"
        "Heuristic **musical key** (`VARCHAR`), e.g. `'C major'`.\n\n"
        "```sql\n"
        "SELECT audio.main.estimated_key('/music/track.wav');\n"
        "```\n\n"
        "> Krumhansl-Schmuckler chroma heuristic -- a hint, not ground truth. "
        "May confuse relative major/minor. `NULL` for undecodable input."
    ),
    keywords=["key", "musical key", "tonality", "major", "minor", "chroma", "harmonic", "camelot", "scale"],
    category="tonal",
    example_queries=[
        (
            "Heuristic musical key (e.g. 'C major') of an inline WAV tone.",
            f"SELECT audio.main.estimated_key({_TONE_BLOB_SQL})",
        ),
    ],
)


# ===========================================================================
# duration(audio) -> DOUBLE
# ===========================================================================


class DurationPathFunction(ScalarFunction):
    """``duration(path)`` -- audio duration in seconds (NULL if undecodable)."""

    class Meta:
        """Function metadata."""

        name = "duration"
        description = "Audio duration in seconds, from a file path (NULL if undecodable)"
        categories = ["audio", "metadata"]
        tags = _TAGS_DURATION
        examples = [_tone_example("duration", "Duration in seconds of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.duration, pa.float64(), pa.DoubleArray)


class DurationBytesFunction(ScalarFunction):
    """``duration(blob)`` -- audio duration in seconds from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "duration"
        description = "Audio duration in seconds, from a BLOB of audio bytes (NULL if undecodable)"
        categories = ["audio", "metadata"]
        tags = _TAGS_DURATION
        examples = [_tone_example("duration", "Duration in seconds of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.duration, pa.float64(), pa.DoubleArray)


# ===========================================================================
# sample_rate(audio) -> INT
# ===========================================================================


class SampleRatePathFunction(ScalarFunction):
    """``sample_rate(path)`` -- native sample rate in Hz."""

    class Meta:
        """Function metadata."""

        name = "sample_rate"
        description = "Native sample rate in Hz, from a file path (NULL if undecodable)"
        categories = ["audio", "metadata"]
        tags = _TAGS_SAMPLE_RATE
        examples = [_tone_example("sample_rate", "Sample rate (Hz) of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.Int32Array, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.sample_rate, pa.int32(), pa.Int32Array)


class SampleRateBytesFunction(ScalarFunction):
    """``sample_rate(blob)`` -- native sample rate in Hz from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "sample_rate"
        description = "Native sample rate in Hz, from a BLOB of audio bytes (NULL if undecodable)"
        categories = ["audio", "metadata"]
        tags = _TAGS_SAMPLE_RATE
        examples = [_tone_example("sample_rate", "Sample rate (Hz) of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.Int32Array, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.sample_rate, pa.int32(), pa.Int32Array)


# ===========================================================================
# channels(audio) -> INT
# ===========================================================================


class ChannelsPathFunction(ScalarFunction):
    """``channels(path)`` -- number of audio channels."""

    class Meta:
        """Function metadata."""

        name = "channels"
        description = "Number of audio channels (1=mono, 2=stereo), from a file path"
        categories = ["audio", "metadata"]
        tags = _TAGS_CHANNELS
        examples = [_tone_example("channels", "Channel count of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.Int32Array, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.channels, pa.int32(), pa.Int32Array)


class ChannelsBytesFunction(ScalarFunction):
    """``channels(blob)`` -- number of audio channels from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "channels"
        description = "Number of audio channels (1=mono, 2=stereo), from a BLOB of audio bytes"
        categories = ["audio", "metadata"]
        tags = _TAGS_CHANNELS
        examples = [_tone_example("channels", "Channel count of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.Int32Array, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.channels, pa.int32(), pa.Int32Array)


# ===========================================================================
# tempo(audio) -> DOUBLE
# ===========================================================================


class TempoPathFunction(ScalarFunction):
    """``tempo(path)`` -- estimated tempo (BPM)."""

    class Meta:
        """Function metadata."""

        name = "tempo"
        description = "Estimated tempo in BPM (heuristic), from a file path"
        categories = ["audio", "rhythm"]
        tags = _TAGS_TEMPO
        examples = [_tone_example("tempo", "Estimated tempo (BPM) of an inline WAV clip")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.tempo, pa.float64(), pa.DoubleArray)


class TempoBytesFunction(ScalarFunction):
    """``tempo(blob)`` -- estimated tempo (BPM) from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "tempo"
        description = "Estimated tempo in BPM (heuristic), from a BLOB of audio bytes"
        categories = ["audio", "rhythm"]
        tags = _TAGS_TEMPO
        examples = [_tone_example("tempo", "Estimated tempo (BPM) of an inline WAV clip (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.tempo, pa.float64(), pa.DoubleArray)


# ===========================================================================
# rms_energy(audio) -> DOUBLE
# ===========================================================================


class RmsEnergyPathFunction(ScalarFunction):
    """``rms_energy(path)`` -- mean RMS energy."""

    class Meta:
        """Function metadata."""

        name = "rms_energy"
        description = "Mean root-mean-square energy of the signal, from a file path"
        categories = ["audio", "energy"]
        tags = _TAGS_RMS
        examples = [_tone_example("rms_energy", "Mean RMS energy of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.rms_energy, pa.float64(), pa.DoubleArray)


class RmsEnergyBytesFunction(ScalarFunction):
    """``rms_energy(blob)`` -- mean RMS energy from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "rms_energy"
        description = "Mean root-mean-square energy of the signal, from a BLOB of audio bytes"
        categories = ["audio", "energy"]
        tags = _TAGS_RMS
        examples = [_tone_example("rms_energy", "Mean RMS energy of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.rms_energy, pa.float64(), pa.DoubleArray)


# ===========================================================================
# zero_crossing_rate(audio) -> DOUBLE
# ===========================================================================


class ZeroCrossingRatePathFunction(ScalarFunction):
    """``zero_crossing_rate(path)`` -- mean zero-crossing rate."""

    class Meta:
        """Function metadata."""

        name = "zero_crossing_rate"
        description = "Mean zero-crossing rate (fraction of sign changes), from a file path"
        categories = ["audio", "spectral"]
        tags = _TAGS_ZCR
        examples = [_tone_example("zero_crossing_rate", "Mean zero-crossing rate of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.zero_crossing_rate, pa.float64(), pa.DoubleArray)


class ZeroCrossingRateBytesFunction(ScalarFunction):
    """``zero_crossing_rate(blob)`` -- mean zero-crossing rate from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "zero_crossing_rate"
        description = "Mean zero-crossing rate (fraction of sign changes), from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        tags = _TAGS_ZCR
        examples = [_tone_example("zero_crossing_rate", "Mean zero-crossing rate of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.zero_crossing_rate, pa.float64(), pa.DoubleArray)


# ===========================================================================
# spectral_centroid(audio) -> DOUBLE
# ===========================================================================


class SpectralCentroidPathFunction(ScalarFunction):
    """``spectral_centroid(path)`` -- mean spectral centroid (Hz)."""

    class Meta:
        """Function metadata."""

        name = "spectral_centroid"
        description = "Mean spectral centroid in Hz (spectrum centre of mass), from a file path"
        categories = ["audio", "spectral"]
        tags = _TAGS_CENTROID
        examples = [_tone_example("spectral_centroid", "Mean spectral centroid (Hz) of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.spectral_centroid, pa.float64(), pa.DoubleArray)


class SpectralCentroidBytesFunction(ScalarFunction):
    """``spectral_centroid(blob)`` -- mean spectral centroid (Hz) from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "spectral_centroid"
        description = "Mean spectral centroid in Hz (spectrum centre of mass), from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        tags = _TAGS_CENTROID
        examples = [_tone_example("spectral_centroid", "Mean spectral centroid (Hz) of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.spectral_centroid, pa.float64(), pa.DoubleArray)


# ===========================================================================
# spectral_bandwidth(audio) -> DOUBLE
# ===========================================================================


class SpectralBandwidthPathFunction(ScalarFunction):
    """``spectral_bandwidth(path)`` -- mean spectral bandwidth (Hz)."""

    class Meta:
        """Function metadata."""

        name = "spectral_bandwidth"
        description = "Mean spectral bandwidth in Hz, from a file path"
        categories = ["audio", "spectral"]
        tags = _TAGS_BANDWIDTH
        examples = [_tone_example("spectral_bandwidth", "Mean spectral bandwidth (Hz) of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.spectral_bandwidth, pa.float64(), pa.DoubleArray)


class SpectralBandwidthBytesFunction(ScalarFunction):
    """``spectral_bandwidth(blob)`` -- mean spectral bandwidth (Hz) from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "spectral_bandwidth"
        description = "Mean spectral bandwidth in Hz, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        tags = _TAGS_BANDWIDTH
        examples = [_tone_example("spectral_bandwidth", "Mean spectral bandwidth (Hz) of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.DoubleArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.spectral_bandwidth, pa.float64(), pa.DoubleArray)


# ===========================================================================
# mfcc(audio[, n]) -> DOUBLE[]  -- LIST return REQUIRES explicit Returns(arrow_type=...)
# ===========================================================================

_MFCC_LIST = pa.list_(pa.float64())
_N_DOC = "Number of MFCC coefficients (default 13, clamped to [1, 128])."


class MfccPathFunction(ScalarFunction):
    """``mfcc(path)`` -- mean of each of 13 MFCCs."""

    class Meta:
        """Function metadata."""

        name = "mfcc"
        description = "Mean of each of 13 MFCC coefficients, from a file path"
        categories = ["audio", "spectral"]
        tags = _TAGS_MFCC
        examples = [_tone_example("mfcc", "13 MFCC means of an inline WAV tone"), _MFCC_DISTANCE_EXAMPLE]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.Array[Any], Returns(arrow_type=_MFCC_LIST)]:
        """Map each input row to its output value."""
        return _map_path(path, features.mfcc, _MFCC_LIST, pa.ListArray)


class MfccPathNFunction(ScalarFunction):
    """``mfcc(path, n)`` -- mean of each of ``n`` MFCCs."""

    class Meta:
        """Function metadata."""

        name = "mfcc"
        description = "Mean of each of n MFCC coefficients, from a file path"
        categories = ["audio", "spectral"]
        tags = _TAGS_MFCC
        examples = [
            FunctionExample(
                sql=f"SELECT audio.main.mfcc({_TONE_BLOB_SQL}, 20)",
                description="20 MFCC means of an inline WAV tone",
            ),
        ]

    @classmethod
    def compute(
        cls,
        path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)],
        n: Annotated[int, ConstParam(_N_DOC)],
    ) -> Annotated[pa.Array[Any], Returns(arrow_type=_MFCC_LIST)]:
        """Map each input row to its output value."""
        return _map_path(path, lambda a: features.mfcc(a, n), _MFCC_LIST, pa.ListArray)


class MfccBytesFunction(ScalarFunction):
    """``mfcc(blob)`` -- mean of each of 13 MFCCs from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "mfcc"
        description = "Mean of each of 13 MFCC coefficients, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        tags = _TAGS_MFCC
        examples = [_tone_example("mfcc", "13 MFCC means of an inline WAV tone (BLOB)"), _MFCC_DISTANCE_EXAMPLE]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.Array[Any], Returns(arrow_type=_MFCC_LIST)]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.mfcc, _MFCC_LIST, pa.ListArray)


class MfccBytesNFunction(ScalarFunction):
    """``mfcc(blob, n)`` -- mean of each of ``n`` MFCCs from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "mfcc"
        description = "Mean of each of n MFCC coefficients, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        tags = _TAGS_MFCC
        examples = [
            FunctionExample(
                sql=f"SELECT audio.main.mfcc({_TONE_BLOB_SQL}, 20)",
                description="20 MFCC means of an inline WAV tone (BLOB)",
            ),
        ]

    @classmethod
    def compute(
        cls,
        blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)],
        n: Annotated[int, ConstParam(_N_DOC)],
    ) -> Annotated[pa.Array[Any], Returns(arrow_type=_MFCC_LIST)]:
        """Map each input row to its output value."""
        return _map_bytes(blob, lambda a: features.mfcc(a, n), _MFCC_LIST, pa.ListArray)


# ===========================================================================
# estimated_key(audio) -> VARCHAR  (heuristic)
# ===========================================================================


class EstimatedKeyPathFunction(ScalarFunction):
    """``estimated_key(path)`` -- chroma-based key estimate (heuristic)."""

    class Meta:
        """Function metadata."""

        name = "estimated_key"
        description = "Heuristic chroma-based musical key estimate, e.g. 'C major', from a file path"
        categories = ["audio", "harmony"]
        tags = _TAGS_KEY
        examples = [_tone_example("estimated_key", "Estimated musical key of an inline WAV tone")]

    @classmethod
    def compute(cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]) -> Annotated[pa.StringArray, Returns()]:
        """Map each input row to its output value."""
        return _map_path(path, features.estimated_key, pa.string(), pa.StringArray)


class EstimatedKeyBytesFunction(ScalarFunction):
    """``estimated_key(blob)`` -- chroma-based key estimate from raw bytes."""

    class Meta:
        """Function metadata."""

        name = "estimated_key"
        description = "Heuristic chroma-based musical key estimate, e.g. 'C major', from a BLOB of audio bytes"
        categories = ["audio", "harmony"]
        tags = _TAGS_KEY
        examples = [_tone_example("estimated_key", "Estimated musical key of an inline WAV tone (BLOB)")]

    @classmethod
    def compute(cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]) -> Annotated[pa.StringArray, Returns()]:
        """Map each input row to its output value."""
        return _map_bytes(blob, features.estimated_key, pa.string(), pa.StringArray)


SCALAR_FUNCTIONS: list[type] = [
    DurationPathFunction,
    DurationBytesFunction,
    SampleRatePathFunction,
    SampleRateBytesFunction,
    ChannelsPathFunction,
    ChannelsBytesFunction,
    TempoPathFunction,
    TempoBytesFunction,
    RmsEnergyPathFunction,
    RmsEnergyBytesFunction,
    ZeroCrossingRatePathFunction,
    ZeroCrossingRateBytesFunction,
    SpectralCentroidPathFunction,
    SpectralCentroidBytesFunction,
    SpectralBandwidthPathFunction,
    SpectralBandwidthBytesFunction,
    MfccPathFunction,
    MfccPathNFunction,
    MfccBytesFunction,
    MfccBytesNFunction,
    EstimatedKeyPathFunction,
    EstimatedKeyBytesFunction,
]

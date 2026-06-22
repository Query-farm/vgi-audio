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
from typing import Annotated

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import features
from .features import AudioInput

# ---------------------------------------------------------------------------
# Mapping helpers: apply a pure ``AudioInput -> X`` function across an input
# array, passing NULL straight through. One pair (path / bytes) per feature.
# ---------------------------------------------------------------------------


def _map_path[T](
    arr: pa.StringArray, fn: Callable[[AudioInput | None], T], arrow_type: pa.DataType
) -> pa.Array:
    out = [fn(AudioInput.from_path(x)) for x in arr.to_pylist()]
    return pa.array(out, type=arrow_type)


def _map_bytes[T](
    arr: pa.BinaryArray, fn: Callable[[AudioInput | None], T], arrow_type: pa.DataType
) -> pa.Array:
    out = [fn(AudioInput.from_bytes(x)) for x in arr.to_pylist()]
    return pa.array(out, type=arrow_type)


_PATH_DOC = "Filesystem path to an audio file the worker will open."
_BLOB_DOC = "Raw audio bytes (e.g. the contents of a WAV/FLAC file)."


# ===========================================================================
# duration(audio) -> DOUBLE
# ===========================================================================


class DurationPathFunction(ScalarFunction):
    """``duration(path)`` -- audio duration in seconds (NULL if undecodable)."""

    class Meta:
        name = "duration"
        description = "Audio duration in seconds, from a file path (NULL if undecodable)"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.duration('/tmp/tone.wav')", description="Duration of a WAV file"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.duration, pa.float64())


class DurationBytesFunction(ScalarFunction):
    """``duration(blob)`` -- audio duration in seconds from raw bytes."""

    class Meta:
        name = "duration"
        description = "Audio duration in seconds, from a BLOB of audio bytes (NULL if undecodable)"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.duration(content) FROM read_blob('*.wav')",
                description="Duration from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.duration, pa.float64())


# ===========================================================================
# sample_rate(audio) -> INT
# ===========================================================================


class SampleRatePathFunction(ScalarFunction):
    """``sample_rate(path)`` -- native sample rate in Hz."""

    class Meta:
        name = "sample_rate"
        description = "Native sample rate in Hz, from a file path (NULL if undecodable)"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.sample_rate('/tmp/tone.wav')", description="Sample rate of a WAV"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.Int32Array, Returns()]:
        return _map_path(path, features.sample_rate, pa.int32())


class SampleRateBytesFunction(ScalarFunction):
    """``sample_rate(blob)`` -- native sample rate in Hz from raw bytes."""

    class Meta:
        name = "sample_rate"
        description = "Native sample rate in Hz, from a BLOB of audio bytes (NULL if undecodable)"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.sample_rate(content) FROM read_blob('*.wav')",
                description="Sample rate from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.Int32Array, Returns()]:
        return _map_bytes(blob, features.sample_rate, pa.int32())


# ===========================================================================
# channels(audio) -> INT
# ===========================================================================


class ChannelsPathFunction(ScalarFunction):
    """``channels(path)`` -- number of audio channels."""

    class Meta:
        name = "channels"
        description = "Number of audio channels (1=mono, 2=stereo), from a file path"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.channels('/tmp/tone.wav')", description="Channel count of a WAV"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.Int32Array, Returns()]:
        return _map_path(path, features.channels, pa.int32())


class ChannelsBytesFunction(ScalarFunction):
    """``channels(blob)`` -- number of audio channels from raw bytes."""

    class Meta:
        name = "channels"
        description = "Number of audio channels (1=mono, 2=stereo), from a BLOB of audio bytes"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT audio.channels(content) FROM read_blob('*.wav')",
                description="Channels from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.Int32Array, Returns()]:
        return _map_bytes(blob, features.channels, pa.int32())


# ===========================================================================
# tempo(audio) -> DOUBLE
# ===========================================================================


class TempoPathFunction(ScalarFunction):
    """``tempo(path)`` -- estimated tempo (BPM)."""

    class Meta:
        name = "tempo"
        description = "Estimated tempo in BPM (heuristic), from a file path"
        categories = ["audio", "rhythm"]
        examples = [
            FunctionExample(
                sql="SELECT audio.tempo('/tmp/click.wav')", description="Estimated BPM of a track"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.tempo, pa.float64())


class TempoBytesFunction(ScalarFunction):
    """``tempo(blob)`` -- estimated tempo (BPM) from raw bytes."""

    class Meta:
        name = "tempo"
        description = "Estimated tempo in BPM (heuristic), from a BLOB of audio bytes"
        categories = ["audio", "rhythm"]
        examples = [
            FunctionExample(
                sql="SELECT audio.tempo(content) FROM read_blob('*.wav')",
                description="Estimated BPM from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.tempo, pa.float64())


# ===========================================================================
# rms_energy(audio) -> DOUBLE
# ===========================================================================


class RmsEnergyPathFunction(ScalarFunction):
    """``rms_energy(path)`` -- mean RMS energy."""

    class Meta:
        name = "rms_energy"
        description = "Mean root-mean-square energy of the signal, from a file path"
        categories = ["audio", "energy"]
        examples = [
            FunctionExample(sql="SELECT audio.rms_energy('/tmp/tone.wav')", description="Mean RMS energy"),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.rms_energy, pa.float64())


class RmsEnergyBytesFunction(ScalarFunction):
    """``rms_energy(blob)`` -- mean RMS energy from raw bytes."""

    class Meta:
        name = "rms_energy"
        description = "Mean root-mean-square energy of the signal, from a BLOB of audio bytes"
        categories = ["audio", "energy"]
        examples = [
            FunctionExample(
                sql="SELECT audio.rms_energy(content) FROM read_blob('*.wav')",
                description="Mean RMS from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.rms_energy, pa.float64())


# ===========================================================================
# zero_crossing_rate(audio) -> DOUBLE
# ===========================================================================


class ZeroCrossingRatePathFunction(ScalarFunction):
    """``zero_crossing_rate(path)`` -- mean zero-crossing rate."""

    class Meta:
        name = "zero_crossing_rate"
        description = "Mean zero-crossing rate (fraction of sign changes), from a file path"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(sql="SELECT audio.zero_crossing_rate('/tmp/tone.wav')", description="Mean ZCR"),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.zero_crossing_rate, pa.float64())


class ZeroCrossingRateBytesFunction(ScalarFunction):
    """``zero_crossing_rate(blob)`` -- mean zero-crossing rate from raw bytes."""

    class Meta:
        name = "zero_crossing_rate"
        description = "Mean zero-crossing rate (fraction of sign changes), from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.zero_crossing_rate(content) FROM read_blob('*.wav')",
                description="Mean ZCR from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.zero_crossing_rate, pa.float64())


# ===========================================================================
# spectral_centroid(audio) -> DOUBLE
# ===========================================================================


class SpectralCentroidPathFunction(ScalarFunction):
    """``spectral_centroid(path)`` -- mean spectral centroid (Hz)."""

    class Meta:
        name = "spectral_centroid"
        description = "Mean spectral centroid in Hz (spectrum centre of mass), from a file path"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.spectral_centroid('/tmp/tone.wav')", description="Mean spectral centroid"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.spectral_centroid, pa.float64())


class SpectralCentroidBytesFunction(ScalarFunction):
    """``spectral_centroid(blob)`` -- mean spectral centroid (Hz) from raw bytes."""

    class Meta:
        name = "spectral_centroid"
        description = "Mean spectral centroid in Hz (spectrum centre of mass), from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.spectral_centroid(content) FROM read_blob('*.wav')",
                description="Centroid from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.spectral_centroid, pa.float64())


# ===========================================================================
# spectral_bandwidth(audio) -> DOUBLE
# ===========================================================================


class SpectralBandwidthPathFunction(ScalarFunction):
    """``spectral_bandwidth(path)`` -- mean spectral bandwidth (Hz)."""

    class Meta:
        name = "spectral_bandwidth"
        description = "Mean spectral bandwidth in Hz, from a file path"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.spectral_bandwidth('/tmp/tone.wav')", description="Mean spectral bandwidth"
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_path(path, features.spectral_bandwidth, pa.float64())


class SpectralBandwidthBytesFunction(ScalarFunction):
    """``spectral_bandwidth(blob)`` -- mean spectral bandwidth (Hz) from raw bytes."""

    class Meta:
        name = "spectral_bandwidth"
        description = "Mean spectral bandwidth in Hz, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.spectral_bandwidth(content) FROM read_blob('*.wav')",
                description="Bandwidth from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        return _map_bytes(blob, features.spectral_bandwidth, pa.float64())


# ===========================================================================
# mfcc(audio[, n]) -> DOUBLE[]  -- LIST return REQUIRES explicit Returns(arrow_type=...)
# ===========================================================================

_MFCC_LIST = pa.list_(pa.float64())
_N_DOC = "Number of MFCC coefficients (default 13, clamped to [1, 128])."


class MfccPathFunction(ScalarFunction):
    """``mfcc(path)`` -- mean of each of 13 MFCCs."""

    class Meta:
        name = "mfcc"
        description = "Mean of each of 13 MFCC coefficients, from a file path"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(sql="SELECT audio.mfcc('/tmp/tone.wav')", description="13 MFCC means"),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_MFCC_LIST)]:
        return _map_path(path, features.mfcc, _MFCC_LIST)


class MfccPathNFunction(ScalarFunction):
    """``mfcc(path, n)`` -- mean of each of ``n`` MFCCs."""

    class Meta:
        name = "mfcc"
        description = "Mean of each of n MFCC coefficients, from a file path"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(sql="SELECT audio.mfcc('/tmp/tone.wav', 20)", description="20 MFCC means"),
        ]

    @classmethod
    def compute(
        cls,
        path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)],
        n: Annotated[int, ConstParam(_N_DOC)],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_MFCC_LIST)]:
        return _map_path(path, lambda a: features.mfcc(a, n), _MFCC_LIST)


class MfccBytesFunction(ScalarFunction):
    """``mfcc(blob)`` -- mean of each of 13 MFCCs from raw bytes."""

    class Meta:
        name = "mfcc"
        description = "Mean of each of 13 MFCC coefficients, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.mfcc(content) FROM read_blob('*.wav')",
                description="13 MFCC means from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_MFCC_LIST)]:
        return _map_bytes(blob, features.mfcc, _MFCC_LIST)


class MfccBytesNFunction(ScalarFunction):
    """``mfcc(blob, n)`` -- mean of each of ``n`` MFCCs from raw bytes."""

    class Meta:
        name = "mfcc"
        description = "Mean of each of n MFCC coefficients, from a BLOB of audio bytes"
        categories = ["audio", "spectral"]
        examples = [
            FunctionExample(
                sql="SELECT audio.mfcc(content, 20) FROM read_blob('*.wav')",
                description="20 MFCC means from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls,
        blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)],
        n: Annotated[int, ConstParam(_N_DOC)],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_MFCC_LIST)]:
        return _map_bytes(blob, lambda a: features.mfcc(a, n), _MFCC_LIST)


# ===========================================================================
# estimated_key(audio) -> VARCHAR  (heuristic)
# ===========================================================================


class EstimatedKeyPathFunction(ScalarFunction):
    """``estimated_key(path)`` -- chroma-based key estimate (heuristic)."""

    class Meta:
        name = "estimated_key"
        description = "Heuristic chroma-based musical key estimate, e.g. 'C major', from a file path"
        categories = ["audio", "harmony"]
        examples = [
            FunctionExample(sql="SELECT audio.estimated_key('/tmp/tone.wav')", description="Estimated key"),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc=_PATH_DOC)]
    ) -> Annotated[pa.StringArray, Returns()]:
        return _map_path(path, features.estimated_key, pa.string())


class EstimatedKeyBytesFunction(ScalarFunction):
    """``estimated_key(blob)`` -- chroma-based key estimate from raw bytes."""

    class Meta:
        name = "estimated_key"
        description = (
            "Heuristic chroma-based musical key estimate, e.g. 'C major', from a BLOB of audio bytes"
        )
        categories = ["audio", "harmony"]
        examples = [
            FunctionExample(
                sql="SELECT audio.estimated_key(content) FROM read_blob('*.wav')",
                description="Estimated key from bytes",
            ),
        ]

    @classmethod
    def compute(
        cls, blob: Annotated[pa.BinaryArray, Param(doc=_BLOB_DOC)]
    ) -> Annotated[pa.StringArray, Returns()]:
        return _map_bytes(blob, features.estimated_key, pa.string())


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

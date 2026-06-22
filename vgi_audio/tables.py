"""Set-returning audio table functions for DuckDB.

These return **many rows** (``beats``) or a single structured row
(``audio_info``), so they are exposed as **table functions** -- the form that
accepts a positional/typed argument through the bind lifecycle. As with the
scalars, the polymorphic audio argument is a ``VARCHAR`` path *or* a ``BLOB`` of
bytes, exposed as two same-name overloads resolved by argument type:

    SELECT * FROM audio.beats('/tmp/click.wav');
    SELECT * FROM audio.beats(content)      FROM read_blob('*.wav');   -- (via a join)
    SELECT * FROM audio.audio_info('/tmp/tone.wav');

Robustness: undecodable / hostile input yields **no rows** (the underlying pure
logic returns ``None``; the worker never crashes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from . import features
from .features import AudioInput
from .schema_utils import field

_PATH_ARG: Arg = Arg(0, arrow_type=pa.string(), doc="Filesystem path to an audio file the worker will open.")
_BLOB_ARG: Arg = Arg(0, arrow_type=pa.binary(), doc="Raw audio bytes (e.g. the contents of a WAV/FLAC file).")


# ===========================================================================
# beats(audio) -> (seq BIGINT, time DOUBLE)
# ===========================================================================


@dataclass(kw_only=True)
class _BeatsPathArgs:
    """``beats(path)``."""

    path: Annotated[str, _PATH_ARG]


@dataclass(kw_only=True)
class _BeatsBytesArgs:
    """``beats(blob)``."""

    blob: Annotated[bytes, _BLOB_ARG]


_BEATS_SCHEMA = pa.schema(
    [
        field("seq", pa.int64(), "0-based beat index.", nullable=False),
        field("time", pa.float64(), "Beat onset time in seconds.", nullable=False),
    ]
)


def _emit_beats(audio: AudioInput | None, params: ProcessParams, out: OutputCollector) -> None:
    times = features.beat_times(audio)
    if times is None:
        times = []  # undecodable -> no rows (still emit an empty batch + finish)
    out.emit(
        pa.RecordBatch.from_pydict(
            {"seq": list(range(len(times))), "time": times},
            schema=params.output_schema,
        )
    )
    out.finish()


@init_single_worker
@bind_fixed_schema
class BeatsPathFunction(TableFunctionGenerator[_BeatsPathArgs]):
    """Beat onset times of an audio file as ``(seq, time)`` rows."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _BEATS_SCHEMA

    class Meta:
        name = "beats"
        description = "Beat onset times (seq, time) of an audio file path"
        categories = ["audio", "rhythm"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM audio.beats('/tmp/click.wav')",
                description="Beat onset times of a click track",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_BeatsPathArgs]) -> TableCardinality:
        return TableCardinality(estimate=None, max=None)

    @classmethod
    def process(cls, params: ProcessParams[_BeatsPathArgs], state: None, out: OutputCollector) -> None:
        _emit_beats(AudioInput.from_path(params.args.path), params, out)


@init_single_worker
@bind_fixed_schema
class BeatsBytesFunction(TableFunctionGenerator[_BeatsBytesArgs]):
    """Beat onset times of raw audio bytes as ``(seq, time)`` rows."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _BEATS_SCHEMA

    class Meta:
        name = "beats"
        description = "Beat onset times (seq, time) of a BLOB of audio bytes"
        categories = ["audio", "rhythm"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM audio.beats((SELECT content FROM read_blob('click.wav')))",
                description="Beat onset times from audio bytes",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_BeatsBytesArgs]) -> TableCardinality:
        return TableCardinality(estimate=None, max=None)

    @classmethod
    def process(cls, params: ProcessParams[_BeatsBytesArgs], state: None, out: OutputCollector) -> None:
        _emit_beats(AudioInput.from_bytes(params.args.blob), params, out)


# ===========================================================================
# audio_info(audio) -> (duration DOUBLE, sample_rate INT, channels INT)  -- one row
# ===========================================================================


@dataclass(kw_only=True)
class _InfoPathArgs:
    """``audio_info(path)``."""

    path: Annotated[str, _PATH_ARG]


@dataclass(kw_only=True)
class _InfoBytesArgs:
    """``audio_info(blob)``."""

    blob: Annotated[bytes, _BLOB_ARG]


_INFO_SCHEMA = pa.schema(
    [
        field("duration", pa.float64(), "Duration in seconds.", nullable=False),
        field("sample_rate", pa.int32(), "Native sample rate in Hz.", nullable=False),
        field("channels", pa.int32(), "Number of audio channels.", nullable=False),
    ]
)


def _emit_info(audio: AudioInput | None, params: ProcessParams, out: OutputCollector) -> None:
    info = features.audio_info(audio)
    if info is None:
        # undecodable -> no rows.
        out.emit(
            pa.RecordBatch.from_pydict(
                {"duration": [], "sample_rate": [], "channels": []},
                schema=params.output_schema,
            )
        )
    else:
        d, sr, ch = info
        out.emit(
            pa.RecordBatch.from_pydict(
                {"duration": [d], "sample_rate": [sr], "channels": [ch]},
                schema=params.output_schema,
            )
        )
    out.finish()


@init_single_worker
@bind_fixed_schema
class AudioInfoPathFunction(TableFunctionGenerator[_InfoPathArgs]):
    """``(duration, sample_rate, channels)`` for an audio file path (one row)."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _INFO_SCHEMA

    class Meta:
        name = "audio_info"
        description = "Single-row (duration, sample_rate, channels) of an audio file path"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM audio.audio_info('/tmp/tone.wav')",
                description="Metadata of a WAV file",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_InfoPathArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[_InfoPathArgs], state: None, out: OutputCollector) -> None:
        _emit_info(AudioInput.from_path(params.args.path), params, out)


@init_single_worker
@bind_fixed_schema
class AudioInfoBytesFunction(TableFunctionGenerator[_InfoBytesArgs]):
    """``(duration, sample_rate, channels)`` for raw audio bytes (one row)."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _INFO_SCHEMA

    class Meta:
        name = "audio_info"
        description = "Single-row (duration, sample_rate, channels) of a BLOB of audio bytes"
        categories = ["audio", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM audio.audio_info((SELECT content FROM read_blob('tone.wav')))",
                description="Metadata from audio bytes",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_InfoBytesArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[_InfoBytesArgs], state: None, out: OutputCollector) -> None:
        _emit_info(AudioInput.from_bytes(params.args.blob), params, out)


TABLE_FUNCTIONS: list[type] = [
    BeatsPathFunction,
    BeatsBytesFunction,
    AudioInfoPathFunction,
    AudioInfoBytesFunction,
]

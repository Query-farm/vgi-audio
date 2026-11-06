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

import base64
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from vgi.arguments import Arg
from vgi.catalog import ConstraintInfo, Table, constraint_record
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

from . import features, meta
from ._example_audio import CLICK_FLAC_B64, TONE_WAV_B64
from .features import AudioInput
from .schema_utils import field

_PATH_ARG: Arg[str] = Arg(0, arrow_type=pa.string(), doc="Filesystem path to an audio file the worker will open.")
_BLOB_ARG: Arg[bytes] = Arg(0, arrow_type=pa.binary(), doc="Raw audio bytes (e.g. the contents of a WAV/FLAC file).")

# Inline self-contained example inputs (see ``vgi_audio._example_audio``): a
# click track with clear onsets for ``beats`` (so it emits multiple rows) and a
# tone for ``audio_info``. ``from_base64(...)`` yields the ``BLOB`` overload's
# input, so the strict linter can EXECUTE every example with no file on disk.
_CLICK_BLOB_SQL = f"from_base64('{CLICK_FLAC_B64}')"
_TONE_BLOB_SQL = f"from_base64('{TONE_WAV_B64}')"

# VGI509: at least one object ships guaranteed-runnable executable examples.
# Each is fully self-contained (inline blob, catalog-qualified) so VGI906 can
# run it against the live worker.
_EXECUTABLE_EXAMPLES = (
    "["
    '{"description": "Beat onset times of an inline 120 BPM click track.", '
    f'"sql": "SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq"}},'
    '{"description": "Duration / sample rate / channels of an inline WAV tone.", '
    f'"sql": "SELECT duration, sample_rate, channels FROM audio.main.audio_info({_TONE_BLOB_SQL})"}}'
    "]"
)


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

_BEATS_COLUMNS_MD = (
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `seq` | BIGINT | 0-based beat index (ascending). |\n"
    "| `time` | DOUBLE | Beat onset time in seconds. |"
)

# VGI307/VGI414: a table function with no backing table must declare its static
# result schema through the structured `vgi.result_columns_schema` (the retired
# free-form `vgi.result_columns_md` is no longer read). VGI910 checks this matches
# what the function actually returns, so the types mirror `_BEATS_SCHEMA` exactly
# (int64 -> BIGINT, float64 -> DOUBLE).
_BEATS_COLUMNS_SCHEMA = json.dumps(
    [
        {"name": "seq", "type": "BIGINT", "description": "0-based beat index (ascending)."},
        {"name": "time", "type": "DOUBLE", "description": "Beat onset time in seconds."},
    ]
)

_TAGS_BEATS = meta.object_tags(
    title="Beat Onset Times Table",
    doc_llm=(
        "## audio.beats\n\n"
        "A **table function** that returns the **detected beat onset times** of a "
        "recording, one row per beat: `(seq BIGINT, time DOUBLE)`. Accepts a "
        "`VARCHAR` path or a `BLOB` of audio bytes.\n\n"
        "**When to use.** Build a beat grid for visualisation or editing, segment "
        "a track on its beats, count beats, or align events to the rhythm.\n\n"
        "**Output.** `seq` is the 0-based, ascending beat index; `time` is the "
        "onset in seconds. Beats are found by librosa's beat tracker on a bounded "
        "mono signal -- a **heuristic**, strongest on material with a clear "
        "pulse. Returns **no rows** for `NULL`/undecodable input or audio with no "
        "detectable beat; it never raises."
    ),
    doc_md=(
        "# beats\n\n"
        "Table function -> detected **beat onset times**, one row per beat.\n\n"
        f"{_BEATS_COLUMNS_MD}\n\n"
        "Heuristic beat tracking on a bounded mono signal; returns **no rows** for "
        "undecodable input or audio without a clear pulse."
    ),
    keywords=["beats", "beat tracking", "onsets", "beat grid", "rhythm", "downbeat", "tempo", "timeline"],
    category="rhythm",
    example_queries=[
        (
            "Beat onset times of an inline 120 BPM click track, in order.",
            f"SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq",
        ),
    ],
)


def _emit_beats(audio: AudioInput | None, params: ProcessParams[Any], out: OutputCollector) -> None:
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
        """Function metadata."""

        name = "beats"
        description = "Beat onset times (seq, time) of an audio file path"
        categories = ["audio", "rhythm"]
        tags = {
            **_TAGS_BEATS,
            "vgi.result_columns_schema": _BEATS_COLUMNS_SCHEMA,
            "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=f"SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq",
                description="Beat onset times of an inline click track",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_BeatsPathArgs]) -> TableCardinality:
        """Estimated and maximum row count for the planner."""
        return TableCardinality(estimate=None, max=None)

    @classmethod
    def process(cls, params: ProcessParams[_BeatsPathArgs], state: None, out: OutputCollector) -> None:
        """Emit the result rows for this audio input."""
        _emit_beats(AudioInput.from_path(params.args.path), params, out)


@init_single_worker
@bind_fixed_schema
class BeatsBytesFunction(TableFunctionGenerator[_BeatsBytesArgs]):
    """Beat onset times of raw audio bytes as ``(seq, time)`` rows."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _BEATS_SCHEMA

    class Meta:
        """Function metadata."""

        name = "beats"
        description = "Beat onset times (seq, time) of a BLOB of audio bytes"
        categories = ["audio", "rhythm"]
        tags = {**_TAGS_BEATS, "vgi.result_columns_schema": _BEATS_COLUMNS_SCHEMA}
        examples = [
            FunctionExample(
                sql=f"SELECT seq, time FROM audio.main.beats({_CLICK_BLOB_SQL}) ORDER BY seq",
                description="Beat onset times from an inline click track (BLOB)",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_BeatsBytesArgs]) -> TableCardinality:
        """Estimated and maximum row count for the planner."""
        return TableCardinality(estimate=None, max=None)

    @classmethod
    def process(cls, params: ProcessParams[_BeatsBytesArgs], state: None, out: OutputCollector) -> None:
        """Emit the result rows for this audio input."""
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

_INFO_COLUMNS_MD = (
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `duration` | DOUBLE | Duration in seconds. |\n"
    "| `sample_rate` | INTEGER | Native sample rate in Hz. |\n"
    "| `channels` | INTEGER | Number of audio channels (1=mono, 2=stereo). |"
)

# VGI307/VGI414: structured static result schema (types mirror `_INFO_SCHEMA`:
# float64 -> DOUBLE, int32 -> INTEGER). Checked against the live return by VGI910.
_INFO_COLUMNS_SCHEMA = json.dumps(
    [
        {"name": "duration", "type": "DOUBLE", "description": "Duration in seconds."},
        {"name": "sample_rate", "type": "INTEGER", "description": "Native sample rate in Hz."},
        {"name": "channels", "type": "INTEGER", "description": "Number of audio channels (1=mono, 2=stereo)."},
    ]
)

_TAGS_INFO = meta.object_tags(
    title="Combined Audio Metadata Row",
    doc_llm=(
        "## audio.audio_info\n\n"
        "A **table function** returning a recording's core metadata in a single "
        "row: `(duration DOUBLE, sample_rate INTEGER, channels INTEGER)`. Accepts "
        "a `VARCHAR` path or a `BLOB` of audio bytes.\n\n"
        "**When to use.** Get duration, sample rate, and channel count in one "
        "call instead of three separate scalars -- handy when ingesting a batch "
        "of files into a metadata table.\n\n"
        "**Output.** Exactly one row on success. Returns **no rows** for "
        "`NULL`/undecodable input (it never raises). Equivalent to selecting "
        "`duration`, `sample_rate`, and `channels` together."
    ),
    doc_md=(
        "# audio_info\n\n"
        "Table function -> one row of core **audio metadata**.\n\n"
        f"{_INFO_COLUMNS_MD}\n\n"
        "One row on success; **no rows** for undecodable input. A one-shot "
        "alternative to the `duration` / `sample_rate` / `channels` scalars."
    ),
    keywords=["audio info", "metadata", "duration", "sample rate", "channels", "probe", "mediainfo", "summary"],
    category="metadata",
    example_queries=[
        (
            "Duration, sample rate, and channel count of an inline WAV tone in one row.",
            f"SELECT duration, sample_rate, channels FROM audio.main.audio_info({_TONE_BLOB_SQL})",
        ),
    ],
)


def _emit_info(audio: AudioInput | None, params: ProcessParams[Any], out: OutputCollector) -> None:
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
        """Function metadata."""

        name = "audio_info"
        description = "Single-row (duration, sample_rate, channels) of an audio file path"
        categories = ["audio", "metadata"]
        tags = {**_TAGS_INFO, "vgi.result_columns_schema": _INFO_COLUMNS_SCHEMA}
        examples = [
            FunctionExample(
                sql=f"SELECT duration, sample_rate, channels FROM audio.main.audio_info({_TONE_BLOB_SQL})",
                description="Metadata of an inline WAV tone",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_InfoPathArgs]) -> TableCardinality:
        """Estimated and maximum row count for the planner."""
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[_InfoPathArgs], state: None, out: OutputCollector) -> None:
        """Emit the result rows for this audio input."""
        _emit_info(AudioInput.from_path(params.args.path), params, out)


@init_single_worker
@bind_fixed_schema
class AudioInfoBytesFunction(TableFunctionGenerator[_InfoBytesArgs]):
    """``(duration, sample_rate, channels)`` for raw audio bytes (one row)."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _INFO_SCHEMA

    class Meta:
        """Function metadata."""

        name = "audio_info"
        description = "Single-row (duration, sample_rate, channels) of a BLOB of audio bytes"
        categories = ["audio", "metadata"]
        tags = {**_TAGS_INFO, "vgi.result_columns_schema": _INFO_COLUMNS_SCHEMA}
        examples = [
            FunctionExample(
                sql=f"SELECT duration, sample_rate, channels FROM audio.main.audio_info({_TONE_BLOB_SQL})",
                description="Metadata from an inline WAV tone (BLOB)",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_InfoBytesArgs]) -> TableCardinality:
        """Estimated and maximum row count for the planner."""
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[_InfoBytesArgs], state: None, out: OutputCollector) -> None:
        """Emit the result rows for this audio input."""
        _emit_info(AudioInput.from_bytes(params.args.blob), params, out)


# ===========================================================================
# example_clips  --  a browsable discovery table (VGI146)
#
# The rest of the worker is table *functions* that take an audio argument, so an
# agent has nothing to browse before it knows how to supply audio bytes. This
# zero-argument table surfaces the worker's two built-in example clips as real,
# selectable rows -- name, a one-line description, the decoded core metadata, and
# the raw `audio` BLOB itself -- so an analyst can read sample data and copy a
# clip straight into any feature function without a file on disk.
# ===========================================================================


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


# (name, description, base64 WAV) for each built-in example clip.
_EXAMPLE_CLIPS: tuple[tuple[str, str, str], ...] = (
    ("tone", "A short 60 Hz mono sine tone (a steady, tonal test signal).", TONE_WAV_B64),
    ("click", "A 120 BPM mono click track (sharp, evenly-spaced onsets).", CLICK_FLAC_B64),
)

_CLIPS_SCHEMA = pa.schema(
    [
        field("name", pa.string(), "Short identifier of the example clip.", nullable=False),
        field("description", pa.string(), "What the clip is.", nullable=False),
        field("duration", pa.float64(), "Duration in seconds.", nullable=False),
        field("sample_rate", pa.int32(), "Native sample rate in Hz.", nullable=False),
        field("channels", pa.int32(), "Number of audio channels.", nullable=False),
        field("beat_count", pa.int32(), "Number of beats librosa detects in the clip.", nullable=False),
        field("audio", pa.binary(), "Raw WAV audio bytes -- pass straight to any feature function.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class ExampleClipsFunction(TableFunctionGenerator[_NoArgs]):
    """The worker's built-in example audio clips, one row per clip."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLIPS_SCHEMA

    class Meta:
        """Function metadata."""

        name = "example_clips"
        description = "Built-in example audio clips (name, metadata, raw bytes) for trying the feature functions"
        categories = ["audio", "metadata"]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        """Estimated and maximum row count for the planner."""
        return TableCardinality(estimate=len(_EXAMPLE_CLIPS), max=len(_EXAMPLE_CLIPS))

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        """Emit one row per built-in example clip."""
        names: list[str] = []
        descriptions: list[str] = []
        durations: list[float | None] = []
        sample_rates: list[int | None] = []
        channels: list[int | None] = []
        beat_counts: list[int] = []
        audio: list[bytes] = []
        for clip_name, clip_desc, b64 in _EXAMPLE_CLIPS:
            raw = base64.b64decode(b64)
            clip = AudioInput.from_bytes(raw)
            info = features.audio_info(clip)
            dur, sr, ch = info if info is not None else (None, None, None)
            beats = features.beat_times(clip)
            names.append(clip_name)
            descriptions.append(clip_desc)
            durations.append(dur)
            sample_rates.append(sr)
            channels.append(ch)
            beat_counts.append(len(beats) if beats is not None else 0)
            audio.append(raw)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "name": names,
                    "description": descriptions,
                    "duration": durations,
                    "sample_rate": sample_rates,
                    "channels": channels,
                    "beat_count": beat_counts,
                    "audio": audio,
                },
                schema=params.output_schema,
            )
        )
        out.finish()


_CLIPS_DOC_LLM = (
    "## audio.example_clips\n\n"
    "A **browsable table** (no arguments) listing the worker's built-in example "
    "audio clips, one row each: `name`, a one-line `description`, the decoded "
    "`duration`/`sample_rate`/`channels`, the detected `beat_count`, and the raw "
    "`audio` `BLOB`.\n\n"
    "**When to use.** Start here to see real sample data and to grab a ready-made "
    "audio blob: select the `audio` column for a clip and pass it directly to any "
    "feature function (`duration`, `tempo`, `mfcc`, `beats`, `audio_info`, ...) "
    "without needing a file on disk. The `tone` clip is a steady sine (good for "
    "the metadata/spectral scalars); the `click` clip has a clear 120 BPM pulse "
    "(good for `tempo` and `beats`)."
)

_CLIPS_DOC_MD = (
    "# example_clips\n\n"
    "A browsable discovery table of the worker's built-in example audio clips.\n\n"
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `name` | VARCHAR | Short clip identifier (`tone`, `click`). |\n"
    "| `description` | VARCHAR | What the clip is. |\n"
    "| `duration` | DOUBLE | Duration in seconds. |\n"
    "| `sample_rate` | INTEGER | Native sample rate in Hz. |\n"
    "| `channels` | INTEGER | Number of audio channels. |\n"
    "| `beat_count` | INTEGER | Number of beats librosa detects in the clip. |\n"
    "| `audio` | BLOB | Raw WAV bytes to feed any feature function. |\n\n"
    "Select the `audio` column of a row and pass it straight into a scalar or "
    "table function to try the worker with no file on disk."
)


def _trusted(table: str, constraint_id: str, kind: str, columns: list[list[str]]) -> ConstraintInfo:
    """Build an ENFORCED + ALL_ROWS constraint record for a ``main``-schema table.

    vgi.v3 drops the legacy ``primary_key=`` / ``not_null=`` Table fields: the worker
    attests constraints with canonical records instead. A PRIMARY_KEY record also makes
    its columns NOT NULL, so only non-key required columns need a NOT_NULL record. The
    example clips are generated from fixed built-in data, so every row satisfies these
    constraints (that is what ENFORCED + ALL_ROWS attests).

    Args:
        table: Owning table name.
        constraint_id: Stable provider-owned constraint id, unique within the table.
        kind: ``"PRIMARY_KEY"`` or ``"NOT_NULL"``.
        columns: Raw column paths, e.g. ``[["name"]]``.

    Returns:
        The canonical constraint record.
    """
    return constraint_record(
        ["main"], table, constraint_id, kind, columns, enforcement="ENFORCED", validation="ALL_ROWS"
    )


CLIPS_TABLE = Table(
    name="example_clips",
    function=ExampleClipsFunction,
    comment="Built-in example audio clips (name, core metadata, raw bytes) for trying the feature functions.",
    constraints=(
        _trusted("example_clips", "pk", "PRIMARY_KEY", [["name"]]),
        *(
            _trusted("example_clips", f"{col}_not_null", "NOT_NULL", [[col]])
            for col in ("description", "duration", "sample_rate", "channels", "beat_count", "audio")
        ),
    ),
    column_comments={
        "name": "Short identifier of the example clip (`tone` or `click`).",
        "description": "One-line description of the clip.",
        "duration": "Clip duration in seconds.",
        "sample_rate": "Native sample rate in Hz.",
        "channels": "Number of audio channels (1 = mono).",
        "beat_count": "Number of beats librosa detects in the clip (0 if none).",
        "audio": "Raw WAV audio bytes -- pass straight to any feature function (e.g. `audio.main.tempo(audio)`).",
    },
    tags={
        "vgi.title": "Built-in Example Audio Clips",
        "vgi.doc_llm": _CLIPS_DOC_LLM,
        "vgi.doc_md": _CLIPS_DOC_MD,
        "vgi.keywords": json.dumps(
            ["example", "sample", "demo", "clip", "tone", "click", "test audio", "discovery", "blob"]
        ),
        "vgi.category": "metadata",
        "domain": "audio",
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "List the built-in example clips with their core metadata.",
                    "sql": (
                        "SELECT name, description, duration, sample_rate, channels "
                        "FROM audio.main.example_clips ORDER BY name"
                    ),
                },
                {
                    "description": "Estimate the tempo of the built-in click track by feeding its bytes to tempo().",
                    "sql": ("SELECT audio.main.tempo(audio) AS bpm FROM audio.main.example_clips WHERE name = 'click'"),
                },
            ]
        ),
    },
)


TABLE_FUNCTIONS: list[type] = [
    BeatsPathFunction,
    BeatsBytesFunction,
    AudioInfoPathFunction,
    AudioInfoBytesFunction,
]

# Browsable discovery tables (each wraps a zero-argument table function). Kept
# separate from TABLE_FUNCTIONS because these are registered on the schema as
# `tables`, not as callable table functions.
DISCOVERY_TABLES: list[Table] = [CLIPS_TABLE]

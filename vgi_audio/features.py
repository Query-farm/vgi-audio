"""Pure audio-feature logic over librosa / soundfile / numpy.

This module has **no Arrow or VGI dependency** so it is directly unit-testable.
Everything here takes an :class:`AudioInput` (a ``VARCHAR`` path the worker
opens, or a ``BLOB`` of raw audio bytes) and returns plain Python scalars /
lists, or ``None`` when the feature cannot be computed.

Robustness contract (THE core convention -- read first)
------------------------------------------------------
Audio supplied to a SQL worker is **untrusted**. A single malformed, truncated,
absurdly large, or maliciously crafted blob must **never** crash the worker
process. Therefore:

* Every public function is **total**: it catches *all* exceptions from decoding
  / analysis and returns ``None`` (scalars) or an empty result (tables) rather
  than raising. ``None`` input -> ``None`` output.
* Decoded audio is **bounded** -- we refuse to materialise more than
  :data:`MAX_DURATION_SECONDS` of audio (a corrupt header can claim billions of
  frames; we stop before exhausting memory).
* ``librosa`` is imported **once** at module load (its first import is slow) and
  cached for the process lifetime; see :func:`_librosa`.

Format note: WAV/FLAC/OGG decode natively via ``soundfile`` (libsndfile, no
external tools). Compressed formats (mp3, m4a, ...) require ``ffmpeg`` /
``audioread`` at runtime; if neither is present those decodes fail *softly*
(return ``None``) rather than crashing.
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

# ---------------------------------------------------------------------------
# Bounds against hostile input.
# ---------------------------------------------------------------------------

#: Hard cap on how much audio we will decode into memory, in seconds. A corrupt
#: container can advertise an enormous frame count; refusing past this keeps a
#: bad blob from exhausting RAM. ~30 min at any sample rate is plenty for
#: feature extraction; longer inputs are analysed on a truncated prefix.
MAX_DURATION_SECONDS: float = 1800.0

#: Analysis is always done at this sample rate (mono). librosa resamples on
#: load; fixing it keeps spectral features comparable and bounds work per frame.
ANALYSIS_SR: int = 22050

#: Default number of MFCC coefficients.
DEFAULT_N_MFCC: int = 13

#: Hard cap on requested MFCC count (defensive against absurd ``n``).
MAX_N_MFCC: int = 128

# Pitch-class names for the chroma-based key estimate.
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (major / minor), normalised at use.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# Expensive, cached librosa import.
# ---------------------------------------------------------------------------

_librosa_module: Any = None


def _librosa() -> ModuleType:
    """Import :mod:`librosa` once and cache it for the process lifetime.

    librosa's first import is notably slow (it pulls in numba / scipy), so we
    defer and memoise it. Importing eagerly at worker start-up is fine too; this
    just guarantees the cost is paid at most once.
    """
    global _librosa_module
    if _librosa_module is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import librosa  # noqa: PLC0415 - intentionally lazy + cached

            _librosa_module = librosa
    return cast("ModuleType", _librosa_module)


def warm_up() -> None:
    """Eagerly pay the librosa import cost (called once at worker start)."""
    _librosa()


# ---------------------------------------------------------------------------
# Polymorphic input: a VARCHAR path OR a BLOB of bytes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioInput:
    """A path to an audio file, or raw audio bytes -- exactly one is set."""

    path: str | None = None
    data: bytes | None = None

    @classmethod
    def from_path(cls, path: str | None) -> AudioInput | None:
        """Wrap a filesystem path, passing ``None`` straight through."""
        if path is None:
            return None
        return cls(path=path)

    @classmethod
    def from_bytes(cls, data: bytes | None) -> AudioInput | None:
        """Wrap raw audio bytes, passing ``None`` straight through."""
        if data is None:
            return None
        return cls(data=data)

    def _open(self) -> Any:
        """Return something ``soundfile`` / ``librosa`` can read."""
        if self.path is not None:
            return self.path
        return io.BytesIO(self.data or b"")


# ---------------------------------------------------------------------------
# Decoding -- the single choke point where all hostile input is contained.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Raw:
    """Native (pre-resample) view: native sample rate + channel count."""

    sample_rate: int
    channels: int
    frames: int  # may be 0 if unknown


def _probe(audio: AudioInput) -> _Raw | None:
    """Read native sample-rate / channel / frame metadata via soundfile.

    Returns ``None`` for anything soundfile can't open (e.g. mp3 without ffmpeg,
    or garbage bytes). Never raises.
    """
    try:
        import soundfile as sf  # noqa: PLC0415

        with sf.SoundFile(audio._open()) as f:
            return _Raw(sample_rate=int(f.samplerate), channels=int(f.channels), frames=int(len(f)))
    except Exception:
        return None


def _load_mono(audio: AudioInput) -> tuple[np.ndarray, int] | None:
    """Decode to a bounded mono ``float32`` waveform at :data:`ANALYSIS_SR`.

    Returns ``(y, sr)`` or ``None`` on any failure. The signal is truncated to
    :data:`MAX_DURATION_SECONDS` so a corrupt/huge input can't exhaust memory.
    """
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(
                audio._open(),
                sr=ANALYSIS_SR,
                mono=True,
                duration=MAX_DURATION_SECONDS,
            )
    except Exception:
        return None
    if y is None or not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32, copy=False), int(sr)


def _finite(value: float) -> float | None:
    """Coerce a numpy/py float to a finite Python float, else ``None``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Metadata scalars (native sample rate / channels via soundfile probe; duration
# preferring the native probe, falling back to a decode).
# ---------------------------------------------------------------------------


def sample_rate(audio: AudioInput | None) -> int | None:
    """Native sample rate in Hz, or ``None``."""
    if audio is None:
        return None
    raw = _probe(audio)
    if raw is not None:
        return raw.sample_rate
    # Compressed formats soundfile can't probe: fall back to a decode at the
    # native rate via librosa (sr=None preserves the source rate).
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _y, sr = librosa.load(audio._open(), sr=None, mono=True, duration=1.0)
        return int(sr)
    except Exception:
        return None


def channels(audio: AudioInput | None) -> int | None:
    """Number of audio channels (1 = mono, 2 = stereo, ...), or ``None``."""
    if audio is None:
        return None
    raw = _probe(audio)
    if raw is not None:
        return raw.channels
    # librosa down-mixes to mono; if we can decode it at all it has >=1 channel.
    return 1 if _load_mono(audio) is not None else None


def duration(audio: AudioInput | None) -> float | None:
    """Duration in seconds, or ``None``.

    Uses the native frame count when soundfile can probe it (exact, cheap);
    otherwise decodes and measures. Always reflects the *true* duration even
    though analysis elsewhere caps at :data:`MAX_DURATION_SECONDS`.
    """
    if audio is None:
        return None
    raw = _probe(audio)
    if raw is not None and raw.frames > 0 and raw.sample_rate > 0:
        return _finite(raw.frames / raw.sample_rate)
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _finite(librosa.get_duration(path=audio._open()))
    except Exception:
        pass
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    return _finite(len(y) / sr) if sr else None


# ---------------------------------------------------------------------------
# Feature scalars (computed on the bounded mono signal).
# ---------------------------------------------------------------------------


def tempo(audio: AudioInput | None) -> float | None:
    """Estimated tempo in BPM (heuristic), or ``None``."""
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tempo_arr = _tempo_fn(librosa)(y=y, sr=sr)
        return _finite(np.atleast_1d(tempo_arr)[0])
    except Exception:
        return None


def _tempo_fn(librosa: ModuleType) -> Any:
    """Locate librosa's tempo estimator across versions.

    The estimator has moved between releases: ``librosa.feature.rhythm.tempo``
    (newer), ``librosa.feature.tempo``, and ``librosa.beat.tempo`` (older). Pick
    whichever exists.
    """
    rhythm = getattr(librosa.feature, "rhythm", None)
    if rhythm is not None and hasattr(rhythm, "tempo"):
        return rhythm.tempo
    if hasattr(librosa.feature, "tempo"):
        return librosa.feature.tempo
    return librosa.beat.tempo


def rms_energy(audio: AudioInput | None) -> float | None:
    """Mean root-mean-square energy of the signal, or ``None``."""
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, _sr = loaded
    librosa = _librosa()
    try:
        rms = librosa.feature.rms(y=y)
        return _finite(np.mean(rms))
    except Exception:
        return None


def zero_crossing_rate(audio: AudioInput | None) -> float | None:
    """Mean zero-crossing rate (fraction of sign changes), or ``None``."""
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, _sr = loaded
    librosa = _librosa()
    try:
        zcr = librosa.feature.zero_crossing_rate(y=y)
        return _finite(np.mean(zcr))
    except Exception:
        return None


def spectral_centroid(audio: AudioInput | None) -> float | None:
    """Mean spectral centroid in Hz (the spectrum's centre of mass), or ``None``."""
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        return _finite(np.mean(sc))
    except Exception:
        return None


def spectral_bandwidth(audio: AudioInput | None) -> float | None:
    """Mean spectral bandwidth in Hz, or ``None``."""
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return _finite(np.mean(sb))
    except Exception:
        return None


def mfcc(audio: AudioInput | None, n_mfcc: int = DEFAULT_N_MFCC) -> list[float] | None:
    """Mean of each of ``n_mfcc`` MFCC coefficients, as a list, or ``None``.

    ``n_mfcc`` is clamped to ``[1, MAX_N_MFCC]`` defensively.
    """
    if audio is None:
        return None
    try:
        n = int(n_mfcc)
    except (TypeError, ValueError):
        n = DEFAULT_N_MFCC
    n = max(1, min(MAX_N_MFCC, n))
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n)
        means = np.mean(coeffs, axis=1)
        out = [_finite(v) for v in means.tolist()]
        if any(v is None for v in out):
            return None
        return [v for v in out if v is not None]
    except Exception:
        return None


def estimated_key(audio: AudioInput | None) -> str | None:
    """Heuristic chroma-based musical key estimate, e.g. ``'C major'``.

    Uses a Krumhansl-Schmuckler correlation against major/minor key profiles.
    This is an *approximation* -- treat it as a hint, not ground truth. Returns
    ``None`` if the signal can't be decoded or analysed.
    """
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile = np.mean(chroma, axis=1)
        if not np.all(np.isfinite(profile)) or float(np.sum(profile)) <= 0.0:
            return None
        profile = profile / np.sum(profile)
        maj = _MAJOR_PROFILE / np.sum(_MAJOR_PROFILE)
        minr = _MINOR_PROFILE / np.sum(_MINOR_PROFILE)
        best_score = -np.inf
        best_key: str | None = None
        for shift in range(12):
            rolled = np.roll(profile, -shift)
            maj_score = float(np.corrcoef(rolled, maj)[0, 1])
            min_score = float(np.corrcoef(rolled, minr)[0, 1])
            if math.isfinite(maj_score) and maj_score > best_score:
                best_score, best_key = maj_score, f"{_PITCH_CLASSES[shift]} major"
            if math.isfinite(min_score) and min_score > best_score:
                best_score, best_key = min_score, f"{_PITCH_CLASSES[shift]} minor"
        return best_key
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table-function helpers.
# ---------------------------------------------------------------------------


def beat_times(audio: AudioInput | None) -> list[float] | None:
    """Beat onset times in seconds (ascending), or ``None`` on failure.

    ``None`` signals "couldn't analyse" (worker surfaces no rows); an empty list
    means "analysed fine, but no beats detected".
    """
    if audio is None:
        return None
    loaded = _load_mono(audio)
    if loaded is None:
        return None
    y, sr = loaded
    librosa = _librosa()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            times = librosa.frames_to_time(beat_frames, sr=sr)
        out: list[float] = []
        for t in np.atleast_1d(times).tolist():
            v = _finite(t)
            if v is not None:
                out.append(v)
        return out
    except Exception:
        return None


def audio_info(audio: AudioInput | None) -> tuple[float, int, int] | None:
    """``(duration, sample_rate, channels)`` in one shot, or ``None``."""
    if audio is None:
        return None
    d = duration(audio)
    sr = sample_rate(audio)
    ch = channels(audio)
    if d is None or sr is None or ch is None:
        return None
    return d, sr, ch

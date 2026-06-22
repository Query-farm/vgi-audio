"""Deterministic synthetic audio generators for the test suite.

All test audio is generated in-process from numpy so tests are hermetic and
reproducible (no committed nondeterministic blobs). WAV bytes are produced via
``soundfile`` into an in-memory buffer.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def sine_wav_bytes(
    *,
    freq: float = 440.0,
    sr: int = 22050,
    seconds: float = 2.0,
    channels: int = 1,
    amplitude: float = 0.5,
) -> bytes:
    """A pure sine tone at ``freq`` Hz as in-memory WAV bytes."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    mono = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    data = mono if channels == 1 else np.column_stack([mono] * channels)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def click_track_bytes(*, bpm: float = 120.0, sr: int = 22050, seconds: float = 6.0) -> bytes:
    """A metronome click track at ``bpm`` as in-memory WAV bytes.

    Each beat is a short decaying burst of broadband noise -- librosa's beat
    tracker locks onto these onsets, so ``tempo`` recovers ``bpm`` closely.
    """
    n = int(sr * seconds)
    y = np.zeros(n, dtype=np.float32)
    interval = sr * 60.0 / bpm
    click_len = int(sr * 0.02)  # 20 ms click
    rng = np.random.default_rng(0)
    env = np.exp(-np.linspace(0.0, 8.0, click_len)).astype(np.float32)
    burst = (rng.standard_normal(click_len).astype(np.float32) * env) * 0.8
    pos = 0.0
    while int(pos) + click_len < n:
        start = int(pos)
        y[start : start + click_len] += burst
        pos += interval
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_wav_file(path: str, data: bytes) -> None:
    """Write raw WAV ``data`` to ``path``."""
    with open(path, "wb") as f:
        f.write(data)

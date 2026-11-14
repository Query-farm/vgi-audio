"""Generate controlled WAV inputs and present SQL results; never extract features."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Markdown, display

COLORS = {"Reference": "#397D68", "Half amplitude": "#6390B4", "Harmonic mix": "#CA843E"}
SOURCE = "Source: generated PCM WAVs · vgi-audio v0.4.0 / librosa 1.0.0 · measured through Haybarn / VGI"


def generate_inputs(directory: Path, harmonic_ratio: float) -> None:
    """Write known synthetic signals, keeping feature extraction in the worker."""
    if isinstance(harmonic_ratio, bool) or not isinstance(harmonic_ratio, (int, float)):
        raise ValueError("HARMONIC_RATIO must be a number from 0 to 1")
    if not math.isfinite(harmonic_ratio) or not 0 <= harmonic_ratio <= 1:
        raise ValueError("HARMONIC_RATIO must be finite and between 0 and 1")
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    time = np.arange(sample_rate * 3, dtype=np.float64) / sample_rate
    clips = [("reference", 1.0, 0.0), ("quiet", 0.5, 0.0), ("bright", 1.0, harmonic_ratio)]
    clips.extend((f"sweep_{index}", 1.0, harmonic_ratio * index / 4) for index in range(5))
    for name, gain, ratio in clips:
        # This defines the experiment, not a measured RMS or spectral feature.
        # Distinct integer-cycle sinusoids are orthogonal over the 3-second clip.
        signal = (
            0.4
            * gain
            / np.sqrt(1 + ratio**2)
            * (np.sin(2 * np.pi * 440 * time) + ratio * np.sin(2 * np.pi * 1320 * time))
        )
        pcm = np.rint(signal * (2**31 - 1)).astype("<i4")
        with wave.open(str(directory / f"{name}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(4)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())


def prose(text: str) -> None:
    """Display result-derived Markdown with folded implementation."""
    display(Markdown(text))


def theme() -> None:
    """Apply consistent plot styling without changing analytical values."""
    sns.set_theme(
        style="whitegrid",
        palette=list(COLORS.values()),
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str) -> None:
    """Attach context and provenance to a readable chart."""
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.08, 0.91, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.10, 1, 0.83))
    fig.text(0.08, 0.03, SOURCE, fontsize=8, color="#6C7D73")
    plt.show()

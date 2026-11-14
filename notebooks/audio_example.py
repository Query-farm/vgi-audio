"""Author controlled audio inputs and present SQL measurements; never extract features."""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vgi_notebooks import finish

COLORS = {"Reference": "#397D68", "Half amplitude": "#6390B4", "Harmonic mix": "#CA843E"}
SOURCE = "Source: authored PCM WAVs · VGI Audio v0.4.0 · features measured through Haybarn / VGI"


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


def level_comparison(comparison, HARMONIC_RATIO):
    """Compare measured amplitude and frequency balance on separate scales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    colors = [COLORS[label] for label in comparison.label]
    for ax, column, xlabel in zip(
        axes, ["rms", "centroid_hz"], ["Mean frame RMS (digital amplitude)", "Mean spectral centroid (Hz)"], strict=True
    ):
        values = comparison[column].astype(float)
        bars = ax.barh(comparison.label, values, color=colors, height=0.55)
        labels = [f"{v:.4f}" if column == "rms" else f"{v:.1f} Hz" for v in values]
        ax.bar_label(bars, labels=labels, padding=5, fontsize=10)
        ax.set_xlim(0, max(values) * 1.3)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    finish(
        fig,
        "Similar RMS does not imply similar spectral balance",
        f"Generated 3 s mono clips · third-harmonic amplitude ratio {HARMONIC_RATIO:g} · each panel has its own units",
        SOURCE,
        name="01-level-versus-balance",
    )


def harmonic_sweep(sweep, HARMONIC_RATIO):
    """Show measured and designed relationships across harmonic weights."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(
        sweep.harmonic_ratio, sweep.centroid_hz, "o-", color=COLORS["Harmonic mix"], label="Worker measurement"
    )
    axes[0].plot(sweep.harmonic_ratio, sweep.ideal_centroid_hz, "--", color="#59716A", label="Ideal two-tone model")
    axes[0].set_ylabel("Spectral centroid (Hz)")
    axes[0].legend(fontsize=9)
    axes[1].plot(sweep.harmonic_ratio, sweep.relative_level_db, "o-", color=COLORS["Reference"])
    axes[1].axhline(0, color="#9BA9A4", linestyle="--", linewidth=1)
    axes[1].set_ylabel("RMS level relative to reference (dB)")
    axes[1].set_ylim(-0.15, 0.15)
    for ax in axes:
        ax.set_xlabel("Third-harmonic / fundamental amplitude ratio")
        ax.set_xlim(-0.03, max(0.1, HARMONIC_RATIO) + 0.03)
    finish(
        fig,
        "More upper-harmonic weight moves the centroid",
        "Left: measured and ideal frequency balance · right: zoomed ±0.15 dB scale around the fixed RMS design",
        SOURCE,
        name="02-harmonic-sweep",
    )


def validate_window(end):
    """Validate a reporting window without changing or trimming audio."""
    if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or not 0 < end <= 2:
        raise ValueError("WINDOW_END_S must be a finite number greater than zero and at most two")


def write_builtin_inputs(frame):
    """Preserve actual catalog audio bytes for literal-path table arguments."""
    if set(frame.name) != {"click", "tone"} or len(frame) != 2:
        raise ValueError("Expected the two built-in example clips")
    directory = Path("artifacts/sample_audio")
    directory.mkdir(parents=True, exist_ok=True)
    for row in frame.itertuples():
        raw = bytes(row.audio)
        if hashlib.sha256(raw).hexdigest() != row.sha256:
            raise ValueError("Built-in audio differs from its SQL SHA256")
        path = directory / (row.name + ".bin")
        path.write_bytes(raw)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError("Written audio differs from its source bytes")


def event_timeline(population, beats, end):
    """Show retained and excluded event positions beside decoded clip extents."""
    fig, ax = plt.subplots(figsize=(12, 6.9))
    for y, row in enumerate(population.itertuples()):
        if pd.isna(row.duration_s):
            ax.text(0.04, y, "No decoded metadata", va="center", color="#6c7971")
            continue
        ax.plot([0, row.duration_s], [y, y], color="#e0e7e2", linewidth=12, solid_capstyle="butt")
        events = beats.loc[beats.name == row.name]
        if events.empty:
            ax.text(row.duration_s + 0.07, y, "No detected beats", va="center", color="#6c7971")
        for event in events.itertuples():
            ax.scatter(event.time, y, s=95, color="#397d68" if event.selected else "#aab5ad", zorder=3)
            ax.text(event.time, y - 0.16, f"{event.time:.3f} s", ha="center", fontsize=10)
    ax.axvline(end, color="#c3823e", linestyle="--", linewidth=1.4)
    ax.set(xlim=(-0.04, 2.12), ylim=(len(population) - 0.5, -0.6), xlabel="Time from clip start (seconds)")
    ax.set_yticks(range(len(population)), population.name)
    ax.scatter([], [], color="#397d68", label="Inside reporting window")
    ax.scatter([], [], color="#aab5ad", label="Outside reporting window")
    ax.plot([], [], color="#c3823e", linestyle="--", label="Excluded upper endpoint")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.80), ncol=3, fontsize=10)
    finish(
        fig,
        "A reporting window selects events, not audio samples",
        f"Selected times satisfy 0 ≤ time < {end:g} s · gray bars show complete decoded clip duration",
        "Source: VGI Audio v0.4.0 built-in synthetic clips; invalid bytes authored in SQL.\n"
        "Full worker beat lists are materialized once; no clip cropping or re-estimation occurs.",
        name="03-detected-beat-window",
    )


def tempo_comparison(population, end):
    """Contrast full-clip tempo with interval-derived rates and unavailable estimates."""
    fig, ax = plt.subplots(figsize=(12, 6.9))
    for y, row in enumerate(population.itertuples()):
        if pd.isna(row.tempo_bpm):
            ax.text(5, y, "No decoded tempo or beat-grid estimate", va="center", color="#6c7971")
            continue
        ax.scatter(row.tempo_bpm, y - 0.12, color="#527ab5", s=95, marker="o")
        ax.text(row.tempo_bpm + 4, y - 0.12, f"{row.tempo_bpm:.2f}", va="center", fontsize=10)
        if pd.notna(row.grid_bpm):
            ax.scatter(row.grid_bpm, y + 0.12, color="#c3823e", s=95, marker="D")
            ax.text(row.grid_bpm + 4, y + 0.12, f"{row.grid_bpm:.2f} ({int(row.gaps)} gaps)", va="center", fontsize=10)
        else:
            ax.text(
                5,
                y + 0.17,
                f"No interval estimate: {int(row.selected_beats)} selected beats",
                va="center",
                color="#6c7971",
                fontsize=10,
            )
    ax.scatter([], [], color="#527ab5", marker="o", label="Whole-clip tempo()")
    ax.scatter([], [], color="#c3823e", marker="D", label="60 / median selected beat gap")
    ax.set(xlim=(0, 205), ylim=(len(population) - 0.5, -0.6), xlabel="Beats per minute (BPM) · different estimators")
    ax.set_yticks(range(len(population)), population.name)
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.80), ncol=2)
    finish(
        fig,
        "A numeric tempo does not imply a detected beat list",
        f"Beat-gap estimate uses events before {end:g} s; tempo() still analyzes each complete clip",
        "Source: actual VGI Audio v0.4.0 / librosa outputs for two built-in synthetic clips.\n"
        "One or two observed intervals are a small sample, not an accuracy or tempo-confidence assessment.",
        name="04-tempo-versus-event-gaps",
    )

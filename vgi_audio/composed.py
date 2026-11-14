"""Composed example queries: realistic SQL that combines several worker objects.

Each constant is a ``(description, sql)`` pair defined **once** here and referenced
from every place it is published -- the schema-level ``vgi.example_queries`` list
(``worker.py``) and the example list of the object that drives the query
(``tables.py`` / ``scalars.py``), where the vgi-lint agent simulator shows it.

Every query is offline and deterministic: it reads the built-in ``example_clips``
discovery table or an inline ``from_base64(...)`` blob, never a file on disk. A VGI
table function only binds literal arguments (no column, LATERAL, or subquery
argument), so a query that feeds ``beats`` embeds the click clip inline.
"""

from __future__ import annotations

from ._example_audio import click_blob_sql

# beats (inline click) -> inter-beat gaps -> beat-grid BPM, cross-checked against
# the tempo() scalar applied to the same clip from example_clips. Live result:
# grid_bpm 120.2 (median of 0.511 s / 0.488 s gaps) vs tracker_bpm 117.5.
BEAT_GRID_VS_TEMPO: tuple[str, str] = (
    "Derive a BPM from the median gap between consecutive beat onsets of the built-in 120 BPM "
    "click track and set it beside the tempo() estimate for the same clip.",
    "WITH grid AS ("
    "SELECT time - lag(time) OVER (ORDER BY seq) AS gap "
    f"FROM audio.main.beats({click_blob_sql()})"
    ") "
    "SELECT round(60.0 / median(gap), 1) AS grid_bpm, "
    "(SELECT round(audio.main.tempo(audio), 1) FROM audio.main.example_clips WHERE name = 'click') "
    "AS tracker_bpm "
    "FROM grid",
)

# example_clips -> rms_energy gate -> spectral_centroid + estimated_key on what
# survives. Live result: one row, tone (rms 0.411, centroid 73.2 Hz, 'B minor');
# the click track (rms ~0.024) is gated out.
LOUD_CLIPS_PROFILE: tuple[str, str] = (
    "Skip quiet clips (mean RMS energy at or below 0.1), then report brightness and estimated key "
    "for the built-in clips that remain.",
    "SELECT name, round(audio.main.rms_energy(audio), 3) AS rms, "
    "round(audio.main.spectral_centroid(audio), 1) AS centroid_hz, "
    "audio.main.estimated_key(audio) AS musical_key "
    "FROM audio.main.example_clips "
    "WHERE audio.main.rms_energy(audio) > 0.1 "
    "ORDER BY name",
)

# example_clips self-join -> mfcc on both sides -> Euclidean distance over
# MFCC-1..12 (MFCC-0 is loudness, which the mfcc docs say to drop for timbre).
MFCC_TIMBRE_DISTANCE: tuple[str, str] = (
    "Timbre distance between each pair of built-in clips: Euclidean distance between their "
    "default MFCC means with MFCC-0 (overall loudness) dropped; smaller means more alike.",
    "SELECT a.name AS clip, b.name AS other_clip, "
    "round(list_distance(audio.main.mfcc(a.audio)[2:], audio.main.mfcc(b.audio)[2:]), 1) AS timbre_distance "
    "FROM audio.main.example_clips AS a "
    "JOIN audio.main.example_clips AS b ON a.name < b.name "
    "ORDER BY timbre_distance, clip, other_clip",
)

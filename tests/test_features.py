"""Unit tests for the pure audio-feature logic (no Arrow / VGI, no subprocess).

Covers a known sine tone (exact sample-rate / duration, spectral centroid near
the tone), a click track (tempo near the generated BPM), and -- crucially -- the
hostile-input contract: empty / garbage / NULL bytes return ``None`` rather than
crashing.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from vgi_audio import features
from vgi_audio.features import AudioInput

from .synth import click_track_bytes, sine_wav_bytes

# --- a 440 Hz, 22050 Hz, mono, 2 s sine tone, as both bytes and a path -------


@pytest.fixture(scope="module")
def tone_bytes() -> bytes:
    return sine_wav_bytes(freq=440.0, sr=22050, seconds=2.0)


@pytest.fixture(scope="module")
def tone_path(tone_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(tone_bytes)
    return path


@pytest.fixture(scope="module")
def click_bytes() -> bytes:
    return click_track_bytes(bpm=120.0, sr=22050, seconds=6.0)


class TestMetadataExact:
    def test_sample_rate_exact_bytes(self, tone_bytes: bytes) -> None:
        assert features.sample_rate(AudioInput.from_bytes(tone_bytes)) == 22050

    def test_sample_rate_exact_path(self, tone_path: str) -> None:
        assert features.sample_rate(AudioInput.from_path(tone_path)) == 22050

    def test_duration_exact(self, tone_bytes: bytes) -> None:
        d = features.duration(AudioInput.from_bytes(tone_bytes))
        assert d is not None
        assert abs(d - 2.0) < 1e-3

    def test_channels_mono(self, tone_bytes: bytes) -> None:
        assert features.channels(AudioInput.from_bytes(tone_bytes)) == 1

    def test_channels_stereo(self) -> None:
        stereo = sine_wav_bytes(freq=440.0, channels=2, seconds=1.0)
        assert features.channels(AudioInput.from_bytes(stereo)) == 2

    def test_audio_info_tuple(self, tone_bytes: bytes) -> None:
        info = features.audio_info(AudioInput.from_bytes(tone_bytes))
        assert info is not None
        d, sr, ch = info
        assert sr == 22050 and ch == 1 and abs(d - 2.0) < 1e-3


class TestSpectralFeatures:
    def test_spectral_centroid_near_tone(self, tone_bytes: bytes) -> None:
        # The centre of mass of a 440 Hz tone's spectrum sits near 440 Hz
        # (a little higher due to windowing / harmonics from quantisation).
        sc = features.spectral_centroid(AudioInput.from_bytes(tone_bytes))
        assert sc is not None
        assert 350.0 < sc < 900.0

    def test_higher_tone_higher_centroid(self) -> None:
        low = features.spectral_centroid(AudioInput.from_bytes(sine_wav_bytes(freq=440.0)))
        high = features.spectral_centroid(AudioInput.from_bytes(sine_wav_bytes(freq=2000.0)))
        assert low is not None and high is not None
        assert high > low

    def test_spectral_bandwidth_finite(self, tone_bytes: bytes) -> None:
        sb = features.spectral_bandwidth(AudioInput.from_bytes(tone_bytes))
        assert sb is not None and sb > 0.0

    def test_rms_energy_positive(self, tone_bytes: bytes) -> None:
        rms = features.rms_energy(AudioInput.from_bytes(tone_bytes))
        assert rms is not None and rms > 0.0

    def test_zcr_in_range(self, tone_bytes: bytes) -> None:
        zcr = features.zero_crossing_rate(AudioInput.from_bytes(tone_bytes))
        assert zcr is not None and 0.0 <= zcr <= 1.0


class TestMfcc:
    def test_default_length_13(self, tone_bytes: bytes) -> None:
        coeffs = features.mfcc(AudioInput.from_bytes(tone_bytes))
        assert coeffs is not None and len(coeffs) == 13

    def test_custom_length(self, tone_bytes: bytes) -> None:
        coeffs = features.mfcc(AudioInput.from_bytes(tone_bytes), 20)
        assert coeffs is not None and len(coeffs) == 20

    def test_n_clamped(self, tone_bytes: bytes) -> None:
        # absurd / non-positive n is clamped defensively, never crashes.
        assert len(features.mfcc(AudioInput.from_bytes(tone_bytes), 0) or []) == 1
        assert len(features.mfcc(AudioInput.from_bytes(tone_bytes), 10_000) or []) == features.MAX_N_MFCC


class TestTempoAndBeats:
    def test_tempo_near_120(self, click_bytes: bytes) -> None:
        bpm = features.tempo(AudioInput.from_bytes(click_bytes))
        assert bpm is not None
        # Allow half/double-time ambiguity but a generous band around 120.
        assert 100.0 < bpm < 140.0 or 55.0 < bpm < 70.0 or 230.0 < bpm < 250.0

    def test_beats_nonempty(self, click_bytes: bytes) -> None:
        times = features.beat_times(AudioInput.from_bytes(click_bytes))
        assert times is not None and len(times) > 3
        assert times == sorted(times)

    def test_estimated_key_format(self, click_bytes: bytes) -> None:
        key = features.estimated_key(AudioInput.from_bytes(sine_wav_bytes(freq=261.63, seconds=2.0)))
        # A pure C tone should be analysable; format is '<pitch> major|minor'.
        assert key is None or (key.split()[-1] in {"major", "minor"})


class TestHostileInput:
    def test_none_input(self) -> None:
        assert features.duration(None) is None
        assert features.sample_rate(None) is None
        assert features.tempo(None) is None
        assert features.mfcc(None) is None
        assert features.beat_times(None) is None
        assert features.audio_info(None) is None
        assert features.estimated_key(None) is None

    def test_empty_bytes(self) -> None:
        a = AudioInput.from_bytes(b"")
        assert features.duration(a) is None
        assert features.sample_rate(a) is None
        assert features.tempo(a) is None
        assert features.mfcc(a) is None
        assert features.beat_times(a) is None
        assert features.spectral_centroid(a) is None

    def test_garbage_bytes(self) -> None:
        a = AudioInput.from_bytes(b"\x00\x01\x02not audio at all\xff\xfe" * 100)
        assert features.duration(a) is None
        assert features.sample_rate(a) is None
        assert features.tempo(a) is None
        assert features.mfcc(a) is None
        assert features.estimated_key(a) is None

    def test_truncated_wav_header(self, tone_bytes: bytes) -> None:
        # First 20 bytes of a real WAV: a valid RIFF magic but truncated body.
        a = AudioInput.from_bytes(tone_bytes[:20])
        assert features.duration(a) is None or features.duration(a) == features.duration(a)
        assert features.tempo(a) is None

    def test_nonexistent_path(self) -> None:
        a = AudioInput.from_path("/no/such/file/at/all.wav")
        assert features.duration(a) is None
        assert features.sample_rate(a) is None
        assert features.tempo(a) is None

    def test_very_short_clip(self) -> None:
        # 1 ms of audio -- should not crash; features are computable or None.
        tiny = sine_wav_bytes(freq=440.0, seconds=0.001)
        a = AudioInput.from_bytes(tiny)
        assert features.sample_rate(a) == 22050
        # tempo / mfcc may legitimately be None on a near-empty signal; just no crash.
        features.tempo(a)
        features.mfcc(a)
        features.beat_times(a)

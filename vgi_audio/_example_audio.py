"""Self-contained synthetic audio blobs for runnable SQL examples.

Every audio function carries metadata examples (the per-function ``examples``
and the catalog/schema ``vgi.executable_examples`` / ``vgi.example_queries``)
that the ``vgi-lint`` strict profile **executes** against the live worker and
expects to return rows. Audio examples normally need a file on disk, which a
linter running in a separate process cannot guarantee exists. To stay hermetic,
we embed two tiny audio clips as base64 and feed them to the worker inline via
DuckDB's ``from_base64(...)`` (which yields a ``BLOB``), exercising the ``BLOB``
overload of each feature with no filesystem dependency.

The clips use a **very low native sample rate** so the byte count -- and thus the
base64 -- stays tiny; ``features._load_mono`` resamples every input to 22.05 kHz
for analysis regardless, so the recovered features are still meaningful. Keeping
the tone in particular *small and copy-pastable* matters for the agent-suitability
suite (``vgi.agent_test_tasks`` / ``vgi-lint simulate``): an analyst discovering
the worker must be able to read a function's documented example in full and re-run
it, which a multi-kilobyte blob (truncated in the analyst's tool output) defeats.

* :data:`TONE_WAV_B64` -- a 0.5 s 60 Hz mono sine at 200 Hz native (~0.3 KB of
  base64). Decodes to a real waveform, so the metadata / signal / spectral /
  harmony scalars all return a value (``duration`` = 0.5, a finite
  ``spectral_centroid``, an ``mfcc`` list, an ``estimated_key``, ...).
* :data:`CLICK_FLAC_B64` -- a 2.0 s 120 BPM metronome click track at 6 kHz native,
  with clear onsets so the rhythm functions (``tempo``, ``beats``) recover real
  values and ``beats`` emits multiple rows. Beat tracking needs enough temporal
  resolution, so this clip is necessarily larger than the tone; it is FLAC (the
  mostly-silent track compresses ~10x vs WAV, ~3 KB of base64) so every example
  that embeds it fits the 4,000-character agent-context bound (VGI417).

Regenerate with ``python -m vgi_audio._example_audio`` (prints whether the
hard-coded literals still match the deterministic synthesis below).
"""

from __future__ import annotations

#: Base64 of a 0.5 s 60 Hz mono sine WAV (200 Hz native, PCM-16). Small enough to
#: be read/copied whole by a discovering agent; decodes to a real signal so every
#: metadata / spectral / harmony feature returns a value.
TONE_WAV_B64 = (
    "UklGRuwAAABXQVZFZm10IBAAAAABAAEAyAAAAJABAAACABAAZGF0YcgAAAAAAApJ29Lb0gpJAAD1tiQtJC31tgAACknb0tvSCkkA"
    "APW2JC0kLfW2AAAKSdvS29IKSQAA9bYkLSQt9bYAAApJ29Lb0gpJAAD1tiQtJC31tgAACknb0tvSCkkAAPW2JC0kLfW2AAAKSdvS"
    "29IKSQAA9bYkLSQt9bYAAApJ29Lb0gpJAAD1tiQtJC31tgAACknb0tvSCkkAAPW2JC0kLfW2AAAKSdvS29IKSQAA9bYkLSQt9bYA"
    "AApJ29Lb0gpJAAD1tiQtJC31tg=="
)


#: Base64 of a 2.0 s 120 BPM mono click-track FLAC (6 kHz native, PCM-16). Clear
#: onsets so ``tempo`` / ``beats`` recover real values and ``beats`` emits rows.
CLICK_FLAC_B64 = (
    "ZkxhQwAAACIQABAAAAKoAALrAXcA8AAALuA9uCNM9TPcQHVO83YE6InghAAAKCAAAAByZWZlcmVuY2UgbGliRkxBQyAxLjQuMyAy"
    "MDIzMDYyMwAAAAD/+MwIAAa+TgAAAFABQQAA+wz4VAAADrO0FUVVMVT+nmbVhqfRioQKh8i0Gi3RUrQoQS6iRrYQKSMjszgeWPhk"
    "qeDB4iMnUhYpZ93snmmNlTMRQAAAAAC4AAEAAABQAWALDMIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEA///"
    "//////////////////w/////////////////////w/////////////////////w/////////////////////w///////////////"
    "//////w/////////////////////w/////////////////////w/////////////////////w/////////////////////w/////"
    "////////////////w/////////////////////w/////////////////////w/////////////////////w/////////////////"
    "////w/////////////////////w/////////////////////w/////////////////////w/////////////////////w///////"
    "//////////////w/////////////////////w/////////////////////w/////////////////////1ggggggggggggggggggg"
    "ggggggggggggggggggggggggggggggggggggggBAB0AAAAYn4OQ6ChAlL0PRUmtapZEiDK3FDLMwSRKXsawu2XlKVcULkLlbk2ST"
    "f+2xfdcVpVMLQAACcAQABoGwTOYIIIIA/////////////////////w/////////////////////w/////////////////////w//"
    "///////////////////w/////////////////////w/////////////////////w/////////////////////w//////////////"
    "///////wkov/+MwIAQarTgAAAAAAAAAAAAAAAAAAAAC0FUVVMVT+nmbVhqfRiB//////////////////////////////////////"
    "///h//////////////////////////////////////////4f/////////////////////////////////////////+H/////////"
    "/////////////////////////////////h//////////////////////////////////////////4f//////////////////////"
    "///////////////////+H//////////////////////////////////////////pCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhC"
    "EIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQACAADQAAAAAAAAIg+ADIA0AYIFQ+RaDRboqVoUIJdR"
    "I1sIFJGR2ZwPLHwyVPBg8RGTqQsUs+72TzTGypmIoAAAAABcAACAAAAoALAFhmEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQ"
    "hCEIQhCEIQhCEIQhCEIQhCEIQgH//////////////////////////////////////////h//////////////////////////////"
    "////////////4f/////////////////////////////////////////+H//////////////////////////////////////////h"
    "//////////////////////////////////////////4f/////////////////////////////////////////+H/////////////"
    "/////////////////////////////h//////////////////////////////////////////4NL0//h8CAIO3wbGTgAAAAAAAAAA"
    "AAAAAAAAAAC0FU1UMXT7HqbShrfRiB//////////////////////////////////////h///////////////////////////////"
    "////////4f//////////////////////////////////////+kIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCE"
    "IQhCEIQhCEIQhCEIQhCEIQhCEIQhAAIAANAAAAAAAAAmDoAKgBIAhgRGbGDXW9pJZZwQKMnyZ4KOGwR6JAl4ZH8jghQ+vghpWvnJ"
    "m1VAg4TC7IwAAAAAC4AAGgAADwAkAWGYQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCEIQhCAf///"
    "///////////////////////////////////+H///////////////////////////////////////h///////////////////////"
    "////////////////4f//////////////////////////////////////+H///////////////////////////////////////h//"
    "/////////////////////////////////////4f//////////////////////////////////////+H/////////////////////"
    "//////////////////h///////////////////////////////////////4f//////////////////////////////////////+H"
    "///////////////////////////////////////h///////////////////////////////////////4LoY="
)


def _audio_b64(samples: object, sample_rate: int, fmt: str = "WAV") -> str:
    """Encode a float32 mono waveform as a base64 PCM-16 audio file string.

    Args:
        samples: A 1-D ``numpy`` float32 array of mono samples in ``[-1, 1]``.
        sample_rate: The native sample rate to stamp into the file header.
        fmt: The ``soundfile`` container format (``"WAV"`` or ``"FLAC"``).

    Returns:
        The base64 of the encoded PCM-16 mono file.
    """
    import base64
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sample_rate, format=fmt, subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _synth_tone() -> str:
    """Synthesize :data:`TONE_WAV_B64` -- a 0.5 s 60 Hz sine at 200 Hz native.

    Returns:
        The base64 of the tone clip (matches :data:`TONE_WAV_B64`).
    """
    import numpy as np

    sr = 200
    t = np.arange(int(sr * 0.5)) / sr
    return _audio_b64(0.6 * np.sin(2 * np.pi * 60.0 * t), sr)


def _synth_click() -> str:
    """Synthesize :data:`CLICK_FLAC_B64` -- a 2.0 s 120 BPM click track at 6 kHz.

    Returns:
        The base64 of the click clip (matches :data:`CLICK_FLAC_B64`).
    """
    import numpy as np

    sr = 6000
    duration = 2.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    n = int(0.010 * sr)
    click = np.sin(2 * np.pi * 1000.0 * np.arange(n) / sr) * np.hanning(n)
    k = 0
    while True:
        idx = int(k * 0.5 * sr)
        k += 1
        if idx >= len(y):
            break
        seg = min(n, len(y) - idx)
        y[idx : idx + seg] += click[:seg]
    return _audio_b64(y, sr, "FLAC")


def tone_blob_sql() -> str:
    """SQL expression decoding :data:`TONE_WAV_B64` to a ``BLOB``.

    Returns:
        A ``from_base64('...')`` expression yielding the tone clip as a ``BLOB``.
    """
    return f"from_base64('{TONE_WAV_B64}')"


def click_blob_sql() -> str:
    """SQL expression decoding :data:`CLICK_FLAC_B64` to a ``BLOB``.

    Returns:
        A ``from_base64('...')`` expression yielding the click clip as a
        ``BLOB``.
    """
    return f"from_base64('{CLICK_FLAC_B64}')"


if __name__ == "__main__":  # pragma: no cover - regeneration helper
    print("TONE_WAV_B64 matches synthesis:", _synth_tone() == TONE_WAV_B64)
    print("CLICK_FLAC_B64 matches synthesis:", _synth_click() == CLICK_FLAC_B64)

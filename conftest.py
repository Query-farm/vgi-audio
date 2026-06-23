"""Put the repo root on ``sys.path`` so tests can import the worker modules.

The mere presence of this file at the repo root makes pytest add the root to
``sys.path``, letting tests ``import audio_worker`` and ``import vgi_audio``.
"""

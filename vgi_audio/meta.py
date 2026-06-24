"""Shared helpers for the per-object discovery/description metadata.

The ``vgi-lint`` strict profile expects these tags on **every** function and
table:

* ``vgi.title`` (VGI124) -- human-friendly display name (must differ from the
  machine name, VGI125).
* ``vgi.doc_llm`` (VGI112) -- a Markdown narrative aimed at LLM/agent
  consumers (what it does, when to use it, inputs/outputs, edge cases).
* ``vgi.doc_md`` (VGI113) -- a Markdown narrative for human docs
  (overview + usage + notes).
* ``vgi.keywords`` (VGI126) -- comma-separated search terms / synonyms.
* ``vgi.source_url`` (VGI128) -- link to the file that implements the object.

:func:`source_url` builds the canonical GitHub blob URL and :func:`object_tags`
assembles the five-tag dict each function/table merges into its ``Meta.tags``.
"""

from __future__ import annotations

#: Base GitHub blob URL for source files in this repo (pinned to ``main``).
SOURCE_BASE = "https://github.com/Query-farm/vgi-audio/blob/main"


def source_url(relative_path: str) -> str:
    """Build the ``vgi.source_url`` for a repo-relative source file.

    Args:
        relative_path: Path of the implementing file relative to the repo root,
            e.g. ``"vgi_audio/scalars.py"``.

    Returns:
        The canonical GitHub blob URL for that file on ``main``.
    """
    return f"{SOURCE_BASE}/{relative_path}"


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery/description tags.

    Args:
        title: Human-friendly display name (``vgi.title``); must differ from the
            object's machine name.
        doc_llm: Markdown narrative for LLM/agent audiences.
        doc_md: Markdown narrative for human documentation.
        keywords: Comma-separated search terms / synonyms.
        relative_path: Implementing file relative to the repo root.

    Returns:
        A tag dict ready to merge into a function/table ``Meta.tags``.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }

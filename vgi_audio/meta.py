"""Shared helpers for the per-object discovery/description metadata.

The ``vgi-lint`` strict profile expects these tags on **every** function and
table:

* ``vgi.title`` (VGI124) -- human-friendly display name (must differ from the
  machine name, VGI125).
* ``vgi.doc_llm`` (VGI112) -- a Markdown narrative aimed at LLM/agent
  consumers (what it does, when to use it, inputs/outputs, edge cases).
* ``vgi.doc_md`` (VGI113) -- a Markdown narrative for human docs
  (overview + usage + notes).
* ``vgi.keywords`` (VGI126/VGI138) -- search terms / synonyms, serialised as a
  **JSON array of strings** (e.g. ``["audio","tempo"]``), not a comma-separated
  string.

``vgi.source_url`` is intentionally **not** set per object (VGI139): the
catalog-level ``source_url`` is the single source-of-truth link, so duplicating
it on every function/table is redundant and is dropped here.

:func:`keywords_json` serialises a keyword list and :func:`object_tags`
assembles the per-object tag dict each function/table merges into its
``Meta.tags``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def keywords_json(keywords: Sequence[str]) -> str:
    """Serialise search keywords as a JSON array of strings (VGI138).

    Args:
        keywords: The individual search terms / synonyms for the object.

    Returns:
        A JSON array string such as ``["audio", "tempo"]`` suitable for the
        ``vgi.keywords`` tag value.
    """
    return json.dumps(list(keywords))


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: Sequence[str],
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    Args:
        title: Human-friendly display name (``vgi.title``); must differ from the
            object's machine name.
        doc_llm: Markdown narrative for LLM/agent audiences.
        doc_md: Markdown narrative for human documentation.
        keywords: Search terms / synonyms, serialised as a JSON array (VGI138).

    Returns:
        A tag dict ready to merge into a function/table ``Meta.tags``.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_json(keywords),
    }

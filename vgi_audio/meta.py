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


def example_queries_json(examples: Sequence[tuple[str, str]]) -> str:
    """Serialise ``(description, sql)`` pairs as a ``vgi.example_queries`` value.

    The native ``duckdb_functions().examples`` carrier only holds bare SQL --
    per-example descriptions are dropped there (VGI515). Carrying the examples in
    this JSON tag instead keeps each one's human-readable description; when both
    carriers hold the same SQL the linter dedupes to the described (tag) copy.

    Args:
        examples: ``(description, sql)`` pairs. The ``sql`` should match the
            object's ``Meta.examples`` SQL byte-for-byte so the two carriers
            dedupe to a single, described example.

    Returns:
        A JSON list of ``{"description": ..., "sql": ...}`` objects suitable for
        the ``vgi.example_queries`` tag value.
    """
    return json.dumps([{"description": description, "sql": sql} for description, sql in examples])


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: Sequence[str],
    category: str,
    example_queries: Sequence[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    Args:
        title: Human-friendly display name (``vgi.title``); must differ from the
            object's machine name.
        doc_llm: Markdown narrative for LLM/agent audiences.
        doc_md: Markdown narrative for human documentation.
        keywords: Search terms / synonyms, serialised as a JSON array (VGI138).
        category: The object's primary ``vgi.category`` -- one of the names in the
            schema's ``vgi.categories`` registry (VGI409/VGI411).
        example_queries: Optional ``(description, sql)`` pairs serialised into a
            ``vgi.example_queries`` tag so every example carries a description
            (VGI515). Omit for objects that carry the tag themselves.

    Returns:
        A tag dict ready to merge into a function/table ``Meta.tags``.
    """
    tags = {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_json(keywords),
        "vgi.category": category,
    }
    if example_queries:
        tags["vgi.example_queries"] = example_queries_json(example_queries)
    return tags

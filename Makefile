# vgi-audio — dev and test targets.
#
# Usage:
#   make test       # unit/integration (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit  # pytest only
#   make test-sql   # DuckDB sqllogictest .test files via haybarn-unittest
#   make lint       # ruff format check + ruff check + mypy
#
# test-sql is self-contained: it points VGI_AUDIO_WORKER at the worker run as
# a uv stdio subprocess (exactly how DuckDB drives it after ATTACH) and runs the
# files under test/sql/. haybarn-unittest is a uv tool:
#   uv tool install 'haybarn-unittest<2'   # installs ~/.local/bin/haybarn-unittest

# Worker command DuckDB uses for ATTACH (overridable). Use the project venv's
# python (the frozen, locked SDK from uv.lock) rather than `uv run <script>`:
# `uv run` on a PEP-723 script builds a separate, cached environment that can
# resolve a *different* vgi-python than the one the lockfile pins, so the venv
# python is the single source of truth for both the E2E and the metadata lint.
WORKER_STDIO    ?= $(CURDIR)/.venv/bin/python $(CURDIR)/audio_worker.py

# haybarn-unittest lives in the uv tools bin; keep it on PATH.
HAYBARN_BIN     ?= $(HOME)/.local/bin
TEST_DIR         = .
TEST_PATTERN     = test/sql/*

.PHONY: test test-unit test-sql lint

test: test-unit test-sql

test-unit:
	uv run pytest -q

test-sql:
	PATH="$(HAYBARN_BIN):$$PATH" \
		VGI_AUDIO_WORKER="$(WORKER_STDIO)" \
		haybarn-unittest --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_audio/

# llama-router build/test entry points.
#
# `make test` is the exact full-suite command: the deterministic command gate
# used by the Spectre City python-service workflow and the canonical way to
# verify any change to this repository.
#
# The interpreter is pinned explicitly because the suite must run in whatever
# environment the gate executes in, not only in an interactive shell. On
# Normandy the system `python3` (3.14) lacks the project's runtime deps,
# while python3.12 carries pyyaml and pytest (see requirements.txt).
PYTHON ?= python3.12

.PHONY: test

test:
	$(PYTHON) -m pytest tests/ -v

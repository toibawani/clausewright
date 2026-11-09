PY ?= .venv/bin/python

.PHONY: setup lint typecheck test safety check eval run

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pre_commit install

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest -q

safety:
	$(PY) scripts/check_commit_safety.py --all

check: lint typecheck test

eval:
	$(PY) -m clausewright.eval run --split dev

run:
	$(PY) -m uvicorn clausewright.api.app:app --host 127.0.0.1 --port 8000

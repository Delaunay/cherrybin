PROJECT = cherrybin

.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

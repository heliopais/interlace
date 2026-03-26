.PHONY: install test lint typecheck format check build clean

install:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

check: lint typecheck test

build:
	uv build

clean:
	rm -rf dist .mypy_cache .ruff_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

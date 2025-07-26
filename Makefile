vendor:
	uv sync

types:
	uv run mypy .

run:
	uv run python agent.py
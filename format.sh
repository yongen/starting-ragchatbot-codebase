#!/bin/bash
# Code formatting script

echo "Running black formatter..."
uv run black backend/ frontend/

echo "Running ruff linter and auto-fixes..."
uv run ruff check --fix backend/ frontend/

echo "✨ Code formatting complete!"

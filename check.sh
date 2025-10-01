#!/bin/bash
# Code quality check script

echo "🔍 Running code quality checks..."
echo ""

echo "1️⃣  Checking code formatting with black..."
uv run black --check backend/ frontend/
BLACK_EXIT=$?
echo ""

echo "2️⃣  Running ruff linter..."
uv run ruff check backend/ frontend/
RUFF_EXIT=$?
echo ""

echo "3️⃣  Running type checker with mypy..."
uv run mypy backend/
MYPY_EXIT=$?
echo ""

# Summary
echo "================================"
echo "Quality Check Summary"
echo "================================"

if [ $BLACK_EXIT -eq 0 ]; then
    echo "✅ Black formatting: PASSED"
else
    echo "❌ Black formatting: FAILED (run ./format.sh to fix)"
fi

if [ $RUFF_EXIT -eq 0 ]; then
    echo "✅ Ruff linting: PASSED"
else
    echo "❌ Ruff linting: FAILED (run ./format.sh to auto-fix)"
fi

if [ $MYPY_EXIT -eq 0 ]; then
    echo "✅ Mypy type checking: PASSED"
else
    echo "❌ Mypy type checking: FAILED"
fi

# Exit with error if any check failed
if [ $BLACK_EXIT -ne 0 ] || [ $RUFF_EXIT -ne 0 ] || [ $MYPY_EXIT -ne 0 ]; then
    exit 1
fi

echo ""
echo "🎉 All quality checks passed!"

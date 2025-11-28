
set -eu

# Python
echo "==> Python: ruff format/check"
ruff format pydowl
ruff check pydowl
ruff format tests
ruff check tests

echo "==> Python: mypy"
mypy --config-file ./mypy.ini

echo "==> Python: tests"
PYTHONPATH=. pytest tests -q

echo "All good ✅"

.PHONY: help install install-dev clean lint format type-check test build publish

help:  ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__/ .pytest_cache/ .coverage .mypy_cache/

lint:  ## Run linting (ruff, flake8, vulture)
	ruff check .
	flake8 .
	# Vulture: check only our code, ignore tests to reduce noise
	vulture innit innit_*.py --min-confidence 80 || true

lint-fix:  ## Fix linting issues
	ruff check --fix .

format:  ## Format code (black)
	black innit innit_*.py

format-check:  ## Check code formatting
	black --check innit innit_*.py

type-check:  ## Run type checking (mypy)
	mypy innit innit_detector.py innit_tinygrad.py innit_tinygrad_fixed.py innit_client.py

test:  ## Run tests with coverage
	pytest

quality:  ## Run all quality checks
	$(MAKE) format-check lint type-check

quality-fix:  ## Fix all quality issues
	$(MAKE) format lint-fix

build:  ## Build package
	python -m build

publish:  ## Publish to PyPI
	@if [ -z "$$PYPI_API_TOKEN" ]; then \
	  echo "❌ PYPI_API_TOKEN not set"; exit 1; \
	fi
	python -m pip install twine >/dev/null 2>&1 || true
	twine upload -u __token__ -p $$PYPI_API_TOKEN dist/*

publish-test:  ## Publish to TestPyPI
	@if [ -z "$$TEST_PYPI_API_TOKEN" ]; then \
	  echo "❌ TEST_PYPI_API_TOKEN not set"; exit 1; \
	fi
	python -m pip install twine >/dev/null 2>&1 || true
	twine upload --repository testpypi -u __token__ -p $$TEST_PYPI_API_TOKEN dist/*

check-model:  ## Test model download and inference
	python -c "from innit_detector import InnitDetector; d = InnitDetector(); print(d.predict('Hello world!'))"

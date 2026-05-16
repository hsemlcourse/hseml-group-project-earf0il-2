.PHONY: help install lint test test_deploy test_all format all clean nb-test

help:
	@echo "Available targets:"
	@echo "  install   - install dependencies from requirements.txt"
	@echo "  lint      - run flake8 over src/ and tests/"
	@echo "  test      - run pytest"
	@echo "  nb-test   - execute notebooks via papermill (smoke-test)"
	@echo "  all       - lint + test"
	@echo "  clean     - remove caches and __pycache__"

install:
	pip install -r requirements.txt

lint:
	flake8 src tests

test_all: test test_deploy nb-test

test:
	pytest -q tests

test_deploy:
	pytest -q deploy/tests

nb-test:
	papermill notebooks/01_eda.ipynb /tmp/01_eda_out.ipynb -k python3
	papermill notebooks/02_baseline.ipynb /tmp/02_baseline_out.ipynb -k python3

all: lint test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

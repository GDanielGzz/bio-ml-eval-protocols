.PHONY: help install test fast figures lint clean

help:
	@echo "make install   - editable install with dev extras"
	@echo "make test      - full suite, including the example scripts"
	@echo "make fast      - unit tests only (skips the ~1 min example runs)"
	@echo "make figures   - regenerate every figure the documents cite"
	@echo "make clean     - remove build artefacts and generated figures"

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

fast:
	python -m pytest -m "not slow"

figures:
	python examples/leakage_inflation.py
	python examples/paired_comparison.py

clean:
	rm -rf build dist *.egg-info .pytest_cache figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
